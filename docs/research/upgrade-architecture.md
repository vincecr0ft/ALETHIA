# ALETHIA upgrade architecture

Concrete design for the non-JEPA upgrades in [upgrade-plan.md](upgrade-plan.md).
JEPA (U14) is being executed by a parallel agent — that work lives in
[02-foundation-model/jepa-fm.md](02-foundation-model/jepa-fm.md) and
`experiments/intention-vs-deepsets/jepa_fm.py` and is intentionally
out-of-scope for this document.

This document fixes interfaces (function signatures, span attribute
keys, file edit sites) so every remaining upgrade can be implemented
against a stable contract.

## 0. Module map (what's new, what's modified)

```
modules/surrogate/intention/
  eigen.py            NEW    pure-function eigendecomp + LIS extraction on A
  fisher.py           NEW    empirical Fisher of the analytic oracle on c-prior
  acquisition.py      MODIFY add epig_acquire_m_eigen, importance-resampled pool
  drift.py            MODIFY kappa_drift returns (eig_state, ...) not just kappa
  calibration.py      —      unchanged
  model.py            MODIFY add A_eigen() helper (numpy)

modules/surrogate/
  oracle_smeft.py     MODIFY surface (sm, interf, bsm_sq) channels + joint score

experiments/full-chain-run/
  run.py              MODIFY emit chain.drift.eigen span; pool sampler swap
  identifiability_probe.py
                      MODIFY add Fisher-rotated probe; posterior-covariance probe
  bimodal_drift.py    NEW    U6 driver — bimodal withhold, EPIG vs random
  pfn_baseline.py     NEW    U3 PFN training + Table 1 row
  scaling_sweep.py    NEW    U11 sweep over d_psi
  eigen_query.py      NEW    Phoenix HTTP query implementing the MCP rubric

paper/alethia.tex     MODIFY new Section 5.4 + new Table 2 + Section 7 paragraphs
paper/refs.bib        MODIFY 16 additions (see upgrade-plan §8)
```

No file in `modules/surrogate/intention/` changes its public symbols
beyond *additions*. The Intention-FM training entry points in
`experiments/intention-vs-deepsets/` are touched only to add a new
script (`pfn_baseline.py`) — existing experiment scripts are not edited.
This keeps the diff small enough that the parallel JEPA work cannot
collide.

## 1. The spine: eigen.py

The load-bearing technical artefact is one new module. Pure functions
over numpy arrays; no torch, no Phoenix imports. This keeps it cheap
to call from drift, acquisition, the disclosure probe, and the
Phoenix-side query script.

### 1.1 Interface

```python
# modules/surrogate/intention/eigen.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EigenState:
    """Full eigendecomposition of the ridge design matrix A on a context.

    A = Psi(M_ctx).T @ Psi(M_ctx) + alpha * I   in R^{d x d}
    Eigenpairs are returned ascending by eigenvalue.
    """
    lam: np.ndarray      # (d,) eigenvalues, ascending
    U:   np.ndarray      # (d, d) eigenvectors as columns, U[:, j] <-> lam[j]
    alpha: float
    d: int

    @property
    def kappa(self) -> float:
        return float(self.lam[-1] / max(self.lam[0], 1e-30))

    def low_subspace(self, k: int) -> np.ndarray:
        """Return U_low: (d, k), the k smallest-eigenvalue directions."""
        return self.U[:, :k]

    def high_subspace(self, k: int) -> np.ndarray:
        return self.U[:, -k:]

    def lis_rank(self, tau: float = 1e-2) -> int:
        """# eigenvalues with lam_j > tau * lam_max."""
        thr = tau * self.lam[-1]
        return int(np.sum(self.lam > thr))


def eigen_state(Psi: np.ndarray, alpha: float) -> EigenState:
    """Eigendecompose Psi.T @ Psi + alpha I."""
    d = Psi.shape[1]
    A = Psi.T @ Psi + alpha * np.eye(d)
    lam, U = np.linalg.eigh(A)   # ascending
    return EigenState(lam=lam, U=U, alpha=alpha, d=d)


def eigen_resample_weights(
    Psi_pool: np.ndarray, low: np.ndarray, lam_low: np.ndarray
) -> np.ndarray:
    """Importance weights r(m) = sum_j (psi(m)·u_j)^2 / lam_j over low subspace.

    Psi_pool: (P, d). low: (d, k). lam_low: (k,). Returns (P,) >= 0.
    """
    proj = Psi_pool @ low                    # (P, k)
    w = (proj ** 2) / np.clip(lam_low, 1e-30, None)
    return w.sum(axis=1)


def eigenvector_stability(U_prev: np.ndarray, U_curr: np.ndarray) -> np.ndarray:
    """|cos(u_prev,j, u_curr,j)| for each column j, after sign-flip alignment.

    Returns (k,) in [0, 1]; near 1 = stable, near 0 = eigenvector rotated.
    """
    cos = np.einsum("dj,dj->j", U_prev, U_curr)
    return np.abs(cos)
```

### 1.2 IntentionFM helper

`modules/surrogate/intention/model.py` gains one method:

```python
@torch.no_grad()
def A_eigen(self, M_ctx: np.ndarray) -> EigenState:
    """Convenience: eigen_state(self.psi_np(M_ctx), self.alpha)."""
    from .eigen import eigen_state
    return eigen_state(self.psi_np(M_ctx), self.alpha)
```

`kappa_A` is unchanged for backwards-compat but is now expressible as
`self.A_eigen(M_ctx).kappa`. No callers are forced to migrate.

## 2. Drift integration: spans + new return contract

### 2.1 kappa_drift becomes structured

Currently `kappa_drift(...) -> (fired, kappa, proj_ratio)`. Change to:

```python
def kappa_drift(
    model, M_ctx, recent_M, train_proj_var, *,
    kappa_threshold=1e4, proj_ratio_threshold=3.0,
) -> tuple[bool, float, float, EigenState]:
    """As before, plus the full EigenState on the current context."""
```

This is the *only* drift.py edit. The new fourth return value is the
single payload that every downstream upgrade (U5, U6, U7, U10) consumes.

### 2.2 chain.drift.eigen Phoenix span

Emitted inside `experiments/full-chain-run/run.py` immediately after
the `kappa_drift` call (replacing the existing inline attributes block
around line ~316). Attribute schema:

```
aletheia.eigen.lambda           float[d]    full ascending spectrum of A
aletheia.eigen.U_top             float[d,k] top-k eigenvectors (cols)
aletheia.eigen.U_low             float[d,k] bottom-k eigenvectors (cols)
aletheia.eigen.lam_top_k         float[k]    eigenvalues paired with U_top
aletheia.eigen.lam_low_k         float[k]    eigenvalues paired with U_low
aletheia.eigen.kappa             float       lam_max / lam_min (existing)
aletheia.eigen.lis_rank          int         # eigenvalues > 1e-2 * lam_max
aletheia.eigen.stability_top      float[k]   |cos| with previous cycle's U_top
aletheia.eigen.stability_low      float[k]   |cos| with previous cycle's U_low
aletheia.drift.cov.fired         bool        (existing)
aletheia.drift.cov.kappa         float       (existing)
aletheia.drift.cov.vmin_projection_ratio
                                 float       (existing)
```

`k = 3` is the default for top/low — small enough to fit comfortably
in span attributes (Phoenix span attributes are not designed for large
tensors), large enough to span the typical leading and trailing eigen-
direction sets at d_psi = 16. The full `lambda` vector is small enough
(d ≤ 64 in any plausible run) that it can ride as a flat list attribute.

### 2.3 The eigen_query.py MCP-equivalent

A standalone script under `experiments/full-chain-run/eigen_query.py`
that hits Phoenix's HTTP API and implements the rubric described in
upgrade-plan §5.2:

```
eigen_query.py --since-cycle T0 --theta 1e-3 --top-k 3
  -> prints JSON {
       "fired_cycles": [...],
       "u_low_union": float[d, m],     # column-orthonormalised union
       "lambda_low_union": float[m],
     }
```

This is the artefact that gets demonstrated as "Phoenix-as-control-
plane". It is callable from a notebook, from MCP if the wiring lands,
and from `bimodal_drift.py` to feed eigen-redirected acquisition off
real span history rather than the current-cycle decomposition.

## 3. Eigen-redirected acquisition

`modules/surrogate/intention/acquisition.py` gains one function:

```python
def epig_acquire_m_eigen(
    model, M_ctx, Y_ctx, M_pool, M_target, *,
    eigen: EigenState, k_low: int = 3, k_pick: int = 5,
    resample: bool = True, rng: np.random.Generator,
) -> np.ndarray:
    """EPIG acquisition with eigen-redirected candidate pool.

    1. Compute importance weights r(m) over M_pool from eigen.low_subspace(k_low).
    2. If resample: redraw the pool (with replacement) proportional to r(m),
       then score the redrawn pool with the existing EPIG criterion.
       Otherwise: weight EPIG scores by r(m) directly.
    3. Return the indices into M_pool of the k_pick selected points
       (de-duplicated under resample).
    """
```

`run.py` integration: at the existing site around lines 358–370,
add a third branch alongside `random` and `leverage`:

```python
elif ACQUISITION == "epig_eigen":
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    eig = model.A_eigen(M_ctx)
    picked = epig_acquire_m_eigen(
        model, M_ctx, Y_ctx, M_pool, M_target,
        eigen=eig, k_low=3, k_pick=K_EPIG, rng=rng)
```

The existing `ACQUISITION` env var grows one accepted value; no other
behavior changes. Sherman-Morrison context update, conformal refit,
drift detectors — all unchanged.

## 4. Fisher-eigenbasis disclosure (U7)

### 4.1 fisher.py

```python
# modules/surrogate/intention/fisher.py
"""Empirical Fisher information of an oracle on a c-prior, in Wilson space.

This is the disclosure-side companion to eigen.py. eigen.py rotates the
acquisition geometry into the eigenbasis of the *model's* design matrix
A. fisher.py rotates the disclosure target into the eigenbasis of the
*oracle's* Fisher information on c. They are independent rotations
serving different upgrades (U5 vs U7), kept in separate modules to
make that explicit.
"""
from __future__ import annotations
import numpy as np

def empirical_fisher_c(
    oracle, c_samples: np.ndarray, m_samples: np.ndarray, *,
    fd_step: float = 1e-3,
) -> np.ndarray:
    """E[ grad_c log mu(c, m) (grad_c log mu(c, m)).T ] over the c-prior.

    Estimated by central finite differences on oracle.truth(c, m).
    Returns (N_WC, N_WC) symmetric PSD matrix.
    """

def fisher_basis(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose F = V D V.T. Returns (D_desc, V_desc), descending."""
```

The choice of finite-difference for the gradient (vs autograd through
`oracle.truth`) is deliberate: the analytic oracle wraps a numpy
calculator that does not back-propagate. A 4×2-evaluation finite-
difference is cheap enough at the typical N=400 c-prior sample, and
keeps `fisher.py` import-free of torch.

### 4.2 Disclosure probe rotation

`experiments/full-chain-run/identifiability_probe.py` gains a second
phase. The current code (the c-basis probe at lines ~94–157) stays as
a backstop; a new section appends:

```
=== Phase 2: Fisher-eigenbasis probe ===
1. Sample c_prior ~ U([-0.7, 0.7]^4) excluding the withheld band.
2. Compute F = empirical_fisher_c(oracle, c_prior, m_prior).
3. D, V = fisher_basis(F).   # V columns = c-tilde directions, descending D
4. Rotate: c_tilde_train = c_train @ V; c_tilde_in = c_in @ V; c_tilde_out = c_out @ V
5. Probe: w_train -> c_tilde_train via RidgeCV
6. Report per-eigendirection R^2 on inside and withheld.
```

`identifiability_summary.json` gains a sibling block:

```json
"fisher_basis": {
  "eigenvalues": [...],
  "V": [[...], [...], ...],
  "r2_inside_tilde":   {"c_tilde_1": ..., ..., "c_tilde_4": ...},
  "r2_withheld_tilde": {...}
}
```

### 4.3 Posterior-covariance disclosure (U10)

In the same probe file. Replace `RidgeCV` with a Bayesian linear
regression that returns (mean, cov). Use `sklearn.linear_model.BayesianRidge`
per-output target, OR the closed-form posterior from a Gaussian prior
matching the selected RidgeCV `alpha_`:

```
posterior_mean = (W.T W + alpha I)^{-1} W.T C_train
posterior_cov  = sigma_hat^2 * (W.T W + alpha I)^{-1}
```

For each held-out scenario report:
- point estimate (existing)
- 68%/95% credible interval per c-direction (new)
- empirical coverage of the 68% interval across the held-out set (new)

This is what turns the disclosure result into an SBI claim.

## 5. PFN baseline (U3)

`experiments/full-chain-run/pfn_baseline.py` (NEW). Self-contained:

```
1. Sample N_train = 500 scenarios from the same SMEFT prior used by
   Intention pretraining; for each, draw K=12 context points and a
   query block of Q=32 from the analytic oracle.
2. Train a transformer ICL head:
     - input: (M_ctx, Y_ctx, M_q), serialised as a (K + Q, 2)-token
       sequence where context tokens carry (m, y) and query tokens
       carry (m, 0) with a binary "is-query" flag.
     - 2 transformer layers, 4 heads, d_model = 32, GELU MLP.
     - Output head: per-query y_hat scalar.
     - Param count target: 5328 +/- 20% to match the headline Intention model.
3. Train with Adam lr=1e-3, 1500 steps, batch=24.
4. Evaluate on the same 50 in-distribution + 50 out-of-distribution
   scenarios as Table 1. Emit table_1_pfn_row.json with median R^2,
   p5 R^2 (in / out) and MSE.
```

The PFN row lands in Table 1 (`tab:headline`) as a fifth architecture
between DeepSets and the cheat regressor. The discussion paragraph in
Section 7 is rewritten depending on outcome:

- if Intention beats PFN by > 0.05 in p5 R^2: keep closed-form-ridge
  framing, PFN sits alongside DeepSets as "generic ICL that does not
  exploit the polynomial structure".
- if PFN matches Intention within 0.02 in p5 R^2: reframe Section 7
  to position ALETHIA as the closed-form-ridge specialisation of PFNs
  on regression problems (i.e. the contribution is the *constraint* +
  the *closed-form head*, not "the only architecture that works").

The rewrite is a paragraph either way; the decision is made once the
PFN numbers exist (Phase 3, no sooner).

## 6. Bimodal drift experiment (U6)

`experiments/full-chain-run/bimodal_drift.py` (NEW). Driver script,
reuses `run.py` machinery wherever possible.

```
1. Define a new pretraining mask: exclude
     |c_lq^(3)| in [0.6, 1.0]  AND simultaneously
     |c_Hq^(3)| in [0.4, 0.8].
   This is a 2D withhold rather than the current 1D.

2. Retrain psi_theta on the masked prior. Save to
   experiments/full-chain-run/output_bimodal/intention_fm.pt.

3. Loop config matches run.py but with target_c = (0, -0.5, 0.8, 0)
   (both withheld operators active simultaneously) and ORACLE_BUDGET=500.

4. Run THREE configurations under identical seeds:
     ACQUISITION=random
     ACQUISITION=epig
     ACQUISITION=epig_eigen

5. Plot rmse_band_trace for all three on one axis (the recovery
   curve). Save to docs/research/plots/bimodal_recovery.png.

6. Emit bimodal_drift_summary.json with cycle-to-recovery, final RMSE,
   and area-under-curve metrics for each acquisition mode.
```

Success criterion (upgrade-plan §U6): `epig_eigen` visibly beats
`random` on bimodal. If not, the Section 6.4 rewrite frames the
result as a designed null — "we built a candidate failure-mode probe
and EPIG = random even under bimodal drift on 1D m_ll" — which is
itself publishable.

## 7. Morphing channels + joint score (U8, U9)

### 7.1 Surface morphing channels in the oracle

`modules/surrogate/oracle_smeft.py` gains a method:

```python
def truth_channels(self, c, m) -> dict[str, np.ndarray]:
    """Return {'sm_only', 'interference', 'bsm_squared'} per-event.

    Thin wrapper around modules.analytic_smeft.smeft.differential_xs
    that returns the named channels rather than collapsing them to mu.
    Shape: each value is (n,).
    """
```

Zero training cost; only used by U9 and (potentially) by JEPA-style
self-supervision in the future.

### 7.2 Joint-score auxiliary loss

In `experiments/intention-vs-deepsets/intention_learned.py` (and the
training loop in `experiments/full-chain-run/run.py` that pretrains
Intention before the closed-loop trial), add an auxiliary loss term:

```
L = L_mse + lambda_score * L_score
L_score = || nabla_c (linear_probe(w_implicit(c)) - c) ||^2
        averaged over a small batch of c-perturbations per training scenario
```

The joint score `grad_c log mu(c, m)` is available analytically from
the morphing polynomial; the auxiliary loss says "the probe's output
should track c such that its gradient w.r.t. c matches the analytic
score". This is the Brehmer-Cranmer-Louppe-Pavez "Mining Gold" trick
specialised to the Intention head.

The aux loss is opt-in via a CLI flag `--joint-score-aux` so that
the existing run results are not invalidated; the baseline-vs-aux
comparison becomes its own ablation row.

## 8. Scaling sweep (U11)

`experiments/full-chain-run/scaling_sweep.py` (NEW). One overnight run:

```
for d_psi in [4, 8, 16, 32, 64, 128]:
    train psi_theta for 1500 steps, 1 seed (or 3 if budget allows)
    eval held-out median R^2 in-box and out-of-box
    eval disclosure probe per-operator R^2
    write scaling_sweep_summary.json[d_psi] = {...}
plot:
  panel A: median R^2 vs d_psi
  panel B: per-operator disclosure R^2 vs d_psi
save: docs/research/plots/scaling_sweep.png
```

One-panel scaling figure in Section 7 (or Appendix B, depending on
space).

## 9. Paper edits (concrete)

### 9.1 New Section 5.4 "Operator-level disclosure on the Fisher eigenbasis"

Inserted after Section 5.3 (the learned-basis analysis), before
Section 6 (the closed-loop section). ~1.5 pages of text plus the new
Table 2 (full operator disclosure, both raw-c-basis and Fisher-
rotated) and a new Figure (predicted vs true scatter, both bases).

Sourcing rule: every number in the table caption must trace back to
`experiments/full-chain-run/output/identifiability_summary.json` (the
5328-param run). The `output_supercharged/` numbers from the 20896-
param run remain available in the ablation section but are clearly
labelled as such.

### 9.2 Section 7 paragraphs

Three additions, each ~one paragraph:

- Concurrent SMEFT FM (arXiv:2512.15862): "Concurrently with this
  work, [author] et al. propose a SMEFT-tailored ICL transformer
  trained on [scope]. Their architecture is a standard attention head
  with c-conditioning [confirm at read time]; ALETHIA's contribution
  is the closed-form-ridge head, the explicit no-c constraint, and
  the closed-loop runtime."

- TabPFN-as-SBI (arXiv:2504.17660): "Schmitt and Radev position TabPFN
  as a posterior-predictive SBI engine; ALETHIA is structurally the
  closed-form-ridge specialisation of that approach on a physics
  regression target. The closed form is what enables disclosure
  probing in the Fisher eigenbasis (Section 5.4) and the Sherman-
  Morrison update in the closed-loop runtime (Section 6)."

- Fitmaker / Fisher-eigenbasis in SMEFT (Costa-Marzocca-Mimasu-Salko,
  Ellis-Madigan-Mimasu-Sanz-You): one sentence locating ALETHIA in
  the SMEFT-global-fit Fisher-eigenbasis tradition.

### 9.3 Bibliography

Sixteen additions per upgrade-plan §8. Group them with existing
\bibitems by section (AL / SBI / ICL / Fisher / physics).

## 10. Execution order (dependency graph, not a calendar)

The work is organised as four phases plus a parallel paper-only track.
A phase opens when every gating item from the prior phase is satisfied;
within a phase the items can run concurrently. The JEPA agent runs
independently against `experiments/intention-vs-deepsets/jepa_fm.py`
and is not blocked by anything in this plan.

### Phase 0 — Spine (gating: blocks everything downstream)

Nothing else proceeds until these exist and are tested.

- `modules/surrogate/intention/eigen.py` with `EigenState`,
  `eigen_state`, `eigen_resample_weights`, `eigenvector_stability`.
- `modules/surrogate/intention/fisher.py` with `empirical_fisher_c`,
  `fisher_basis`.
- `IntentionFM.A_eigen(M_ctx)` helper on `model.py`.
- Unit tests over each: round-trip eigendecomposition, importance-
  weight non-negativity, Fisher-basis orthonormality.

### Phase 1 — Span emission and acquisition rewire

Unblocked once Phase 0 is in. These items can run in parallel.

- Wire `chain.drift.eigen` span in `experiments/full-chain-run/run.py`
  with the attribute schema of §2.2; update `kappa_drift` to return
  `EigenState` (§2.1).
- `epig_acquire_m_eigen` in `acquisition.py`; the
  `ACQUISITION=epig_eigen` branch in `run.py` (§3).
- `experiments/full-chain-run/eigen_query.py` — Phoenix HTTP rubric
  (§2.3). Verifies the span schema round-trips.
- Smoke-run the chain with `ACQUISITION=epig_eigen`; confirm spans
  arrive in Phoenix and the chain still recovers.

### Phase 2 — Disclosure-side rotation + posterior covariance

Unblocked once Phase 0 is in (independent of Phase 1).

- Fisher-basis rotation phase in `identifiability_probe.py` (§4.2);
  re-run on the existing 5328-param model.
- Posterior-covariance probe (§4.3); 68%/95% credible intervals +
  empirical-coverage report.

### Phase 3 — Designed experiments and ablations

Unblocked once Phases 1 and 2 are in. Each item below is independent.

- **PFN baseline** (§5). Standalone training run; lands a row in
  Table 1. Branches Section 7's framing based on outcome.
- **Bimodal drift experiment** (§6). Retrain on the 2D mask, then
  run `ACQUISITION=random`, `epig`, `epig_eigen` under shared seeds.
  Drives the Section 6.4 rewrite.
- **Morphing channels** in the oracle (§7.1) → enables joint-score
  aux loss + retrain + disclosure re-run (§7.2).
- **Scaling sweep** over `d_psi` (§8). Cheapest item; can run
  asynchronously the moment Phase 0 lands.

### Parallel — Paper-only track

Independent of every code phase. The only gating dependency is that
final numbers exist in the relevant JSON before they get cited.

- U1 + U2: new Section 5.4 + Table 2, sourced from
  `experiments/full-chain-run/output/identifiability_summary.json`
  (5328-param run).
- U4 + U12: Section 7 paragraphs (concurrent SMEFT FM, TabPFN-as-SBI,
  fitmaker / Fisher-eigenbasis in SMEFT).
- Bibliography additions (16 entries, upgrade-plan §8).
- Eigen-trajectory figure (§5.5) — drawn from a recorded chain run
  with `epig_eigen`.

### Cut order if scope contracts

If something has to drop, the priority for keeping is U1, U2, U3,
U4, U5, U7, U10 — these address every audit criticism and complete
the eigen-direction spine. U6 (bimodal), U9 (joint score), U11
(scaling), U8 (morphing channels exposed but unused) all drop
cleanly without invalidating the paper.

### External deadline

Hackathon submission: **2026-06-11, 14:00 PT**. That is the only
calendar fact this document acknowledges; everything above is a
dependency graph, not a calendar.

## 11. Risk register

| Risk | Mitigation |
|------|------------|
| PFN matches Intention in p5 R^2 (U3) | Reframe Section 7 paragraph; closed-form-ridge specialisation framing is a defensible contribution either way |
| epig_eigen = random on bimodal (U6) | Frame as designed null result; Section 6.4 rewrites cleanly around it |
| Joint-score aux fails to lift c_Hq disclosure (U9) | Demote to Tier-3; aux loss becomes negative ablation result |
| Phoenix span attribute size limits hit on full lambda vector | Truncate to top-k + low-k in span; store full lambda only in run.log |
| MCP wiring slips (U5) | eigen_query.py runs against Phoenix HTTP API directly; MCP demoed as design, not live |
| JEPA agent's experiment_jepa.py touches model.py | None expected — JEPA is its own training script alongside existing experiments; only collision risk is jepa_fm.py if it imports IntentionFM. Coordinate at the paper-track merge point. |

## 12. What this document is not

This is interface design, not a literature review (see upgrade-plan
§1–§8 for that) and not a paper draft (the .tex prose lives in
paper/alethia.tex). The unit of change for every item here is a
named file under a clear path with a concrete signature; this is the
contract the implementation work hangs off.
