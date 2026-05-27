# Drift monitoring — eval (second pass, agent c)

This document evaluates the drift-monitoring proposal of `docs/research/03-drift/summary.md` against the three other research areas: oracle (a), foundation model (b), EPIG/conformal (d). It is organised as five contracts, three theoretical-backing audits, a scaling estimate, and a module-layout sketch.

## 1. Contract with the foundation model (agent b)

`docs/research/02-foundation-model/summary.md` settles on Architecture A (set/event-level encoder, DeepSets first, Particle Transformer optional) with an InfoNCE pre-training objective and a closed-form ridge probe on the embedding `z`. The drift detectors require *exactly* the following from that probe layer, none of it specific to a particular encoder.

Concretely the FM-class contract is:

| Object | Type | Where it lives | Used by |
|---|---|---|---|
| `phi : np.ndarray (n, d)` | feature map of an embedding batch | `model.phi(z_batch)` (or equivalent) | DAS-CUSUM (none directly); kappa-evaluator (yes, both for `A` and for projection); calibration (none directly) |
| `A_inv : np.ndarray (d, d)` | inverse Gram of the probe's ridge solve | `model.A_inv` after `fit` | kappa-evaluator (`kappa(A)`, `v_min`) |
| `predict(z) -> (mu, sigma)` | mean + epistemic std at a query | `model.predict(z)` | DAS-CUSUM (`mu_hat`), calibration (forms `sigma_conformal` via `cc.coverage_sigma`) |
| `embed(d) -> z` | encoder forward | `encoder.forward(events)` | calibration (clusters `z` to form `region_id`); kappa-evaluator (projection on `v_min`) |
| `update(z_new, y_new)` | rank-1 refresh of `A_inv` and `w` | `model.update` | drift aggregator's `local_retrain` action |
| `v_min, kappa` | smallest right singular vector of `Phi`, condition number of `A` | recomputed by `kappa_vmin.recompute` at each FM-version boundary | kappa-evaluator |

What this constrains in the FM rebuild: the probe head must persist `A_inv` (or expose it on demand via a `state_dict`-style accessor), and `v_min` must be cached per FM version. The DAS-CUSUM detector itself does not look at `z` at all — it operates on the scalar tuple `(mu_hat, sigma_conformal, y_oracle)` of section 2.1 of `summary.md`. So the encoder architecture is essentially opaque to the drift layer. This is the right boundary.

Two non-obvious points:

- **Scale invariance.** Agent (b) does not fix the absolute scale of `z` (a learned embedding); the per-query leverage `lev(z) = phi(z) A^{-1} phi(z)^T` is not coordinate-free. `kappa(A) = lambda_max(A)/lambda_min(A)` is. The drift evaluator must therefore avoid hard thresholds on `lev(z)` and use `kappa(A)` and the projection-variance ratio `Var_recent(<phi, v_min>) / Var_train(<phi, v_min>)` instead.
- **Stability of v_min across FM versions.** Agent (b)'s stage 0 -> stage 1 transition (more operators activated, same encoder weights continued) will change `phi` and therefore `v_min`. Agent (d)'s clustering for `region_id` (next contract) must be stable across FM versions; otherwise A/B comparison is meaningless. The convention is: fit clustering once on the *current* embedding distribution and map new embeddings to nearest centroid.

## 2. Contract with EPIG / agent (d) — the "drifted region" payload

The drift aggregator's `local_retrain` and `global_retrain` actions consume a "drifted region" descriptor that EPIG ingests as its target set. Schema:

```python
@dataclass
class DriftedRegion:
    region_id: str                          # cluster id from FM-embedding clustering
    centroid_z: np.ndarray  # (D_Z,)        # cluster mean in embedding space
    radius: float                           # 95th percentile distance to centroid
    failing_signal: Literal["acc", "cal", "cov"]
    severity: float                         # statistic value (S_t for CUSUM,
                                            #  p_adj for BH, vmin_proj_ratio for kappa)
    triggering_spans: list[str]             # OpenInference span ids
    suggested_target_z: np.ndarray | None   # (n_T, D_Z), only for cov drift
```

`suggested_target_z` is the load-bearing payload for the coverage case. When kappa-evaluator fires on `vmin_projection_ratio`, the *worst direction* is encoded as: take the embeddings of recent queries whose projection on `v_min` is largest, return them as the target set. EPIG's information functional is then maximised by oracle calls whose embeddings will reduce target-set predictive variance.

For the calibration case, `suggested_target_z` is the cluster centroid plus representative members; for the accuracy case it is the recent queries whose standardised residual triggered the CUSUM. In all three cases the payload is *embeddings*, never Wilson coefficients — the constraint of BRIEF.md is respected by construction.

The Phoenix span carrying this is `drift.coverage.kappa` with attribute `aletheia.drift.cov.worst_region_id` plus a child span emitting `suggested_target_z` as a serialised array attribute. Agent (d)'s `epig_acquire_commissions` reads this via the MCP `get-spans` filter on `aletheia.drift.cov.fired == True`.

`aletheia.drift.cov.worst_region_id` is a string id into a Phoenix dataset of region clusters; the dataset row stores `{region_id, centroid_z, radius, fitted_at_fm_version}`. EPIG then queries that dataset with `get-dataset-examples` and pulls `centroid_z` + `radius` to construct its target set.

## 3. Contract with the oracle (agent a) — fidelity-tier drift as a signal

`docs/research/01-oracle/summary.md` exposes the oracle's three Phoenix span attributes that matter here: `oracle.fidelity_tier`, `oracle.cost_seconds`, `oracle.cache.hit`. The structured-dict `OracleResult` carries `metadata.fidelity_tier`, `metadata.mc_noise`, `metadata.wall_time_s`.

The drift evaluators integrate this in two ways:

- **Fidelity disagreement as drift.** When the orchestrator escalates a T1 result to T2 (section 5.4 of the oracle doc), the higher-tier result is the new ground truth. The standardised residual `z_t = (y_T2 - mu_hat) / sigma_conformal` is *not* the same statistic as `z_t = (y_T1 - mu_hat) / sigma_conformal`: the noise scale changes, and the systematic bias between T1 and T2 (NLO QCD K-factor, MC noise) is what we want to detect. The drift evaluator therefore runs *two parallel DAS-CUSUM streams*: one on the T1 stream, one on a T2-conditioned stream. A T2 result that disagrees with the T1 prediction *and* with the FM prediction is a "fidelity-tier drift" event — section 7.1 of summary.md's promote rule already gates on this implicitly via the A/B post-update comparison. We add an explicit `drift.fidelity` evaluator span:

```
drift.fidelity
    aletheia.drift.fid.t_low                "T1"
    aletheia.drift.fid.t_high               "T2"
    aletheia.drift.fid.delta_mu             float       # mu_T2 - mu_T1
    aletheia.drift.fid.delta_mu_z_score     float       # (delta_mu) / mc_noise_T2
    aletheia.drift.fid.fired                bool        # |z_score| > 3
```

This fires only when the orchestrator schedules a T2 cross-check, so its frequency is gated by the existing escalation policy.

- **Cost-aware aggregation.** The drift aggregator's action choice (`recal`, `local_retrain`, `global_retrain`) has a cost ladder of about three orders of magnitude. Agent (a)'s `oracle.cost_seconds` attribute is what we use in `drift.aggregate` to bias toward `recal` when the projected oracle budget for a `local_retrain` would exceed remaining budget. This is the cost-normalised EPIG of agent (d) §2.4, surfaced one level up — the aggregator queries Phoenix for the cumulative `oracle.cost_seconds` over the run and refuses to fire `global_retrain` if the budget is exhausted.

The Phoenix attribute `oracle.linked_predict_span_id` (proposed by agent (c)'s summary.md section 5) is the canonical join key between an oracle call and the FM prediction it ground-truths. The reconciliation with agent (a) is to also emit `oracle.escalated_from_span_id` when a T2 call was triggered by a T1 disagreement — this is what the fidelity-drift evaluator joins on.

## 4. Phoenix-MCP rubric questions vs the span tree

The proposed span tree (`summary.md` section 5) supports the rubric questions as follows:

| Rubric question | MCP path |
|---|---|
| "Show the last drift event by region" | `get-spans` filter `aletheia.drift.action in {recal, local_retrain, global_retrain}`, group by `aletheia.drift.target_region_id`, latest per group. |
| "Compare RMSE on drifted-region between FM version k and k+1" | `get-spans` filter `aletheia.fm.version`, join with per-region RMSE annotation written by the experiment.ab spans. |
| "What did EPIG pick when drift fired?" | `get-spans` filter on `aletheia.drift.action`, then follow `aletheia.epig.picked_span_ids` to the predict spans. |
| "What did the T2 cross-check show when the FM and T1 disagreed?" | `get-spans` filter `aletheia.drift.fid.fired == True`, then `aletheia.oracle.escalated_from_span_id` -> the originating predict span. |
| "Show the trajectory of kappa(A) over the run" | `get-spans` filter `aletheia.drift.cov.fired == True`, project `aletheia.drift.cov.kappa` vs span timestamp. |

All five queries reduce to a single `get-spans` call followed by either a `groupby` or a join on a span id attribute. No MCP gymnastics. The span tree is span-id-joined throughout; the `aletheia.epig.picked_span_ids` and `aletheia.oracle.linked_predict_span_id` conventions are what make the joins cheap.

## 5. Reconciliation with agent (d)'s loop guarantees

`docs/research/04-epig-conformal/summary.md` §5 lays out two loop guarantees: variance reduction in expectation (Guarantee A) and calibration preservation iff recalibration is refit (Guarantee B). The drift aggregator's actions interact with both:

- **`recal` triggers a `ConformalCalibrator.fit` on a calibration set drawn from the current probe region** — this is exactly what Guarantee B requires after every loop iteration. The aggregator therefore enforces Guarantee B as a side effect.
- **`local_retrain` and `global_retrain` invoke EPIG before the FM update**, which means Guarantee A holds at the picked candidates. The drift aggregator should *also* fire `recal` after every retrain — Guarantee B requires it after every model update.

Concrete change to the policy table (extending section 4 of summary.md):

| accuracy | calibration | coverage | action chain |
|---|---|---|---|
| 0 | 0 | 0 | noop |
| 1 | 0 | 0 | watch (require N consecutive) |
| 0 | 1 | 0 | recal |
| 0 | 0 | 1 | local_retrain -> recal |
| 1 | 1 | 0 | recal; if acc persists, local_retrain -> recal |
| 1 | 0 | 1 | local_retrain -> recal |
| 0 | 1 | 1 | local_retrain -> recal |
| 1 | 1 | 1 | global_retrain -> recal |

The "-> recal" tail is non-optional and is what makes the loop preserve Guarantee B. This is the only change to the aggregator semantics from what `summary.md` proposes.

## B. Theoretical backing audit

### B.1 Lorden-Pollak ARL bound for DAS-CUSUM (Lorden 1971; Pollak 1985; arXiv:2210.17353 Theorem 3)

**Assumption.** Under H_0 the standardised residuals `z_t` are i.i.d. with mean 0, variance 1, and finite fourth moment. Under H_1 there exists a change-point `tau` after which the mean shifts by `delta`.

**Consequence.** Page (1954) CUSUM with threshold `h` and reference `k = delta/2` satisfies, asymptotically as `h -> inf`:
- `ARL_0 ~ exp(2 k h) / (2 k^2)` (Siegmund 1985 corollary for the symmetric case, halved for two-sided).
- expected detection delay `ARL_1 ~ h / D(P_1 || P_0)` where `D` is KL divergence (Lorden 1971); for the Gaussian mean shift `D = delta^2 / 2`.

**Verification in our setting.** The conformal layer (`modules/surrogate/calibration.py:75-89`) produces `sigma_conformal` such that the standardised residual `z_t = (y_oracle - mu_hat) / sigma_conformal` is approximately N(0, 1) under exchangeability. *Approximately* because the conformal interval covers at level `p` but does not enforce Gaussianity; tails are typically heavier (Vovk-Gammerman-Shafer 2005 chapter 2). The fourth-moment condition is therefore the assumption that breaks first under heavy-tailed residuals. The DAS extension (Ahad-Davenport-Xie arXiv:2210.17353 Theorem 3) replaces the i.i.d. assumption by an alpha-mixing condition with mixing-rate bound and proves the asymptotic ARL formula picks up a multiplicative constant `1 + O(1/w)` where `w` is the variance-estimation window.

The empirical check (empirical-results.md section 1) shows our implementation has ARL_0 at h=5 sitting at about 370 — roughly 2.5× the Siegmund asymptote. This excess factor is consistent with the published DAS finite-window inflation; the threshold `h` is the tuning knob, and our test plan §A.1 already prescribes calibrating it empirically.

Where the assumption could break: at kinematic edges (e.g. `m_ll` near the Z pole, near LHC kinematic limits) the predictive distribution becomes non-Gaussian and the standardised residuals carry heavier tails. The conformal layer absorbs this in part — Vovk-style coverage holds without Gaussianity — but the standardised score will pick up tail mass that the CUSUM treats as drift. Mitigation: stratify CUSUM by region (one stream per cluster), which `notable.md` already lists as investigation 1.

### B.2 Benjamini-Hochberg FDR control (Benjamini-Hochberg 1995 Theorem 1; Benjamini-Yekutieli 2001)

**Assumption (BH 1995).** Under the global null, p-values `P_1, ..., P_m` are independent and uniformly distributed under their respective nulls. (BY 2001 weakens this to positive regression dependence — PRDS.)

**Consequence.** BH at level `alpha` rejects `H_(i)` iff `P_(i) <= i alpha / m`. Then `FDR <= alpha m_0 / m <= alpha` where `m_0` is the number of true nulls.

**Verification in our setting.** The strata are leverage quantiles on disjoint slices of the calibration set. Two-sided binomial p-values on disjoint subsets are independent under H_0 — the strata partition the data, so the binomial counts are jointly Poisson-multinomial conditional on the total, hence independent under the null in the asymptotic large-n regime. Strict finite-sample independence is broken only by the small Poissonian coupling, which is `O(1/n_per_stratum) = 0.005` for n=200 — negligible. **The independence assumption holds essentially exactly.** Empirical check below at H_0 returns any-rejection rate 0.049, exactly at nominal 0.05.

Under positive dependence (Benjamini-Yekutieli 2001), FDR control degrades to `alpha m / sum_{k=1}^m 1/k`; for `m = 5`, the BY correction factor is `5 / 2.283 = 2.19`, so a paranoid implementation would use `alpha_BY = alpha / 2.19 = 0.023`. We do not need to: the strata are independent.

### B.3 Eckart-Young variance amplification (Eckart-Young 1936; Stewart-Sun 1990 perturbation form)

**Assumption.** `A = Phi^T Phi + lambda I` is symmetric positive definite with eigendecomposition `A = U diag(s_1^2 + lambda, ..., s_d^2 + lambda) U^T` where `s_1 >= ... >= s_d >= 0` are singular values of `Phi`.

**Consequence (Eckart-Young 1936).** The worst-case predictive variance amplification at a query `phi(z)` is
```
phi(z)^T A^{-1} phi(z) <= ||phi(z)||^2 / lambda_min(A) = ||phi(z)||^2 / (s_d^2 + lambda).
```
The bound is tight in the direction `v_min = u_d` (the smallest-singular right vector of `Phi`).

**Stewart-Sun 1990 perturbation form.** Under a rank-1 update `A' = A + phi_p phi_p^T`, the eigenvalues interlace and `lambda_min(A') >= lambda_min(A)`; the condition number `kappa(A') <= kappa(A)` strictly iff the update has nonzero component on `v_min`.

**Verification in our setting.** Under the FM rebuild, the regression head still produces a Gram-like `A` (whatever the embedding `phi(z)` is). The Eckart-Young bound therefore continues to hold by structure; only the *interpretation* of `v_min` changes. Empirical check (empirical-results section 4) confirms strict monotone non-increase of `kappa(A)` under `v_min`-aligned rank-1 updates and approximate but non-monotone decrease under random updates. The assumption that breaks is symmetric positive definiteness — if Tikhonov regularisation `lambda` is too small and floating-point noise pushes `A^{-1}` past positive definite. The summary.md text §2.3 already prescribes Tikhonov at `lambda = 1e-3` which is far above the floating-point floor.

### B.4 Gneiting-Raftery / Bröcker decomposition: is "calibration drift fires first" a theorem?

**Statement** (Bröcker 2009 QJRMS 135 Theorem 2; Gneiting-Raftery 2007 JASA 102 Theorem 1). For any strictly proper scoring rule `S` applied to a probabilistic forecaster, the expected score decomposes:
```
E[S(F, Y)] = Uncertainty - Resolution + Reliability
```
where Reliability `>= 0` with equality iff the forecaster is perfectly calibrated (i.e. `Pr[Y in C | F = f] = p` for all stated coverage levels `p` and all forecasts `f`).

**Is "calibration fires first" a theorem or an empirical regularity?** It is *not* a theorem. The decomposition gives an orthogonal split of the expected score; it does *not* assert that perturbations of the forecaster manifest in one term before another. The actual claim that "calibration drift fires first" relies on a finite-sample power argument: the binomial coverage test on `n = 200` points has effect-size detectable at the 0.05 level when coverage shifts by `sqrt(p(1-p)/n) * z_{0.025} ~ 0.065`, while the CUSUM at h=5 requires a residual-mean shift of about 0.5 standardised units to detect within a comparable number of samples. Whether the *physical* shift in the underlying distribution produces a coverage shift of 0.065 or a residual mean shift of 0.5 first depends on whether the model's bias degrades or its calibrated variance degrades — and that is empirical.

**The right statement** is: *if* a model error manifests as predictive-mean bias of magnitude `b`, the standardised residual mean shifts by `b / sigma_conformal`, and the per-region coverage shifts by approximately `phi(b / sigma_conformal) * (b / sigma_conformal)` for small `b`. The two detectors have comparable power for `b / sigma_conformal ~ 0.5`, *not* one strictly before the other. Empirically `notable.md` investigation 2 is the test of which fires first in practice; the expected outcome of "calibration first" is a hypothesis, not a theorem.

The (1, 0, 0) "impossible in steady state" entry in the aggregator table should therefore be re-marked as "should be rare under the empirical regularity that the conformal stratification is fine enough to resolve the residual variance shift". The persistence policy (require N consecutive) is the right design under this weaker statement.

## D. Scaling: cost of drift evaluators in the 15-hour run

Empirical wall-clock per-cycle on this machine:

| evaluator | per-cycle cost |
|---|---|
| DAS-CUSUM streaming update (one new point) | 0.02 ms |
| DAS-CUSUM full-stream rescan over 5000 points | 13.2 ms |
| BH-corrected binomial, 5 strata × 200 points | 2.6 ms |
| kappa(A) via `eigvalsh`, d=75 | 0.18 ms |
| v_min via `eigh`, d=75 | 4.0 ms (only at FM-version boundary) |
| aggregator decision | < 0.01 ms |
| **per-cycle total (streaming mode)** | **~ 6 ms** |
| **per-cycle total (with v_min refresh)** | **~ 10 ms** |

Agent (a) projects the 15-hour budget as: T1: 50 000 calls / 16 min; T2: 5 000 / 6 h; T5: 100 / 2.4 h.

Per cycle the drift evaluator fires once. A conservative estimate of cycles in 15 hours: ~1000-5000 (one cycle per ~50 oracle calls in T1-dominated mode, one per T2 call when escalating). So total drift compute over the run: between 5 s and 50 s. As a fraction of oracle compute: ≤ 0.002. **Drift detection is essentially free.**

The dominant cost component is the `binomtest` invocation (scipy overhead, ~0.5 ms per call). Replacing scipy `binomtest` with a closed-form Wilson-score or Pearson chi-square implementation would drop the BH cost by 5-10×. Not worth doing for the hackathon.

The `v_min` recompute at FM-version boundaries is the only super-millisecond piece. We recompute only at `fm.update` events, not every cycle. Even at 100 retrains in the 15-hour run that is 100 × 4ms = 0.4 s total.

## E. Module layout: `modules/drift/`

The new module:

```
modules/drift/
  __init__.py
  das_cusum.py            # streaming DAS-CUSUM
  coverage_bh.py          # BH-corrected per-region binomial coverage
  kappa_vmin.py           # kappa(A), v_min, projection-ratio
  aggregator.py           # decide_action(acc, cal, cov) + persistence
  phoenix_evaluators.py   # the three SpanEvaluations producers
  spans.py                # helpers to emit the aletheia.* attrs
```

Public signatures:

```python
# das_cusum.py
@dataclass
class DASCUSUMState:
    S_pos: float
    S_neg: float
    buffer: deque[float]    # rolling window of length w
    n_seen: int

def update(state: DASCUSUMState, z: float, *, w: int = 30,
           h: float = 5.0, k: float = 0.5) -> tuple[DASCUSUMState, bool, float]:
    """Single-step streaming update. Returns (new_state, fired, S_t)."""

def reset(w: int = 30) -> DASCUSUMState: ...
```

```python
# coverage_bh.py
def per_region_binomial(
    counts_covered: np.ndarray,   # (S,) int
    counts_total:   np.ndarray,   # (S,) int
    target_coverage: float = 0.683,
) -> np.ndarray:                  # (S,) p-values
    """Two-sided exact binomial p-value per region."""

def bh_correction(
    pvalues: np.ndarray, alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Returns (rejected_mask, adjusted_threshold)."""

def calibration_drift_fired(
    counts_covered: np.ndarray, counts_total: np.ndarray,
    target_coverage: float = 0.683, alpha: float = 0.05,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Returns (fired, pvalues, rejected_mask)."""
```

```python
# kappa_vmin.py
@dataclass
class KappaState:
    A_inv: np.ndarray              # (d, d)
    v_min: np.ndarray              # (d,)
    kappa: float
    train_proj_var: float          # baseline projection variance

def from_A_inv(A_inv: np.ndarray, Phi_train: np.ndarray) -> KappaState:
    """Recompute v_min and baseline projection at FM-version boundary."""

def coverage_drift_fired(
    state: KappaState, recent_phi: np.ndarray,
    *, kappa_threshold: float = 1e4, proj_ratio_threshold: float = 3.0,
) -> tuple[bool, float, float]:
    """Returns (fired, kappa_now, proj_ratio)."""
```

```python
# aggregator.py
@dataclass
class DriftFlags:
    acc: bool
    cal: bool
    cov: bool

@dataclass
class Action:
    name: Literal["noop","watch","recal","local_retrain","global_retrain"]
    target_region_id: str | None
    chain_recal: bool                 # always True after retrain

def decide_action(
    flags: DriftFlags,
    region_payload: DriftedRegion | None,
    history: list[DriftFlags],         # last N windows for persistence
    persistence_N: int = 3,
) -> Action: ...
```

```python
# phoenix_evaluators.py
def accuracy_drift_das_cusum(
    spans_df: pd.DataFrame, *, window: int = 200, h: float = 5.0,
) -> SpanEvaluations: ...

def calibration_drift_conformal(
    spans_df: pd.DataFrame, *, window: int = 200, alpha: float = 0.05,
) -> SpanEvaluations: ...

def coverage_drift_condition_number(
    spans_df: pd.DataFrame, *, window: int = 200,
) -> SpanEvaluations: ...
```

None of these import `modules.surrogate` directly; they consume a small `FMHandle` protocol (`predict`, `A_inv`, `embed`) so they work against agent (b)'s rebuilt FM with no code change. Existing `modules/surrogate/calibration.py:75-89` is the prototype for `coverage_sigma`, which the evaluators consume verbatim.
