# Oracle — research summary (agent a)

This document covers what the ALETHIA oracle is, what it can do, what it should
do, what its internal structure represents physically, and how to operate it
under realistic load. The shared brief (`docs/research/BRIEF.md`) governs the
constraint that the foundation model cannot see Wilson coefficients; the
oracle is therefore the only object that legitimately knows about them. The
load-bearing interface in the whole project is the oracle's output shape, not
its input.

References like `modules/analytic_smeft/smeft.py:165` cite file and line.
Physics is cited by arXiv id.

---

## 1. Structure — what is inside `modules/analytic_smeft/`

### 1.1 The three files

`modules/analytic_smeft/smeft.py` is the closed-form leading-order
neutral-current Drell-Yan calculator with dimension-6 SMEFT operators in the
Warsaw basis. It produces the differential cross-section `dsigma/dm_ll` (and
optionally `dsigma/dpT_l`) at the parton level, convolved with PDFs, in
picobarns per GeV. No Monte Carlo, no parton shower, no detector; just numpy
plus scipy and Gauss-Legendre quadrature.

`modules/analytic_smeft/pdfs.py` is the PDF abstraction. Two concrete
backends: `LHAPDFSet` (defaults to CT18NNLO via the vendored LHAPDF install
at `vendor/MG5_aMC/HEPTools/lhapdf6_py3/`) and `AnalyticPDF` (a closed-form
toy `x f(x) = N x^a (1-x)^b` ansatz with no DGLAP evolution). Shared
`PDFSet` interface (`pdfs.py:29`) with one required method `xf(pid, x, Q)`.

`modules/analytic_smeft/constants.py` holds PDG 2024 electroweak constants:
`M_Z = 91.1876`, `GAMMA_Z = 2.4952`, `SIN2_THETA_W = 0.23122`,
`ALPHA_EM = 1/127.951`, `V_HIGGS = 246.22 GeV`, `GEV2_TO_PB = 3.8937966e8`,
quark charges and weak isospins. Pure inputs, nothing computed.

### 1.2 Amplitude decomposition

For partonic `q qbar -> l+ l-` the amplitude is split per quark flavour `q`
and per chirality channel `(X, Y)` (`smeft.py:80`), four channels total.

The SM amplitude (`smeft.py:181-183`):

    A_SM_XY(s) = e^2 Q_q Q_l / s
               + (e^2 / sin^2 cos^2 θ_W) g_q^X g_l^Y * 1/(s − M_Z^2 + i M_Z Γ_Z)

The dim-6 correction (`smeft.py:186-190`):

    A_BSM_XY(s) = c4f_XY(c) / Λ^2
                + (v^2 / Λ^2) (κ_q g_l^Y + g_q^X κ_l) * 1/(s − M_Z^2 + i M_Z Γ_Z)

First piece: four-fermion contact, constant in `s`, no propagator
suppression. Coefficient `c4f_XY(c)` is a linear combination of the seven
four-fermion operators selected by the chirality channel
(`smeft.py:146-159`). Second piece: linearised Z-vertex shift from the
seven vertex operators (`smeft.py:126-143`), proportional to `v^2/Λ^2`,
still propagator-suppressed. This is the Greljo–Marzocca decomposition,
arXiv:1704.09015 eqs. 2.4-2.7.

### 1.3 The three squared-amplitude pieces

Squaring `A_SM + A_BSM` channel-by-channel, summing chiralities, multiplying
by `s_hat / (48 π)` (`smeft.py:195-202`):

    σ_q(s, c) = σ_q^SM(s) + σ_q^int(s, c) + σ_q^BSM2(s, c)

`σ^SM = |A_SM|^2`, `σ^int = 2 Re(A_SM* A_BSM)` (linear in `c`, `O(Λ^-2)`),
`σ^BSM2 = |A_BSM|^2` (quadratic in `c`, `O(Λ^-4)`).

This decomposition is mathematically exact at the partonic matrix-element
level. It is also exactly what Wilson-coefficient morphing
(arXiv:1611.06536, arXiv:1808.00442) exploits: squared amplitudes are at
most quadratic in `c`. The brief forbids the FM from being trained against
this ansatz; the oracle *implements it directly* — that is what makes it
an oracle.

### 1.4 The 14 Warsaw-basis operators

Seven four-fermion semileptonic: `c_lq^(1), c_lq^(3), c_eu, c_ed, c_lu,
c_ld, c_qe`. Seven Higgs-current vertex: `c_phi_l^(1), c_phi_l^(3),
c_phi_e, c_phi_q^(1), c_phi_q^(3), c_phi_u, c_phi_d`. Complete set for
flavour-universal NC Drell-Yan at dim 6 (Grzadkowski et al.,
arXiv:1008.4884) once dipole and four-quark operators are dropped. Triplets
`c_lq^(3)` and `c_phi_q^(3)` split up vs down quarks with opposite sign
through `τ^3` projection (`smeft.py:141, 153`).

Surrogate front-end uses only four (`cHq3, cHq1, clq3, clq1`,
`modules/surrogate/features.py:32`); oracle defaults the other ten to zero
(`smeft.py:94-106`). Name mapping in `oracle_smeft.py:36-44`.

### 1.5 Parton luminosity and PDF independence of μ

Hadronic convolution at `smeft.py:208-225`:

    Φ_q(τ) = (1/τ) ∫_τ^1 dx/x [x f_q(x) x f_qbar(τ/x) + (q ↔ qbar)]

evaluated by 64-node Gauss–Legendre in ln x. Hadronic differential:

    dσ/dm_ll = (2 m_ll / s) Σ_q σ_q(m_ll^2, c) Φ_q(m_ll^2 / s)

Critical for the surrogate: SM, interference, BSM² partonic pieces multiply
the **same** parton luminosity. The ratio

    μ(c, m) = (σ_SM + σ_int + σ_BSM2) / σ_SM

is PDF-independent at LO except through flavour-dependent weighting (a
few-percent effect). The surrogate can use the analytic toy PDF safely
(`oracle_smeft.py:21-22`). Empirically (200-call test, this machine), toy
and CT18NNLO `μ` agree at the ~1% level in the `m_ll = 1500 GeV` bin while
absolute cross-sections differ by ~5×.

### 1.6 Energy growth

Four-fermion contact is constant in `s`; SM photon falls as `1/s`, Z as
`1/(s − M_Z^2)`. Therefore `σ^int / σ_SM ~ (m_ll/Λ)^2` and
`σ^BSM2 / σ_SM ~ (m_ll/Λ)^4` — the EFT energy growth (Farina et al.,
arXiv:1609.08157). The vertex contribution carries the SM Z propagator so
it does *not* grow with energy; it sits at constant `(v/Λ)^2` shift. The
two SMEFT structures are observationally distinguishable by their
`m_ll`-dependence, not just by their coefficient.

`tests/test_analytic_smeft.py:110-145` pins `BSM^2/SM ~ (m_ll/Λ)^4` with
constant prefactor in `(500, 2000) GeV`, monotone toward the tail.

### 1.7 What the structure says the oracle can answer

By exposing the three pieces independently, the oracle answers, with one
PDF convolution per `(m, pid)` pair:

1. Total `μ(c, m)` to floating-point precision.
2. Pure SM at any `m` (denominator, cheaply cacheable).
3. Linear interference at any `m`, separately per operator if needed.
4. `BSM^2` at any `m`, separately per operator-pair if needed.
5. Pointwise differential (`differential_xs`, `smeft.py:309-366`) or
   bin-averaged via 8-node Gauss–Legendre (`smeft.py:289-303`, `369-452`).

What it cannot answer at LO analytic: NLO QCD (K ~ 1.3 at `m_ll ~ 1 TeV`),
NLO EFT running, photon-induced contributions, parton shower softening,
detector smearing, lepton acceptance cuts beyond the trivial angular
integral. Those need an MG-class backend.

---

## 2. Capability — what the oracle CAN do

### 2.1 The two adapters that exist

`AnalyticSMEFTOracle` (`modules/surrogate/oracle_smeft.py:47-123`) wraps
`differential_xs`, exposes `truth(c, m)` and `__call__(c, m, noise=True)`.

Empirically (this machine, 200-call batches):

| backend                              | latency per call |
|--------------------------------------|------------------|
| analytic toy PDF, quadratic order    | ~0.6 ms          |
| CT18NNLO, quadratic order            | ~19 ms           |

The LHAPDF path is dominated by per-`x` `xfxQ` calls in a Python loop
(`pdfs.py:162-170`). Vectorising would close most of the gap.

`MadGraphSMEFTOracle` (`oracle_madgraph.py:147-333`) wraps MG5_aMC v3.7.1
with SMEFTsim_U35_alphaScheme_UFO. One MG run per `(c, m)` query: edits
`param_card.dat`, narrows the `mmll/mmllmax` window in `run_card.dat`,
scrubs the env via `scripts/clean_env.sh`, launches
`./bin/generate_events -f`, parses cross-section from the banner, divides
by cached SM run. With `nevents=1000`: 10–60 s per query, ~3% MC noise.
Drop-in compatible with the analytic adapter.

### 2.2 What else could be plugged in (cost ordering)

1. **LO + Pythia parton shower**: + few seconds per MG run. The `μ` ratio
   is not significantly modified by showering above `m_ll ~ 500 GeV`
   (Mangano et al., arXiv:1610.07922) — mostly a distributional refinement.
2. **LO + Delphes detector**: + few seconds. Smears lepton momenta;
   per-mille for muons, percent for electrons; bin migration in `m_ll`.
3. **NLO QCD via MG5_aMC@NLO + SMEFT@NLO UFO** (Degrande et al.,
   arXiv:2008.11743): 5–15% K-factor shift in `μ`, mass-dependent. Minutes
   per query at `nevents=10k`.
4. **NLO EFT operator running** (Jenkins–Manohar–Trott, arXiv:1310.4838,
   arXiv:1312.2014, arXiv:1404.0312): per-cent effect, but the natural
   place to introduce scale-dependence; RG equations linear in `c` so
   compose with morphing.
5. **NLO QCD + parton shower (MC@NLO)**: tens of minutes per query.
6. **Higher-order EFT (dim-8)**: different UFO; breaks strict
   quadratic-in-`c` decomposition. Out of scope, but the oracle interface
   is unaffected because the FM never sees `c`.

The **interface** above the oracle does not change as fidelity changes.
The internal tier picker (T0 = closed form, T1 = closed-form + real PDF,
T2 = MG LO, T3 = T2 + shower, T4 = T3 + detector, T5 = NLO, T6 = full
sim) returns the same observable representation. The FM stays blind to
`c` and to tier.

### 2.3 Cost-vs-fidelity gradient

| tier | description                  | latency / call | MC noise on μ | extra physics                  |
| ---- | ---------------------------- | -------------- | ------------- | ------------------------------ |
| T0   | analytic, toy PDF            | ~0.6 ms        | 0 (exact)     | LO NC DY, 14 ops               |
| T1   | analytic, CT18NNLO LO        | ~19 ms         | 0 (exact)     | + realistic PDFs               |
| T2   | MG5 LO, nevents=1000         | ~30 s          | ~3%           | + spin correl, MG matrix       |
| T3   | MG5 LO + Pythia              | ~60 s          | ~3%           | + parton shower                |
| T4   | MG5 LO + Pythia + Delphes    | ~90 s          | ~3%           | + detector                     |
| T5   | MG5 NLO (SMEFT@NLO)          | ~10 min        | ~1% at 1e4 evt| + NLO QCD, NLO EFT             |
| T6   | NLO + shower + detector      | ~1 h+          | ~1%           | full sim chain                 |

Two orders of magnitude between T0 and T2 is the design lever: T0/T1
saturates bulk EPIG acquisitions, T2 cross-validates a small subsample, T3+
fires only on drift events or final numbers.

### 2.4 Upper bound on physics

The 14-operator Warsaw basis is **complete** for flavour-universal dim-6
SMEFT effects on NC Drell-Yan. Adding processes (CC, di-boson, top, Higgs)
needs new partonic amplitudes but the same `SM + int + BSM2`
decomposition is exact. The hard ceiling is not the basis (extensible) but
the **dim-6 truncation**: dim-8 introduces extra energy growth at amplitude
level and breaks the quadratic-in-`c` closed form. The current architecture
still works but the morphing basis inside the analytic backend needs
extending.

---

## 3. What SHOULD be done — recommended interface

### 3.1 The oracle's three jobs

1. Generate samples from `theory(c)` at a chosen fidelity tier given
   `(c, m, observable, fidelity)`. The FM consumes the *output*, not `c`.
2. Enable EPIG-driven active acquisition: drift flags a region, EPIG picks
   k candidates, the oracle evaluates physics at the chosen `(c, m)`. The
   FM never sees `c`; EPIG ranks `φ` not `c`; the oracle is the evaluator.
3. Provide ground truth for drift verification when the FM disagrees with
   measured data (within the tier's accuracy).

### 3.2 The output shape question

What does one oracle call return? Options:

- **Scalar `μ(c, m)`**: minimal, current. Fast cache. Throws away
  differential structure.
- **Binned histogram over `m_ll`**: closer to what a measurement exposes.
  Easy cache.
- **Unbinned events**: full kinematics; only MG natively. Expensive.
- **Structured dict**: contains all three pieces (`sm_only`,
  `interference`, `bsm_squared`) at the requested kinematic grid plus
  metadata.

**Recommendation: structured dict.** Shape:

```python
OracleResult = {
    "kinematics": {
        "observable": "m_ll" | "pT_l",
        "grid": np.ndarray,            # (n,) point values (TeV)
        "binning": np.ndarray | None,  # optional bin edges if averaged
    },
    "pieces": {
        "sm_only":      np.ndarray,    # (n,) pb/GeV or pb/bin
        "interference": np.ndarray,    # (n,)
        "bsm_squared":  np.ndarray | None,
    },
    "total":  np.ndarray,              # SM + int + BSM2
    "mu":     np.ndarray,              # total / sm_only — headline
    "wilson_coefficients": dict,
    "metadata": {
        "fidelity_tier": "T0" | "T1" | "T2" | ...,
        "backend":       "analytic" | "madgraph",
        "pdf":           "CT18NNLO" | "analytic" | ...,
        "lambda_gev":    1000.0,
        "sqrt_s_gev":    13000.0,
        "order":         "linear" | "quadratic",
        "wall_time_s":   float,
        "mc_noise":      float | None,
        "nevents":       int  | None,
        "cache_hit":     bool,
        "oracle_call_id": str,         # uuid for cache + Phoenix span
    }
}
```

**The output shape is identical across fidelity tiers.** Only
`metadata.fidelity_tier`, `metadata.mc_noise`, `metadata.wall_time_s`
differ. This is the load-bearing point: tier swaps do not propagate
through the interface.

The default FM extraction is `total - sm_only` divided by `sm_only`
(current `μ`). A future FM that consumes the differential reads
`pieces["interference"]` and `pieces["bsm_squared"]` separately.

### 3.3 One oracle call

One `(c, m_grid, observable, fidelity)` tuple producing one
`OracleResult`. `m_grid` is scalar or vector. T0/T1 cost is linear in
`len(m_grid)`. T2+ cost is dominated by the fixed-cost `generate_events`;
passing a wider `mmll` window and binning the resulting events captures
multiple `m` points per MG call. That is the natural batch/cache unit at
high fidelity.

### 3.4 Sketched interface

```python
class Oracle(Protocol):
    def call(self, c, m, *,
             observable="m_ll", fidelity="T1",
             order="quadratic", noise=True,
             timeout_s=None, budget_s=None,
             return_pieces=True) -> OracleResult: ...

    def truth(self, c, m, **kw) -> OracleResult: ...

    @property
    def supported_fidelity(self) -> tuple[str, ...]: ...

    def warmup(self, fidelity="T1") -> None:
        """Lazy-init PDFs/MG; idempotent."""
```

Current `AnalyticSMEFTOracle.truth` returns the scalar `μ`. Wrapping it
in the richer return type is a 30-line additive change to the adapter; no
change to `smeft.py`; existing tests still pass on the scalar accessor
`result["mu"]`.

---

## 4. Caching

The oracle is the expensive thing; the FM is free. Exact, observable,
cheap cache is mandatory.

### 4.1 Keying

```python
key = sha256(canonical_json({
    "c": tuple(round(c_i, 9) for c_i in c),
    "m": tuple(round(m_j, 9) for m_j in m),
    "observable": observable,
    "fidelity": fidelity,
    "order": order,
    "sqrt_s_gev": sqrt_s_gev,
    "lambda_gev": lambda_gev,
    "pdf": pdf,
    "nevents": nevents if tier >= "T2" else None,
    "seed":    seed    if tier >= "T2" else None,
    "code_version": ...,   # hash over smeft.py, constants.py, pdfs.py, oracle_*.py
}))
```

Rounding to 1e-9 is conservatively below any physically resolvable scale
in `c` (~ 1 mrad) and `m_ll` (~ μGeV). For T0/T1 the `seed` is omitted —
the analytic path is deterministic modulo a reseedable noise stream. For
T2+, `seed` is in the key because MG uses pseudo-random numbers.

### 4.2 Storage

Two-level:

- **In-memory LRU**, ~1000–10000 entries, ~10 KB each ⇒ 10–100 MB.
- **On-disk parquet** sharded by tier:
  ```
  cache/oracle/
    T0/shard_NNNN.parquet
    T1/shard_NNNN.parquet
    T2/shard_NNNN.parquet
    index.sqlite          # key -> (shard, row_offset)
  ```

Parquet because the workload is "give me all rows with fidelity=T1 and
m_ll > 1 TeV"; columnar reads handle that well; pandas/polars are
first-class. SQLite is unsuitable for the bulk; SQLite is the perfect
fit for the index alone (single-writer, O(1) membership). Joblib memmap
is too rigid (no schema evolution). Plain `.npz` does not version.

### 4.3 Invalidation

Fidelity-tier bumps do **not** invalidate older rows: tier is in the key.
The cache is append-only modulo eviction.

What invalidates a row semantically:

- SM constants edited (`constants.py`).
- Matrix element edited (`smeft.py`).
- PDF set version bumped (CT18NNLO → CT18NNLO_v2).
- For T2+, UFO version bumped.

Each captured by the `code_version` hash in the key (over a fixed file
list, possibly `git rev-parse HEAD:modules/analytic_smeft/`). Bumping any
of those partitions the cache automatically; old rows survive for
debugging.

### 4.4 Hierarchy

All tiers in parallel. **Do not evict the cheap when the expensive
arrives.** The cheap tier is the source of EPIG decisions and FM bulk
training; the expensive tier exists to spot-check. After a T2 call at
point `p`, both T1 and T2 results for `p` coexist; drift detection
compares them.

### 4.5 Growth model

| call count | cache size | Phoenix |
|------------|-----------|---------|
| 10k        | 10–100 MB | ~50 MB  |
| 100k       | ~1 GB     | ~500 MB |
| 1M         | ~10 GB    | ~5 GB   |

1M is beyond a single-laptop hackathon. 10–100k is the realistic 15-hour
range; <1 GB total.

### 4.6 Hit-rate under EPIG querying

EPIG picks points the model is most uncertain about → mostly cache
**misses**. Exception: the **target set** for EPIG is fixed across many
rounds. Practical pattern:

1. At start, evaluate oracle at full target set and cache. One-shot.
2. EPIG selects pool points; oracle calls those (misses).
3. As seen-pool grows, target-set evaluations stay cached and
   pool-to-target updates use only the model state, not the oracle.

Drift checks evaluate existing rows ⇒ 100% hit by construction.

SM piece `σ_SM(m)` is `c`-independent ⇒ ~100% hit at the same `m` across
all queries. The MG oracle already exploits this
(`oracle_madgraph.py:267-300`); the analytic oracle does not need to.

### 4.7 Phoenix span attributes

```
oracle.fidelity_tier   = "T1"
oracle.backend         = "analytic" | "madgraph"
oracle.observable      = "m_ll"
oracle.n_points        = 1
oracle.order           = "quadratic"
oracle.cache.hit       = True | False
oracle.cache.key       = "ab12cd..."
oracle.cache.bytes     = 4096
oracle.cost_seconds    = 0.018
oracle.mc_noise        = 0.0
oracle.lambda_gev      = 1000.0
oracle.sqrt_s_gev      = 13000.0
oracle.wilson_norm     = ||c||_2          # NOT individual c_i
oracle.m_min_tev       = 0.3
oracle.m_max_tev       = 0.3
oracle.mu_min          = 1.0
oracle.mu_max          = 1.0
```

`oracle.wilson_norm` rather than individual `c_i` keeps the FM-irrelevant
information out of the trace UI. The three rubric-relevant attributes are
`oracle.cache.hit` (cost projection), `oracle.fidelity_tier`
(tier-conditional cost), `oracle.cost_seconds` (budget exhaustion).

---

## 5. Interactive mode

### 5.1 Lazy LHAPDF init

`LHAPDFSet.__init__` (`pdfs.py:157-160`) takes ~1 s first call;
subsequent are warm. Publish a `warmup()` method that performs that first
call so the notebook init pays the cost up front.

### 5.2 Warm MG worker pool

`MadGraphSMEFTOracle._launch_and_parse` (`oracle_madgraph.py:224-261`)
spawns a fresh `generate_events` per call. First MG invocation in a fresh
process directory: ~30 s (source compilation); subsequent in the same dir:
10–20 s. For interactive:

1. Pre-create N parallel process dirs at startup (one per core).
2. Compile SM-only event generator in each (one warmup MG run with all
   `c = 0`).
3. Distribute oracle calls via `concurrent.futures.ProcessPoolExecutor`
   over per-directory workers. Pool size = physical cores − 1.

### 5.3 Bounded latency

T0 < 1 ms ⇒ synchronous. T1 ~ 20 ms is interactive but laggy for
`n_points > 50` ⇒ batch.

Default `fidelity="T1"`, `timeout_s=0.1`. Beyond 100 ms the call returns
its partial result with `metadata.partial=True` and the orchestrator
escalates. This is the per-call kill-switch.

### 5.4 Async escalation

When drift fires:
1. `result_T1 = oracle.call(c, m, fidelity="T1")` (~20 ms, blocking).
2. If FM prediction at `(c, m)` falls outside conformal interval around
   `result_T1["mu"]`, escalate.
3. `future_T2 = oracle.call_async(c, m, fidelity="T2")` (~30 s, async).
4. UI shows "T2 pending"; Phoenix span remains open.
5. On completion: fold into cache; emit span; wake consumer.

`call_async` returns `concurrent.futures.Future[OracleResult]`. Progress
is logged into the span as `oracle.progress = 0.1, 0.2, …` parsed from
MG's stdout (optional for hackathon).

### 5.5 Kill-switch

`timeout_s` per call. T0/T1 ignore (always sub-second). T2+ pass into
`subprocess.run(timeout=...)` as in the current adapter
(`oracle_madgraph.py:235`). On timeout: kill, clean partial run dir, raise
`OracleTimeoutError`. Orchestrator retries at lower fidelity or marks
unresolved.

---

## 6. Scaling — 10–20 hour run budget

### 6.1 Tiered scheduler

For a 15-hour run with one CPU for the agent and ~7 cores for MG workers:

- T0: 0.0006 s/call ⇒ 9×10⁷ calls/hour. Never the bottleneck.
- T1: 0.019 s/call ⇒ 2.8×10⁶ calls/hour. Oversupplied for FM + EPIG.
- T2: 7 cores × 15 h × 3600 s / 30 s = ~12 600 calls.
- T5: 7 × 15 × 3600 / 600 = ~630 calls.

Recommended allocation:

| tier | budget        | use                                                         |
|------|---------------|-------------------------------------------------------------|
| T0   | unlimited     | EPIG target-set variance, leverage scan                     |
| T1   | ~50 000 calls | bulk FM training, EPIG acquisitions                         |
| T2   | ~5 000 calls  | drift verification, T1 cross-validation                     |
| T5   | ~100 calls    | end-of-run NLO anchor, publication-grade numbers            |

### 6.2 Scheduling logic

1. Default every call to T1.
2. Promote to T2 when `||c||_2 > 0.5` (large EFT effect) or `m_ll > 2 TeV`
   (energy-growing regime, NLO QCD K-factor matters).
3. When `check_drift` fires coverage drift, schedule a T2 batch in the
   drifted region for cross-validation.
4. When residual between FM and T2 systematically deviates, schedule T5
   cross-check.
5. T0 for EPIG variance landscape; T1 for FM training; T2+ for what T1
   cannot answer.

### 6.3 Cost projection

In the 15-hour budget:

- T1 wall time: 50 000 × 0.019 s = ~16 min. Negligible.
- T2 wall time: 5000 × 30 s / 7 = ~6 h.
- T5 wall time: 100 × 600 / 7 = ~2.4 h.
- Total MG-bound: ~8.5 h; ~6.5 h slack for FM training cycles and demos.

The risk is **not compute exhaustion** but coordination — making sure the
agent does not block on T5 when T2 would suffice, and that many writers
play nice with the cache. Single-writer-many-reader per parquet shard is
enough; `index.sqlite` is the single writer.

---

## 7. Disagreement with the existing plan

`aletheia_verification_plan.md` at the repo root specifies the FM joint
feature map `phi_joint(c, m) = phi_c(c) ⊗ phi_x(m)`
(`modules/surrogate/features.py:60-65`) and trains `Y = μ(c, m)` via
closed-form ridge. The brief's hard constraint forbids this: `c` must not
be an FM feature.

This document is consistent with the brief. The **oracle** is exactly where
`c` legitimately lives. The morphing ansatz `phi_c` is exactly right
*inside* the oracle's analytic backend (because the squared dim-6 matrix
element really is `{1, c_a, c_a c_b}` polynomial in `c`) and exactly wrong
*as a feature for the FM*. Both facts coexist: morphing is correct physics;
the FM should not be a fit to it.

Recommendations:

1. Keep `modules/analytic_smeft/` as-is — correct closed-form physics
   layer.
2. Add an `OracleResult` wrapper (section 3) — additive.
3. Drop `phi_c` and `phi_joint` from the FM forward pass; they survive
   *inside* the oracle (where they already are, as the partonic-amplitude
   decomposition). Subject of agent (b).
4. Build the cache layer (section 4) as a new `modules/oracle/cache.py`.
5. Wire the OpenInference attributes (section 4.7).

(1) and (2) are non-breaking. (3) is the rewrite the brief mandates. (4)
and (5) are new machinery.

---

## 8. Citations

- Greljo & Marzocca, JHEP 09 (2017) 105, arXiv:1704.09015.
- Farina, Panico, Pappadopulo, Ruderman, Torre, Wulzer, Phys.Lett.B 772
  (2017) 210, arXiv:1609.08157.
- Grzadkowski, Iskrzyński, Misiak, Rosiek, JHEP 10 (2010) 085,
  arXiv:1008.4884.
- Jenkins, Manohar, Trott, parts I-III: arXiv:1310.4838, arXiv:1312.2014,
  arXiv:1404.0312.
- Degrande, Durieux, Maltoni, Mimasu, Vryonidou, Zhang, arXiv:2008.11743.
- Mangano, Moretti, Piccinini, Pittau, arXiv:1610.07922.
- Smith, Bickford Smith, Rainforth, arXiv:2304.08151 (used by agent d).
- Hartland, Maltoni, Nocera et al., arXiv:1901.05965 (SMEFiT global EFT
  fits — comparison point).

---

## 9. Closing

The oracle is the only object that legitimately knows about Wilson
coefficients. Its internal `sm_only / interference / bsm_squared`
decomposition over 14 Warsaw operators is the exact physics of dim-6 SMEFT
matrix elements and is what makes T0/T1 fast and deterministic. The MG
backend is the fidelity-escalation handle at T2+.

The recommended interface returns a structured dict with identical shape
across fidelity tiers; the cache key fully determines reproducibility;
interactive mode is achievable with warmup + worker pool; the 15-hour
budget allocation (50 k T1 + 5 k T2 + 100 T5) is comfortable provided the
scheduler defaults to T1 and escalates only on flagged drift.

Cache is keyed on (rounded `c`, rounded `m`, observable, fidelity, order,
λ, √s, pdf, code_version, [seed, nevents]); stored as in-memory LRU +
sharded parquet + sqlite index; never evicts older tiers when newer
arrive; emits cache.hit / fidelity_tier / cost_seconds to Phoenix;
near-0% hit on EPIG-acquired calls and ~100% on target-set re-evaluations
and drift verifications.
