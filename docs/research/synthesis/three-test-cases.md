# ALETHIA — three executable test cases (synthesis)

> **Addendum (2026-05-27).** The FM encoder recommendation in this
> document — "DeepSets first, Particle Transformer when budget allows" —
> has been revised by the empirical comparison at
> [../02-foundation-model/intention-vs-deepsets.md](../02-foundation-model/intention-vs-deepsets.md).
> A learned-basis Intention closed-form-attention head outperforms
> matched-parameter DeepSets-FM by ~700× in mean MSE on held-out
> SMEFT scenarios and beats even a regressor that gets `c` for free.
> The recommended encoder for Test 2 is the Intention head with a
> learned `ψ_θ` MLP basis; DeepSets is reserved for the per-event
> sub-encoder of a hybrid in the multi-dimensional extension. The
> structure of the three test cases below is unchanged; only the
> "what to put inside the FM encoder slot" is revised.

This document binds the four research areas — oracle (a), foundation model
(b), drift (c), EPIG / conformal (d) — into a single executable plan with
three test cases that compose. Test 1 is wiring-only and takes under five
minutes; Test 2 is theory-against-a-known-problem at the 10–30 minute
scale; Test 3 is the hackathon-grade 10–20 hour demo run. Each test gates
the next.

All paths are absolute; uv invocations are prefixed with
`export PATH="$HOME/snap/code/240/.local/bin:$PATH"` per
`/home/vince/ALETHIA/docs/research/BRIEF.md:126-130`.

---

## Prologue

### Joint architecture (one paragraph)

The oracle (`/home/vince/ALETHIA/modules/analytic_smeft/smeft.py`
plus the `OracleResult` wrapper proposed in
`/home/vince/ALETHIA/docs/research/01-oracle/summary.md:251-280` and the
`sample_events` extension of
`/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:7-26`) is
the only object that legitimately knows about Wilson coefficients. Each
Commission `(wilson_region, kinematic_window, fidelity_tier)` triggers
sampling of c-tagged event batches at a chosen tier (T0 analytic toy
PDF, T1 analytic CT18NNLO, T2 MadGraph LO, T5 MadGraph NLO). A frozen
foundation model encoder
(`EventSetFM`, the DeepSets-then-Particle-Transformer architecture
recommended in
`/home/vince/ALETHIA/docs/research/02-foundation-model/summary.md:72-101`)
maps each event batch to an embedding `z ∈ ℝ^d`; **c never enters this
forward pass**, only the InfoNCE pairing signal that defines the loss
(`/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:67-83`).
A closed-form ridge probe `LinearProbe` (the surviving math of
`IntentionFM` in `/home/vince/ALETHIA/modules/surrogate/model.py:25-78`
with `phi_joint(c, m)` replaced by `phi(z) = z`) fits `Z → y` for any
scalar target (μ at fixed m, log μ, per-operator c attribute) and exposes
`A_inv`, `v_min`, `kappa`, `leverage`. The drift layer
(`/home/vince/ALETHIA/docs/research/03-drift/summary.md:134-235`) runs
three evaluators — DAS-CUSUM on standardised residuals, BH-corrected
per-region binomial coverage, condition-number / `v_min`-projection — on
the probe state and emits a `DriftedRegion` payload. EPIG over
commissions
(`/home/vince/ALETHIA/docs/research/04-epig-conformal/summary.md:105-138`)
consumes that payload, ranks candidate commissions by closed-form
Gaussian-linear information gain on a target embedding set, and feeds
the chosen commissions back to the oracle. After every model update the
ConformalCalibrator is refit on a fresh calibration set drawn from the
current probe region. The whole loop is one ADK turn instrumented by
OpenInference; Phoenix at `http://localhost:6006` is the trace store and
MCP server.

### Load-bearing contracts (the API the synthesis pins)

The four areas' second-pass evaluations agree on six load-bearing contracts.
The test cases below assume these as preconditions; any divergence in
implementation breaks the chain.

- **`Commission` type.** `frozen dataclass` with fields
  `wilson_region`, `kinematic_window`, `fidelity_tier`,
  `observable`, `order`
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:11-19`).
  This is the unit of work for the oracle; it is what EPIG ranks; it is
  what `drift.aggregate -> action=local_retrain` consumes via the
  `DriftedRegion` payload.

- **`FMHandle` protocol.** The smallest interface the drift and EPIG
  modules consume:
  `predict(z) -> (mu, sigma)`, `embed(events) -> z`,
  `A_inv: (d, d)`, `v_min: (d,)`, `kappa: float`,
  `leverage(z) -> np.ndarray`, `update(z_new, y_new)`,
  `target_set_for_region(region_id) -> (n_T, d)`
  (`/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:153-176`,
  `/home/vince/ALETHIA/docs/research/03-drift/eval.md:11-19`). The
  encoder architecture is opaque under this protocol; agent (b) may swap
  DeepSets for Particle Transformer without touching drift or EPIG.

- **`DriftedRegion` payload.** `region_id`, `centroid_z`, `radius`,
  `failing_signal ∈ {acc, cal, cov}`, `severity`, `triggering_spans`,
  `suggested_target_z`
  (`/home/vince/ALETHIA/docs/research/03-drift/eval.md:30-43`). The
  `suggested_target_z` array is what EPIG ingests as `z_target`. This
  payload is the bridge between drift's worst direction and EPIG's
  target set — the un-formalised joint assumption flagged by
  `/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:60-65`.

- **Conformal refit-after-update invariant.** `ConformalCalibrator.fit`
  must be called on every `model.update`; `coverage_sigma` must
  raise `OutdatedCalibratorError` if the model fingerprint
  has changed since calibration
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:60-85`,
  empirical evidence
  `/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:65-85`:
  no-refit cov_68 mean 0.777 vs refit 0.698 vs nominal 0.683 with
  cross-leverage 0.286). This is enforced at the runtime, not only by
  the drift evaluator.

- **Calibration-on-broader-probe-region invariant.** Cal set must cover
  the support the FM will be queried on, not the support it was trained
  on
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/summary.md:198-202`,
  empirical
  `/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:87-117`:
  BAD cal on `|c| ≤ 1` evaluated on `|c| ∈ [1.5, 2]` gives
  cov_68 = 0.523, GOOD cal on `|c| ≤ 2` gives 0.729). Tracked over
  iterations: when EPIG pulls training support outward, the cal set
  composition must track the probe-region composition with proportional
  per-cluster pools
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:111-122`).

- **Headline span attribute: target-set predictive entropy `H_T`.**
  `H_T = 0.5 Σ_{i ∈ T} log(2 π e σ² lev_{T_i})` on the `fm.update` span,
  attribute name `aletheia.fm.target_entropy_H_T_post`
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:86-107`).
  The 1/N decay regime
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:120-149`,
  log-log slope -1.66 to N=1600) is the single number that says "EPIG is
  working".

### Standing of the existing aletheia_verification_plan.md

`/home/vince/ALETHIA/aletheia_verification_plan.md` was written before
the brief's hard constraint was settled
(`/home/vince/ALETHIA/docs/research/BRIEF.md:10-43`). Read it as a
roadmap with three categories of content.

What it got right and should be kept verbatim:
- §0 algebraic frame for the Intention head: ridge with Sherman-Morrison
  updates, leverage as quadratic form, GP/KRR duality, EPIG closed form
  with σ² cancellation. The math survives the rewrite; only the feature
  map changes (`/home/vince/ALETHIA/aletheia_verification_plan.md:11-15`).
- Component 1 Phoenix tests (span hierarchy, trace persistence, MCP
  self-query, experiment compare, input/output capture). These are
  Phoenix-substrate checks; they are independent of the FM redesign and
  belong as-is in `tests/test_phoenix_integration.py`
  (`/home/vince/ALETHIA/aletheia_verification_plan.md:23-41`).
- Component 2 analytic SMEFT tests (SM limit, linear interference,
  sign-flip, quadratic growth, no-NaN). The oracle does not change;
  these run as-is and already pass per
  `/home/vince/ALETHIA/docs/research/01-oracle/test-plan.md:11-65`
  (`/home/vince/ALETHIA/aletheia_verification_plan.md:51-75`).
- Component 4 `check_drift` math tests (DAS-CUSUM ARL, BH FDR, kappa
  monotonicity). The Monte Carlo validations of
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:14-117`
  ratify these against theory.

What it got wrong and must be rewritten:
- Component 3 "Foundation model" tests assume `phi_joint(c, m)` as the
  feature map, with `D_C = 15`, `D_X = 5`, `D_JOINT = 75`, and `Y = μ`
  as the regression target
  (`/home/vince/ALETHIA/aletheia_verification_plan.md:81-122`). This is
  exactly the forbidden pattern — c on the forward pass and the morphing
  ansatz as the loss target. The empirical demonstration of how badly
  this fails is
  `/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:27-47`:
  R² = -144 per-operator when an operator is held at zero during
  training. Replace all of Component 3 with the constraint-checking,
  representation-quality probe tests of
  `/home/vince/ALETHIA/docs/research/02-foundation-model/test-plan.md:8-50`
  (signature check, permutation invariance, linear-probe R² on held-out
  c, downstream-task swap, scaling).
- Component 5 `compute_epig` interface signature
  `epig_acquire_in_drifted_region(model, drift_region_bounds, k, ...)`
  takes `drift_region_bounds: dict[str, tuple[float, float]]` over
  Wilson coordinates
  (`/home/vince/ALETHIA/aletheia_verification_plan.md:292-305`). Under
  the constraint, regions live in embedding `z` space, not in `c` space.
  Replace with `epig_acquire_commissions(probe, commissions,
  embed_preview, z_target, k)`
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/summary.md:106-135`).
- Component 6 end-to-end scenario withholds `|c_lq^(3)| ∈ [0.6, 1.0]`
  from training and asserts that "coverage drift fires when query batch
  enters the withheld region". Under the constraint there is no query
  batch indexed by c; queries are oracle commissions producing
  embeddings. The right statement is "coverage drift fires when probe
  embeddings hit the cluster that corresponds to that band". This
  document keeps the spirit of the scenario, restated in Test 3 below.

What it left out and must be added:
- The conformal refit-after-update runtime check
  (contract listed above).
- The calibration-on-broader-probe-region tracking-the-probe-distribution
  invariant.
- The `H_T` span attribute as the headline EPIG monitor.
- The DriftedRegion → EPIG `z_target` bridge.

Net advice: keep §0, Component 1, Component 2, Component 4 of the old
plan; rewrite Component 3 and Component 5 against the contracts above;
restate Component 6 in embedding-space terms as the Test 3 scenario in
this document. The full file should remain in the repo as a historical
artefact pointed to by this synthesis.

---

## Test Case 1 — Machinery smoke check

**What it demonstrates.** Every component of the joint design is wired
together: the oracle returns an `OracleResult` whose `pieces` decompose
correctly; a (toy) FM encoder produces an embedding from sampled events
without c on its forward pass; a `LinearProbe` fits on the embedding and
exposes `A_inv` / `v_min` / `leverage`; a `ConformalCalibrator` refits
without raising `OutdatedCalibratorError`; an `epig_acquire_commissions`
call picks one commission from a candidate pool; the three drift
evaluators run on a synthetic span stream and one of them fires when the
synthetic change point is crossed; the OpenInference instrumentation
emits all expected spans into an in-memory exporter sink. No physics
claim is made and no scaling is asserted — only the wiring.

**Prerequisites (must exist before running).**

1. `modules/surrogate/oracle_smeft.py` enriched with the `call(c, m, *,
   observable, fidelity, order, return_pieces) -> OracleResult` method
   per `/home/vince/ALETHIA/docs/research/01-oracle/eval.md:64-69` and
   the `sample_events(c, n_events, *, m_range_tev, observable, rng)
   -> np.ndarray` method per
   `/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:11-19`.
   These are additive: existing
   `/home/vince/ALETHIA/modules/surrogate/oracle_smeft.py:84-123`
   keeps the `truth`/`__call__` scalar interface.
2. `modules/surrogate/encoder/deepsets.py` per the layout in
   `/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:119-149`:
   class `EventSetFM(nn.Module)` with `forward(events: Tensor) -> Tensor`
   and `embed(events: np.ndarray) -> np.ndarray`. For Test 1 the encoder
   does not need to be trained; the smoke test uses random weights. The
   forward-pass signature must not accept `c`.
3. `modules/surrogate/probe.py` defining
   `class LinearProbe` per
   `/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:165-176`.
   This is a thin rewrite of
   `/home/vince/ALETHIA/modules/surrogate/model.py:25-78`
   that consumes `Z, Y` rather than `(C, M, Y)`.
4. `modules/surrogate/calibration.py` extended with the model-fingerprint
   stamp and `OutdatedCalibratorError` per
   `/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:160-187`.
5. `modules/drift/` per the layout of
   `/home/vince/ALETHIA/docs/research/03-drift/eval.md:188-296` containing
   `das_cusum.py`, `coverage_bh.py`, `kappa_vmin.py`, `aggregator.py`.
6. `modules/surrogate/acquisition_commissions.py` exposing
   `epig_acquire_commissions(probe, commissions, embed_preview,
   z_target, k, mc_samples=1)` per
   `/home/vince/ALETHIA/docs/research/04-epig-conformal/summary.md:106-135`.
7. `tests/test_smoke_machinery.py` (new) implementing the assertions
   below.
8. An in-memory OTLP exporter so the test does not require the Phoenix
   container. Pattern:

   ```python
   from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
       InMemorySpanExporter,
   )
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import SimpleSpanProcessor
   exporter = InMemorySpanExporter()
   provider = TracerProvider()
   provider.add_span_processor(SimpleSpanProcessor(exporter))
   ```

   `agent/instrumentation.py:62-85` should accept a
   `tracer_provider=` override; for the smoke test the provider is the
   in-memory one above and `PHOENIX_TRACING=0` is set so the live
   exporter is not registered.

**Commands.**

```bash
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
PHOENIX_TRACING=0 \
  uv run pytest -xvs tests/test_smoke_machinery.py
```

**Closed-form predictions the test checks.**

- *Oracle wiring:* `OracleResult.pieces["sm_only"] +
  pieces["interference"] + pieces["bsm_squared"]` equals
  `OracleResult.total` to bit-for-bit precision
  (`/home/vince/ALETHIA/docs/research/01-oracle/empirical-results.md:42-46`).
- *FM constraint:* `inspect.signature(EventSetFM.forward).parameters`
  contains no parameter whose name matches `c`, `wilson`, or any of
  `('cHq3','cHq1','clq3','clq1')`; AST scan of the encoder class
  body finds no reference to those symbols
  (`/home/vince/ALETHIA/docs/research/02-foundation-model/test-plan.md:8-23`).
- *Probe linear algebra:* `LinearProbe.A_inv` is symmetric and positive
  definite (`np.linalg.eigvalsh(A_inv).min() > 0`); `v_min` is the
  eigenvector at the smallest eigenvalue of `A`; `leverage(z) =
  z @ A_inv @ z.T` is non-negative
  (`/home/vince/ALETHIA/modules/surrogate/model.py:67-70` adapted to
  `phi(z) = z`).
- *Conformal refit guard:* After `probe.update(z_new, y_new)`, the cached
  calibrator must raise `OutdatedCalibratorError` on
  `coverage_sigma(probe, z_test, 0.683)`; refitting clears the error
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:75-85`).
- *EPIG non-negativity:* `epig_acquire_commissions` returns exactly
  `k=1` commission; the IG of that commission against the target set is
  positive
  (`/home/vince/ALETHIA/modules/surrogate/acquisition.py:25-67` algebra).
- *Drift evaluator wiring:* fed a 200-sample i.i.d. `N(0,1)` synthetic
  span stream with no change point, none of `accuracy_drift_das_cusum`,
  `calibration_drift_conformal`, `coverage_drift_condition_number` fires
  (combined boolean flag `000`). Fed a 200-sample stream with a
  `N(0.5, 1)` shift at sample 100, `accuracy_drift_das_cusum` fires
  by sample 200 with detection delay under 110 (theoretical bound from
  `/home/vince/ALETHIA/docs/research/03-drift/test-plan.md:39-50`).
- *Instrumentation:* the in-memory exporter contains exactly one
  `agent.cycle` chain span, with child spans named
  `oracle.query`, `fm.predict`, `drift.evaluate`, `drift.aggregate`,
  `epig.select`, `fm.update` per the tree in
  `/home/vince/ALETHIA/docs/research/03-drift/summary.md:381-440`.
  Each span carries the `aletheia.*` attributes its sub-area requires.

**Pass / fail criteria.**

| check                                       | threshold                                |
|---------------------------------------------|------------------------------------------|
| recomposition `pieces` sum == `total`        | bit-for-bit (`np.array_equal`)           |
| `c` not in encoder signature                | exact, by inspect                        |
| no c-like symbol in encoder AST             | exact, by ast walker                     |
| `A_inv` symmetric pos-def                   | `eigvalsh.min() > 1e-12`                 |
| `OutdatedCalibratorError` raised post-update | exception type match                     |
| EPIG returns k commissions, IG ≥ 0           | strict inequality                        |
| DAS-CUSUM null-case fires                   | rate < 0.07 over 50 seeds                |
| DAS-CUSUM shift-case fires                  | rate > 0.93 over 50 seeds                |
| span tree present in exporter               | all 6 span names present                 |
| `H_T` attribute on `fm.update` span         | float, finite                            |

**Estimated wall time.** Under 5 minutes on a CPU laptop (the dominant
cost is the InfoNCE-untrained DeepSets forward pass on the smoke event
batches — N=512 events, 12k parameters, ~0.4 s per forward pass per
`/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:49`).
50 seeds of the drift synthetic stream are vectorised in NumPy and add
about 30 seconds total
(`/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:119-128`).

**What this test does NOT verify (passed to Test 2).**

- The FM has learned anything: weights are random, R² on c is meaningless.
- EPIG actually outperforms random acquisition: only one pick is made.
- Conformal coverage hits nominal level on the broader probe region:
  the calibration set is too small to estimate coverage.
- DAS-CUSUM's ARL_0 hits the calibrated target of ~1000: only a coarse
  fire/no-fire check is done.
- The closed-form EPIG matches a brute-force posterior refit: Test 2
  pins this to 1e-12.
- Phoenix MCP can answer the rubric queries: requires the live container.

---

## Test Case 2 — Theory check on a known problem

**What it demonstrates.** When the math gives a closed-form answer, the
implementation reproduces it numerically. Specifically, on the
`DummyAnalyticOracle` from
`/home/vince/ALETHIA/modules/surrogate/ground_truth.py` (a toy whose μ
is exactly polynomial in `phi_joint(c, m)`) and on the analytic SMEFT
oracle at fixed kinematics, five closed-form claims are pinned:
(i) the OLD `IntentionFM` regressor recovers μ to machine precision when
trained on a basis-saturating dataset — the regressor baseline that
shows the morphing ansatz is exactly invertible; (ii) the NEW FM +
probe recovers μ to a quantifiable accuracy at a stated compute budget,
with per-operator probe R² behaviour as predicted by the InfoNCE
mutual-information bound; (iii) closed-form EPIG agrees with brute-force
posterior refit to 1e-12; (iv) conformal coverage with refit is within
±5pp of nominal on the broader probe region while without refit drifts
by more than 5pp under a high-cross-leverage update batch; (v) the three
drift evaluators each fire on synthetic streams with known change points
and stay silent on streams with no change point, with empirical false-
alarm and detection-delay rates inside the published Lorden-Pollak and
Benjamini-Hochberg bounds.

**Prerequisites.**

1. All of Test 1's prerequisites passing, with one addition: the
   `LinearProbe.fit` path must support the **OLD** `IntentionFM`-style
   feature map `phi_joint(c, m)` under a compatibility flag so the (i)
   baseline is reachable from the same code. Concretely:
   `LinearProbe.fit_phi(Phi, Y)` where `Phi` is the precomputed
   `phi_joint(C, M)` matrix; this is the matrix the OLD design produces
   and the NEW design must reproduce when (and only when) testing the
   regressor baseline. Internally identical to
   `/home/vince/ALETHIA/modules/surrogate/model.py:39-51`.
2. Encoder pre-trained for at least 250 InfoNCE batches per
   `/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:67-83`:
   B=16, N=512, lr=2e-3, mean-pool DeepSets with the auxiliary log-rate
   scalar of section 4 there, d=32, ~12k parameters. Saved to
   `/home/vince/ALETHIA/cache/enc_stage0.pt`.
3. Brute-force posterior refit harness in
   `tests/test_epig_closed_form.py` matching
   `/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:14-41`.
4. A scripted "pathological" EPIG batch matching
   `/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:65-85`:
   100 picks in the top-leverage stratum, mean cross-leverage ≈ 0.286.
5. Synthetic drift streams per
   `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:14-92`:
   1000 null trials and 1000 shift trials at δ ∈ {0.25, 0.5, 0.75, 1.0,
   1.5} for DAS-CUSUM; 5-stratum BH check with 200 points per stratum.

**Commands.**

```bash
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
PHOENIX_TRACING=0 \
  uv run pytest -xvs tests/test_theory_check.py \
                     tests/test_epig_closed_form.py \
                     tests/test_conformal_refit_guarantee.py \
                     tests/test_drift_synthetic_streams.py
```

`tests/test_theory_check.py` runs the (i) regressor baseline plus the
(ii) FM + probe headline; `tests/test_epig_closed_form.py` runs (iii);
`tests/test_conformal_refit_guarantee.py` runs (iv);
`tests/test_drift_synthetic_streams.py` runs (v).

**Closed-form predictions checked.**

(i) **Regressor baseline (the old design, here to show the constraint
violation in numerical form).**
- *Algebra:* the ridge estimator
  `w = A^{-1} Phi^T Y` with `A = Phi^T Phi + λI` and
  `Phi = phi_joint(C, M)` is exact when `Y` is in the column space of
  `Phi`, i.e. when `Y` is polynomial in `phi_joint(c, m)`. The dummy
  oracle and the analytic SMEFT oracle at fixed kinematics satisfy this
  by construction
  (`/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:27-47`).
- *Prediction:* trained-set R² ≥ 0.998 and held-out (same c-box) R² ≥
  0.997 with `n_train = 50`.
- *Constraint-violation diagnostic:* hold one operator at zero during
  training; per-operator R² on that direction is strongly negative
  (≤ -20 in
  `/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:42`).
  This is the regressor-not-FM failure mode, recorded numerically.

(ii) **FM + probe (the new design).**
- *Algebra:* the InfoNCE-trained encoder's mutual information with c is
  bounded below by `log K - L_NCE` per van den Oord et al. arXiv:1807.03748
  Theorem 1; in our run `K = 16` and `L_NCE ≈ 1.28` at convergence,
  giving `I(z; c) ≥ 1.49 nats`. Tian-Krishnan-Isola arXiv:2005.10243 §5.3
  predicts linear-probe `R²(c_k | z) ≈ 1 - exp(-2 I(c_k; z))`
  (`/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:88-94`).
- *Prediction:* with MI concentrated on the shape-changing operator
  `c_lq^(3)`, per-operator linear-probe R² satisfies
  `R²(c_lq^(3)) ≥ 0.55` in-distribution `|c| ≤ 0.7` and `R² ≥ 0.30`
  extrapolation `|c| ∈ (0.7, 1.0]`; vertex operators
  `c_phi_q^(3)`, `c_phi_q^(1)` have `|R²| < 0.10` (mean-pool DeepSets is
  rate-blind, per
  `/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:73-84`).
  With the auxiliary log-rate scalar of
  `/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:121-135`,
  `R²(c_lq^(3)) ≥ 0.75` after 80 batches.
- *Compute budget:* 15 minutes wall time for the 250-batch run on CPU,
  per the same reference.
- *Downstream-task swap:* fix the encoder; refit the probe with target
  `log(μ at m=1.0 TeV)` instead of `c_lq^(3)`. R² ≥ 0.30. This is the
  operational definition of an FM
  (`/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:87-97`).

(iii) **EPIG closed form vs brute-force refit.**
- *Algebra:* `IG_T(p) = 0.5 log(lev_T / (lev_T - k_Tp^2 / (1 + lev_p)))`
  must equal `0.5 log(lev_T_pre / lev_T_post)` where `lev_T_post` is
  recomputed from a freshly fitted ridge on the augmented dataset
  (`/home/vince/ALETHIA/modules/surrogate/acquisition.py:25-67`).
- *Prediction:* over 200 random (candidate, target) pairs,
  `max |diff| ≤ 1e-12`, `median |diff| ≤ 1e-14`
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:30-41`).

(iv) **Conformal refit invariant.**
- *Algebra:* Barber-Candès-Ramdas-Tibshirani arXiv:2202.13415 Theorem 1
  bounds the coverage deficit by the Lipschitz constant of the score CDF
  times the Wasserstein-1 distance between pre- and post-update score
  distributions; the latter scales linearly with the cross-leverage
  between picks and cal set
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/summary.md:248-260`).
- *Prediction (pathological case, 20 seeds, 100 picks in top-leverage
  stratum, mean cross-leverage 0.286):* no-refit cov_68 mean = 0.777
  (±9.4pp from nominal 0.683); refit cov_68 mean = 0.698 (within 1.5pp
  of nominal)
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:69-79`).
- *Prediction (broader-probe-region invariant):* cal on `|c| ≤ 1`
  evaluated on `|c| ∈ [1.5, 2]` gives cov_68 = 0.523 (-16pp);
  cal on `|c| ≤ 2` evaluated on the same band gives cov_68 = 0.729
  (within ±5pp)
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:108-117`).
- *Runtime check:* `OutdatedCalibratorError` raised on the first
  `coverage_sigma` after `probe.update(...)` without an intervening
  `cc.fit(...)` (per the fingerprint contract).

(v) **Drift detectors on synthetic streams with known change points.**
- *DAS-CUSUM under H₀:* 1000 streams of 5000 N(0,1) samples; threshold
  `h = 5.0`, window `w = 30`, reference `k = 0.5`, warmup `2w`.
  Empirical ARL_0 = 370 vs Siegmund prediction 148 (factor 2.5 finite-
  window inflation, per Ahad-Davenport-Xie arXiv:2210.17353 Theorem 3,
  reproduced in
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:19-45`).
- *DAS-CUSUM under H₁ (mean shift):* 1000 streams of 100 N(0,1) +
  400 N(0.5,1). Detect rate ≥ 0.75; median delay ≈ 100 (theoretical
  Lorden-Pollak ARL_1 = h/(δ²/2) = 40, finite-window factor pushes
  empirical to ~100, per
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:47-65`).
- *BH-corrected coverage under H₀:* 1000 trials of 5 strata × 200
  samples. Any-rejection rate ≤ 0.07 (Benjamini-Hochberg 1995 nominal
  0.05; empirical 0.049 in
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:69-79`).
- *BH-corrected coverage under H₁ (one stratum at 1.5× sigma
  inflation):* stratum-3 detection power ≥ 0.95; other strata
  false-positive rate ≤ 0.07 (empirical 1.000 and 0.018–0.024 in
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:81-89`).
- *κ(A) under v_min-aligned updates:* strictly monotone non-increasing
  over 30 rank-one updates (zero violations); under random-direction
  updates, monotonicity broken (≥ 1 strict increase). Eckart-Young 1936
  plus Stewart-Sun 1990 interlacing, reproduced in
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:99-115`.
- *Target-set entropy `H_T` scaling:* a 7-point sweep
  `N ∈ {25, 50, 100, 200, 400, 800, 1600}` with focused target around
  `c_0 = +1.2, m = 1.5` TeV. Combined fit `H_T = a + b/N + c/log N`
  gives `b/c ≈ 79`; pure-1/N fit has SSE within 2% of combined; pure-
  1/log N is 6× worse
  (`/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:120-149`).
  Log-log slope of `Σ lev_T` vs N is `-1.66 ± 0.2`. This is the
  EPIG-effective regime.

**Pass / fail criteria.**

| sub-test         | criterion                                                              |
|------------------|------------------------------------------------------------------------|
| (i) regressor R² | held-out R² ≥ 0.997 at n_train = 50                                    |
| (i) constraint   | per-op R² ≤ -20 on operator held at zero during training               |
| (ii) probe R²    | R²(c_lq^(3)) in-distribution ≥ 0.55; vertex |R²| < 0.10                |
| (ii) downstream  | task-swap R² ≥ 0.30 with no encoder retrain                            |
| (iii) EPIG       | max |diff| ≤ 1e-12 over 200 pairs                                      |
| (iv) refit       | refit cov_68 in [0.668, 0.728]; no-refit |Δ| ≥ 5pp on pathological     |
| (iv) broader cal | broader-cal cov_68 ≥ 0.679; narrow-cal cov_68 ≤ 0.638 on extrapolation |
| (iv) runtime     | `OutdatedCalibratorError` raised exactly once on the unrefit path      |
| (v) DAS H₀       | ARL_0 ∈ [200, 800] over 1000 streams                                   |
| (v) DAS H₁ δ=0.5 | detect rate ≥ 0.70; median delay ∈ [80, 130]                           |
| (v) BH H₀        | any-rejection rate ≤ 0.07                                              |
| (v) BH H₁        | stratum-3 power ≥ 0.95; others ≤ 0.07                                  |
| (v) κ monotone   | 0 violations under v_min-aligned, ≥ 1 under random over 30 updates     |
| (v) H_T scaling  | pure-1/N SSE within 5% of combined; combined `b/c` ≥ 30                |

Any single criterion failing forces a fix before Test 3 runs.

**Estimated wall time.** 10–30 minutes total on CPU. Breakdown:
- (i) regressor baseline: ~30 s.
- (ii) probe protocol with 250-batch FM training: 15–20 min.
- (iii) EPIG closed-form vs brute-force, 200 pairs: 30 s.
- (iv) refit guarantee + broader-cal: 60 s.
- (v) DAS-CUSUM 1000 trials: 6 s; BH 1000 trials: 3 s; κ monotonicity:
  1 s; H_T scaling sweep (N up to 1600): 4 min.

**What this test does NOT verify (passed to Test 3).**

- The closed loop runs in real time with Phoenix as the trace store and
  MCP server: Test 2 mocks both.
- The promotion rule for A/B-tested FM versions: Test 2 evaluates a
  single FM/probe pair.
- Drift evaluators fire correctly on a real (not synthetic) residual
  stream: Test 2 uses Gaussian streams that DAS-CUSUM is calibrated for.
- The engineered drift event ("withhold a c-region from training and
  verify the loop closes") runs end-to-end: Test 3.
- The 15-hour budget projection is realistic: Test 3 is the integration.
- The MCP-mediated reflexive use ("the agent itself queries Phoenix
  mid-run"): Test 3.
- The Particle Transformer scaling falsification (whether `1 - R² ∝
  N^{-c}` with `c > 0.05`): Test 3 captures the FM-vs-regressor scaling
  signature that
  `/home/vince/ALETHIA/docs/research/02-foundation-model/empirical-results.md:99-116`
  found flat at small scales.

---

## Test Case 3 — Full chain, 10–20 hour run

**What it demonstrates.** ALETHIA, end-to-end, on a clock budget
consistent with the hackathon demo: ~50 000 T1 + ~5 000 T2 + 100 T5
oracle calls
(`/home/vince/ALETHIA/docs/research/01-oracle/summary.md:520-548`,
adjusted to the PDF-fix factor of 6 from
`/home/vince/ALETHIA/docs/research/01-oracle/empirical-results.md:22-27`).
An engineered drift event — `|c_lq^(3)| ∈ [0.6, 1.0]` withheld from
stage-0 training — is detected by the coverage evaluator, the EPIG layer
selects new commissions inside the band, the oracle returns labels at
appropriate fidelity, the probe and conformal layer refit, the A/B
experiment promotes the new model, and Phoenix records the whole tree
of spans. The agent itself uses the Phoenix MCP to read its own trace
history mid-run, satisfying the rubric's reflexive criterion.

**Compute budget and stage schedule.** Single-machine, 8-core CPU
workstation per
`/home/vince/ALETHIA/docs/research/01-oracle/empirical-results.md:61-83`:

- *Stage 0 (FM pre-train, ~3 h).* 4 operators activated (`cHq3`, `cHq1`,
  `clq3`, `clq1`), `|c| ∈ [0, 0.6]` (the `c_lq^(3) ∈ [0.6, 1.0]` band is
  withheld). InfoNCE training of the rate-aware DeepSets encoder
  (8k batches, B=16, N=512), then upgrade to a small Particle
  Transformer (10⁶ parameters, 2k batches). All training at T1 fidelity
  (analytic + CT18NNLO PDF). Per-operator probe R² recorded at every
  500 batches; saved to
  `/home/vince/ALETHIA/cache/enc_stage0.pt`.
- *Stage 1 (initial probe + cal fit, ~30 min).* Sample 2 000 commissions
  uniformly on the un-withheld region; oracle at T1; embed; fit
  `LinearProbe.fit(Z, μ)` (target = μ at m=1.5 TeV, the EFT-energy-growth
  diagnostic bin). Fit `ConformalCalibrator` on a held-out 800-point cal
  set drawn from `|c| ≤ 0.8` (deliberately broader than training, but
  still excluding the withheld band — as in the
  broader-probe-region pattern but without leaking the drift). Snapshot
  state as `fm.v0`.
- *Stage 2 (active loop, ~9 h).* Repeat:
  1. Draw a probe batch of 500 commissions from the broader probe region
     including the withheld band; predict, oracle at T1, evaluate drift.
  2. When drift fires, schedule a T2 batch (escalation per
     `/home/vince/ALETHIA/docs/research/01-oracle/summary.md:528-540`)
     to verify the failure; if persistent for `N = 3` consecutive
     windows per the persistence rule of
     `/home/vince/ALETHIA/docs/research/03-drift/summary.md:351-362`,
     EPIG selects k=20 commissions inside the failing cluster's
     `suggested_target_z`.
  3. Oracle at T1 for the EPIG batch (escalate to T2 for the top 5%
     by `||c||_2 > 0.5` per
     `/home/vince/ALETHIA/docs/research/01-oracle/summary.md:528-540`).
  4. `probe.update(z_new, y_new)`; `cc.fit(probe, Z_cal_new, ...)` with
     the cal set composition tracking the probe-region composition per
     `/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:111-122`.
  5. Spawn a Phoenix A/B experiment comparing the pre- and post-update
     probe on a fixed 1 000-point probe set; apply the promotion rule of
     `/home/vince/ALETHIA/docs/research/03-drift/summary.md:559-572`.
  6. Bump `fm.version` if promoted; otherwise log `promote_failed`.
- *Stage 3 (NLO anchor + headline, ~2–3 h).* On the final promoted
  version, run 100 T5 oracle calls at the high-confidence points
  (`||c||_2 < 0.3`, `m_ll > 2 TeV` per the same energy-growth bin) and
  fit a fidelity-correction term per
  `/home/vince/ALETHIA/docs/research/03-drift/eval.md:53-69`. Capture
  the headline plot.

**What gets logged to Phoenix and how often.**

- Every oracle call emits one `oracle.query` span with
  `aletheia.oracle.fidelity_tier`, `aletheia.oracle.backend`,
  `aletheia.oracle.cost_seconds`, `aletheia.oracle.cache.hit`,
  `aletheia.oracle.wilson_norm = ||c||_2`,
  `aletheia.oracle.linked_predict_span_id`,
  `aletheia.oracle.commission_id`
  (`/home/vince/ALETHIA/docs/research/01-oracle/eval.md:80-83` plus the
  individual-c-suppression rule of
  `/home/vince/ALETHIA/docs/research/01-oracle/summary.md:441-447`).
- Every probe predict batch emits one `surrogate.predict` (renamed
  `fm.predict`) span with `aletheia.fm.version`, `aletheia.fm.n_query`,
  `aletheia.fm.embed_norm_*`, `aletheia.fm.leverage_*`,
  `aletheia.fm.mu_mean`, `aletheia.fm.sigma_mean`,
  `aletheia.fm.region_id`
  (`/home/vince/ALETHIA/docs/research/03-drift/summary.md:386-393`).
- Every drift evaluator firing emits a `drift.evaluate` chain plus three
  child spans (`drift.accuracy.das_cusum`, `drift.calibration.bh`,
  `drift.coverage.kappa`) with the per-evaluator attributes of
  `/home/vince/ALETHIA/docs/research/03-drift/summary.md:399-416`. The
  evaluator runs once per loop iteration; ~5 000 cycles total per
  `/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:130-140`.
- Every `fm.update` span carries `aletheia.fm.target_entropy_H_T_pre`,
  `aletheia.fm.target_entropy_H_T_post`, `aletheia.fm.kappa`,
  `aletheia.fm.kappa_delta`, plus `aletheia.fm.target_entropy_by_region`
  as a list per
  `/home/vince/ALETHIA/docs/research/04-epig-conformal/eval.md:88-107`.
  ~100 `fm.update` spans expected in the run.
- Every A/B comparison spawns an `experiment.ab` chain with
  `aletheia.ab.promote`, `aletheia.ab.improvement_drifted`,
  `aletheia.ab.no_regression`
  (`/home/vince/ALETHIA/docs/research/03-drift/summary.md:432-438`).
- Total span volume: ~50 000 + 5 000 + 100 oracle calls ≈ 55 100
  oracle spans, ~10 000 fm.predict spans (probe batches of size 500),
  ~5 000 drift.evaluate spans, ~100 fm.update + experiment.ab pairs.
  Trace volume on disk via the docker-compose Phoenix container at
  `/home/vince/ALETHIA/docker-compose.yml:14-32`: ~500 MB
  (`/home/vince/ALETHIA/docs/research/01-oracle/summary.md:402-408`).

**Engineered drift event.** Withhold all commissions with
`c_lq^(3) ∈ [0.6, 1.0]` from stage-0 InfoNCE training, from the
stage-1 probe fit, and from the cal set
(`|c_lq^(3)| ≤ 0.6` is enforced on every sampler the encoder and
calibrator see). The probe queries in stage 2 sample uniformly over
`|c_lq^(3)| ∈ [0, 1.0]`, so 40% of probe batches hit the withheld band.
The expected sequence:

1. Probe queries in the band produce embeddings z whose projection on
   the current probe's `v_min` is unusually large (the encoder has not
   placed those samples on a well-populated manifold direction).
2. `coverage_drift_condition_number` fires:
   `kappa(A)` itself may sit below the soft threshold `10⁴`
   (`/home/vince/ALETHIA/docs/research/03-drift/empirical-results.md:144-155`),
   but the `vmin_projection_ratio` will exceed 3 within the first
   500-query batch after the withheld region is sampled.
3. After `N = 3` consecutive windows
   (`/home/vince/ALETHIA/docs/research/03-drift/summary.md:351-362`),
   `drift.aggregate` emits `action = local_retrain` with
   `target_region_id` = the cluster of high-`v_min`-projection embeddings,
   and `suggested_target_z` = the worst-projection sample from that
   cluster.
4. `epig_acquire_commissions(probe, candidates, embed_preview,
   z_target=suggested_target_z, k=20)` selects 20 commissions; at least
   18 of them have `c_lq^(3) ∈ [0.6, 1.0]` (the test asserts this).
5. Oracle returns labels at T1 (15 of 20) and T2 for the 5 highest-
   `||c||_2` commissions.
6. `probe.update(z_new, y_new)` is called; the conformal layer is refit
   immediately on a freshly drawn cal set whose composition mirrors the
   updated probe-region traffic (per the cal-tracking invariant).
7. Coverage on the previously failing region — empirical cov_68 on
   embeddings with `c_lq^(3) ∈ [0.6, 1.0]` — returns to nominal 0.683
   within ±5pp; `H_T_post` < `H_T_pre`; `kappa_post < kappa_pre`.
8. A/B promotion: RMSE on the drifted region improves by ≥ 5%, no
   region regresses past the BH-corrected bootstrap CI at FDR 0.05;
   coverage on the drifted region is within ±2pp of nominal at both
   0.683 and 0.954
   (`/home/vince/ALETHIA/docs/research/03-drift/summary.md:559-572`).

**Phoenix MCP queries issued by the agent itself, mid-run.** The agent
script invokes the Phoenix MCP server (config at
`/home/vince/ALETHIA/.gemini/settings.json:2-15`) at three pre-arranged
points. The MCP tools used are listed in
`/home/vince/ALETHIA/docs/research/03-drift/summary.md:82-88`.

1. *After 500 cycles* (≈ 1.5 h): `list-projects`, `list-traces`,
   `get-spans` filtered on `aletheia.drift.action != "noop"`. The agent
   summarises "how many drift events have fired, by region" and includes
   this in its next FM.update decision narrative. Reflexive criterion 1.
2. *After every promotion event*: `list-experiments-for-dataset` on
   the probe-set dataset, `get-experiment-by-id` on the latest two; the
   agent compares per-region RMSE deltas and writes a one-paragraph
   "what changed" annotation back to the `experiment.ab.decision` span
   via `phoenix.Client().log_evaluations`. Reflexive criterion 2.
3. *At the run's end*: `get-spans` filtered on `name == "fm.update"`,
   projecting `aletheia.fm.target_entropy_H_T_post` over time;
   `get-spans` filtered on `aletheia.drift.action ==
   "local_retrain"` to produce the per-region trajectory list. The agent
   emits a final natural-language run summary which is captured as the
   demo voiceover. Reflexive criterion 3.

**Headline plot.** A 4-panel figure built from the Phoenix span data via
MCP query + a single `scripts/plot_headline.py`:

| panel | content                                                                                  |
|-------|------------------------------------------------------------------------------------------|
| (a)   | `H_T` vs cumulative oracle calls N, log-log axes; the EPIG-effective slope is -1 to -2   |
| (b)   | per-region coverage at 0.683 and 0.954, trajectories over time, with the drifted-region  |
|       | recovery clearly visible                                                                 |
| (c)   | `kappa(A)` trajectory with annotated promotion events                                    |
| (d)   | EPIG efficiency vs random baseline: cumulative `H_T` reduction per oracle call           |
|       | for the EPIG branch vs a paired random-acquisition branch run on the same probe set      |

The empirical scaling
(`/home/vince/ALETHIA/docs/research/04-epig-conformal/empirical-results.md:120-149`)
predicts panel (a) shows log-log slope ≈ -1.66 in the EPIG-effective
regime, slowing to ≈ -1 at the 10⁴+ scale as the pool's T-relevant
information saturates. Panel (d) should give EPIG efficiency ≥ 1.3×
random per `/home/vince/ALETHIA/docs/research/02-foundation-model/notable.md:21-23`.

**Pass / fail criteria (rubric-presentable).**

| criterion                                                | target                                        |
|----------------------------------------------------------|-----------------------------------------------|
| run completes in wall time                               | ≤ 20 h                                        |
| total spans logged to Phoenix                            | ≥ 50 000                                       |
| coverage drift fires after withheld band exposed         | within 5 probe batches (≤ 2 500 queries)      |
| EPIG selects inside the withheld band                    | ≥ 90% of picks in `c_lq^(3) ∈ [0.6, 1.0]`     |
| promotion succeeds at least once                         | ≥ 1 `experiment.ab.promote = True`            |
| coverage on previously failing region post-promotion     | within ±5pp of nominal at 0.683 and 0.954     |
| `H_T_post < H_T_pre` on the promotion update             | strict inequality, 95% of updates             |
| log-log slope of `H_T` vs N                              | between -2.0 and -0.7                         |
| EPIG efficiency vs random baseline                       | ≥ 1.3× cumulative `H_T` reduction             |
| Phoenix MCP queries land at the agreed points            | all 3 events captured in trace                |
| no `OutdatedCalibratorError` raised after a promotion    | exact (the orchestrator must refit on update) |
| no Wilson coefficient appears on any `fm.predict` span   | filter `aletheia.fm.*.c*` returns zero spans  |

The last criterion is the constraint-check at runtime: if any span
attribute named `aletheia.fm.c_*` exists, the FM has seen `c` on its
forward path and the rebuild has regressed.

**Pre-run checklist.**

1. Phoenix container running. `make phoenix-up`; verify
   `curl -fsS http://localhost:6006 > /dev/null`. The container is
   defined at `/home/vince/ALETHIA/docker-compose.yml:14-32`.
2. Phoenix MCP server reachable. `.gemini/settings.json` already has
   `@arizeai/phoenix-mcp` configured per
   `/home/vince/ALETHIA/docs/research/03-drift/summary.md:60`.
3. LHAPDF vendored. `ls /home/vince/ALETHIA/vendor/lhapdf` should show
   the CT18NNLO files (`/home/vince/ALETHIA/scripts/install_lhapdf.sh`
   builds it if missing). The PDF-fix from Diff 1 of
   `/home/vince/ALETHIA/docs/research/01-oracle/eval.md:55-58` must be
   applied so each oracle call costs 3.2 ms rather than 19 ms.
4. MadGraph installed for T2/T5 calls. `ls vendor/MG5_aMC` shows the
   distribution; `oracle_madgraph.py` import succeeds.
5. uv env synced. `uv sync` from
   `/home/vince/ALETHIA/pyproject.toml`; verify
   `uv run python -c "from modules.surrogate.encoder import EventSetFM"`.
6. Encoder weights present. `ls cache/enc_stage0.pt`; if missing, stage 0
   runs as part of the test (adds ~3 h).
7. Test 1 and Test 2 green within the last 24 h. If not, re-run them
   first; Test 3 assumes them as preconditions.
8. Plot script. `ls scripts/plot_headline.py` produces the 4-panel
   figure from Phoenix data.

**Commands.**

```bash
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
make phoenix-up
curl -fsS http://localhost:6006 > /dev/null     # health check
uv sync                                          # env sync
uv run pytest -xvs tests/test_smoke_machinery.py \
                     tests/test_theory_check.py \
                     tests/test_epig_closed_form.py \
                     tests/test_conformal_refit_guarantee.py \
                     tests/test_drift_synthetic_streams.py
# preconditions green; launch the long run
uv run python scripts/run_alethia_endtoend.py \
    --stage0-batches 8000 \
    --stage1-pool 2000 \
    --stage2-cycles 5000 \
    --t1-budget 50000 \
    --t2-budget 5000 \
    --t5-budget 100 \
    --withhold-clq3 0.6 1.0 \
    --phoenix-project alethia \
    --plot-out artifacts/headline.png
```

**Estimated wall time.** 10–20 h total. Breakdown per
`/home/vince/ALETHIA/docs/research/01-oracle/empirical-results.md:61-83`:

- Stage 0 FM pre-train: 3 h CPU.
- Stage 1 setup: 30 min.
- Stage 2 active loop: 8–10 h, dominated by T2 MG runs (5 000 calls /
  7 cores × 30 s ≈ 6 h, with the 10× T2 amortisation of
  `/home/vince/ALETHIA/docs/research/01-oracle/empirical-results.md:81-83`
  it can drop to ~40 min, but allow 6 h for safety).
- Stage 3 NLO anchor: 2.4 h (100 T5 calls / 7 cores × 600 s).
- Slack: 2–6 h for debugging and the headline plot.

**Risk register.** Three risks to monitor mid-run via MCP:

1. *Cache footprint approaching 1 GB.* If
   `get-spans` summing `aletheia.oracle.cache.bytes` exceeds 800 MB,
   trim T1 rows older than the latest promotion (the script supports
   `--cache-trim-after-promote`).
2. *T2 worker pool blocking on a single hung MG run.* The
   `oracle.cost_seconds` distribution should be tight around 30 s;
   anything above 120 s is hung. Kill via the `timeout_s` knob.
3. *Drift never fires because the withheld-band exposure is too gentle.*
   If after 1 000 probe batches no `drift.aggregate` span has
   `action != noop`, the script should warn and increase the withheld-
   band probe weighting from 40% to 70%.

**What this test does NOT verify (acknowledged scope).**

- That the FM scaling is FM-shaped (`1 - R² ∝ N^{-c}` with `c > 0.05`):
  this is a separate falsification per
  `/home/vince/ALETHIA/docs/research/02-foundation-model/eval.md:110-117`
  requiring 10³ to 10⁵ batches across two encoder architectures.
- That the dim-8 SMEFT extension fits the same architecture: the brief
  expects only dim-6 here
  (`/home/vince/ALETHIA/docs/research/01-oracle/summary.md:215-220`).
- That the cost-normalised EPIG dominates uniformly: agent (d)'s
  notable #5 (`/home/vince/ALETHIA/docs/research/04-epig-conformal/notable.md:13`)
  is a follow-up experiment.
- That Phoenix scales beyond a single agent's traces: the rubric is
  scored on this single-agent demo; multi-agent persistence is out of
  scope.
- That the analytic oracle and MadGraph agree for `c_lq^(1)`: the PDF-
  independence claim has a 13.5% counterexample at `(c_lq^(1) = 0.5,
  m_ll = 0.5 TeV)` per
  `/home/vince/ALETHIA/docs/research/01-oracle/empirical-results.md:34-40`.
  Test 3 stays inside the `c_lq^(3)` direction where the analytic oracle
  is trustworthy.

---

## Composition

Test 1 prerequisites are minimal: only the additive interfaces
(`OracleResult`, `sample_events`, `LinearProbe`, `OutdatedCalibratorError`,
`modules/drift/`, `epig_acquire_commissions`). Test 2 assumes Test 1
passes — the same code paths plus a pre-trained encoder, the brute-force
posterior refit harness, and the synthetic stream generators. Test 3
assumes Test 2 passes — the same code paths plus Phoenix up, MadGraph
installed, the engineered drift scenario script, and the headline plot
generator. Failure of any criterion in Test n forces a fix before
Test n+1; the chain is the gate. The deliverables of all three tests
together are: (Test 1) one CI-runnable smoke test that proves wiring;
(Test 2) a pytest suite that pins every closed-form prediction in the
joint design to a numerical value; (Test 3) one captured Phoenix trace
and one headline figure that constitute the hackathon demo material.
