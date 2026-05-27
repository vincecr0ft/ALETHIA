# ALETHEIA: Component Verification and `check_drift` / `compute_epig` Extension Plan

> **Status: SUPERSEDED (2026-05-26).** This plan assumes the foundation
> model takes Wilson coefficients on its forward pass and is fit against
> the morphing ansatz `σ(c) ≈ σ_SM + Σ A_i c_i + Σ B_ij c_i c_j`. That
> assumption now violates the project's hard constraint. The revised
> design is in [docs/research/](docs/research/README.md); the operational
> test plan is at
> [docs/research/synthesis/three-test-cases.md](docs/research/synthesis/three-test-cases.md).
> §0 (algebraic frame), Component 1 (Phoenix), and Component 2 (analytic
> SMEFT) of the document below carry over. Component 3 (the FM) and
> Components 4–6 (drift, EPIG, end-to-end) are superseded.

Instruction document for Claude Code. Three existing components (Phoenix demo, MadGraph + analytic SMEFT, Intention foundation model) get mathematical-completeness test suites. Two new modules (`check_drift`, `compute_epig`) get full specs plus their own test suites. End-to-end integration test closes the loop. Every test below targets a closed-form algebraic prediction the architecture makes, not a vibe.

---

## 0. Algebraic frame the tests target

Every component sits inside one of three closed-form objects. The tests target the algebra, not crash-avoidance.

The **Intention head** is the operator `T(Q) = Q A^{-1} Phi^T Y` with `A = Phi^T Phi + lam I`, where `Phi` is the joint feature map `phi(c, m) = phi_c(c) ⊗ phi_x(m)`. This is Garnelo & Czarnecki Theorem 2.7 (arXiv:2305.10203) in regression notation: `Q = phi(x*)`, `K = Phi`, `V = Y`. Theorem 2.8 of that paper establishes Linear Attention as the `α → ∞` limit; finite `α` is strictly more expressive. The kernel-ridge / Gaussian-process dual (Rasmussen & Williams 2006, Chapter 6) gives the predictive variance `Var(f(x)) = σ² · phi(x) A^{-1} phi(x)^T = σ² · lev(x)` in closed form at noise-variance/regularisation correspondence `σ² ↔ α`. Posterior updates after one new `(phi_p, y_p)` go through Sherman-Morrison: `A_new^{-1} = A^{-1} - (A^{-1} phi_p)(phi_p^T A^{-1}) / (1 + phi_p^T A^{-1} phi_p)`. Exact to numerical precision.

The **three drift signals** land directly on this algebra. Accuracy drift is a change-point in the residual stream `||Y - Phi w||₂`. Calibration drift is the empirical coverage of the conformal interval, which under the GP dual is a function of `lev(x)` and `α`. Coverage drift is the eigenvalue distribution of `A`; equivalently, the condition number `κ(A)`, which through SVD upper-bounds the predictive-variance amplification factor. Each drift signal is a function the FM already computes; `check_drift` is a thin wrapper plus sequential statistics.

**EPIG** (Smith, Bickford Smith, Rainforth 2023, arXiv:2304.08151) collapses to closed form under the Bayesian linear regression: `IG_T(p) = 0.5 · log( lev_T / (lev_T - k_Tp² / (lev_p + 1)) )` with `k_Tp = phi_T A^{-1} phi_p^T`. Cauchy-Schwarz on the `A^{-1}` inner product makes IG non-negative and bounded. The `σ²` cancellation is exact under the homoscedastic assumption used in the ridge fit. No Monte Carlo retraining required.

The central architectural hypothesis of the programme, which Component 3's identifiability test must adjudicate: for a θ-stratified training distribution sweeping the morphing basis, the row-space of `K` admits a linear probe to the Wilson coefficient basis up to a unitary rotation in polynomial-coefficient space. This is a conjecture, not a theorem; the test is its empirical adjudication, and its result determines whether the FM-as-theory-imprint framing survives in the writeup.

---

## Component 1: Phoenix demo

### State

Arize-ai/gemini-hackathon starter forked. ADK agent wired with OpenInference. Traces flow to local Docker Phoenix. Phoenix MCP config in `.gemini/settings.json`.

### Tests (`tests/test_phoenix_integration.py`)

1. **`test_span_hierarchy`**: With a stub tool, trace tree shows `agent_run → tool_call → sub_tool_call`. Sub-spans inherit `trace_id`. Failure mode: instrumentation drops nested spans, judges see a flat trace.

2. **`test_trace_persistence`**: Run agent. Restart Phoenix container. Trace history survives via volume-mounted SQLite.

3. **`test_mcp_self_query`**: From inside an agent tool, invoke the Phoenix MCP `get_recent_spans` tool. Confirm the agent reads its own trace history. This is the reflexive loop the rubric scores.

4. **`test_experiment_compare`**: Register two prompt variants as Phoenix experiments. Run 20 invocations each. Verify per-variant score breakdown and that latency, error, and custom-metric columns populate.

5. **`test_input_output_capture`**: Stub tool with a structured Pydantic input. Confirm full input/output JSON appears in span attributes, not truncated. Phoenix's `max_attribute_length` setting must accommodate the largest expected SMEFT payload (Wilson vector + bin array, ~5 KB).

### Remediation rule

If 1–5 fail on the unmodified starter, file an issue on `Arize-ai/gemini-hackathon` before forking. Do not paper over with custom instrumentation; the rubric weights "meaningful use of tracing and MCP" enough that regressed-starter is unwinnable.

---

## Component 2: MadGraph and analytic SMEFT

### State

`modules/analytic_smeft.py` implementing closed-form LO Drell-Yan with Warsaw-basis Wilson coefficients (cHq3, cHq1, clq3, clq1 minimum; extension set documented). MadGraph subprocess wrapper as second implementation of the same interface, gated behind `try/except` on LHAPDF availability.

### Mathematical tests (`tests/test_analytic_smeft.py`, must pass without MadGraph)

1. **`test_sm_limit`**: All Wilson coefficients zero. Output matches a tabulated reference SM Drell-Yan differential cross section in m_ll bins [200, 400, 600, 1000, 2000] GeV to 5%.

2. **`test_linear_interference`**: Set `c_lq^(1)` at {0.01, 0.1, 1.0}, others zero. Fit `(σ - σ_SM)` linearly in `c_lq^(1)` per bin, require R² > 0.999.

3. **`test_sign_flip`**: `c → -c` flips the interference sign in every bin. Verify exact algebraic relation `(σ(c) + σ(-c) - 2σ_SM) / 2` equals the quadratic piece to numerical precision (relative tolerance 1e-10).

4. **`test_quadratic_growth`**: With `order='quadratic'`, the BSM² piece in tail bins (m_ll > 1 TeV) grows as `(m_ll / Λ)²` relative to SM. Fit a power law on [800, 2000] GeV; exponent 2.0 ± 0.1.

5. **`test_no_nan`**: For `|c_i| ≤ 1.0` across all operators, no bin returns NaN or inf at `Λ = 1` TeV.

### Cross-validation tests (`tests/test_madgraph_crossval.py`, MadGraph required, marked `slow`)

6. **`test_analytic_vs_madgraph_grid`**: 20 randomly sampled Wilson configurations through both pipelines. Per-bin relative disagreement `(analytic - MG) / MG` satisfies median < 5%, 95th percentile < 15%. Outliers in highest-|c| points are expected; flag, don't fail, if pattern matches missing higher-order EFT.

7. **`test_disagreement_grows_with_c`**: Fit `log|disagreement|` vs `log|c|`. Slope positive and consistent with the missing `Λ^-4 / Λ^-6` ratio. This characterises the regime in which the agent can trust the analytic oracle; basis for fidelity-routing decision in the end-to-end test.

8. **`test_regression_pinned`**: Pin specific `(Wilson config, bin) → cross-section` values from a reference run. Future refactors must reproduce: analytic to relative tolerance 1e-10, MadGraph to 1e-3 with `rng_seed=42` fixed.

### Physics-sanity tests

9. **`test_dimensional_analysis`**: Output units are pb/GeV. Integrate over a bin: units check, integrated cross section in [50, 200] pb range expected.

10. **`test_PDF_propagation`** (optional, MadGraph only, marked `slow`): Vary PDF set across CT18NNLO members; verify the PDF uncertainty band is reported and falls in published Drell-Yan ranges.

---

## Component 3: Foundation model

### State

`smeft_surrogate/model.py` with `IntentionFM` class. Joint feature map `phi(c, m) = phi_c(c) ⊗ phi_x(m)`:
- `phi_c(c)` is the exact SMEFT polynomial basis `{1, c_a, c_a·c_b}`. For `N_WC = 4`: `D_C = 1 + 4 + 10 = 15`.
- `phi_x(m)` is a polynomial in `log(m / M_ref)` of degree `K_X - 1 = 4`. `D_X = 5`.
- Joint dim `D_JOINT = D_C · D_X = 75`.

Closed-form ridge: `w = A^{-1} Phi^T Y`, `A = Phi^T Phi + λI`. Predictive variance: `σ²_total = (noise_frac · |μ|)² · (1 + leverage)`. Methods: `fit`, `predict`, `leverage`, `update` (Sherman-Morrison), `acquire` (sequential greedy on leverage), `state_dict` / `from_state_dict` for versioning. `ConformalCalibrator` separate class, leverage-stratified split-conformal with exchangeability-corrected rank `ceil((n+1) · coverage) / n`.

### Algebraic tests (`tests/test_intention_fm.py`)

1. **`test_closed_form_recovery`**: Generate `y = Phi w_true + eps`, `eps ~ N(0, σ²)`. Fit at `λ = σ² / ||w_true||²` (Bayes-optimal regulariser). Posterior mean recovers `w_true` within `σ / √N` in expectation. Posterior covariance matches `σ² A^{-1}`. Run 20 seeds, verify via empirical mean and covariance of fitted weights.

2. **`test_sherman_morrison_consistency`**: Fit on `N=200` points. Update with one new point via Sherman-Morrison rank-one. Compare to fresh fit on `N+1` points. Posterior mean and covariance agree to relative tolerance 1e-10.

3. **`test_predictive_variance_geometry`**: Query points inside the convex hull of training K-rows have smaller predictive variance than points outside, by the closed-form Mahalanobis-distance ratio under metric `A^{-1}`. Rank 200 query points by `lev(x)`; Spearman ρ with physical extrapolation distance > 0.85.

4. **`test_condition_number_growth`**: Generate training Ks with controlled collinearity (rotate to `k` near-degenerate principal directions, controlled by collinearity parameter `δ ∈ [0, 1]`). As `δ → 0` (collinearity increases), `κ(A)` grows; predictive variance at any held-out query amplifies by a factor matching the SVD-predicted bound `(1 + σ_min^{-2})` to 5%. This is the algebraic content of coverage drift.

5. **`test_GP_dual_predictive_variance`**: For `λ` matched to noise variance, the closed-form `qᵀA^{-1}q` predictive variance equals the GP-posterior variance under the linear kernel on `phi`. Compare to a hand-coded GP regressor with linear kernel; relative tolerance 1e-8. Validates the GP/KRR duality the writeup invokes.

### Calibration tests (`tests/test_conformal.py`)

6. **`test_raw_marginal_coverage`**: Homoscedastic ridge predictive interval at 68% nominal. Empirical coverage on 1000 held-out same-distribution points falls in [0.65, 0.71]. Top-20%-leverage stratum drops below 0.65. Both assertions must hold; the second is the documented motivation for the conformal layer and prevents the failure being silently masked.

7. **`test_stratified_conformal_coverage`**: After fitting `ConformalCalibrator` with 5 strata, per-stratum coverage at 0.683 and 0.954 falls within ±3pp of nominal in each stratum, evaluated on a held-out test set drawn from the **broader probe region** (not only training region). Calibration set must also be drawn over the full probe region. This is the trap from the conformal docstring.

8. **`test_calibration_set_extrapolation_failure`**: Negative-control. Fit conformal on `|c| ≤ 1`, test on `|c| ∈ [1, 2]`. Verify high-leverage stratum under-covers, by design. Documents the failure mode; prevents silent regression.

### Identifiability tests (the central architectural hypothesis)

9. **`test_smeft_polynomial_probe`**: Train FM on a θ-stratified analytic-SMEFT corpus (5000 Wilson configurations from a Latin hypercube over `|c| ≤ 0.5`). Fit a linear probe from `K = phi(C_train)` row-space to the held-out Wilson coefficient vector. Pass: R² > 0.9 on at least 3 of 4 Wilson coefficients. R² < 0.7 falsifies the central architectural hypothesis; the writeup must acknowledge this if it occurs.

10. **`test_smeft_probe_vs_softmax_baseline`**: Same probe on same data, with `phi_c` replaced by a softmax-attention mapping `{c → Attention(c, K_basis, V_basis)}` of comparable parameter count. Intention should win by `ΔR² > 0.2`. Not a "proof" of Intention superiority; an empirical characterisation. The Garnelo-Czarnecki separation (their Figure 3) is empirical not a formal lower bound; the writeup must reflect that.

### Update and persistence

11. **`test_state_dict_roundtrip`**: `from_state_dict(state_dict(model))` produces a model giving bitwise-identical predictions on a fixed query batch. Required for Phoenix experiment A/B-ing of FM versions.

12. **`test_update_invariance_to_order`**: Two streams of new `(C, M, Y)` points fed in different orders produce the same final `A^{-1}`, `w`, predictions. Mathematically guaranteed by associativity; the test catches bugs in the Sherman-Morrison cache.

---

## Component 4: `check_drift` module (TO BUILD)

### What it is

Single module exposing three drift detectors as pure functions over the stream of `(query, prediction, oracle)` tuples Phoenix has logged. Returns drift flags plus statistics the orchestrator writes back as Phoenix metrics. No internal state across calls; Phoenix is the state store.

### Interface (`modules/check_drift.py`)

```python
def accuracy_drift_DAS_CUSUM(
    residuals: np.ndarray,          # (y_oracle - mu_pred) / sigma_pred, streaming
    h: float | None = None,         # threshold; auto-calibrate from history if None
    window: int = 500,
) -> dict:
    """
    Data-Adaptive Symmetric CUSUM (Ahad, Davenport, Xie 2022, arXiv:2210.17353).
    Pre- and post-change variance estimated from the residual stream itself,
    making the detector symmetric and threshold-tunable from a single h.
    Returns:
      {
        "drift_detected": bool,
        "change_point": int | None,
        "cusum_pos": float,
        "cusum_neg": float,
        "h_used": float,
        "expected_run_length_h0": float,
      }
    """


def calibration_drift_conformal(
    predictions: np.ndarray,        # mu, shape (n,)
    sigmas: np.ndarray,             # raw model sigma, shape (n,)
    observations: np.ndarray,       # y_oracle, shape (n,)
    leverage: np.ndarray,           # lev = phi A^{-1} phi^T, shape (n,)
    region_label: np.ndarray,       # int region id, shape (n,)
    target_coverage: float = 0.683,
    fdr: float = 0.05,
) -> dict:
    """
    Per-region empirical coverage with Benjamini-Hochberg correction across
    regions. Region is typically a leverage-stratum or a Wilson-space cluster.
    Reference: Araz & Spannowsky arXiv:2512.17048; Gibbs-Candès JMLR 2024 for
    the online-adaptive extension if streaming.
    Returns:
      {
        "per_region_coverage": np.ndarray,
        "per_region_pvalue": np.ndarray,
        "regions_drifted": np.ndarray,    # bool, BH-corrected
        "global_drift_detected": bool,
      }
    """


def coverage_drift_condition_number(
    A_history: list[np.ndarray],    # A_t = Phi_t^T Phi_t + lam I per tick
    kappa_threshold: float | None = None,
    relative_increase: float = 5.0,
) -> dict:
    """
    Condition-number monitoring. kappa(A) upper-bounds the variance
    amplification factor through the SVD of A. Coverage drift = kappa
    growing beyond baseline, indicating incoming queries probe poorly-
    supported directions in feature space.
    Returns:
      {
        "drift_detected": bool,
        "kappa_history": np.ndarray,
        "kappa_current": float,
        "worst_direction": np.ndarray,    # right singular vector at sigma_min
      }
    """


def aggregate_drift_decision(
    accuracy: dict,
    calibration: dict,
    coverage: dict,
) -> dict:
    """
    Aggregator. Returns:
      {
        "any_drift": bool,
        "drift_type": "none" | "accuracy" | "calibration" | "coverage" | "multiple",
        "recommended_action": "retrain_global" | "retrain_local" | "watch" | "none",
        "drifted_region": dict | None,    # bounds passed to epig_acquire_in_drifted_region
      }
    """
```

### Mathematical tests (`tests/test_check_drift.py`)

1. **`test_DAS_CUSUM_ARL_H0`**: Under no change (residuals i.i.d. N(0, 1)), expected run length to false alarm matches the Lorden-Pollak bound within 20% Monte Carlo error over 1000 streams. Pin `h` such that ARL_0 = 1000, confirm.

2. **`test_DAS_CUSUM_detection_delay`**: Inject mean shift `δ` at known time. Expected detection delay scales as `log(h) / KL(post || pre)`; verify within Monte Carlo error. Symmetric extension over classical CUSUM is the whole reason for DAS.

3. **`test_DAS_CUSUM_symmetry`**: Inject equal-magnitude positive and negative shifts. Detection delays agree to Monte Carlo precision. Classical Page CUSUM would fail one direction.

4. **`test_conformal_coverage_nominal`**: 1000 calibration points drawn exchangeably with test points. Per-region coverage at 0.683 nominal lands within ±3pp across 5 leverage strata, average over 100 seeds.

5. **`test_BH_fdr_control`**: 50 regions simultaneously tested, no real drift. FDR across 1000 trials stays below 0.05 within Monte Carlo error. Verify against the Benjamini-Hochberg theoretical guarantee.

6. **`test_condition_number_monotonicity`**: Append training rows along an existing eigenvector of `A`: `κ(A)` decreases monotonically. Append rows orthogonal to all existing eigenvectors with magnitude below numerical noise: `κ(A)` grows. Both monotonicities to numerical precision.

7. **`test_cross_detector_agreement`**: Construct a regime where oracle truth diverges from the FM via a `Λ^-4` piece the analytic surrogate doesn't model. Both calibration and accuracy detectors fire. Calibration drift fires FIRST (proper-scoring-rule decomposition: accuracy collapses into calibration in expectation, Gneiting-Raftery 2007; Bröcker 2009). Verify ordering.

8. **`test_kappa_predicts_variance_amplification`**: For training sets with controlled `κ(A) ∈ [10, 10^6]`, the predictive variance at a fixed worst-direction query grows as `κ(A) · σ²` to 5%. Validates `κ(A)` as the coverage-drift signal.

9. **`test_aggregator_recommendation_rules`**: 
   - Only coverage drift → `retrain_local` with drifted_region bounds populated.
   - Only calibration drift, no coverage drift → `recalibrate_conformal`. (Add this action to the enum.)
   - Accuracy AND calibration AND coverage → `retrain_global`.
   - None → `none`.

### Phoenix wiring tests (`tests/test_drift_phoenix.py`)

10. **`test_drift_metrics_emit`**: All three detectors emit metrics tagged on the Phoenix span. Mock OTLP collector receives `drift.accuracy.cusum_pos`, `drift.calibration.global_pvalue`, `drift.coverage.kappa`.

11. **`test_drift_triggers_evaluator`**: On detected drift, the Phoenix evaluator marks span `needs_retraining` and writes a structured payload identifying the drifted region. Evaluator output schema is consumable by `compute_epig`.

---

## Component 5: `compute_epig` module (TO FORMALISE)

### State

`acquisition.py` in the demo includes `epig_acquire` as a sequential-greedy Sherman-Morrison routine. Closed-form derivation already documented in the module docstring. The module needs to be:
- extracted to a standalone production module
- given Phoenix-evaluator integration (consume `drifted_region` from `check_drift`)
- test-suited

### Interface (`modules/compute_epig.py`)

```python
def compute_epig(
    model: IntentionFM,
    C_pool: np.ndarray,
    M_pool: np.ndarray,
    C_target: np.ndarray,
    M_target: np.ndarray,
    k: int,
    method: str = "sequential_greedy",      # or "batch_one_shot"
) -> dict:
    """
    Returns:
      {
        "selected_indices": np.ndarray,   # (k,)
        "epig_scores": np.ndarray,        # (n_pool,)
        "predicted_variance_reduction": np.ndarray,    # (k,)
        "method": str,
      }
    """


def epig_score_single(
    model: IntentionFM,
    phi_p: np.ndarray,        # (D_JOINT,)
    Phi_T: np.ndarray,        # (n_target, D_JOINT)
) -> float:
    """
    Closed form:
      EPIG(p | T) = mean over i in T of
        0.5 · log( lev_T_i / (lev_T_i - k_Tp_i^2 / (lev_p + 1)) )
    where k_Tp = phi_T A^{-1} phi_p^T, lev_T = phi_T A^{-1} phi_T^T.
    sigma^2 cancels under the homoscedastic ridge assumption.
    """


def epig_acquire_in_drifted_region(
    model: IntentionFM,
    drift_region_bounds: dict[str, tuple[float, float]],
    k: int,
    n_pool: int = 1000,
    n_target: int = 100,
    rng_seed: int = 42,
) -> dict:
    """
    Convenience wrapper consuming the drift_region payload from check_drift.
    Samples pool and target candidates from the drifted region, returns
    top-k for oracle invocation.
    """
```

### Mathematical tests (`tests/test_compute_epig.py`)

1. **`test_non_negativity`**: EPIG score ≥ 0 for every candidate against every target set. Information-acquiring never increases uncertainty. Verify across 1000 random `(model, pool, target)` triples.

2. **`test_self_acquisition`**: A candidate with identical `phi` to a target point has strictly higher EPIG than an unrelated candidate. Validates kernel structure.

3. **`test_closed_form_vs_monte_carlo`**: Small problem (`n_pool=20, n_target=20`). Closed-form EPIG matches a Monte Carlo estimate that re-fits the model on each candidate via Sherman-Morrison and computes the posterior-entropy reduction directly. Relative tolerance 5%. Closed form is production; MC is gold reference.

4. **`test_sequential_greedy_no_duplication`**: Sequential greedy with Sherman-Morrison cache must never select the same pool index twice. Verify via `avail` mask.

5. **`test_sequential_better_than_one_shot`**: For `k > 1`, sequential greedy outperforms one-shot top-k. Pool engineered to have one obvious high-leverage cluster; sequential must avoid k-fold clustering. Predicted variance reduction on the target ≥ 1.5× one-shot.

6. **`test_EPIG_collapses_to_leverage_for_uniform_target`**: Target uniformly distributed over the same support as the pool. EPIG ranking agrees with pure leverage ranking, Spearman ρ > 0.95. Confirms EPIG → D-optimal design under uniform target; the connection to the coverage-drift signal.

7. **`test_EPIG_diverges_from_leverage_for_focused_target`**: Target concentrated in a small region of `(c, m)` space. EPIG and leverage rankings disagree materially (Spearman ρ < 0.7). Justifies the orchestrator's choice between `leverage_acquire` and `epig_acquire`.

8. **`test_cauchy_schwarz_bound`**: Verify `k_Tp² ≤ lev_T · lev_p` for every `(T, p)` pair. Algebraic guarantee that makes the log argument bounded in `(0, 1]`. Most likely failure mode this test catches: bug in `phi_joint` feature-map construction.

9. **`test_sigma_invariance`**: Scale `y` and `noise_frac` together by a constant. EPIG scores invariant to numerical precision. Verifies the `σ²` cancellation in the closed form.

### Integration tests

10. **`test_drift_to_epig_pipeline`**: `check_drift` identifies a drifted region. `epig_acquire_in_drifted_region` returns `k` candidates within the region bounds (assert). Candidates have EPIG scores higher than candidates outside the region. End-to-end smoke test of the coverage-drift → EPIG → oracle handoff.

11. **`test_phoenix_evaluator_emit`**: EPIG scores written to span attributes as `acquisition.epig.scores` and `acquisition.epig.selected`. Trace propagation: subsequent oracle-call spans inherit the EPIG decision span as parent.

---

## Component 6: End-to-end loop test

This is the rubric-winning artefact and the demo trace. Component 6 runs only after Components 1–5 all pass.

### Scenario

Withhold training data from a specific Wilson region: `|c_lq^(3)| ∈ [0.6, 1.0]` with `m_ll > 1` TeV. Pretrain on the complement.

### Verification order (`tests/test_end_to_end.py`)

1. **Coverage drift fires first**: `κ(A)` grows when query batch enters the withheld region. `check_drift` emits coverage drift with the correct `drifted_region` payload.

2. **EPIG selects inside the withheld region**: `epig_acquire_in_drifted_region` returns `k` candidates with `c_lq^(3) ∈ [0.6, 1.0]` and `m_ll ∈ [1, 2.5]` TeV. Rejection rate of outside-region candidates > 95%.

3. **Oracle invocation with fidelity routing**: Agent invokes `simulate_analytic_smeft` for low-|c| candidates, `simulate_madgraph` cross-validation for the highest-|c| candidates (informed by Component 2 test 7). Verify routing logic from logged tool selections.

4. **Closed-form head update**: `IntentionFM.update` folds new `(K, V)` rows via Sherman-Morrison. Update completes in < 1 s on dev laptop for batches up to 50 new points.

5. **Calibration drift normalises**: After update, `calibration_drift_conformal` on the previously-drifted region returns coverage within ±3pp of nominal across all strata.

6. **A/B Phoenix experiment promotes the new head**: Register pre-update and post-update models as experiment variants. Drive 100 queries through each over the full probe region. Experiment readout shows: (a) reduced RMSE on the previously-drifted region, (b) restored calibration, (c) no regression on non-drifted regions. Promotion rule: improvement on drifted region AND no significant regression elsewhere, BH-corrected across all regions at FDR=0.05.

### Failure modes to verify explicitly

- Step 2 must not select outside the drifted region. If it does, the EPIG implementation is wrong or the `drift_region_bounds` payload is malformed.
- Step 5 must not over-correct. If calibration on non-drifted regions degrades, the update has memory effects, most likely a numerical bug in `noise_frac` estimation in `IntentionFM.fit`.
- Step 6 must apply BH correction. Without it, the A/B will spuriously reject the new head on naturally noisy regions.

### Demo capture

The Phoenix trace from one clean run of this scenario is the Devpost demo material. Plan multiple attempts; select the cleanest. Capture after all unit tests pass.

---

## Test infrastructure

- `tests/` directory mirrors `modules/`. One test file per module plus `test_end_to_end.py`.
- `pytest`. Each unit test < 1 s wall time; end-to-end may take 30 s.
- `pytest-benchmark` for the Sherman-Morrison update timing in Component 3.
- `hypothesis` for property-based tests on algebraic identities (closed-form vs Monte Carlo, Sherman-Morrison consistency, BH FDR, Cauchy-Schwarz bound).
- Fix `numpy` seed in every random test. Pin reference values from the first passing run; future changes must explicitly bump pins.
- CI runs full suite on every PR. End-to-end is marked `slow` and runs only on `main`.

## Dependencies to add

- `hypothesis` for property tests
- `scipy.stats` for KL divergence in the DAS-CUSUM analytical comparison and for BH utilities

## Citations for the writeup

Tests reference algebra and theorems from:

- Garnelo & Czarnecki, arXiv:2305.10203 (Intention as ridge regression, Theorem 2.7 and 2.8)
- Smith, Bickford Smith, Rainforth, arXiv:2304.08151 (EPIG)
- Ahad, Davenport, Xie, arXiv:2210.17353 (DAS-CUSUM)
- Araz & Spannowsky, arXiv:2512.17048 (conformal calibration for HEP surrogates)
- Gibbs & Candès, JMLR 2024 (online conformal under arbitrary shift)
- Barber, Candès, Ramdas, Tibshirani, Annals of Statistics 2023 (conformal beyond exchangeability)
- Rasmussen & Williams 2006, Chapter 6 (KRR-GP duality)
- Hébert-Johnson, Kim, Reingold, Rothblum, ICML 2018 (multicalibration, writeup only)
- Gneiting & Raftery, JASA 2007; Bröcker, QJRMS 2009 (proper scoring rule decomposition, justifies calibration drift as the dominant signal)

## Order of execution

1. Component 1 tests (Phoenix integration) before anything else; non-negotiable.
2. Components 2 and 3 tests in parallel, since they are independent.
3. Component 4 (`check_drift`) build and tests; depends on Component 3 for the leverage / `A^{-1}` access pattern.
4. Component 5 (`compute_epig`) formalisation and tests; depends on Components 3 and 4.
5. Component 6 end-to-end; runs only after 1–5 are green.
