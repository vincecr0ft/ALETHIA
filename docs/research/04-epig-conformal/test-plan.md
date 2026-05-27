# Test plan: EPIG, conformal, and the loop

Tests are split into three groups: (A) closed-form algebraic checks that the existing code matches the math, (B) loop-level behavioural checks that the iterative procedure has the asserted guarantees, and (C) negative controls that document the expected failure modes. Where a test requires a code change first, the change is sketched.

## Group A: closed-form checks against the existing code

### A.1 EPIG sequential vs batch variance reduction

Goal: confirm sequential greedy strictly beats batch one-shot on a clustered pool, matching the Schur-complement prediction of §3.

Construct a candidate pool with two tight feature clusters (deliberately A^{-1}-collinear). Build a fixed target set. Pick k=10 by:

- sequential greedy with Sherman-Morrison (the existing epig_acquire);
- batch one-shot (top-k by static EPIG against unchanged A^{-1}); needs a small helper in acquisition.py (does not exist yet, sketch below).

Compute target-set variance after picking (refit the model on cal data plus the picks). Assert sequential reduction >= batch reduction, with a quantitative gap that matches the closed-form prediction (the Schur complement). Run on 50 seeds.

```python
def epig_acquire_batch_oneshot(model, C_pool, M_pool, C_target, M_target, k):
    # top-k by static EPIG, no Sherman-Morrison update
    Phi_pool = phi_joint(C_pool, M_pool); Phi_T = phi_joint(C_target, M_target)
    A_inv = model.A_inv
    lev_T  = np.einsum("id,de,ie->i", Phi_T, A_inv, Phi_T)
    lev_P  = np.einsum("pd,de,pe->p", Phi_pool, A_inv, Phi_pool)
    K_TP   = Phi_T @ A_inv @ Phi_pool.T
    var_red = K_TP**2 / (lev_P[None, :] + 1.0)
    ig = 0.5 * np.log(lev_T[:, None] / (lev_T[:, None] - var_red))
    scores = ig.mean(axis=0)
    return np.argsort(-scores)[:k]
```

File: tests/test_acquisition_sequential_vs_batch.py (new).

### A.2 EPIG Cauchy-Schwarz numerical tightness

Already covered by tests/test_acquisition.py:77-92. Strengthen by reporting the gap (lev_T lev_p - k_Tp^2) / (lev_T lev_p) and asserting median across 1000 candidates is < 0.1 (the bound is generically not tight).

### A.3 EPIG against direct GP posterior

Closed-form check that the IG returned by epig_acquire equals what you get by manually computing the Gaussian posterior variance reduction. Take one candidate, one target, evaluate IG by both routes; assert equality to 1e-10. Catches sign errors and broadcasting bugs.

File: tests/test_acquisition.py (extend).

## Group B: loop-level behavioural checks

### B.1 Strict variance reduction at the target (Guarantee A)

Start from the model fit on stage-0 data. Pick one EPIG candidate; refit; compute target-set variance before and after. Assert strict decrease for every target point with k_Tp != 0. Repeat for 50 seeds; report distribution of decreases.

File: tests/test_loop_guarantee_a.py (new).

### B.2 Coverage preserved iff calibration is refit (Guarantee B)

Two parallel runs:

- Refit cal: run one loop iteration, refit ConformalCalibrator, evaluate coverage on a fresh test set drawn from current probe region. Expect coverage in [0.65, 0.71] at nominal 0.683.
- No-refit cal: run one loop iteration, do NOT refit cal, evaluate coverage. Expect coverage outside [0.65, 0.71].

Quantify the gap as a function of cumulative cross-leverage between picks and the calibration set.

File: tests/test_loop_guarantee_b.py (new). This is the bug-class test for §4.3.

### B.3 Monitoring quantities tracked

Smoke test that the four monitoring quantities of §6 are computable from existing state and produce numeric values:

- H_T = 0.5 sum_i log(2 pi e sigma^2 lev_{T_i});
- per-stratum empirical coverage and reliability;
- condition number kappa(A);
- Sherman-Morrison residual ||A_inv_inc - A_inv_fresh||_F (only when applicable).

File: tests/test_monitoring.py (new). Run across a 10-iteration loop and assert: H_T monotone decreasing on average, coverage stable, kappa non-increasing on average.

## Group C: negative controls

### C.1 Calibration-set extrapolation failure (§7)

Fit cal on |c| <= 1. Evaluate on |c| in [1.5, 2]. Assert that the highest-leverage stratum under-covers at nominal 0.954 by more than 5pp. Report the Wasserstein-1 distance between cal and test marginal distributions for context.

File: tests/test_extrapolation_under_coverage.py (new).

### C.2 Sherman-Morrison cache staleness (§5.4)

When IntentionFM.update is replaced by a Sherman-Morrison path, this test runs 1000 incremental updates, refreshes the exact inverse, and asserts ||A_inv_inc - A_inv_fresh||_F / ||A_inv_fresh||_F < 1e-6. Currently moot (update refits) but should be added before Sherman-Morrison is introduced as an optimisation.

File: tests/test_sherman_morrison_residual.py (new, gated).

### C.3 Heteroscedastic noise breaks the closed form (§1.4)

Synthetic oracle with input-dependent noise (e.g. sigma(x) = 1 + 5 |x_0|). Show that EPIG-ranked picks are not the best in terms of realised target variance reduction; rerank by sigma(x)-corrected IG and show this is recovered. Documents the assumption.

File: tests/test_heteroscedastic_failure.py (new).

## Empirical scaling experiment (for second pass)

Generate target-set entropy as a function of cumulative oracle calls N in {50, 100, 200, 400, 800, 1600, 3200}. Fit H_T(N) = a + b/N + c/log N. Report b/c. Conduct on:

- the dummy analytic oracle (cheap; full scaling visible);
- a small MadGraph oracle run (expensive; partial scaling).

Closed-form prediction: in the EPIG-effective regime b/c >> 1; in the saturated regime b/c << 1.

File: scripts/scaling_epig.py (new).

## Tests that should be removed or rewritten

- tests/test_acquisition.py:55-74 tests that EPIG with focused target picks toward target. Useful but limited; should be augmented by Group B tests for the actual guarantee.
- tests/test_calibration.py:41-45 only asserts factor ordering; replace by a real conditional-coverage test on the broader probe region.
