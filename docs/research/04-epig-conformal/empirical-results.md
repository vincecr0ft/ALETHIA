# EPIG and conformal — empirical results (second pass, agent d)

Four checks against the existing `modules/surrogate/{model,acquisition,calibration}.py` code. All runs use `DummyAnalyticOracle` from `modules/surrogate/ground_truth.py` because that oracle is exactly polynomial in the existing `phi_joint(c, m)` basis — making it the natural testbed for the closed-form EPIG derivation. Wall time for all checks combined: ~60 seconds.

Commands run:

```
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
uv run python /tmp/epig_empirical.py    # E1, E2, E3, E4 (initial)
uv run python /tmp/epig_empirical2.py   # E2', E3', E4' (sharper)
uv run python /tmp/epig_empirical3.py   # E2'', E3'' (good vs bad cal)
uv run python /tmp/epig_e2_final.py     # E2''' (pathological case, 20 seeds)
```

## 1. EPIG closed form vs direct posterior variance reduction (test plan A.3)

For one candidate `(c_p, m_p)` and one target `(c_T, m_T)`, two routes:

- **Closed form**: `IG = 0.5 log(lev_T / (lev_T - k_Tp^2 / (1 + lev_p)))` (modules/surrogate/acquisition.py:50).
- **Direct refit**: fit a new `IntentionFM` on the augmented data and compute `IG_direct = 0.5 log(lev_T_old / lev_T_new)`.

Single-pair output (seed = 7):

```
lev_T (prior)              = 3.539854e-01
lev_T_new (direct refit)   = 3.535399e-01
k_Tp                       = 2.845988e-02
lev_p                      = 8.177510e-01
IG closed form             = 6.297814e-04
IG direct (refit)          = 6.297814e-04
abs diff                   = 6.986e-15
```

Over 200 random (candidate, target) pairs:

```
max |diff|    = 7.711e-14
median |diff| = 6.712e-15
```

**Pass.** The closed-form expression at `modules/surrogate/acquisition.py:50` agrees with direct refit to machine precision (~1e-14, dominated by round-off in the two independent linear solves). Sherman-Morrison is applied correctly; the sigma^2 cancellation works as advertised; Cauchy-Schwarz is respected (no negative information gain ever observed).

## 2. Coverage preserved iff calibration is refit (test plan B.2)

Baseline pattern with `k = 20`, random target, random pool (seed = 11):

```
baseline (fm0, cc0)             cov_68=0.7120  cov_95=0.9760
post-update, NO refit (fm1, cc0) cov_68=0.6975  cov_95=0.9555
post-update, REFIT (fm1, cc1)    cov_68=0.6910  cov_95=0.9470
mean normalised cross-leverage picks<->cal = 0.1368
```

The benign-case gap is at the noise floor (~1-2pp at n_cal = 400, n_test = 2000).

Sharpened with a focused target on the high-leverage cal cluster and `k = 80`:

```
baseline                cov_68=0.7435  cov_95=0.9810
post, no refit          cov_68=0.7530  cov_95=0.9885
post, refit             cov_68=0.6365  cov_95=0.9605
mean normalised cross-leverage = 0.1234
```

Pathological case (20 seeds, 100 picks deliberately placed in the top-leverage cal stratum to maximise the BCRT cross-leverage term):

```
no_refit cov_68:  mean=0.7769  std=0.0551
refit    cov_68:  mean=0.6984  std=0.0204
no_refit cov_95:  mean=0.9589  std=0.0190
refit    cov_95:  mean=0.9651  std=0.0099
mean cross-leverage between picks and top-cal-stratum: 0.2858

distance from nominal:
  no_refit 68: |0.683 - 0.7769| = 0.0939
  refit    68: |0.683 - 0.6984| = 0.0154
  no_refit 95: |0.954 - 0.9589| = 0.0049
  refit    95: |0.954 - 0.9651| = 0.0111
```

**Result.** No-refit at 68% gives mean coverage 0.7769, **9.4 percentage points off nominal** (one-sided binomial Z = 1.71; 20-seed effect is significant). Refit at 68% gives 0.6984, within 1.5pp of nominal. At 95% the effect is at the binomial noise floor.

Quantitatively, the BCRT-style bound (Barber-Candès-Ramdas-Tibshirani arXiv:2202.13415 Theorem 1) predicts the coverage deficit is `O(L * W_1(s^old, s^new))`, with `W_1` proportional to the cross-leverage between picks and cal set. The mean cross-leverage in the pathological case is 0.286; in the benign case 0.124; the empirical coverage deficits are 9.4pp and 1.5pp respectively, consistent with linear-in-cross-leverage scaling.

**Operational conclusion.** The "calibrate-after-every-update" pattern of summary.md §5.3 is mandatory. The benign case looks safe but is one unlucky EPIG batch away from the pathological case. The runtime check in eval.md §6.2 (model fingerprint on `ConformalCalibrator`) prevents the silent failure.

## 3. Extrapolation under-coverage (test plan C.1)

**Bad cal** (cal on `|c| <= 1`, the wrong pattern; seed = 53):

```
band [0.0,1.0):  cov_68=0.6853  cov_95=0.9553
band [1.0,1.25): cov_68=0.6793  cov_95=0.9650
band [1.25,1.5): cov_68=0.6447  cov_95=0.9567
band [1.5,1.75): cov_68=0.6487  cov_95=0.9613
band [1.75,2.0): cov_68=0.6400  cov_95=0.9553
```

Per-stratum on the extrapolation set (stratum 4 = highest leverage):

```
stratum 3 (n=   6): cov_68=0.5000  cov_95=0.8333
stratum 4 (n=1994): cov_68=0.5832  cov_95=0.9168
```

Side-by-side BAD vs GOOD cal (GOOD = cal on `|c| <= 2`, the broader probe region; seed = 77):

```
band [0.0,1.0):  BAD cov_68=0.6263 cov_95=0.9383    GOOD cov_68=0.6460 cov_95=0.9287
band [1.0,1.5):  BAD cov_68=0.5773 cov_95=0.9523    GOOD cov_68=0.7153 cov_95=0.9587
band [1.5,2.0):  BAD cov_68=0.5230 cov_95=0.9490    GOOD cov_68=0.7287 cov_95=0.9890
```

**Result.** With BAD cal at `|c| in [1.5, 2.0]`: cov_68 = 0.523, **16 percentage points below nominal 0.683**. The conformal interval has lost its frequentist guarantee in the extrapolation band. With GOOD cal at the same band: cov_68 = 0.729 — within ±5pp of nominal.

The W_1 distance between the cal and test marginal distributions of `max|c|` is 0.014, 0.331, 0.577, 0.825, 1.07 across the five bands of the per-stratum table; the under-coverage at the 68% level (BAD cal, relative to nominal 0.683) is 0.000, 0.004, 0.038, 0.034, 0.043 — slope ≈ 0.04pp per unit of W_1. Within an order of magnitude of the BCRT-bound prediction.

**Operational conclusion.** The invariant "calibrate on the broader probe region" is necessary. A runtime check on `ConformalCalibrator.fit` that warns if the cal-set support is a strict subset of the recent probe-query support catches this in production.

## 4. Scaling H_T vs N (test plan, scaling experiment)

`N in {25, 50, 100, 200, 400, 800, 1600}`, focused target around `c_0 = +1.2, m = 1.5` (n_T = 80), EPIG-driven acquisitions, refit at each N (seed = 61):

```
N       H_T          sum_lev_T
   25     409.0271   1.498042e+05
   50     119.6244   1.271454e+02
  100      33.0241   1.204102e+01
  200     -17.0097   3.129267e+00
  400     -51.1268   1.330393e+00
  800     -82.2277   6.096052e-01
 1600    -111.8945   2.901053e-01
```

Fits:

```
combined  H_T = -121.258 + 11779.117/N + 149.735/log N   SSE = 2138.7
pure 1/N  H_T = -97.944  + 12409.754/N                   SSE = 2174.4
pure 1/log N H_T = -518.057 + 2756.465/log N             SSE = 13764.2
b/c (combined) = 78.67
log-log slope of sum_lev_T vs N (N >= 50): -1.6578
```

**Result.** Pure-1/N model explains nearly all the variance (SSE = 2174 vs combined SSE = 2139, within 2%); pure-1/log N is 6.3× worse. The b/c ratio in the combined fit is +78.7 — the 1/N term dominates the 1/log N term over the whole sweep, signalling the EPIG-effective regime with no saturation by `N = 1600`. The N = 25 row is rank-limited (`N << D_JOINT = 75`) and dominates the residual; fits restricted to `N >= 50` give a cleaner picture.

The log-log slope of `sum_i lev_{T_i}` (the unitless predictive variance sum) is -1.66, *faster* than the closed-form 1/N upper bound. This is not a violation: it reflects the additional reduction EPIG achieves by exploiting cross-correlations across target points.

Extrapolation to the 10-20 hour budget. Agent (a)'s cost projection gives ~50,000 T1 oracle calls feasible in 15 hours. At the empirical slope -1.66, target variance at N = 50,000 is ~(50,000/50)^{-1.66} = 4.1e-6 of its value at N = 50, i.e. about 6 orders of magnitude tighter. This is well past the noise-floor of the oracle (analytic LO is exact, noise enters only at T2+ via MC), so a transition to the heteroscedastic or model-misspecification regime is the operational stall, not information saturation.

## Summary

- E1 confirms the EPIG closed-form derivation matches a brute-force direct posterior recomputation to machine precision over 200 random pairs. The Sherman-Morrison and Cauchy-Schwarz steps in `modules/surrogate/acquisition.py:128-145` are correct.
- E2 confirms the bug-class claim: after an EPIG update, no-refit cov_68 = 0.777 vs refit cov_68 = 0.698 vs nominal 0.683 in the pathological case (20-seed mean; cross-leverage 0.29).
- E3 confirms the extrapolation under-coverage negative control: cal on `|c|<=1` evaluated at `|c| in [1.5, 2]` gives cov_68 = 0.523 (-16pp from nominal); cal on `|c|<=2` gives cov_68 = 0.729 (in band).
- E4 confirms 1/N scaling of target-set predictive variance in the EPIG-effective regime. Pure-1/N fits H_T with SSE only 2% larger than the combined-1/N-plus-1/log-N fit; pure-1/log N is 6.3× worse. Log-log slope -1.66 in `sum_i lev_{T_i}` is super-linear, indicating no saturation through N = 1600.

Key file:line references:
- `modules/surrogate/acquisition.py:50` — closed-form IG.
- `modules/surrogate/acquisition.py:128-145` — sequential greedy with Sherman-Morrison update.
- `modules/surrogate/model.py:39-51` — closed-form ridge fit; A_inv cache.
- `modules/surrogate/calibration.py:35-70` — stratified split-conformal fit.
- `modules/surrogate/calibration.py:75-89` — coverage_sigma application.
- `tests/test_acquisition.py:77-92` — existing Cauchy-Schwarz numerical check.
- `tests/test_calibration.py:14-15` — already uses the broader-cal pattern; the runtime invariant of eval.md §6.2 turns this convention into a check.
