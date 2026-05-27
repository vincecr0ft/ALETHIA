# Drift monitoring test plan (agent (c))

The aim of the tests below is to demonstrate that each of the three
detectors and the aggregation policy behave correctly when fed with
synthetic streams whose ground-truth drift pattern is known. Wherever
possible the test reduces to a closed-form prediction the math
guarantees; otherwise it is a small Monte Carlo against a clearly
specified alternative.

All tests live under `tests/test_drift_*.py` (new files; one per
detector + one for the aggregator + one for the Phoenix wiring). They
import the new module `modules/drift/` (to be created; see structure at
the end).

The DAS-CUSUM, BH-corrected binomial coverage test, and condition-number
evaluator are independent of the FM constraint argument (section 3.2 of
summary.md), so all detector tests can be written against the *current*
`IntentionFM` and ported unchanged when agent (b) ships the c-free FM.

## A. Detector tests

### A.1 DAS-CUSUM under H_0 (no drift)

**Setup.** Draw N = 5000 z_t i.i.d. N(0, 1). Apply DAS-CUSUM with
window w = 30 and threshold h = 5.0. Repeat 1000 times. Record the
fraction of streams that triggered.

**Closed-form prediction.** Under N(0, 1) residuals and h = 5.0, the
Lorden-Pollak ARL_0 estimate gives ~1000 expected samples to false
alarm, so the per-stream false-alarm rate at N = 5000 should be roughly
1 - exp(-5000 / 1000) ~ 0.993 - too high; this is the right test for
calibrating h. Instead test against a target false-alarm probability of
0.05 over a 200-sample window: estimate the empirical p(triggered) and
verify it matches the theoretical ARL prediction within +/- 0.02.

**Pass criterion.** |empirical FA rate - theoretical| < 0.02 over 1000
streams.

### A.2 DAS-CUSUM under H_1 (mean shift)

**Setup.** Stream of 100 N(0, 1) followed by 100 N(0.5, 1) samples
(a half-standard-deviation drift). Run the detector. Repeat 1000 times.

**Closed-form prediction.** Lorden-Pollak gives expected detection delay
~ 2 * log(ARL_0) / (theta^2/2) = 2 * log(1000) / 0.125 ~ 110 - so we
expect almost all triggers to fall within the second half-window.

**Pass criterion.** >= 95% of streams trigger within 200 samples of the
change point; mean detection delay within 20% of the closed-form value.

### A.3 DAS-CUSUM under variance shift

**Setup.** Stream of 100 N(0, 1) followed by 100 N(0, 2). Same protocol
as A.2.

**Closed-form prediction.** A symmetric CUSUM on |z_t| should detect
this; the DAS variance estimator inflates after the change point and
the standardised statistic moves.

**Pass criterion.** >= 90% triggers within the second half-window. (The
threshold is lower than for mean shift because variance shifts produce
a less-direct CUSUM signal.)

### A.4 BH-corrected binomial coverage under H_0

**Setup.** Synthesise S = 5 strata of 200 (mu, sigma, y) triples where
y ~ N(mu, sigma^2). Stated coverage is 0.683 (matching the conformal
default). For each stratum compute empirical coverage and a two-sided
binomial test. Run BH at alpha = 0.05. Repeat 1000 times.

**Closed-form prediction.** BH controls the FDR at 0.05; under H_0
(all strata well-calibrated) the rate of *any* false rejection should
be <= 0.05.

**Pass criterion.** Empirical rate of any-rejection <= 0.07 (slack for
1000-sample Monte Carlo).

### A.5 BH-corrected binomial coverage under H_1

**Setup.** Same as A.4 but stratum 3 is mis-calibrated: y ~ N(mu, (1.5
sigma)^2), which inflates the interval and over-covers. Other strata
correctly calibrated.

**Pass criterion.** Stratum 3 flagged in >= 85% of trials; no other
stratum flagged in > 7% of trials (BH FDR control).

### A.6 kappa(A) and v_min on a synthetic feature stream

**Setup.** Build Phi in R^{200 x 75} as 200 i.i.d. N(0, I) rows. Compute
kappa(A) for A = Phi^T Phi + 1e-3 I. Then perturb the design: add 50
rows that are i.i.d. on the first 70 dimensions but zero on the last 5.
Recompute kappa(A) and verify it has grown.

**Closed-form prediction.** With 250 rows on the full feature space,
A's smallest eigenvalue is at most lambda + smallest-singular(Phi)^2
and at least lambda; under random Gaussian rows the condition number
sits in a known band derived from Marchenko-Pastur. Adding rows that
span a degenerate sub-space shrinks the smallest eigenvalue toward
lambda and so kappa grows by roughly the ratio of original to
post-degeneration smallest singular values.

**Pass criterion.** kappa(A_after) > 2 * kappa(A_before); v_min after
the perturbation has > 0.95 mass on dimensions 70-74 (the deficient
sub-space).

### A.7 v_min projection ratio as a drift detector

**Setup.** Train an `IntentionFM` on 200 random (c, m). Compute v_min.
Then synthesise two query streams: stream A samples (c, m) uniformly
from the training distribution; stream B samples (c, m) concentrated
so its phi_joint embedding lies along v_min (use a small numerical
optimiser to construct one such (c, m), then jitter it).

**Pass criterion.** The projection-variance ratio
Var_streamB(<phi, v_min>) / Var_streamA(<phi, v_min>) exceeds 5 on
samples of size 50.

## B. Aggregator tests

### B.1 8-row decision table is exhaustive and consistent

**Setup.** Unit-test the aggregator function `decide_action(acc, cal,
cov)` on every (acc, cal, cov) in {0, 1}^3.

**Pass criterion.** Each input maps to the action specified in section
4 of summary.md. The function emits a `target_region_id` whenever the
action is `local_retrain` or `global_retrain` and the inputs include
either a calibration or coverage signal.

### B.2 Persistence policy: (1, 0, 0) only triggers after N consecutive

**Setup.** Feed the aggregator with a stream of `(1, 0, 0)` flags
followed by some randomness; verify it stays in `watch` for the first
N-1 windows and only escalates to `local_retrain` on window N.

**Pass criterion.** No retrain action emitted before window N; one
emitted at window N.

## C. Phoenix wiring tests

These tests assume Phoenix is running at `http://localhost:6006`; if
not (e.g. CI), they `pytest.skip()`.

### C.1 Spans round-trip through Phoenix

**Setup.** Emit a synthetic `agent.cycle` trace with the full tree of
section 5 of summary.md. Use `phoenix.Client().get_spans_dataframe(...)`
to read them back. Verify the parent-child relationships are intact and
the `aletheia.*` attributes are present and correctly typed.

**Pass criterion.** All emitted spans appear within 5s; all
`aletheia.*` attributes match what was emitted.

### C.2 Evaluator writes back annotations

**Setup.** Run `accuracy_drift_das_cusum` against a synthetic span
stream with a known drift point. The evaluator should call
`log_evaluations(SpanEvaluations(...))` with the drift flag.

**Pass criterion.** `phoenix.Client().get_span_annotations(...)`
returns the expected annotation on the expected spans.

### C.3 A/B experiment promote rule

**Setup.** Synthesise a probe set as a Phoenix dataset; run two
experiments where the post-update FM has lower RMSE on region 2 (the
"drifted region") and equal RMSE elsewhere. Run the promote-rule
evaluator.

**Pass criterion.** Promote rule returns `True`; the
`aletheia.ab.improvement_drifted` span attribute is positive and the
`aletheia.ab.no_regression` attribute is True.

### C.4 MCP query: "show me the last drift event by region"

**Setup.** Emit several `drift.aggregate` spans with varying
`aletheia.drift.target_region_id`. From a small client that talks to
the MCP server, run a `get-spans` filter and group by region.

**Pass criterion.** The returned list contains the latest drift event
per region; orderings agree with insertion order.

## D. End-to-end smoke

### D.1 One closed loop cycle with the existing FM

**Setup.** Use the existing `IntentionFM` with c in the forward pass
(constraint violation acknowledged - this is the smoke test, not the
real loop). Synthesise a drift by sampling 50 points from a region
outside the training pool. Run one cycle of: `predict -> oracle ->
drift.evaluate -> drift.aggregate -> epig.select -> oracle ->
fm.update -> experiment.ab`.

**Pass criterion.** The cycle completes in < 60s; all spans appear in
Phoenix; the aggregator's action is `local_retrain` (coverage drift on
the off-pool region is unambiguous); the A/B promote rule fires
positive after the update.

## E. Code that needs to exist first

The tests above assume a small new module `modules/drift/` with this
layout:

    modules/drift/
      __init__.py
      das_cusum.py            # streaming DAS-CUSUM
      coverage_bh.py          # BH-corrected per-region binomial coverage
      kappa_vmin.py           # kappa(A), v_min, projection-ratio
      aggregator.py           # decide_action(acc, cal, cov) + persistence
      phoenix_evaluators.py   # the three SpanEvaluations producers
      spans.py                # helpers to emit the aletheia.* attrs

None of these depend on the existing `surrogate` internals beyond the
public API (`predict`, `leverage`, `A_inv`, calibration `coverage_sigma`).
When agent (b) ships the c-free FM, the only change is that
`kappa_vmin.py` swaps `model.A_inv` for whatever the new model exposes
under the same name (or whatever contract the team settles on).

The tests in A.6 and A.7 use the current `IntentionFM` directly; they
will need their feature-construction calls swapped when the FM
interface changes, but the math (kappa, v_min, projection ratios) is
unchanged.

The smoke test D.1 requires the Phoenix container running locally; the
Makefile target `make phoenix-up` (already in repo) is sufficient. The
test guards on HTTP reachability and skips if Phoenix is down.

## F. Calendar

| step | tests | est. effort |
|------|-------|-------------|
| 1    | A.1-A.5 (DAS-CUSUM + BH) | half a day |
| 2    | A.6-A.7 (kappa, v_min)   | half a day |
| 3    | B.1-B.2 (aggregator)     | half a day |
| 4    | C.1-C.4 (Phoenix wiring) | one day (gated on Phoenix MCP being up) |
| 5    | D.1 (smoke)              | half a day |
| total |                          | three days, parallelisable to one with agents (a)/(b)/(d) |
