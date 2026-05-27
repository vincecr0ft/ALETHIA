# Notable empirical investigations (agent d)

Five numbered investigations. Each is a single hypothesis and a single expected outcome.

1. **Sequential vs batch on clustered pools.** Hypothesis: on a candidate pool with two tight A^{-1}-collinear clusters, sequential greedy EPIG achieves at least 30% greater target-set variance reduction than batch one-shot for k >= 5. Expected outcome: the gap matches the Schur-complement closed form within a few percent and grows with cluster tightness; demonstrates the value of the Sherman-Morrison cache empirically.

2. **Coverage decay after a single EPIG update without recalibration.** Hypothesis: a single EPIG-driven update of the FM, without refitting ConformalCalibrator, drops empirical coverage on the broader probe region by 3-7 percentage points at nominal 0.954. Expected outcome: the decay scales linearly with the mean cross-leverage between picks and the calibration set; refitting calibration restores coverage to within 1pp of nominal. This is the bug-class test for the staleness of the cal layer.

3. **Target-set entropy scaling: 1/N versus 1/log N.** Hypothesis: with a focused target and a broad candidate pool, target-set total predictive entropy decays as O(1/N) for N up to roughly 5 * D_JOINT = 375 oracle calls, then transitions to O(1/log N) as the pool's T-relevant directions saturate. Expected outcome: the transition is visible as a knee in log(H_T) vs log(N); the inferred b/c ratio from the fit H_T = a + b/N + c/log N changes sign around N ~ 5d.

4. **Extrapolation under-coverage as a function of Wasserstein distance.** Hypothesis: per-stratum coverage degrades approximately linearly with W_1(P_cal, P_{x_test}) on the SMEFT box geometry, with slope set by the Lipschitz constant of the score CDF. Expected outcome: coverage at nominal 0.954 drops from 0.94 to about 0.78 as |c| increases from 1.0 (cal support) to 2.0 (twice the cal support); the slope is approximately constant across leverage strata when normalised by stratum-mean leverage.

5. **EPIG with the cost-normalised criterion.** Hypothesis: under a heterogeneous oracle mix (cheap analytic at 1 unit, expensive MadGraph at 100 units), cost-normalised EPIG (EPIG / cost) achieves the same target-set entropy reduction as raw EPIG at roughly 1/20 the wall-clock cost, by preferring cheap exploratory calls until the cheap oracle's information is exhausted. Expected outcome: the cost-normalised curve dominates the raw-EPIG curve in (cost, H_T) space; the crossover point identifies the regime where MadGraph calls start to add genuinely new information.
