# Notable empirical investigations (agent (c))

Five numbered investigations, each with a one-sentence hypothesis and a
one-sentence expected outcome.

1. **DAS-CUSUM threshold transfer from synthetic to real residuals.**
   *Hypothesis:* the h ~ 5.0 threshold calibrated on N(0, 1) synthetic
   streams gives a comparable ARL_0 (~1000) on the actual standardised
   residual stream of `IntentionFM` against the analytic SMEFT oracle,
   because the conformal layer enforces approximate marginal
   N(0, 1)-ness of z_t.
   *Expected outcome:* ARL_0 within a factor of 2 of the synthetic
   calibration; residual non-Gaussianity (heavy tails near kinematic
   edges) inflates the false-alarm rate by less than 50%.

2. **(1, 0, 0) is empirically rare.**
   *Hypothesis:* across a closed-loop run of ~100 cycles, the
   (accuracy, calibration, coverage) = (1, 0, 0) combination accounts
   for less than 5% of all drift firings - because Gneiting-Raftery
   predicts calibration drift fires first in expectation and the
   conformal layer's stratified coverage is sensitive enough to pick
   up the corresponding sigma misallocation.
   *Expected outcome:* (1, 0, 0) frequency in [0%, 5%]; if higher,
   the conformal stratum count S is too small and should be raised
   from 5 to 10.

3. **kappa(A) and v_min projection ratio are complementary, not
   redundant.**
   *Hypothesis:* on simulated query streams that probe "thin
   directions" of the design matrix, kappa(A) and the v_min projection
   ratio fire at *different times* - kappa(A) is a population-level
   signal that needs many off-manifold queries to move, while the
   projection ratio responds to even a handful of concentrated
   queries.
   *Expected outcome:* on a stream of 100 in-manifold + 20 thin-direction
   queries, the projection ratio fires after ~10 thin-direction queries
   while kappa(A) only fires after the FM has been retrained on the
   collected oracle responses (i.e. kappa is post-action, projection
   is pre-action). This is the right operational split.

4. **Local retrain beats global retrain on oracle budget.**
   *Hypothesis:* when the (0, 0, 1) action triggers, an EPIG-driven
   local retrain on the drifted region uses roughly 3-5x fewer oracle
   calls than a global retrain to bring the drifted-region RMSE below
   the promotion threshold, because the EPIG target is concentrated.
   *Expected outcome:* on a synthetic drift in one (c, m) sub-region,
   local retrain hits the promote bar in 50 oracle calls vs ~200 for
   global; A/B promote rule fires in both cases.

5. **Persistence requirement quashes the false-positive retrain rate.**
   *Hypothesis:* requiring N = 3 consecutive (1, *, *) windows before
   any retrain (the "persistence" rule of section 4 of summary.md)
   reduces the total number of retrain events by ~50% on a noisy
   synthetic stream without measurable degradation in long-run RMSE.
   *Expected outcome:* on a 200-cycle synthetic run, persistence rule
   retrains ~10 times vs ~22 times without it, with average RMSE
   within 5% of the no-persistence baseline. The oracle-budget saving
   pays for the small RMSE penalty by roughly an order of magnitude in
   total compute, which is the right trade for the hackathon demo.
