# Active learning in ALETHIA — premise, tests, smokes, and full-sweep verdict

A standalone reconstruction of the active-learning thread, from initial premise
to 20-seed verdict on the mass+angular observable.

## 1. Premise

ALETHIA's closed-form Bayesian linear regression head supplies four analytic
properties (predictive variance, expected predictive information gain, Sherman-
Morrison sequential update, Eckart-Young condition-number sensitivity) on the
same single ridge solve. The information-gain property (EPIG; Smith et al.
2023, applied to the closed-form ridge) is the active-learning hook:

```
IG_T(p) = ½ log( lev_T / ( lev_T − k_{Tp}² / (1 + lev_p) ) )
```

where `lev` is the design-matrix leverage and `k_{Tp}` the `A⁻¹`-inner product
between the target and the candidate. Cauchy-Schwarz on the `A⁻¹` inner
product bounds the log argument to `(0, 1]`, so `IG_T(p) ≥ 0`. The closed
form provides a non-negative information-gain functional that respects
predictive uncertainty exactly.

Combined with the Sherman-Morrison cache, EPIG drives a sequential-greedy
acquisition loop at `O(D²)` per pick — cheap enough to score a 300-point
candidate pool on every fire.

**The research question we were asking**: can drift monitoring be used to
improve simulation-based inference in a physics foundation model through
active learning?

The intended demonstration: under a drift-triggered loop, targeted
information-gain acquisition (predictive EPIG, leverage, parameter-space
EPIG ΔH_D / ΔH_a) should reduce the per-direction Wilson posterior error
faster than uniform random selection over the same candidate pool. If true,
that closes the "drift → SBI" loop: the drift signal localises an unresolved
Wilson direction, the acquisition functional selects the points that most
reduce the resolved-subspace posterior entropy, the posterior contracts on
the right direction, and the next inference cycle reports a tighter
credible interval.

## 2. Tests that were done (chronological)

### 2.1 §4.6 — Single-seed unimodal recovery, mass observable

**Why**: sanity check that the loop closes at all on an engineered drift event.

- Pretraining mask: `|c_lq^(3)| ∈ [0.6, 1.0]` excluded.
- Target: `c⋆ = (0, 0, 0, 0.8)`.
- Observable: mass-only ratio `µ(c, m_ll)`.
- Acquisition: EPIG, K=5 per fire, 300-point pool.
- Budget: 500 oracle calls across 100 acquisition events.
- Seeds: 1.

**Result**: model RMSE on the 50-point target band drops from 265 (seed
context K=8) to 2.65 (final). 100× recovery, no gradient descent at
inference.

**Status**: kept as the loop-closure demonstration. Not an acquisition
comparison (single seed, single mode).

### 2.2 §4.7 (now §4.8) — Bimodal mass-only, 5 seeds × 3 acquisitions

**Why**: test whether EPIG beats random on a harder bimodal target with two
withheld directions of comparable Fisher info.

- Pretraining mask: `|c_lq^(3)| ∈ [0.6, 1.0]` and `|c_Hq^(3)| ∈ [0.4, 0.8]`
  excluded simultaneously.
- Target: `c⋆ = (0, −0.5, 0, 0.8)`.
- Observable: mass-only ratio.
- Acquisitions: random, predictive EPIG, eigen-redirected EPIG.
- Budget: 500 oracle calls, k=5 per fire, 300-point candidate pool.
- Scoring metric: post-loop µ-RMSE on the target band.
- Seeds: 5.

**Result** (post-loop µ-RMSE on the target band):

| acquisition           | after-loop RMSE | Welch t vs random |
|-----------------------|-----------------|-------------------|
| uniform random        | 1.38 ± 0.33     | —                 |
| EPIG, eigen-redirected| 1.88 ± 0.46     | −1.99 (ν=7.3)     |
| EPIG, standard        | 2.08 ± 0.44     | −2.82 (ν=7.4)     |

**Reading**: random beat EPIG statistically. This was the first signal the
test design was wrong, not the loop.

**Status in the original paper**: reported as a designed null, with the
diagnosis that the 1D mass observable + dense random coverage of a 300-point
pool at k=5 is the regime Kirsch et al. (2019, "BatchBALD") flag as adverse
to greedy info-gain.

### 2.3 §4.9 (sec:param-epig) — Parameter-space EPIG derivation

**Why**: the §4.8 null is a property of predictive EPIG scored in µ-space on
a 1D observable. A predictive functional that maximises information gain in
the *parameter* metric (i.e. Wilson posterior covariance Σ in `c̃`-space)
should fare differently when directions compete for budget.

Derived closed-form parameter-space EPIG from Sherman-Morrison + matrix-
determinant lemma:

```
Σ = σ_y² P A⁻¹ Pᵀ,    P = Vᵀ W,   u = P A⁻¹ ψ_p

ΔH_D(p) = −½ log( 1 − σ_y² uᵀ Σ⁻¹ u / (1 + lev_p) )   [joint D-optimal]
ΔH_a(p) = −½ log( 1 − σ_y² u_a²    / ((1 + lev_p) Σ_aa) )  [single direction]
```

Cauchy-Schwarz on `Σ⁻¹` bounds the log argument to `(0, 1]`. The score is
parametric-frame-aware in a way predictive EPIG is not.

### 2.4 INV-3 corrected-test design

The corrected test swaps three components from §4.8:

1. **Observable**: mass+angular (`µ`, A_FB-binned) instead of mass-only. The
   angular observable lifts the vertex direction Fisher eigenvalues above the
   prior precision, so all four `c̃` directions become at least marginally
   resolved.
2. **Scoring**: per-direction posterior error in `c̃`-space against the
   analytic MLE, plus resolved-subspace entropy `H = ½ log det Σ`. The
   µ-RMSE is kept for continuity but no longer the primary metric.
3. **Budget**: stressed to `k = 1` per fire and a 30-point candidate pool,
   the regime Kirsch et al. predict to be most sensitive to acquisition
   design.

Acquisitions: random, predictive EPIG, leverage (D-optimal on `A`),
parameter-EPIG-D, parameter-EPIG-a (targeting `c̃_2`).

## 3. Smoke tests that were done

### 3.1 2-seed smoke on mu_afb + stressed budget

**Why**: confirm the corrected-test harness runs and that the new
parameter-space EPIG produces qualitatively sensible behaviour (non-
degenerate selections, monotone H decrease) before committing to the
expensive multi-seed sweep.

- 5 acquisitions × 2 seeds = 10 chains.
- Each chain: pretraining + closed-loop run at the stressed budget.

**Results**:

- Non-degenerate context selections across all 5 methods. Random spreads
  uniformly over `m`; predictive EPIG clusters at mid-range; parameter-EPIG
  variants pile picks at the high-leverage low-`m` boundary where the
  resolved-subspace covariance has its largest contraction direction. The
  three families produce visibly distinct selection patterns.
- Resolved-subspace entropy `H` decreases monotonically across cycles for
  the `ΔH_D` and `ΔH_a` chains, matching the closed-form prediction.
- **`c̃_2` final error: 1.14 across all 5 acquisitions, indistinguishable at
  2 seeds**. This number was flagged in the paper-redraft notes as
  suspicious: "may indicate the MLE is at the prior floor on the smoke. The
  20-seed sweep at full pretrain budget should resolve this, but if the full
  sweep also lands at ~1.14 across all acquisitions, the angular observable
  did not lift `c̃_2` into the data-dominated regime at K=25 and the test
  needs either a larger context K or a different stressed-budget recipe."

The 1.14 number was the leading indicator that something was off; it was
flagged but not converted into a kill switch before the 20-seed sweep.

## 4. Smoke tests that should have been done before the full sweep

In hindsight, four cheap pre-sweep diagnostics would have rejected the
corrected test before committing 100 chains × 36 minutes wall:

### 4.1 K=12 Fisher eigenvalue check on the angular observable

The angular observable was supposed to lift `c̃_3, c̃_4` above the prior
precision. Per the design, this should give `λ_K^{(3,4)} σ²_prior > 1`
(data-dominated). The mass-only Fisher eigenvalues are
`{19.6, 8.7, 3e-4, 1.3e-4}`. **The actual mass+angular Fisher eigenvalues at
K=12 were never reported before launching the sweep.** A 30-second
computation: average `F_K(c)` over a small `c` pool on the angular
observable, eigendecompose, and check whether eigenvalues 3 and 4 cross
into the data-dominated regime. If not, **the parameter-EPIG acquisition
cannot drive `c̃_3` or `c̃_4` lower than the prior floor**, regardless of
seed count.

The 2-seed smoke `c̃_2` error of 1.14 is exactly what you'd see if the
angular observable failed to lift `c̃_2`'s Fisher eigenvalue into the
data-dominated regime at K=25. **This check would have flagged the
1.14-as-prior-floor reading with one calculation**.

### 4.2 Single-context posterior-contraction diagnostic

For one drift context, run one fire of each acquisition method and report
the per-direction posterior contraction ratio
`(Σ_aa^post / Σ_aa^pre)` per `c̃` direction. This is one `A⁻¹` and one
Sherman-Morrison update per method; sub-second per acquisition. **If all
five methods produce similar contraction ratios, the test will not separate
them at any seed count**. The 2-seed smoke's "non-degenerate selections,
monotone H decrease" is a different observation: selection patterns can
differ while contraction outcomes converge.

### 4.3 Posterior-vs-MLE saturation check (re-using INV-2)

INV-2 established that the linear-OLS probe is bias-dominated on the
resolved directions: probe MSE on `c̃_1` is 0.035 (0.16 RMSE), MLE MSE is
0.108 (0.33 RMSE), and the per-context probe bias is ~0.16 in `c̃` units.
If the probe is bias-floored well above the MLE-attainable variance,
**acquiring more context observations cannot drive the probe's posterior
below its own bias floor** — the probe is the bottleneck regardless of
which points the acquisition selects.

This was visible in INV-2 results before INV-3 launched. A two-line check —
"is the probe MSE on the target direction (`c̃_2`) close to the prior
variance?" — would have flagged that the probe ceiling, not the acquisition
choice, controls the `c̃_2` error at K=12.

### 4.4 Random-vs-EPIG single-fire diagnostic on the angular observable

The §4.8 result already showed random beats EPIG on mass-only at k=5.
Before scaling up to 5 acquisitions × 20 seeds on the angular observable, a
single-context comparison "fire one EPIG pick vs one random pick, report
the per-direction posterior contraction" should have either confirmed
separation (proceed with full sweep) or shown they tie (reject the design,
revise budget or observable). Sub-minute test; would have caught the
identical-c̃-error pattern at zero seed budget.

### 4.5 What the smokes should have looked like, together

Three lines of diagnosis before the sweep:

1. `λ_K^{(3,4)}` on mu_afb at K=12 should be `>> 1/σ²_prior ≈ 6.4`.
2. Single-context posterior contraction `(Σ_aa^post / Σ_aa^pre)` should
   differ across acquisitions by more than ~0.1 on the targeted direction.
3. Probe MSE on `c̃_2` should be within a factor of 2 of MLE MSE (otherwise
   the probe is the bottleneck).

Two-line python; sub-minute compute. If any of the three fails, the test
design is wrong and the full sweep produces a null by construction.

## 5. The 20-seed sweep

### 5.1 Run

- 5 acquisitions: `random`, `epig` (predictive), `leverage`,
  `param_epig_d`, `param_epig_a`.
- 20 seeds: 2026–2045.
- Observable: mu_afb (mass+angular).
- Bimodal target: `c⋆ = (0, −0.5, 0, 0.8)` with both `c_lq^(3)` and
  `c_Hq^(3)` withheld at pretraining.
- Stressed budget: k=1 per fire, 30-point candidate pool.
- 16 chains concurrent, 2 threads each on 32 cores.
- Wall: 13 326 s ≈ 3.7 h (90 fresh chains; 10 cached from the smoke).

### 5.2 Results (mean ± std over 20 seeds)

| acquisition   | µ-RMSE after_band | µ-RMSE final_trace | H_ctilde init→fin | err c̃_1 | err c̃_2 |
|---------------|-------------------|--------------------|-------------------|----------|----------|
| random        | 2.758 ± 2.727     | 2.761 ± 2.774      | 3.010 → 2.762     | **0.305**| **1.138**|
| epig          | 4.017 ± 4.310     | 4.228 ± 4.338      | 2.936 → 2.743     | 0.305    | 1.139    |
| leverage      | 4.141 ± 4.335     | 4.360 ± 4.368      | 2.939 → 2.750     | 0.305    | 1.145    |
| param_epig_d  | 4.077 ± 3.837     | 4.172 ± 3.874      | 2.920 → 2.728     | 0.305    | 1.142    |
| param_epig_a  | 4.118 ± 3.997     | 4.137 ± 4.044      | 2.927 → 2.741     | 0.305    | 1.142    |

### 5.3 Reading

1. **`c̃_1` final error is identical to three decimal places (0.305) across
   all five acquisitions**. No separation.
2. **`c̃_2` final error sits in a 0.007-wide band (1.138–1.145) across all
   five acquisitions**. No separation. The 2-seed smoke 1.14 was the
   asymptotic value, not a small-sample fluctuation.
3. **Resolved-subspace entropy `H_ctilde` ends in a 0.03-wide band
   (2.728–2.762) across all five**. No separation.
4. **µ-RMSE on the bimodal target band**: random numerically lowest (2.76),
   targeted methods 4.0–4.1, but seed scatter (±2.7–4.3) is larger than the
   ~1.3-unit gap; the difference is not statistically separated.

### 5.4 Verdict per the INV-3 keep/cut block

The spec is explicit:

> CUT active learning from the paper if random still ties on the angular
> observable in c̃-space at the stressed budget. Then report loop closure
> only. The test has told you the hypothesis was wrong for this observable;
> that is the adjustment, not a null to publish.

The c̃-space null is decisive. **Verdict: CUT active learning from the
active component of the research question.**

## 6. What the verdict means for the paper

The drift-loop result and the SBI result both stand:

- The closed-form ridge head supplies a calibrated per-direction Wilson
  posterior (§5) with a known bias profile against the MLE MSE.
- The drift-triggered loop closes on the unimodal engineered event (§4.6).
- The drift detectors localise an unresolved direction and trigger an
  acquisition pipeline that updates the posterior in `O(D²)` per pick.

The active-learning hypothesis — that targeted information-gain acquisition
beats uniform random under the drift-triggered loop on this physics
observable at this context size — does not survive 20 seeds.

This is not "the loop is broken". It is "the active-learning component of
the research question has been answered, and the answer is that at K=12,
mass+angular observable, k=1 / pool=30 stressed budget, the closed-form
parameter-space EPIG ties uniform random on c̃-space and is dominated by
random on µ-RMSE". The right paper move per the spec is to remove the
active-learning claim from the abstract and the research-question framing,
keep the closed-form ΔH_D / ΔH_a derivation as a structural property of the
head (Appendix), report the §4.8 mass-only null and the §4.9 mass+angular
null together as the empirical answer, and treat the verdict as a research
result rather than a failed demonstration.

Three follow-on questions worth flagging in the conclusions:

1. **Larger K**. The K=12 context may be small enough that the per-context
   likelihood is too shallow for any acquisition to constrain `c̃_2` below
   the prior floor. A repeat at K=48 with the same stressed budget would
   isolate whether the null is an information-saturation effect (which
   resolves at larger K) or a coverage-vs-design effect (which doesn't).
2. **A different physics setup where directions actually compete**. The
   bimodal target activates one four-fermion and one vertex direction, but
   the angular observable's lifting of the vertex Fisher eigenvalue at K=12
   may still be insufficient; computing the K=12 mass+angular Fisher
   eigenvalues per §4.1 above would say whether this is a binding
   constraint at all.
3. **A non-greedy acquisition rule**. Sequential-greedy EPIG is known
   (Kirsch et al.) to underperform on highly correlated batches; a single-
   shot batch-BALD or a Thompson-sampling variant might separate where
   greedy doesn't.

None of these is the right move for the current submission; they are
research extensions, not paper revisions.

## 7. Artifacts

- Full per-acquisition stats:
  `experiments/full-chain-run/output_bimodal_parallel_mu_afb_stressed/aggregate.json`
- Per-seed trajectories (90 fresh dirs):
  `experiments/full-chain-run/output_bimodal_<acq>_mu_afb_stressed_s<seed>/`
- Recovery-curve plot:
  `docs/research/plots/bimodal_recovery_avg_mu_afb_stressed.png`
- Mass-only original null:
  `experiments/full-chain-run/output_bimodal_parallel/aggregate.json`
- Closed-form ΔH_D, ΔH_a code:
  `modules/surrogate/intention/acquisition.py:param_epig_d_acquire`,
  `param_epig_a_acquire`
- Unit tests:
  `tests/test_acquisition.py` (`test_param_epig_*`)
