# EPIG, conformal prediction, and the improvement loop

Agent (d), first-pass research summary. The job is to verify the mathematics of the iterative training and improving loop: that EPIG-driven acquisition combined with conformal calibration yields a procedure whose predictive entropy on a stated target set shrinks monotonically in expectation while calibration is preserved. Citations are by file:line for code and by arXiv id (or DOI) for the literature.

The whole pipeline can be read as Vovk-style on-line learning under covariate shift: at each round the foundation model is updated, the calibration set is refit on the (possibly shifted) probe region, and a new batch of acquisitions is chosen by maximising an information functional defined on a user-specified target set. The point of this document is to establish that each ingredient is mathematically correct, that they compose without violating each other's guarantees, and to flag the precise places where the existing code drifts off the rails.

---

## 1. EPIG, rigorously

### 1.1 Definition

EPIG (Smith, Bickford Smith, Rainforth 2023, arXiv:2304.08151) is, in the abstract,

EPIG(x_p | T) = E_{x_T ~ p_T} [ I( f(x_T) ; y_p | D ) ],

i.e. the expectation over a target distribution p_T of the mutual information between the latent function values at x_T and the observation y_p that we would obtain at a candidate input x_p, conditional on data D seen so far. Two limiting cases:

- p_T uniform on the same support as the candidate pool. EPIG collapses to per-sample BALD averaged over the input distribution (Houlsby, Huszár, Ghahramani, Lengyel 2011, arXiv:1112.5745); in the Gaussian-linear case this in turn collapses to the leverage criterion (§1.4).
- p_T a Dirac mass at a single point x_T*. EPIG becomes classical optimal experimental design with one design objective.

### 1.2 Closed form under Bayesian linear regression / KRR-GP duality

The model is the closed-form Bayesian linear regression in modules/surrogate/model.py:39. Adopt the conjugate setting

  y = Phi w + eps, eps ~ N(0, sigma^2 I), w ~ N(0, alpha^{-1} I).

The posterior is Gaussian with covariance Sigma_w = sigma^2 A^{-1} where A = Phi^T Phi + lam I, lam = alpha sigma^2 is the ridge parameter, and the predictive variance for the latent f(x) = phi_x w is sigma^2 lev_x with lev_x = phi_x A^{-1} phi_x^T. This is the KRR-GP duality of Rasmussen and Williams (Gaussian Processes for Machine Learning, 2006, Ch. 6); the kernel is k(x,x') = phi_x phi_{x'}^T.

Condition on a new observation (x_p, y_p). The updated posterior covariance is, by Sherman-Morrison applied to A:

  A_new^{-1} = A^{-1} - (A^{-1} phi_p^T phi_p A^{-1}) / (1 + lev_p).

Plug into the leverage at target x_T:

  lev_T^new = lev_T - k_Tp^2 / (1 + lev_p),  k_Tp = phi_T A^{-1} phi_p^T.

The information gain on f(x_T) from observing y_p is the KL divergence between two Gaussians with the same posterior mean (the variance shrinks deterministically conditional on y_p only through the design point x_p, since the mean shift is integrated over the posterior predictive of y_p), and reduces to

  IG_T(p) = 0.5 log( sigma^2 lev_T / [ sigma^2 (lev_T - k_Tp^2/(1 + lev_p)) ] )
          = 0.5 log( lev_T / [ lev_T - k_Tp^2/(1 + lev_p) ] ).

The closed-form quoted in the module docstring (modules/surrogate/acquisition.py:50). The sigma^2 cancellation is exact because the noise enters only through the global scale of the predictive variance; numerator and denominator each carry one factor of sigma^2 and the log eats them.

EPIG is then

  EPIG(p | T) = (1/|T|) sum_{i in T} IG_{T_i}(p),

implemented at modules/surrogate/acquisition.py:127-137.

### 1.3 Non-negativity and the Cauchy-Schwarz bound

A^{-1} is symmetric positive definite and defines an inner product. Cauchy-Schwarz on that inner product:

  k_Tp^2 = (phi_T A^{-1} phi_p^T)^2 <= (phi_T A^{-1} phi_T^T)(phi_p A^{-1} phi_p^T) = lev_T lev_p.

Therefore k_Tp^2 / (1 + lev_p) <= lev_T lev_p / (1 + lev_p) < lev_T, so the denominator inside the log is strictly positive and less than lev_T. The log argument lies in (0, 1] under the convention IG = -0.5 log(post/prior) (equivalently +0.5 log(prior/post)); information gain is non-negative. Equality is achieved only when phi_p is parallel to phi_T in the A^{-1} inner product. The existing test tests/test_acquisition.py:77-92 checks this numerically.

The code clamps the denominator at modules/surrogate/acquisition.py:134-136 to avoid finite-precision negatives. Algebraically the quantity cannot go negative; the clamp is a finite-precision band-aid and is harmless.

### 1.4 When the sigma^2 cancellation fails

The cancellation is structural under three conjugate-Gaussian assumptions.

1. Homoscedastic Gaussian noise. If sigma^2 = sigma^2(x_p), the Sherman-Morrison update becomes A^{-1} - A^{-1} phi_p^T phi_p A^{-1} / (sigma^2(x_p)/sigma_0^2 + lev_p), and IG picks up a factor depending on the candidate-local noise scale. The ridge fit assumes a single global noise scale (noise_frac in modules/surrogate/model.py:49). The heteroscedastic predictive sigma at modules/surrogate/model.py:62-65 is constructed post-hoc and is not the inferential noise of the Bayesian update.
2. Conjugate likelihood. Non-Gaussian observations (Poisson counts, binomial labels) break the closed form. Falls back to Monte Carlo over posterior samples; Smith et al. arXiv:2304.08151 §3.
3. No model misspecification. EPIG is an expected information gain under the prior model. If the prior is wrong, EPIG is confidently wrong; the calculation is internal to the regressor and cannot detect systematic bias.

For the present scope the homoscedastic assumption is operational. The aleatoric noise model at modules/surrogate/model.py:62-65 is a heuristic for calibration; the EPIG calculation correctly uses only the epistemic covariance sigma^2 A^{-1}.

### 1.5 Implementation audit of modules/surrogate/acquisition.py:105-146

- L121-122: build joint feature matrices for the candidate pool and target set. Correct.
- L123: copy A_inv so the model state is not mutated. Correct, important for sequential updates.
- L128-130: einsum forms of lev_T, lev_p, K_TP. Signatures match the algebraic definitions.
- L131: variance reduction K_TP^2 / (1 + lev_P[None, :]) broadcasting (n_T, n_P). Correct.
- L134-137: log ratio with numerical clamps. The clamp at 1e-12 * max(lev_T, 1e-12) bounds the maximum IG per observation at about 0.5 log(1e12) ≈ 13.8 nats. High enough not to bite operationally; should be documented as the clip. The mean over target points implements the average in §1.2.
- L138-141: greedy pick with availability mask. Correct.
- L143-145: Sherman-Morrison update of A_inv so subsequent picks see the model as if the current pick had already been folded in. Correct, and crucial; see §3.

Implementation nit: at L134-136 the relative floor is 1e-12 on the denominator; consider an explicit guard at the level of var_red (var_red <- min(var_red, lev_T (1-eps))) to avoid the log ratio nominally going to infinity. The current behaviour is operationally safe but the failure mode is opaque.

---

## 2. EPIG under the foundation-model constraint

### 2.1 The shape of the candidate space changes

The hard constraint of the project (BRIEF §10-43) is that the FM forward pass cannot see Wilson coefficients. Under the rebuilt FM that agent (b) is producing, the feature row is no longer phi_joint(c, m) but phi(z), where z is the FM embedding of a measured distribution. The existing epig_acquire(model, C_pool, M_pool, C_target, M_target, k) signature is then no longer a faithful description of the candidate space.

Two distinct things are tangled in the current API:

- The candidate space EPIG ranks over: oracle commissions, tuples (target Wilson region or sampler config, kinematic window, fidelity tier). A commission, when executed, returns a measured distribution (an embedding z) and a label (the observable of interest).
- The target space: a fixed set of observables of interest, encoded as a fixed list of embeddings { z_T^{(i)} }. Target embeddings live in the same latent space as candidate embeddings.

### 2.2 EPIG decomposes over the commission graph

A commission kappa produces a stochastic embedding z(kappa) ~ p(z | kappa). The information gain on the latent function at a target embedding z_T from observing the labelled commission (z(kappa), y(kappa)) is the same expression with phi_p <- phi(z(kappa)). EPIG over commissions:

  EPIG(kappa | T) = E_{z(kappa)} (1/|T|) sum_i 0.5 log( lev_{T_i} / (lev_{T_i} - k_{T_i kappa}^2 / (1 + lev_kappa)) ).

The outer expectation is the only structural addition: the oracle's output for a given commission is stochastic, and EPIG must integrate. In practice replace by a small Monte Carlo over pre-cached "preview" commissions if the oracle is expensive, or by the deterministic mean embedding mean z(kappa) if the embedding distribution is tight around its mean (which it is for low-noise oracles like modules/surrogate/oracle_smeft.py).

### 2.3 Sketched API

```python
def epig_acquire_commissions(
    model,                          # rebuilt FM with predict on z
    commissions: list[Commission],  # candidate oracle jobs
    embed_preview: Callable[[Commission], np.ndarray],
                                    # cheap preview, optional MC
    z_target: np.ndarray,           # (n_T, D_Z) target embeddings
    k: int,
    mc_samples: int = 1,
) -> list[Commission]:
    A_inv = model.A_inv.copy()
    chosen, remaining = [], list(commissions)
    phi_T = model.feature_map(z_target)
    while len(chosen) < k:
        scores = []
        for c in remaining:
            zs = [embed_preview(c) for _ in range(mc_samples)]
            phi_p = model.feature_map(np.stack(zs)).mean(axis=0)
            lev_T = np.einsum("id,de,ie->i", phi_T, A_inv, phi_T)
            lev_p = phi_p @ A_inv @ phi_p
            k_Tp  = phi_T @ A_inv @ phi_p
            ig    = 0.5 * np.log(lev_T / (lev_T - k_Tp**2 / (1 + lev_p)))
            scores.append(ig.mean())
        i = int(np.argmax(scores))
        chosen.append(remaining.pop(i))
        p = model.feature_map(embed_preview(chosen[-1]))
        Ap = A_inv @ p
        A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
    return chosen
```

Structurally the same code path; candidate features are constructed by previewing the oracle rather than directly indexing a (C_pool, M_pool) array. Commission carries three fields: wilson_region (bounded region the oracle samples over to produce the embedding), kinematic_window (cuts and binning), fidelity_tier (cheap analytic vs. expensive MadGraph). No commission field enters the FM's predict path; the FM only sees the resulting embedding z. Constraint discharged.

### 2.4 Cost-aware EPIG

A heterogeneous fidelity tier is a natural fit for cost-normalised EPIG: pick the candidate that maximises EPIG(kappa | T) / c(kappa), where c(kappa) is wall-clock or CPU cost. This is the Bayesian decision-theoretic framing of MacKay 1992 (Neural Computation 4); the change is one division. The oracle commission queue should default to this form.

---

## 3. Sequential greedy vs batch

Sequential greedy with Sherman-Morrison: O(d^3 + k n d^2) total. d = D_JOINT = 75 (modules/surrogate/features.py:40), n is the pool size, k is the number of picks. O(d^3) is the initial inversion (already done in fit); each pick is O(n d^2) for scoring and O(d^2) for the rank-one A^{-1} update. Batch one-shot picks the top-k by static EPIG against unchanged A^{-1}: cost O(n d^2), no update.

### 3.1 The variance-reduction argument

Two candidates p1, p2 with identical static IG against a fixed target x_T. If they are A^{-1}-aligned (phi_{p_2} close to phi_{p_1} in the A^{-1} inner product), the marginal IG of p2 after picking p1 is much smaller than its static IG. Algebraically:

  k_Tp_2^(1) = k_Tp_2 - k_Tp_1 k_{p_1 p_2} / (1 + lev_{p_1}).

Batch one-shot picks two A^{-1}-collinear candidates because they have the same individual EPIG; sequential greedy picks p1, then sees that p2 has become redundant and picks something orthogonal.

The variance reduction at the target after k picks is

  lev_T - lev_T^{(k)} = sum_{j=1..k} (k_{T p_j}^{(j-1)})^2 / (1 + lev_{p_j}^{(j-1)}),

each term computed in the updated geometry. For batch one-shot all cross-leverages are computed in the initial geometry, and the realised variance reduction is less than sum_j k_{T p_j}^2 / (1 + lev_{p_j}) by the Schur complement of the candidate Gram matrix. The proof is the determinantal identity used in DPP sampling (Kulesza-Taskar 2012, FnTML 5, arXiv:1207.6083).

### 3.2 Empirical check in the existing tests

tests/test_acquisition.py:55-74 verifies sequential greedy steers picks toward a focused target but does not directly compare sequential vs batch. The variance shrinkage of scripts/surrogate_demos/demo_epig.py:104-140 is sequential and shows the leverage-vs-EPIG ratio but does not isolate the seq-vs-batch effect. I add an explicit test (test plan §1) that constructs a clustered candidate pool, picks k=10 by batch and by sequential, computes the actual target variance reduction, and asserts sequential >= batch.

### 3.3 Operational implication

The current epig_acquire is sequential greedy; this is correct and should not be changed. Batch one-shot is only interesting as a baseline.

---

## 4. Conformal prediction

### 4.1 Split-conformal: first-principles derivation

Split-conformal (Vovk, Gammerman, Shafer 2005, *Algorithmic Learning in a Random World*, Springer; Lei, G'Sell, Rinaldo, Tibshirani, Wasserman 2018, JASA 113, arXiv:1604.04173) gives marginal coverage under exchangeability.

1. Fit predictor mu_hat on a training split. Compute nonconformity scores s_i = |y_i - mu_hat(x_i)| / sigma_hat(x_i) on a held-out calibration set of size n. Calibration scores are an exchangeable sample from the score distribution of any new exchangeable point.
2. Let s_{(q)} be the ceil((n+1)(1-alpha))-th order statistic. By exchangeability, the rank of the test score among the n+1 combined scores is uniform on {1,...,n+1}, so Pr(s_test > s_{(q)}) <= alpha.
3. Conformal interval mu_hat(x_test) +/- s_{(q)} sigma_hat(x_test) has marginal coverage at least 1 - alpha.

Implementation at modules/surrogate/calibration.py:30-33 uses the exchangeability-aware quantile ceil((n+1) coverage) / n, clamped at 1. Correct.

### 4.2 Stratified split-conformal: approximate conditional coverage

Marginal coverage is the weakest guarantee. It does not prevent under-coverage on systematic sub-populations. For our purposes the systematic sub-population is "the high-leverage stratum where the FM is doing covariate extrapolation".

Stratified split-conformal (Romano, Patterson, Candès 2019, NeurIPS, arXiv:1905.03222) partitions the calibration set by a stratification variable g(x) and fits a per-stratum quantile. If exchangeability holds within each stratum, each gets marginal coverage at nominal level. The union is conditional on the stratum. As long as the stratification correlates with the residual structure, this is a practical proxy for conditional coverage.

Implementation at modules/surrogate/calibration.py:55-69 stratifies on leverage quantiles, which is the residual axis that fails first under extrapolation. Correct.

### 4.3 Exchangeability beyond, and the bug

Two extensions are relevant when the loop drives the model and queries off the training distribution.

- Barber, Candès, Ramdas, Tibshirani 2023, "Conformal prediction beyond exchangeability", *Annals of Statistics* 51(2), arXiv:2202.13415. The marginal coverage guarantee degrades by a factor depending on the total variation between calibration and test distributions.
- Gibbs, Candès 2024, "Adaptive conformal prediction inference", *JMLR* 25, arXiv:2106.00170. Online recalibration tracks coverage drift via a Robbins-Monro update on the effective alpha. This is the right thing to integrate with the loop in §5.

Flag: at modules/surrogate/calibration.py:46-49 the docstring says "the calibration set should cover the same input distribution the model will be queried on". The existing test tests/test_calibration.py:14-15 does draw the calibration set on the broader probe region (|c| <= 2) rather than only the training box (|c| <= 1). Correct.

What is not enforced: this property must be preserved across loop iterations. Once the EPIG-driven update has shifted the training-set support, the calibration quantiles will no longer match the realised probe distribution unless the calibration set is also drawn from the current probe region. If ConformalCalibrator.fit is called once at deployment and never re-called, the guarantee gradually expires. This is a bug-class issue: there is no test that the coverage holds after a loop iteration. I add it to the test plan as test_coverage_preserved_after_update.

### 4.4 What the leverage-stratified extension actually requires

Vovk-style theorem restricted to a stratum: given a stratification g, if calibration scores and test scores are exchangeable conditional on g, per-stratum quantiles give per-stratum marginal coverage. Operational assumption: g is computed on x only (not on y). The leverage g(x) = lev(x) is a function of x only and so does not break exchangeability.

Implementation uses the calibration set's own leverage to define stratum edges (calibration.py:55). Fine in isolation, but it makes edges depend on the calibration set; after a model update, edges must be recomputed. The state_dict round-trip preserves edges but does not enforce a refit, contributing to staleness risk.

---

## 5. The loop, and its two guarantees

### 5.1 The procedure, fully specified

1. Pre-train FM on stage-0 oracle data. IntentionFM.fit (modules/surrogate/model.py:39-51).
2. Calibrate conformal layer on held-out calibration set drawn from the broader probe region. ConformalCalibrator.fit (calibration.py:35-70).
3. Run probe queries; collect predictions and conformal sigmas. model.predict, cc.coverage_sigma (evaluation.py:51-71).
4. Drift detection (agent (c)) flags region R where accuracy / calibration / coverage drift is significant.
5. EPIG selects k candidates in R against target set T of downstream observables. epig_acquire (acquisition.py:105-146).
6. Oracle returns labels. AnalyticSMEFTOracle / OracleMadGraph.
7. Sherman-Morrison-update FM (current code refits; see §5.4). IntentionFM.update (model.py:73-78).
8. Recalibrate on a cal set drawn from the current probe region. ConformalCalibrator.fit.
9. Re-probe; go to 4.

### 5.2 Guarantee A: variance reduction in expectation

Claim. Each EPIG-driven update strictly reduces predictive variance on the target set T in expectation under the model's own posterior.

Proof sketch. Sherman-Morrison gives, for one pick p* and each x_T in T,

  sigma^2 lev_T^new = sigma^2 lev_T - sigma^2 k_{T p*}^2 / (1 + lev_{p*}).

The right-hand term is non-negative by Cauchy-Schwarz (§1.3) and strictly positive whenever k_{T p*} != 0, i.e. whenever phi_{p*} has nonzero A^{-1}-inner-product with phi_T. EPIG maximises the mean over T of log(1 / (1 - var-red ratio)), monotonically increasing in the variance reduction; the picked candidate has the largest possible expected reduction. Summing over k picks gives strict reduction in target-set total predictive variance.

Failure modes.

- Heteroscedasticity. Realised noise scale at p* may exceed the assumed homoscedastic noise; realised variance reduction is less than predicted. Still non-negative, but EPIG ranking is no longer optimal.
- Model misspecification. Reduction is in predictive variance, not generalisation error. If the FM is biased, reducing variance does not reduce error.
- Sherman-Morrison cache staleness. Accumulated rank-one updates drift from the exact inverse due to floating-point error. The current IntentionFM.update (model.py:73-78) refits from scratch, sidestepping staleness at the cost of redoing the O(d^3) inversion. Acceptable at d=75.

### 5.3 Guarantee B: calibration is preserved iff calibration is refit

Claim. After a Sherman-Morrison update of A^{-1} and the corresponding update of w_hat, the conformal coverage guarantee is preserved only if ConformalCalibrator.fit is called again on a calibration set drawn from the current probe region.

Sketch. Split-conformal requires exchangeability between cal scores and test scores. The score depends on mu_hat, sigma_hat; both change when the model updates. The old calibration quantiles are quantiles of {s_i^old}; the new test scores live under {s_i^new}. The two agree only in the no-op limit (k_{T p*} = 0 for all T). In general, the calibration set must be re-scored and the per-stratum quantiles re-computed.

The leverage strata themselves are also stale: lev(x) = phi(x) A^{-1} phi(x)^T decreases at every x in the direction of the rank-one update. Stratum edges shift; a calibration point in stratum 4/5 may now belong in stratum 3.

Quantifying the violation. The Wasserstein-1 distance between old and new score distributions on the calibration set is bounded above by (1/n) sum_i |s_i^new - s_i^old|, and each term is bounded by

  |Delta mu_i| / sigma_hat_i^old + (|y_i - mu_hat_i^old| |Delta sigma_i|) / (sigma_hat_i^old sigma_hat_i^new).

Both Delta mu_i and Delta sigma_i are linear in the rank-one update direction A^{-1} phi_{p*}^T at first order; their magnitudes scale with the cross-leverage between p* and x_i. The coverage deficit at level alpha is then bounded by

  |Pr(s^new <= q^old) - (1-alpha)| <= L * W_1(s^old, s^new),

where L is the Lipschitz constant of the score CDF (Barber-Candès-Ramdas-Tibshirani arXiv:2202.13415 §2). Pure Sherman-Morrison without recalibration violates conditional coverage to first order in the cross-leverage between the picked candidate and the calibration set.

Operational consequence. After every loop iteration, refit calibration. Cost is linear in cal set size (O(n_cal d^2)) and small compared to oracle cost.

### 5.4 Sherman-Morrison vs refit in the existing update

IntentionFM.update (model.py:73-78) calls fit on the concatenated data. Robust (no staleness) but loses the O(d^2) per-update advantage; with d=75 and N ~ 10^3 the difference is negligible (refit at O(N d^2 + d^3) ~ 5.4M FLOPs is no slower than O(N d^2) ~ 5.0M FLOPs). For Sherman-Morrison update to be worthwhile, d >> 500; not our regime. Leave refit as default. The Sherman-Morrison machinery inside epig_acquire for acquisition planning is separate and remains necessary.

---

## 6. Monitoring the loop

Quantities computable from existing state, all functions of the FM posterior and the cal set.

**Quantity 1: target-set total predictive entropy.**

  H_T = 0.5 sum_{i in T} log(2 pi e sigma^2 lev_{T_i}).

Diagonal approximation suffices when off-diagonals of Sigma_w restricted to T are dominated by the diagonal (high-dimensional generic regime). Exact form: 0.5 log det(2 pi e sigma^2 Phi_T A^{-1} Phi_T^T); compute when |T| is small. Trajectory of H_T vs oracle calls is the EPIG dashboard.

**Quantity 2: empirical coverage per region.** Bröcker 2009 (QJRMS 135) decomposes Brier score on indicator-covered events into reliability (calibration error), resolution, and uncertainty. Per leverage stratum: empirical coverage and reliability. Reliability should be at noise level after each recalibration.

**Quantity 3: condition number trajectory.** kappa(A) characterises predictive variance amplification. Sherman-Morrison multiplies A^{-1} by I - v v^T / (1 + ||v||^2) for v = A^{-1/2} phi_p^T; eigenvalue ratio changes by at most 1/(1 + ||v||^2) in the relevant direction. kappa(A) should decrease monotonically under EPIG-driven acquisition; spikes indicate near-collinear additions and warrant a Tikhonov bump.

**Quantity 4: Sherman-Morrison residual.** Every N updates, compare A^{-1}_incremental to A^{-1}_fresh. Frobenius residual quantifies floating-point drift; if it exceeds relative threshold (say 1e-6), force a refresh. Moot in current code (update refits) but mandatory once Sherman-Morrison is introduced as an optimisation.

Four panels of the loop monitoring dashboard. Mathematical objects with clear semantics; cheap to compute; no extra oracle calls.

---

## 7. Extrapolation failure as negative control

Standard failure mode of conformal: fit on training-distribution-supported |c| <= 1, evaluate on |c| in [1, 2]; the high-leverage stratum will under-cover.

Under the linear-GP model, conditional coverage on extrapolation points is bounded above by

  Pr(cover | x_test) <= (1 - alpha) * (1 - Delta(x_test)),

with Delta(x_test) the exchangeability defect. By Barber-Candès-Ramdas-Tibshirani Theorem 1 (arXiv:2202.13415),

  Delta(x_test) <= Lip(s_alpha) * W_1(P_cal, P_{x_test}).

For high-leverage x_test the conditional score distribution is dominated by epistemic uncertainty (the leverage term in the predictive variance), and W_1 grows roughly linearly with the distance of x_test from cal support. Coverage degrades approximately linearly with extrapolation depth, until at the extreme the conformal interval has no meaningful frequentist guarantee.

The existing test tests/test_calibration.py:41-45 only weakly checks the negative control (asserts the high-leverage factor is at least the low-leverage factor: necessary but far from sufficient). I add an explicit test_extrapolation_under_coverage (test plan §3) that fits on |c| <= 1 and demonstrates significant under-coverage in |c| in [1.5, 2].

---

## 8. Scaling prediction

**Target-set total predictive entropy.** Under effective rank r (the number of distinct A^{-1}-orthogonal directions in feature space relevant to T), each EPIG pick reduces target entropy by at most a factor (1 + lev_p)/lev_p in one of r orthogonal subspaces. After N picks the target-set variance shrinks like

  sum_{i in T} sigma^2 lev_{T_i}^{(N)} ~ C / N,

i.e. O(1/N), as long as new picks find new A^{-1}-orthogonal directions relevant to T. Optimal parametric rate. When the pool is exhausted of T-relevant information, rate degrades to O(1/log N) as EPIG picks near-redundant candidates. The transition is a useful diagnostic.

**Empirical coverage stabilisation.** Under recalibration with cal set size n_cal, empirical coverage converges to nominal at rate O(1/sqrt(n_cal)) by Dvoretzky-Kiefer-Wolfowitz 1956 (Ann. Math. Stat. 27), with a constant depending on the score CDF smoothness. With n_cal = O(N), coverage stabilises at O(1/sqrt(N)).

**Condition number.** kappa(A) decreases approximately as kappa_0 / (1 + N beta) for some beta > 0 under EPIG-driven acquisition. Stagnation warns the candidate pool no longer provides directions complementary to the current model.

**Empirical scaling test.** The second-pass empirical work should generate target entropy vs oracle calls, fit H_T = a + b/N + c/log N, and report b/c. A clean 1/N regime indicates EPIG is finding fresh information; a regime dominated by 1/log N indicates saturation and the target set should be enlarged or the FM embedding dimension expanded.

---

## 9. References

- Smith, Bickford Smith, Rainforth 2023, arXiv:2304.08151. EPIG.
- Houlsby, Huszár, Ghahramani, Lengyel 2011, arXiv:1112.5745. BALD.
- Rasmussen, Williams 2006, Gaussian Processes for Machine Learning, MIT Press, Ch. 6.
- Vovk, Gammerman, Shafer 2005, Algorithmic Learning in a Random World, Springer.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman 2018, JASA 113, arXiv:1604.04173.
- Romano, Patterson, Candès 2019, NeurIPS, arXiv:1905.03222.
- Barber, Candès, Ramdas, Tibshirani 2023, Annals of Statistics 51(2), arXiv:2202.13415.
- Gibbs, Candès 2024, JMLR 25, arXiv:2106.00170.
- Bröcker 2009, QJRMS 135.
- MacKay 1992, Neural Computation 4(4).
- Kulesza, Taskar 2012, FnTML 5, arXiv:1207.6083.
- Dvoretzky, Kiefer, Wolfowitz 1956, Ann. Math. Stat. 27.
- Gneiting, Raftery 2007, JASA 102.
