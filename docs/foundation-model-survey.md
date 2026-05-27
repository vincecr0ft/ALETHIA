# Foundation-Model Techniques for a SMEFT → Differential Cross-Section Surrogate

> **Status: SUPERSEDED (2026-05-26).** This survey explicitly recommends
> the closed-form ridge over a polynomial-in-Wilson-coefficient feature
> map as "the right model class". That design puts `c` on the FM's
> forward pass, which now violates the project's hard constraint. The
> revised foundation-model design treats the closed-form ridge as a
> *probe* on a learned representation of the measured distribution, not
> as the model itself. See
> [docs/research/02-foundation-model/](research/02-foundation-model/summary.md)
> for the rebuild and
> [docs/research/synthesis/three-test-cases.md](research/synthesis/three-test-cases.md)
> for the operational plan. The literature survey itself (sections 1
> onward) remains useful background.

A critical, actionable research survey for the ALETHIA physics foundation model.

**Problem under survey.** Build a surrogate mapping SMEFT Wilson coefficients (≤14
real-valued, structured input) to binned differential cross sections (~5–20 bins of a
Drell-Yan dilepton spectrum, structured output). Candidate architecture: the "Intention"
mechanism — closed-form regularized least squares cast as a KVQ attention layer
(Garnelo & Czarnecki 2023, arXiv:2305.10203), feature-wise variant. Implementation in JAX
(GPU where available). The model feeds an active-learning loop (drift detection + EPIG
acquisition). A planned key experiment compares the closed-form solution against
gradient-descent training of the same architecture.

**Date of survey:** 2026-05-22. Literature current to early 2026.

**TL;DR verdict.** The physics here is *benign*: the SMEFT cross section is a known
**quadratic polynomial** in the Wilson coefficients (SM + interference + pure-BSM terms).
This is the single most important fact in this document and it should drive every design
decision. A 14-dimensional quadratic has only 120 monomial terms; with an explicit quadratic
feature map the entire surrogate is *exactly* linear and the closed-form ridge / Intention
solution is not an approximation — it is the right model class. Most of the "foundation
model" machinery (deep nets, scaling laws, neural operators) is unnecessary and risks
*underperforming* a correctly-built linear-in-features ridge model. The hard problems are
not capacity but **conditioning, uncertainty calibration, and acquisition** — focus effort
there.

---

## 1. The Intention / closed-form-attention literature

### Key findings

- **Garnelo & Czarnecki 2023 (arXiv:2305.10203), "Exploring the Space of Key-Value-Query
  Models with Intention."** Defines the KVQ space — models sharing attention's
  key/value/query input structure but not its computation. The "Intention" module computes
  `Intention(K,V,Q) = Q (KᵀK + αI)⁻¹ KᵀV`, i.e. the regularized-least-squares (ridge)
  solution: fit `W = (KᵀK + αI)⁻¹ KᵀV` mapping keys→values, then apply it to the query.
  Proven to be a **strict generalization of linear attention** at identical asymptotic
  complexity, validated on few-shot learning and policy distillation. The α→0 limit is
  ordinary least squares; the α-regularized form is ridge regression.
- **Follow-up — "Ordinary Least Squares as an Attention Mechanism" (arXiv:2504.09663).**
  Independently recasts OLS as a restricted attention module by learning the embedding
  space rather than coefficients directly; reinforces that the closed-form least-squares /
  attention correspondence is solid and now studied from multiple directions.
- **Follow-up — "DEALing with Image Reconstruction: Deep Attentive Least Squares"
  (arXiv:2502.04079).** Stacks least-squares-style attention blocks for image
  reconstruction — evidence the primitive composes into deeper architectures.
- **Theoretical context.** A large 2023–2024 literature shows linear self-attention
  *implements* (preconditioned) gradient descent / ridge regression on in-context
  regression: von Oswald et al. "Transformers learn in-context by gradient descent"
  (arXiv:2212.07677); Mahankali et al. "One Step of Gradient Descent is Provably the
  Optimal In-Context Learner with One Layer of Linear Self-Attention" (arXiv:2307.03576);
  "Linear Transformers are Versatile In-Context Learners" (NeurIPS 2024). Implication:
  Intention's closed form is the *exact* one-shot optimum that a linear-attention layer
  can only approach via iterative GD. For our use case this is decisive — see §5.

### Feature-wise vs global variants

- **Global variant:** one scalar α and one shared `(KᵀK + αI)⁻¹` operator across all output
  dimensions/features — one ridge fit, all output bins share the same regularization and
  the same key-space geometry.
- **Feature-wise variant:** α (and effectively the fit) is resolved per feature/output
  channel — each of the ~5–20 cross-section bins gets its own regularization strength and,
  if keys are also feature-resolved, its own conditioning. This is the planned variant and
  it is the **correct choice here**: Drell-Yan bins span orders of magnitude in event yield
  (steeply falling spectrum), so a per-bin α and per-bin output scaling materially help. It
  is essentially independent ridge regressions per bin, optionally sharing the key/feature
  map. Cost is negligible at this size.

### Regularization (α) selection — recommendations

- **Do not leave α as a hand-tuned constant or a single learnable scalar.** With a quadratic
  feature map (~120 features) and possibly few hundred training points, α controls the
  bias/variance tradeoff directly and the optimum varies per bin.
- **Recommended: per-bin marginal-likelihood (empirical-Bayes) selection.** Treat each bin's
  ridge fit as Bayesian linear regression; α = σ²/τ² (noise variance / prior variance).
  Maximize the log marginal likelihood. "Bayes beats Cross Validation: Efficient and
  Accurate Ridge Regression via Expectation Maximization" (arXiv:2310.18860) gives a fast
  EM procedure that beats leave-one-out CV in both speed and quality. This is the highest-
  value recommendation in this section: it makes α data-driven, per-bin, and gives a
  calibrated predictive variance for free (see §4).
- **Alternative: Generalized Cross-Validation (GCV).** Closed-form, no held-out split,
  proven consistent for prediction risk even under misspecification (arXiv:2601.13955).
  Good cheap fallback / cross-check against the EM estimate.
- **σ-Ridge / group empirical Bayes (arXiv:2010.15817)** is the right framework if you want
  *grouped* regularization (e.g. one α for linear-in-c features, another for quadratic).

### Failure modes — what NOT to do

- **Do NOT run the ridgeless / α→0 limit.** With ~120 quadratic features and possibly fewer
  training points you are near or past the interpolation threshold. Ridgeless least squares
  exhibits a **double-descent variance spike at the interpolation point** and arbitrarily
  large test error there ("Surprises in High-Dimensional Ridgeless Least Squares
  Interpolation", arXiv:1903.08560; ICLR 2024 "Double Descent Demystified"). Keeping α
  strictly positive and tuned removes the spike. *Always regularize.*
- **`KᵀK` doubles the condition number.** Forming the normal-equations matrix and inverting
  squares the condition number of the design. With a quadratic feature map over coefficients
  of disparate magnitude the design is already poorly conditioned. **Never call an explicit
  `inv`.** Use a Cholesky solve of `(KᵀK + αI)` or, better, solve the ridge problem via the
  QR/SVD of the *augmented* design `[K; √α I]` so you never square the conditioning. See §5.
- **Linear attention's known weaknesses still apply when α→0:** the module reduces to linear
  attention, which is a weak associative model. Intention's value is precisely the
  regularized regime — do not advertise the OLS limit as the design point.
- **Single global α hides per-bin heteroscedasticity.** A steeply-falling spectrum has bins
  with vastly different signal-to-noise; one α over-smooths the tail and under-smooths the
  peak. Use the feature-wise variant.

---

## 2. Foundation models & surrogates for scientific regression with small structured inputs

### Key findings

- **There is a directly relevant 2025 paper.** "Reusable theory representations for
  colliders: a demonstrator SMEFT foundation model" (arXiv:2512.15862) builds a foundation
  model for *exactly this domain*: it samples the Warsaw-basis dimension-6 Wilson-coefficient
  space at leading order, takes high-resolution differential distributions in invariant mass
  and transverse momentum of neutral-current Drell-Yan as input, and learns a low-dimensional
  latent manifold via a minimally-parameterized encoder with a supervised contrastive loss.
  It augments predictions with physics-motivated Monte Carlo replicas carrying correlated
  uncertainties, and supports downstream classification-with-UQ, anomaly detection, and
  nearest-neighbor retrieval. **Read this paper in full before finalizing the architecture.**
  Note the differences: that model encodes *distributions → latent*, whereas ALETHIA needs
  *coefficients → distribution* (a forward surrogate / emulator). They are complementary;
  the data-augmentation-with-correlated-MC-replicas idea transfers directly (see §3).
- **The SMEFT cross section is polynomial.** dσ ∝ σ_SM + Σᵢ cᵢ σ_int,ᵢ + Σᵢⱼ cᵢcⱼ σ_BSM,ᵢⱼ.
  At dimension-6, leading order, this is *exactly quadratic* in the Wilson coefficients. This
  is standard in SMEFT phenomenology (e.g. the Drell-Yan SMEFT literature: arXiv:2207.01703,
  arXiv:2211.12261). Consequence: with the explicit quadratic feature map
  φ(c) = [1, c₁..c₁₄, cᵢcⱼ for i≤j], the surrogate is *linear in φ* and ridge/Intention
  recovers the true model class with zero approximation error in the noiseless limit.
- **Random-feature / kernel-ridge / GP theory.** Random features with O(√n log n) features
  match exact kernel ridge regression's O(1/√n) error ("Generalization Properties of
  Learning with Random Features", arXiv:1602.04474; "Ridgeless Regression with Random
  Features", arXiv:2205.00477). GPs deliver higher accuracy than deep nets in the tens-to-
  thousands-of-samples regime and give calibrated uncertainty natively (multiple 2024–2025
  surrogate-modeling papers, e.g. the GP+autoencoder line, arXiv:2407.10732).
- **Neural operators are the wrong tool here.** They target mappings between *function
  spaces* (PDE solution operators). Our output is a short fixed-length vector of bin yields,
  not a discretized field; Graph Neural Simulators / FNOs add inductive bias for *spatial/
  temporal* structure we do not have (arXiv:2509.06154). Skip them.
- **Tabular foundation models (TabPFN) for small structured inputs.** TabPFN — a transformer
  pre-trained on millions of synthetic regression tasks — does single-forward-pass Bayesian
  inference and beat GPs/random forests on 8/10 materials datasets in an active-learning
  setting, saving ~52% of evaluations vs GP ("Foundation-Model Surrogates Enable
  Data-Efficient Active Learning for Materials Discovery", arXiv:2603.12567). Worth a
  *baseline* comparison but not the production model: it gives no closed form, no exposed
  posterior covariance for EPIG, and ignores the known polynomial structure.

### Recommendations

- **Make the quadratic structure explicit.** Use φ(c) = [1, linear terms, all cᵢcⱼ products]
  as the feature map fed to Intention. This is the dominant recommendation of the survey:
  it turns the problem into exact linear regression, makes the closed form provably optimal,
  collapses data requirements (≈120 parameters per bin, not millions), and makes every UQ
  and active-learning quantity analytic. Anything that does *not* exploit this is leaving
  guaranteed accuracy on the table.
- **Keep Intention as the regressor on top of φ.** Feature-wise Intention over φ(c) is
  exactly per-bin Bayesian ridge regression — principled, fast, closed-form. The "attention"
  framing buys conceptual unity with the foundation-model narrative without costing anything.
- **If you suspect the truth deviates from a pure quadratic** (NLO/loop effects, dimension-8
  contamination, PDF-induced nonlinearity): append a *small* random-Fourier-feature block
  (§3) to φ rather than switching to a deep net. Stay linear-in-features so the closed form
  survives.
- **Baselines to run for honest evaluation:** (i) exact GP with an anisotropic RBF/Matérn
  kernel (gold standard for calibrated UQ at this scale); (ii) plain quadratic-feature ridge
  with EM-tuned α; (iii) optionally TabPFN. The surrogate must beat or match the GP on CRPS
  and coverage, not just RMSE.
- **What NOT to do:** do not reach for deep MLPs, transformers with learned attention, or
  neural operators as the production surrogate. At ≤14 inputs / ≤20 outputs / a known
  quadratic, they add variance, kill the closed form, complicate UQ, and will likely lose to
  ridge on a fair CRPS comparison. Use them only as deliberately-handicapped baselines.

---

## 3. Improving generalization in low/medium-data scientific ML

### Key findings

- **Input/output normalization is the highest-leverage, lowest-effort fix.** Wilson
  coefficients differ in natural scale and physical units; Drell-Yan bin yields fall by
  orders of magnitude across the spectrum. Unnormalized, `KᵀK` is severely ill-conditioned
  and α cannot regularize all directions sensibly. Normalization before the feature map
  stabilizes solvers and is repeatedly cited as preventing divergence in scientific-ML
  pipelines (RFF-PINN feature-engineering work, arXiv:2502.07209).
- **Random Fourier features (RFF)** approximate shift-invariant kernels, control which
  frequencies a model learns first, and supply a beneficial inductive bias for fast
  adaptation; "Random at First, Fast at Last" (arXiv:2506.02406) shows NTK-guided Fourier
  pre-processing helps tabular DL. RFF are a controlled, *linear-in-features* way to add
  capacity beyond the quadratic without abandoning the closed form.
- **Multi-fidelity** is well-established for expensive simulators: chain cheap low-fidelity
  and expensive high-fidelity models; recent surveys and the D-MFDAL framework
  (arXiv:2305.04392) show large sample-efficiency gains, and continuous-fidelity
  active-learning kernels exist (arXiv:2503.23158).
- **Physics-informed constraints** here are *structural*, not differential-equation
  penalties: non-negativity of cross sections, the exact quadratic form, known SM value at
  c = 0, monotonic high-mass tail behavior, and symmetry/sign relations among interference
  terms.

### Recommendations

- **Normalize inputs and outputs first, and do it physically.**
  - Inputs: standardize each Wilson coefficient (or scale by its physically expected range /
    current experimental bound) so φ(c) entries are O(1) and `KᵀK` is well-conditioned.
  - Outputs: model the cross section **relative to the SM prediction per bin** (ratio
    dσ/dσ_SM, or log-ratio). This is the single most physics-aware normalization available:
    it removes the orders-of-magnitude spread across the falling spectrum, makes the c = 0
    anchor exactly 1, and turns "predict a steeply-falling spectrum" into "predict an O(1)
    distortion." Strongly recommended.
- **Bake in the SM anchor exactly.** Enforce surrogate(c = 0) = SM by construction (the
  constant term of φ, or by always modelling the ratio). Free, exact, removes a whole error
  mode.
- **Augment with correlated-MC replicas (borrowed from arXiv:2512.15862).** Generate
  training-target replicas drawn from the simulator's statistical + PDF + scale uncertainty
  with the correct bin-to-bin correlations. This both regularizes the fit and makes the
  surrogate's learned noise model match the real experiment's covariance — directly
  improving UQ calibration (§4).
- **If adding capacity, add RFF — not depth.** A modest RFF block appended to the quadratic
  φ keeps the model linear-in-features, preserves the closed form and the analytic
  posterior, and can absorb mild NLO/PDF nonlinearity. Tune the RFF bandwidth by marginal
  likelihood alongside α.
- **Use multi-fidelity if the simulator has a fast mode.** A LO/parton-level fast tier plus
  a slow NLO/showered tier with a multi-fidelity correction (a second ridge fit on the
  residual) can cut expensive calls substantially. Defer until the single-fidelity loop
  works.
- **Enforce non-negativity at prediction time** (clip, or model log-ratio so positivity is
  automatic). Negative predicted cross sections are an immediate, visible failure.
- **What NOT to do:** do not use generic data augmentation (noise jitter, mixup) blindly —
  it injects a noise model inconsistent with the simulator's true covariance and *miscalibrates*
  UQ. Augmentation must be physics-derived. Do not add PDE-residual penalties — there is no
  governing PDE here; the "physics" is the polynomial form and positivity, which are better
  imposed structurally.

---

## 4. Uncertainty quantification for surrogates

### Key findings

- **A ridge fit *is* a Bayesian linear model — use that.** Per-bin Intention/ridge with
  Gaussian prior and noise yields a closed-form Gaussian posterior predictive:
  mean μ(c) = φ(c)ᵀŵ, variance v(c) = σ²(1 + φ(c)ᵀ(KᵀK + αI)⁻¹φ(c)). This is the
  **Bayesian last-layer** model in its purest form and it is *exact* (not an approximation)
  because the model is genuinely linear in φ. No ensembles, no MC dropout needed for
  epistemic uncertainty.
- **Deep ensembles** are the standard deep-learning UQ baseline; the Bayesian-last-layer
  view (arXiv:2105.13283) shows they approximate variational inference over the last layer —
  exactly what the closed form gives us directly and exactly. So ensembles add cost without
  adding fidelity here.
- **GPs** give calibrated uncertainty natively at this data scale and are the natural gold-
  standard cross-check (numerous 2024 surrogate-UQ papers).
- **Conformal prediction** ("Uncertainty Quantification of Surrogate Models using Conformal
  Prediction", arXiv:2408.09881) gives *distribution-free, finite-sample marginal coverage*
  at near-zero cost, model-agnostic, with cell-wise calibration that preserves tensor
  structure and holds even out-of-distribution. Limitations: marginal (not conditional)
  coverage, and it assumes exchangeability of calibration and test data — **which active
  learning deliberately breaks** (each AL round shifts the input distribution).
- **Proper scoring rules** — NLPD/NLL and CRPS — are the correct way to *score* predictive
  distributions; CRPS is robust and using CRPS as a training/selection loss can improve both
  accuracy and NLL (shallow-ensemble UQ work, IOPscience 10.1088/2632-2153/ad594a).

### Recommendations

- **Adopt the closed-form Bayesian-linear predictive distribution as the primary UQ.** It is
  exact for the quadratic-feature model, gives per-bin epistemic variance analytically, and —
  critically — exposes the **full posterior covariance** needed for EPIG (§6). This is the
  reason to prefer this architecture over a deep net.
- **Estimate σ² (aleatoric / simulator noise) per bin from the marginal-likelihood / EM fit**
  (§1), or set it from the simulator's known statistical + replica uncertainty. Output the
  *predictive* covariance = epistemic + aleatoric. The simulator MC error is itself the
  aleatoric term — you have a rare luxury: the noise level is known, not guessed.
- **Model the full bin-to-bin output covariance, not just per-bin variance.** Drell-Yan bins
  are correlated (shared PDF/scale uncertainty, migration). A diagonal covariance will
  miscalibrate joint coverage and bias χ² downstream. Use a matrix-variate / multi-output
  ridge or fit the residual covariance from the correlated MC replicas.
- **Wrap the model in conformal prediction as an outer calibration guarantee**, but: (i) use
  a calibration set drawn from the *current* deployment region, (ii) re-conformalize after
  every AL round, and (iii) report it as marginal coverage only. Treat conformal as a
  safety net on top of the Bayesian intervals, not a replacement — the Bayesian posterior is
  what AL needs.
- **Score with NLPD and CRPS, never RMSE alone.** RMSE rewards a confident wrong model;
  CRPS/NLPD penalize miscalibration. Report both, per bin and aggregated.
- **What NOT to do:** do not use deep ensembles or MC dropout here — they approximate, at
  higher cost, the posterior you can write in closed form exactly. Do not trust conformal
  coverage *during* active learning without re-calibration — exchangeability is violated by
  design. Do not report a single scalar "uncertainty" — separate epistemic (drives
  acquisition) from aleatoric (irreducible simulator noise); conflating them is the
  documented reason uncertainty heuristics fail in AL (arXiv:2501.01248, arXiv:2501.08223).

---

## 5. JAX performance & engineering

### Key findings

- **For ≤14 inputs / ~120 quadratic features / ≤20 output bins, the linear algebra is
  tiny.** `(KᵀK + αI)` is at most ~120×120. A Cholesky solve is microseconds. The closed
  form is not a performance concern; correctness and conditioning are.
- **`jit` + `vmap` is the canonical fast pattern**; `vmap` turns matrix-vector products into
  matrix-matrix products (good for GPU) but materializes whole-batch intermediates, so it
  can blow up peak memory (JAX docs; apxml vmap notes). `scan` is for genuinely sequential
  dependencies (e.g. the AL round loop), not for parallel batches.
- **`enable_x64` caveats.** float64 must be set via `jax.config.update("jax_enable_x64",
  True)` *before* JAX initializes; it is a global flag. There are open issues with
  `linalg.solve`, `cholesky`, `svd`, `eig` accuracy/behavior inside `jit` under x64
  (jax-ml/jax #27602, #11433), and `jnp.linalg.lstsq` has been reported less accurate than
  NumPy's. Double precision also costs memory/throughput, badly on consumer GPUs.
- **GPU vs closed form.** Closed-form linear solves at this size run fine on CPU and GPU;
  the GPU gives nothing for a 120×120 system. GPU matters only if you batch *thousands* of
  independent fits (e.g. per-bin × per-AL-round × replica ensembles) — then `vmap` over the
  batch is the win.
- **When GD genuinely beats closed form:** (a) when the feature map itself is *learned*
  (then it is no longer linear and there is no closed form); (b) when the design matrix is
  too large to factorize in memory (not our case); (c) when you want online/streaming
  updates; (d) when a non-Gaussian likelihood or non-quadratic loss is required. None of
  these apply to the production quadratic-feature model — so the closed form should *win*
  the planned head-to-head on both speed and accuracy.

### Recommendations

- **Solve ridge via the augmented-design QR/SVD, not normal equations.** Stack the design
  `Φ` with `√α · I` and solve the least-squares problem on `[Φ; √α I]` by QR (or SVD). This
  computes the ridge solution *without ever forming `ΦᵀΦ`*, so the condition number is not
  squared. This is the single most important engineering recommendation: with a quadratic
  feature map the conditioning headroom matters. Use `jax.scipy.linalg` Cholesky only on the
  already-well-conditioned normalized problem, as a faster path once you have verified
  agreement with the QR result.
- **Never call `jnp.linalg.inv`.** Use `cho_factor`/`cho_solve` or a QR/SVD solve. Computing
  the posterior covariance: factor once, reuse the factor for both the mean and for
  `φ(c)ᵀ(ΦᵀΦ+αI)⁻¹φ(c)` (triangular solves), never an explicit inverse.
- **Enable float64 globally for the linear-algebra core.** The fit is cheap; numerical
  exactness of the closed form (and of the closed-form-vs-GD comparison) is worth far more
  than the throughput. Set the flag at program start. Verify `linalg` ops behave under
  `jit`+x64 on your JAX version with a small known-answer test before trusting them.
- **`vmap` the batch dimensions, `jit` the whole fit-and-predict function.** Batch the ~20
  per-bin fits (and any replica ensemble) with `vmap`; that converts the per-bin matvecs
  into single GEMMs. Keep the outer active-learning round loop as a Python loop or `scan` —
  it is sequential and tiny.
- **Skip mixed precision.** bf16/fp16 helps large neural-net training; for a 120×120 solve
  it only injects error into a computation whose entire value is being exact. Mixed
  precision here is a net negative.
- **Run the closed-form vs GD experiment as a *verification*, not an open question.** Theory
  (von Oswald 2212.07677; Mahankali 2307.03576) says GD on a linear-attention/ridge
  objective converges *to* the closed form. Expectation: the closed form matches the
  GD optimum exactly and is orders of magnitude faster. If GD "wins," that is a *bug signal*
  — most likely the closed form was computed on a badly-conditioned normal-equations matrix
  (squared condition number) or with the wrong α. Use the discrepancy as a conditioning
  diagnostic.
- **Common pitfalls to avoid:** recompilation from changing array shapes (pad to fixed
  shapes across AL rounds); `vmap` memory blow-up on large replica batches (chunk the
  `vmap`); silent float32 fallback if the x64 flag is set too late; trusting
  `jnp.linalg.lstsq` for the production solve (use QR/SVD or validate against NumPy).

---

## 6. Active learning / Bayesian experimental design with surrogates

### Key findings

- **EPIG vs BALD.** BALD maximizes expected information gain about model *parameters*; it
  lacks a notion of an input distribution and over-selects points irrelevant to the region
  of interest. EPIG ("Prediction-Oriented Bayesian Active Learning", Bickford Smith et al.,
  arXiv:2304.08151) maximizes expected information gain about *predictions* at a target
  input distribution — it favors only acquisitions that reduce *downstream predictive*
  uncertainty, and beats BALD across datasets, especially under distribution shift and with
  redundant pool data.
- **What the acquisition needs from the surrogate.** EPIG requires a *joint* predictive
  distribution: the model must expose, for candidate point c* and target points c_t, the
  posterior predictive covariance `Cov(y(c*), y(c_t))`. For a Gaussian model EPIG reduces to
  an entropy/log-determinant expression that is **analytic** — exactly computable from the
  closed-form posterior. This is the second decisive reason to use the closed-form Bayesian-
  linear model: EPIG, BALD, and variance-reduction acquisitions all become closed-form.
- **Epistemic/aleatoric split is mandatory.** Acquisition must target *epistemic* (reducible)
  uncertainty; chasing aleatoric (irreducible simulator noise) wastes the budget. Heuristics
  that conflate the two underperform (arXiv:2501.01248, arXiv:2408.13690 "Active Learning
  under Model Mismatch", big-batch AL arXiv:2501.08223).
- **Drift detection** (the planned Phoenix component) is consistent with EPIG: define the
  target distribution over the physics regime of interest; drift = the deployment
  distribution moving away from where the surrogate has low predictive uncertainty.

### Recommendations

- **Use EPIG, not BALD, as the acquisition function.** Our explicit goal is predictive
  accuracy of the cross-section surrogate over a region of Wilson-coefficient space — EPIG's
  prediction-space objective is the literal match. Define the EPIG target distribution as
  the physically relevant region (e.g. inside current experimental bounds, or the region a
  downstream SMEFT fit is exploring).
- **Exploit the closed form for fast batch acquisition.** Adding a candidate point updates
  `(ΦᵀΦ + αI)⁻¹` by a rank-1 Sherman–Morrison update — no refit needed to score thousands of
  candidates. EPIG/variance-reduction scores over a large candidate pool become a single
  `vmap`'d closed-form computation. This makes large-pool, batched BED cheap.
- **Acquire on epistemic variance only.** Score candidates with the *epistemic* part
  φ(c)ᵀ(ΦᵀΦ+αI)⁻¹φ(c) (and its cross-covariances for EPIG), excluding the fixed aleatoric
  σ². For batches, use the joint covariance / log-det to avoid selecting mutually redundant
  points.
- **Re-tune α and re-conformalize after each acquisition round.** The design changes every
  round; a stale α or stale conformal calibration silently degrades both predictions and
  coverage.
- **Expose, in the surrogate API for the AL loop:** posterior mean, per-bin epistemic
  variance, full posterior covariance (bin-to-bin *and* the parameter-space covariance
  `(ΦᵀΦ+αI)⁻¹` enabling candidate cross-covariances), and the aleatoric σ². If the
  surrogate cannot return the joint covariance, EPIG cannot be computed properly — this is a
  hard architectural requirement.
- **What NOT to do:** do not use plain max-predictive-variance ("uncertainty sampling") —
  it ignores the target distribution and redundancy and reduces to BALD-like failure modes.
  Do not run EPIG off a point-estimate surrogate (a deep net without a posterior) — it has
  no joint predictive distribution to integrate. Do not let acquisition chase aleatoric
  noise.

---

## 7. Statistically-focused evaluation metrics

### Key findings

- **PIT histograms** (Probability Integral Transform) diagnose calibration shape: a uniform
  histogram = calibrated; a U-shape = under-dispersed (over-confident) intervals; a
  dome/bell shape = over-dispersed (under-confident); a slope or edge peak = bias
  (ESANN 2024 PIT-interpretation paper). For binned data use the *randomized* PIT.
- **Reliability diagrams / coverage.** Plot nominal vs empirical coverage of central
  intervals; CORP reliability diagrams add consistency bands for uncertainty on the diagram
  itself (Dimitriadis et al., arXiv:2008.03033). Conformal-prediction physics work
  (arXiv:2408.09881) demonstrates empirical-coverage validation across PDE/MHD/weather/fusion
  surrogates.
- **Proper scoring rules.** NLPD/NLL and CRPS are *proper* — minimized only by the true
  predictive distribution — so they jointly score sharpness and calibration; RMSE is not
  proper for distributions and rewards over-confidence.
- **Physics-native diagnostics.** Pulls (standardized residuals (y_pred − y_true)/σ_pred)
  should be unit-Gaussian: mean 0, width 1. χ²/ndf using the *full predictive covariance*
  should be ≈ 1. These are the HEP-standard checks and map cleanly onto the probabilistic
  metrics above.

### Recommendations — the validation suite for the surrogate

- **Pull distribution per bin and aggregated.** Pulls must be ~N(0,1). A pull width > 1 =
  over-confident surrogate; width < 1 = over-conservative; nonzero mean = bias. This is the
  fastest, most physics-legible calibration check — make it the headline plot.
- **χ²/ndf with the full predictive covariance**, not diagonal. Per-spectrum χ² over the
  ~5–20 bins should average ~1 with the expected χ²-distribution spread. A diagonal-only χ²
  will be wrong because Drell-Yan bins are correlated (§4).
- **PIT histogram (randomized) per bin** — read off bias / over- / under-dispersion at a
  glance. Pair with a **reliability diagram** (nominal vs empirical interval coverage, e.g.
  50/68/90/95%) with CORP consistency bands.
- **CRPS and NLPD as the primary scalar scores**, reported per bin and aggregated, on a
  held-out set drawn from the deployment region. These are the numbers to track across AL
  rounds and to gate redeploys on. RMSE/MAE only as secondary sharpness indicators.
- **Residual diagnostics vs inputs.** Plot residuals and pulls against each Wilson
  coefficient and against bin index / invariant mass — structure (trends, fan shapes)
  reveals model misspecification (e.g. missing NLO curvature, heteroscedasticity the σ²
  model misses).
- **Coverage tracked through the AL loop.** Because AL breaks exchangeability, recompute
  coverage and PIT on a fresh deployment-region sample each round; a drifting PIT histogram
  is an early warning that should gate redeployment (ties directly into the Phoenix drift
  monitoring).
- **What NOT to do:** do not report only RMSE/R² — they hide miscalibration and a
  confidently-wrong model scores well. Do not use diagonal covariance for χ². Do not
  evaluate only on an i.i.d. test split — also evaluate on held-out *physics regimes* (high-mass
  tail, large-|c| region) to expose extrapolation failure, exactly as the conformal-surrogate
  paper stresses OOD testing.

---

## 8. Verifying that a model "scales"

### Key findings

- **Empirical scaling laws extrapolate well when fit correctly.** Power-law fits from small
  models reliably predict larger-model performance; refined methods cut extrapolation error
  substantially (Farseer, arXiv:2506.10972, ~433% error reduction over Chinchilla-style
  fits; "How to Upscale Neural Networks with Scaling Law", arXiv:2502.12051; NeuNeu
  arXiv:2601.19831 frames it as time-series extrapolation).
- **For a linear-in-features model the "scaling law" is classical learning theory, not a
  fitted mystery.** Kernel-ridge / random-feature error scales as O(1/√n) and random-feature
  count O(√n log n) suffices to match exact KRR (arXiv:1602.04474, arXiv:2205.00477). The
  test error of the quadratic-feature ridge model as a function of training size is
  *predictable from theory* — you do not need LLM-style empirical scaling laws.

### Recommendations

- **Build a learning curve (test CRPS/NLPD/χ² vs training-set size) on log-log axes early.**
  Fit a power law `error ≈ a·n^(−b) + c`. The asymptote `c` is the irreducible error
  (model misspecification + simulator noise floor); `b` is the convergence rate. This single
  plot tells you (i) whether more data helps, (ii) the data budget to hit a target accuracy,
  (iii) whether the surrogate is misspecified (a high `c` floor on a problem you believe is
  exactly quadratic = a bug or missing feature, e.g. unmodelled NLO terms).
- **Verify the closed form scales by problem size, not by parameter count.** "Scaling" here
  means: more Wilson coefficients → more quadratic features (D grows as ~D²/2), more bins →
  more independent fits. Profile fit time and memory vs (#coefficients, #bins, #training
  points) on small cases and extrapolate the (cheap, polynomial) cost. There is no
  emergent-capability regime to worry about.
- **Validate small-to-large extrapolation explicitly.** Run the full pipeline (fit + UQ +
  one AL round) on a deliberately small fast configuration (few coefficients, few bins, few
  hundred samples), record wall-clock and accuracy, then predict the large-run cost/accuracy
  from the learning curve and the polynomial cost model — and check the prediction on one
  medium run before committing to the full run.
- **Track CRPS/NLPD vs AL-round** as the operational scaling curve: it should decrease and
  flatten; a flat-from-the-start curve means the acquisition is not finding informative
  points (revisit EPIG target distribution); a noisy curve means α/conformal are not being
  re-tuned per round.
- **What NOT to do:** do not assume "bigger model = better" — for a known quadratic, extra
  capacity past the quadratic feature map only adds variance. Do not extrapolate a scaling
  law from a *single* small point or from points all below the interpolation threshold (the
  double-descent region, §1) — the curve is non-monotonic there and will mislead. Do not
  conflate compute scaling (cheap, polynomial, predictable) with statistical scaling (the
  O(1/√n) learning curve) — they are different axes.

---

## Consolidated, prioritized recommendations

1. **Exploit the quadratic structure: use an explicit quadratic feature map**
   φ(c) = [1, cᵢ, cᵢcⱼ]. The SMEFT cross section is exactly quadratic in dim-6 Wilson
   coefficients at LO — this makes Intention/ridge the *exactly correct* model class, not an
   approximation. (~120 params/bin.)
2. **Model the cross section as a ratio to the SM per bin** (dσ/dσ_SM or log-ratio). Removes
   the orders-of-magnitude spread of the falling spectrum, anchors c = 0 exactly to the SM,
   and improves conditioning and UQ calibration.
3. **Always regularize — never run ridgeless (α→0).** With ~120 features you sit near the
   interpolation threshold; ridgeless triggers the double-descent variance spike.
4. **Select α per bin by marginal-likelihood / EM (empirical Bayes)**, not a hand-tuned or
   single learnable scalar. Fast, beats LOO-CV, and yields a calibrated posterior for free.
   Use the feature-wise Intention variant so each bin gets its own α.
5. **Solve ridge via augmented-design QR/SVD (`[Φ; √α I]`), never the normal equations or
   `inv`.** Forming `ΦᵀΦ` squares the condition number; with a quadratic feature map that is
   the most likely source of a spurious "GD beats closed form" result.
6. **Use the closed-form Bayesian-linear posterior as the native UQ** — predictive mean +
   epistemic covariance + known simulator-noise aleatoric term. Exact here; no ensembles or
   MC dropout needed.
7. **Model the full bin-to-bin output covariance** (multi-output / matrix-variate ridge or
   fit residual covariance from MC replicas). Drell-Yan bins are correlated; a diagonal
   covariance miscalibrates joint coverage and χ².
8. **Use EPIG (not BALD, not plain uncertainty sampling) for acquisition**, with the target
   distribution = the physically relevant Wilson-coefficient region. EPIG is analytic for
   the Gaussian closed-form model.
9. **Acquire on epistemic variance only**, using Sherman–Morrison rank-1 updates to score
   large candidate pools without refitting; `vmap` the scan over candidates.
10. **Score with CRPS and NLPD, never RMSE alone**; make per-bin **pull distributions** and
    **χ²/ndf with full covariance** the headline validation plots, plus randomized-PIT
    histograms and CORP reliability diagrams.
11. **Re-tune α and re-conformalize after every active-learning round** — AL breaks
    exchangeability and shifts the design; stale calibration silently degrades coverage.
12. **Augment training targets with correlated-MC replicas** drawn from the simulator's
    statistical/PDF/scale uncertainty (idea from the SMEFT foundation-model paper,
    arXiv:2512.15862) — this is physics-correct augmentation that improves UQ calibration;
    avoid generic noise/mixup augmentation, which miscalibrates.
13. **Treat the closed-form-vs-GD experiment as verification:** theory says GD converges to
    the closed form. Expect the closed form to match GD's optimum and be far faster; if GD
    "wins," treat it as a conditioning bug (see #5), not a real result.
14. **Build a log-log learning curve (CRPS/NLPD vs n) early** and fit `a·n^(−b)+c`; the
    floor `c` should be near the simulator noise level — a high floor on a "purely
    quadratic" problem signals misspecification (e.g. unmodelled NLO). Use it to set the
    data budget; do not fit a scaling law from points in the double-descent region.
15. **Use `jit` + `vmap` over the ~20 per-bin fits, float64 enabled globally, no mixed
    precision.** The linear algebra is tiny (~120×120); correctness dominates throughput.
    GPU only matters when batching thousands of fits/replicas.

### Key references

- Garnelo & Czarnecki, *Exploring the Space of KVQ Models with Intention*, arXiv:2305.10203
- *Ordinary Least Squares as an Attention Mechanism*, arXiv:2504.09663
- *Deep Attentive Least Squares (DEAL)*, arXiv:2502.04079
- von Oswald et al., *Transformers learn in-context by gradient descent*, arXiv:2212.07677
- Mahankali et al., *One Step of GD is the Optimal In-Context Learner...*, arXiv:2307.03576
- *Reusable theory representations for colliders: a demonstrator SMEFT foundation model*, arXiv:2512.15862
- Hastie et al., *Surprises in High-Dimensional Ridgeless Least Squares Interpolation*, arXiv:1903.08560
- *Bayes beats Cross Validation: Ridge Regression via EM*, arXiv:2310.18860
- *σ-Ridge: group regularized ridge via empirical Bayes*, arXiv:2010.15817
- *Uniform Consistency of GCV for Ridge Regression*, arXiv:2601.13955
- Rudi & Rosasco, *Generalization Properties of Learning with Random Features*, arXiv:1602.04474
- *Ridgeless Regression with Random Features*, arXiv:2205.00477
- *Foundation-Model Surrogates Enable Data-Efficient Active Learning for Materials Discovery*, arXiv:2603.12567
- Bickford Smith et al., *Prediction-Oriented Bayesian Active Learning* (EPIG), arXiv:2304.08151
- *Uncertainty Quantification of Surrogate Models using Conformal Prediction*, arXiv:2408.09881
- Dimitriadis et al., *Evaluating probabilistic classifiers: Reliability diagrams revisited* (CORP), arXiv:2008.03033
- *Disentangled Multi-Fidelity Deep Bayesian Active Learning*, arXiv:2305.04392
- Drell-Yan SMEFT phenomenology: arXiv:2207.01703, arXiv:2211.12261
- *Farseer: A Refined Scaling Law*, arXiv:2506.10972; *How to Upscale with Scaling Law*, arXiv:2502.12051
- JAX docs: GPU performance tips, Default dtypes and the X64 flag; JAXMg multi-GPU solver, arXiv:2601.14466
