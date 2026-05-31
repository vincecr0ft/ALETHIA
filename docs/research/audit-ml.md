# ML-side audit of ALETHIA (paper/alethia.tex)

Scope: ML content only. Physics oracle, EPIG/conformal calibration mathematics,
and chain plumbing are taken as given. Goal is to identify the actual ML
contribution, locate the load-bearing baselines that are missing, and propose
implementable upgrades.

---

## 1. The real questions

The paper sells four claims:

1. A 5,328-parameter "foundation model" learns a basis psi_theta for closed-form
   ridge in m_ll that beats DeepSets, GP-Matern, kernel ridge and a "cheat"
   regressor on held-out SMEFT scenarios.
2. The architectural constraint (Wilson coefficient never on the forward pass)
   is what buys magnitude-extrapolation robustness — the cheat baseline has the
   p5 catastrophic tail to prove it.
3. A linear probe from the per-scenario implicit ridge weights w_implicit to
   c_lq^(3) reaches R^2 = 0.94 on a withheld magnitude band — "strong-form
   disclosure".
4. The whole stack composes into a drift-gated EPIG/conformal closed loop that
   recovers 110x RMSE on an engineered extrapolation.

Stripped of the foundation-model framing, the load-bearing technical claim is
much narrower:

> Meta-training a 3-layer MLP basis psi_theta : R -> R^16 inside a closed-form
> ridge head, on 200 SMEFT scenarios with K=12 context points, yields a per-
> scenario amortised regressor whose linear probe disentangles the dominant
> energy-growing Warsaw operator on an unseen magnitude band.

That is a Prior-Fitted-Network / Neural-Process construction with the
attention block restricted to its analytic ridge form. The question is whether
that restriction plus the SMEFT prior plus the linear-probe disclosure is a
genuine architectural contribution or a re-skinning of Mueller et al. 2022 PFNs
with extra geometry.

The four specific questions that drove this audit:

- (Q1) Is "foundation model" defensible vs "PFN with linear attention"?
- (Q2) Is the DeepSets baseline a strawman, and what's the right one?
- (Q3) HEP-JEPA is cited but never compared. Should it be?
- (Q4) Is the linear-probe identifiability claim an overclaim, or are there
  load-bearing details ([Sec 3.4 of the paper](file:///home/vince/ALETHIA/paper/alethia.tex#L210)) that elevate it?
- (Q5) Does an online-PCA / Fisher-aligned rotation of psi_theta survive
  contact with the literature, and what experiment would settle it?
- (Q6) Why is there no scaling analysis when the paper repeatedly invokes
  "scaling-behaviour analyses now standard"?

---

## 2. Literature snapshot (2022-2026)

Curated to the audit, not exhaustive.

**Prior-Fitted Networks and in-context Bayesian inference**

- Mueller, Hollmann, Arango, Grabocka, Hutter, 2022. *Transformers Can Do
  Bayesian Inference.* arXiv:2112.10510. The PFN construction: pretrain a
  transformer on synthetic datasets from a prior, single forward pass yields
  posterior predictive. This is the parent line. The ALETHIA paper cites it
  once ([Sec 6](file:///home/vince/ALETHIA/paper/alethia.tex#L338)) and never
  compares.
- Nagler, 2023. *Statistical Foundations of Prior-Data Fitted Networks.* ICML
  2023. Proves PFNs converge to true Bayes-optimal posterior predictive at the
  appropriate rate.
- TabPFN-v2 (Hollmann et al., 2025) + NPE-PFN (arXiv:2504.17660, 2025).
  *Effortless Simulation-Efficient Bayesian Inference using Tabular Foundation
  Models.* Repurposes TabPFN as a training-free SBI engine, claims orders-of-
  magnitude simulation efficiency. This is the most direct competitor for the
  amortised-inference framing.
- Do-PFN (arXiv:2506.06039, 2025): PFN for causal effects.
- arXiv:2505.23947, 2025. *Position: The Future of Bayesian Prediction Is
  Prior-Fitted.* Argues PFNs subsume Bayesian prediction.

**Closed-form / linear in-context learning**

- Garnelo, Czarnecki, 2023. *Exploring the Space of KVQ Models with
  Intention.* arXiv:2305.10203. The closed-form ridge head used here.
- Akyurek, Schuurmans, Andreas, Ma, Zhou, 2022. *What learning algorithm is
  in-context learning?* arXiv:2211.15661. Trained transformers on linear
  regression match OLS / ridge to within numerical precision; the Bayes-
  optimal ridge parameter is the one matched.
- von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov,
  Vladymyrov, 2023. *Transformers Learn In-Context by Gradient Descent.*
  Linear self-attention = one step of GD on least-squares.
- Mahankali, Hashimoto, Ma, 2023. *One Step of GD is Provably the Optimal
  In-Context Learner with One Layer of Linear Self-Attention.* arXiv:2307.03576.
- Vladymyrov, von Oswald et al., 2024. *Linear Transformers are Versatile
  In-Context Learners.* NeurIPS 2024.

**HEP foundation models**

- OmniJet-alpha (Birk, Hallin, Kasieczka), 2024. arXiv:2403.05618. Cross-task
  jet FM, autoregressive tokenisation.
- OmniLearn / OmniLearned (Mikuni, Nachman), 2024 + 2025 (arXiv:2510.24066,
  1B-jet pretraining + scaling curves).
- Masked Particle Modeling (Heinrich, Golling et al.), 2024. arXiv:2401.13537.
- Sophon (jet-universe), 2024. 188-class pretraining, Particle Transformer
  backbone.
- HEP-JEPA (Katel, 2025). arXiv:2502.03933. The first published JEPA-style
  jet model: predict masked particle embeddings in latent space.
- JJEPA, 2024. arXiv:2412.05333. Symmetry-independent jet representations.
- arXiv:2512.04149, 2025. Enhancing next-token-prediction pretraining for jet
  FMs.
- **arXiv:2512.15862, December 2025 (one month ago).** *Reusable theory
  representations for colliders: a demonstrator SMEFT foundation model.* This
  is the only other published "SMEFT foundation model" and is concurrent with
  ALETHIA. Architecture is supervised-contrastive on Drell-Yan invariant-mass +
  pT spectra, learns a latent geometry of Warsaw-basis deformations, supports
  classification / anomaly / retrieval. **The paper does not cite or mention it.**

**Neural processes (the pre-history of Intention)**

- Garnelo, Schwarz, Rosenbaum, Viola, Rezende, Eslami, Teh, 2018. *Neural
  Processes.* The parent line. Encoder + latent + decoder; Intention is the
  closed-form linear special case.
- Kim, Mnih, Schwarz, Garnelo, Eslami, Rosenbaum, Vinyals, Teh, 2019.
  *Attentive Neural Processes.* arXiv:1901.05761.
- Lee, Lee, Kim, Kosiorek, Choi, Teh, 2019. *Set Transformer.* The right
  baseline for "permutation-invariant context summarisation" — strictly more
  powerful than mean-pool DeepSets.
- Markou et al., 2022; Gordon et al., 2024. *Sparse Gaussian Neural Processes.*
  arXiv:2504.01650 (2025).

**Linear-probe interpretability**

- Alain, Bengio, 2016. *Understanding intermediate layers using linear
  classifier probes.* The progenitor.
- Belinkov, 2022. *Probing Classifiers: Promises, Shortcomings, Advances.*
  Computational Linguistics. The known-pitfalls survey.
- Hewitt, Liang, 2019. *Designing and Interpreting Probes with Control Tasks.*
- arXiv:2511.16288, 2025. *Spectral Identifiability for Interpretable Probe
  Geometry (SIP).* Connects eigengaps, Fisher information and probe stability
  via a phase transition. **Directly relevant to the eigen-direction proposal
  in section 5 below.**

**Rotated / Fisher-aligned representations (relevant to eigen-direction proposal)**

- Whitening / score-PCA: Stuart et al. 2025, *Rotated Mean-Field Variational
  Inference and Iterative Gaussianization* (UCLA preprint), rotates to the
  eigenbasis of the Hessian / Fisher and shows the rotation maximises a lower
  bound on projected Fisher information.
- Liu, Chen 2020, *Fisher Information Patch-based Parametric PCA.*

---

## 3. Findings

### 3.1 "Foundation model" vs PFN

Concrete evidence from the code:

- The forward pass is exactly equation (3) of Mueller et al. 2022 (PFN
  posterior predictive) restricted to a Gaussian prior over weights and a
  homoscedastic noise model. See [`IntentionFM.forward`,
  modules/surrogate/intention/model.py:57-68](file:///home/vince/ALETHIA/modules/surrogate/intention/model.py#L57).
- The training loop is the PFN training loop: sample a scenario from the prior
  over SMEFT Wilson coefficients ([data_smeft.sample_c]),
  draw context K + query Q from that scenario, MSE-regress the held-out query.
  See [experiment_smeft.py:75-109](file:///home/vince/ALETHIA/experiments/intention-vs-deepsets/experiment_smeft.py#L75).
- The "in-context generalisation" is then the standard PFN claim, with the
  attention block analytically substituted by `torch.linalg.solve` over a
  16-dim feature space.

So the construction is **a PFN with the transformer block replaced by a
closed-form ridge regression in a learned feature map**. This is internally
consistent and reasonable, but the rhetorical move from "PFN" to "foundation
model for SMEFT" is unsupported by the experiments. The Mueller paper is
cited in passing ([line 338](file:///home/vince/ALETHIA/paper/alethia.tex#L338))
and dismissed as "adopting the same in-context view, but with the closed-form
linear attention of equation (3) in place of the generic transformer". This is
exactly the comparison that should be a headline experiment.

The concurrent SMEFT FM paper (arXiv:2512.15862) is not cited at all. That is
the natural sister-work comparison and the omission is a substantive gap.

**Verdict:** This is a PFN with an analytic ridge head and a HEP-specific
prior. The "foundation model" label is rhetorically thin without (a) a
self-supervised pretext that does not see the downstream label and (b) a
fine-tuning experiment to a distinct downstream task. The model has neither.

### 3.2 Strawman analysis of DeepSets baseline

The DeepSets baseline is mean-pool with no attention:
[deepsets_matched.py:50-80](file:///home/vince/ALETHIA/experiments/intention-vs-deepsets/deepsets_matched.py#L50).
The paper's [Sec 6](file:///home/vince/ALETHIA/paper/alethia.tex#L334)
admits "DeepSets is at an architectural disadvantage on a one-dimensional
kinematic regression problem". The right baseline pool for a closed-form
ridge head over a 1D context is:

- **Vanilla PFN.** Transformer encoder + cross-attention to query, trained on
  the same scenario prior. This is the direct apples-to-apples test of
  whether the closed-form restriction helps.
- **Attentive Neural Process.** The non-closed-form ancestor of Intention.
- **Set Transformer.** Drop-in replacement for the DeepSets mean-pool, gives
  the permutation-invariant baseline its strongest form.
- **Standard transformer ICL on (m, Y) pairs.** The linear-regression
  in-context learners studied by Akyurek 2022 / von Oswald 2023 / Mahankali
  2023, trained on this prior.

None of these exists in `experiments/intention-vs-deepsets/`. Only the
DeepSets-mean-pool, the cheat MLP, the GP/Matern, the RBF kernel-ridge and the
fixed polynomial-basis Intention variants are implemented. The directory was
audited by reading the file list and confirming via
[experiments/intention-vs-deepsets/output_smeft/summary.json](file:///home/vince/ALETHIA/experiments/intention-vs-deepsets/output_smeft/summary.json).

**Verdict:** Yes, this is a strawman in its current form. R^2 = 0.9999 vs 0.70
is not a useful comparison — the right comparison is against models that also
use attention over the context. Concrete fix in section 4 below.

### 3.3 HEP-JEPA / latent-pretraining missed

[Sec 6](file:///home/vince/ALETHIA/paper/alethia.tex#L340) lists HEP-JEPA among
"jet-level foundation models" and dismisses them all on the grounds that
ALETHIA "produces a continuous prediction of a cross-section ratio rather than
a class probability". This sidesteps the actual question: **could JEPA-style
latent prediction be used to pretrain psi_theta**, decoupling the basis from
the supervised target?

Concrete adaptation. Mask half of the context entries (M_ctx, Y_ctx); ask the
encoder to predict their latent embeddings from the unmasked half. Use the
resulting psi_theta as the basis for the closed-form ridge fine-tune. This is
the HEP-JEPA recipe applied to 1D kinematic sequences. The proposal in
section 4.2 below.

### 3.4 Linear-probe identifiability claim

The claim ([Sec 5](file:///home/vince/ALETHIA/paper/alethia.tex#L210)) is that
a linear probe in w_implicit (the closed-form ridge weight vector) recovers
c_lq^(3) at R^2 = 0.94 on a withheld magnitude band. Two things to note:

- The probe is in `w_implicit` ∈ R^16, *not* in psi_theta(m). Because
  w_implicit is linear in Y_ctx, and Y_ctx is quadratic in c, the probe is a
  linear map of a quadratic function of c. The paper acknowledges this on
  [lines 218-220](file:///home/vince/ALETHIA/paper/alethia.tex#L218): "a
  linear probe in w_implicit is therefore a genuinely nonlinear-in-c map".
  Fair point but underplayed.
- The "strong-form disclosure" framing imports vocabulary from
  mechanistic-interpretability papers (Anthropic-style features, TCAV) without
  matching their evidentiary burden. A linear probe at R^2 = 0.94 on a single
  operator out of four is not a "disclosure result"; it is a regression of
  one direction of the implicit weight vector onto one operator, evaluated on
  a magnitude band only — not a *direction* held out.
- Belinkov 2022 catalogues the known pitfalls (selectivity, control tasks,
  baseline-probe inflation) that the paper does not run.

**Verdict:** The probe is interesting and the result is real, but the
"strong-form disclosure" framing overreaches. Either drop the language or do
the control experiments. Cheap proposal in section 4.5.

### 3.5 No scaling analysis

[Sec 7](file:///home/vince/ALETHIA/paper/alethia.tex#L349) refers to
"scaling-behaviour analyses now standard in the jet-level foundation model
literature". OmniLearned (arXiv:2510.24066) has a power-law fit over 5 orders
of magnitude of pretraining data. Pang et al. (2025) is cited by OmniLearned
for jet FM scaling laws. ALETHIA has zero scaling curves. The closest is
[Fig 4](file:///home/vince/ALETHIA/paper/alethia.tex#L194), which is a
training-step curve at fixed model size and dataset.

This is the cheapest missing experiment. d_psi ∈ {4, 8, 16, 32, 64, 128, 256}
costs minutes per fit on CPU. Done in section 4.1.

---

## 4. Proposed upgrades (3-7 day windows)

Each item: design, expected outcome, falsification, confidence of landing on
time and actually answering the criticism.

### 4.1 Width / depth scaling curve for psi_theta (1 day, **high confidence**)

Vary d_psi ∈ {2, 4, 8, 16, 32, 64, 128, 256} and hidden ∈ {16, 32, 64, 128}
with fixed train budget. Fit median R^2 and probe-R^2 vs parameter count and
context size K. Report two curves: a regression-side scaling law for in-
distribution R^2 (expected to saturate near 1 quickly, this is a 1D regression)
and a disclosure-side scaling law for the linear-probe R^2.

Effort: <1 day to run, half a day to plot.

Confidence of landing: **high.** All infrastructure is in place. The
in-distribution side will saturate immediately — the interesting curve is
the probe-R^2 vs d_psi. That is the empirical foundation-model signature the
paper is currently asserting without measuring.

Falsification: if probe-R^2 at d_psi = 4 already saturates, the "learned 16-d
basis discovers structure" claim is reduced to "more capacity is not used".

### 4.2 JEPA-style pretext for psi_theta (3-4 days, **medium confidence**)

Two-stage training:

- Stage 1 (pretext): mask 50% of (M_ctx, Y_ctx) pairs in each scenario, fit
  psi_theta + an auxiliary predictor g_phi (small MLP) to predict the
  embeddings psi_theta(M_masked) from psi_theta(M_unmasked) + Y_unmasked. EMA
  target-encoder per HEP-JEPA / I-JEPA. No closed-form ridge in stage 1.
- Stage 2 (fine-tune): freeze (or low-LR) psi_theta, train the closed-form
  ridge head's hyperparameters (alpha, output projection).

Compare to vanilla supervised-ridge psi_theta on three metrics:
- held-out R^2 at small K (4, 6, 8) — does JEPA pretext buy data efficiency?
- probe-R^2 — does the latent-prediction objective expose c_lq^(3) better
  or worse than the supervised one?
- transfer to a held-out operator direction (e.g. train on c_lq^(3), probe
  c_lq^(1)).

Effort: 3-4 days. The JEPA loss, EMA, and target-encoder boilerplate are
~150 LOC. The data pipeline is unchanged.

Confidence of landing: **medium.** Stage-1 JEPA on a tiny 1D problem may
collapse (the standard JEPA pathology); the closed-form ridge in stage 2 may
just relearn the supervised features. But even a *negative* result settles the
question "does masked-latent pretraining help on this prior", which is the
sharper version of the HEP-JEPA omission.

Falsification: if vanilla supervised wins on all three metrics at matched
psi_theta parameter count, drop the "complements HEP-JEPA" framing and report
the experiment.

### 4.3 Vanilla PFN baseline (2 days, **high confidence**)

Implement a 2-layer transformer with cross-attention from query to (context M,
context Y) tokens, trained on the same scenario prior with the same K=12 / Q=32
/ N_train=200 budget. Match the parameter count to 5,328 by tuning d_model and
n_heads. This is the load-bearing missing baseline.

Expected outcome (priors from von Oswald 2023, Akyurek 2022): trained
transformer will match the closed-form ridge head on median R^2 and may beat
it on small-K data efficiency. If it does, the "closed form is special"
framing weakens and the contribution becomes "an interpretable special case
that exposes the implicit ridge weights for probing". That is *also* a
legitimate contribution but a less ambitious one. Either way the experiment
disambiguates.

Confidence of landing: **high.** Standard cross-attention is a one-evening
implementation. The training loop is already there. The only risk is hyper-
parameter tuning for the transformer — fix that with a tiny LR sweep on the
in-distribution validation set.

Falsification of the paper's framing: if vanilla PFN matches Intention on R^2
and probe-R^2 to within bootstrap, the closed-form simplification is a nice
property but not an empirical win.

### 4.4 Set Transformer + Attentive Neural Process baselines (2 days, **medium-high confidence**)

A pair of attention-equipped baselines that the DeepSets mean-pool is the weak
form of:

- Set Transformer (Lee et al. 2019): replace mean-pool in
  [deepsets_matched.py](file:///home/vince/ALETHIA/experiments/intention-vs-deepsets/deepsets_matched.py)
  with ISAB-style self-attention over the context.
- Attentive Neural Process (Kim et al. 2019): cross-attention from query to
  context with the same encoder.

Both fit in the same training loop. Parameter-match to ~5,300.

Confidence of landing: **medium-high.** ANP code is on GitHub; Set Transformer
is a standard layer. The implementations are 3-day work each but parallelisable.

Falsification: if either matches Intention, the Drell-Yan problem is easy
enough that attention-over-context alone (closed-form or not) is the active
ingredient, and the "learned basis psi_theta absorbs structure no kernel
encodes" framing ([Sec 4.2](file:///home/vince/ALETHIA/paper/alethia.tex#L208))
needs softening.

### 4.5 Probe control tasks (1 day, **high confidence**)

Per Hewitt & Liang 2019, run two controls:

- **Selectivity control.** Replace c_lq^(3) labels with a random permutation
  on the same magnitude band. Train the same linear probe. If probe-R^2 on the
  permuted labels exceeds ~0.3, the probe is fitting features that have no
  semantic alignment to c_lq^(3).
- **Random-feature control.** Replace psi_theta with a randomly-initialised
  MLP of the same shape and run the probe on its w_implicit. The gap between
  probe-R^2(trained psi_theta) and probe-R^2(random psi_theta) is the actual
  identifiability signal.

Confidence of landing: **high.** Both are 50 LOC additions to the existing
identifiability script.

Falsification of the paper's "strong-form disclosure" claim: if either control
shows comparable R^2, the disclosure claim collapses to "implicit weight
vectors are correlated with the labels they are derived from", which is
tautological.

### 4.6 The concurrent-SMEFT-FM comparison (1 day, **high confidence**)

Cite arXiv:2512.15862, position ALETHIA against it, and pick one of its
downstream tasks (nearest-neighbour retrieval of Wilson-coefficient
configurations) to run the linear probe on. The constraint that the FM
forward pass not see c rules ALETHIA in to that comparison naturally.

Effort: half a day for citation + comparison paragraph, half a day to run the
retrieval experiment if their public weights are available. If not, drop the
experiment but do the citation.

Confidence of landing: **high** for citation; **medium** for the experiment
depending on code availability.

---

## 5. The eigen-direction / online-PCA proposal

### Statement

When the drift aggregator fires, rotate psi_theta to align its top output
directions with the Fisher information of the current loaded context. The
closed-form ridge head then operates in a basis where the top-k directions
absorb the c-resolvable variation.

### Why this is non-trivial

The closed-form ridge solves w = A^{-1} Psi^T Y. The design matrix
A = Psi^T Psi + alpha I has eigenvectors v_j with eigenvalues lambda_j ≥
alpha. The predictive leverage along v_j is 1/(alpha + lambda_j). High-lambda
directions are well-determined; low-lambda directions are dominated by the
ridge prior. Under distribution shift the *informative* directions in Y space
(in the Fisher sense) may not be the high-lambda directions of psi_theta,
because psi_theta was meta-trained on a different scenario prior.

Two pieces of evidence that this matters:

- The design-matrix condition number kappa(A) is the chain's drift signal
  [Sec 8.2](file:///home/vince/ALETHIA/paper/alethia.tex#L249), and it
  climbs from 1e3 to 3e5 across the run
  ([Fig 6](file:///home/vince/ALETHIA/paper/alethia.tex#L283)). The paper
  reads the climb as "new directions added to the row space", but it is also
  the signature of growing misalignment between psi_theta's eigenstructure
  and the informative directions of the current Y.
- The Spectral Identifiability Principle (arXiv:2511.16288, 2025) makes
  exactly this connection for linear probes: probe stability depends on the
  eigengap between task-relevant directions and the Fisher estimation
  noise. Same maths.

### Connection to existing theory

- Stuart et al. 2025, *Rotated MFVI*: rotating to the Hessian / Fisher
  eigenbasis is the optimal pre-conditioner for variational inference; the
  ridge head here is the conjugate-Gaussian special case.
- Online PCA / streaming PCA: Oja's rule and its modern variants give an
  O(d_psi^2) per-step update that fits naturally with the Sherman-Morrison
  cadence of the closed-form context update
  ([App A.1](file:///home/vince/ALETHIA/paper/alethia.tex#L362)).

### Concrete experiment

- Pretrain psi_theta on the standard SMEFT prior.
- Build the empirical Fisher F = (1/N) sum_i (d log p / d psi)(d log p / d
  psi)^T on the *current loaded context*, for the conjugate-Gaussian
  predictive density induced by the ridge head.
- Eigendecompose F = U D U^T.
- Apply the orthogonal map psi_theta -> U^T psi_theta. (This is just a basis
  rotation on the output of psi_theta; the ridge solve is invariant up to
  the rotation of w, so this is mathematically free until coupled to the
  conformal calibrator and the EPIG leverage.)
- Compare to no-rotation: (a) RMSE on the target band; (b) probe-R^2; (c)
  data-efficiency at small K.

Two variants worth running in parallel:

- **Hard rotation** at each drift firing (analogous to Sherman-Morrison
  but on the basis rather than the design matrix).
- **Soft Oja update** on every Sherman-Morrison call, with learning rate
  decaying over the run.

### Confidence

**Medium.** The mathematics is sound and the intervention is cheap (one
eigendecomposition per drift firing, ~10 us in d_psi = 16). The risk is that
on this 1D Drell-Yan problem the closed-form ridge already extracts the
available information and there is no headroom — i.e. the EPIG leverage is
flat across the kinematic range, exactly as the paper reports in
[Sec 8.4](file:///home/vince/ALETHIA/paper/alethia.tex#L325). The honest
falsifier is: "in a problem where the EPIG comparison-table is flat, rotation
does not help either." A two-dimensional kinematic problem (m_ll x p_T_ll) or
a multi-operator scenario where the c-resolvable directions are different in
different bands is the regime where the experiment becomes discriminating.

### Effort

- 1 day to implement the rotation step inside `IntentionFM`.
- 1 day to wire the empirical-Fisher computation into the drift aggregator.
- 1-2 days to run on the existing drift event + one new multi-operator drift
  event.
- 1 day to write up.

Total: 4-5 days. This is the most genuinely novel of the proposals on this
list and the one most plausibly elevating the contribution from "PFN on
Drell-Yan" to "PFN with online-Fisher-aligned basis".

---

## 6. Other angles worth flagging

### 6.1 Closed-form attention as a Gaussian-process / NTK limit

The paper already notes the Gaussian-process duality
([line 127](file:///home/vince/ALETHIA/paper/alethia.tex#L127)) but does not
push it. The infinite-width limit of psi_theta (Hron et al. 2020 for attention
NTK; Lee et al. 2019 for MLP NTK) is a fixed kernel, and the meta-training is
*learning the NTK by adjusting psi_theta's finite-width statistics*. This
reframes the closed-form ridge head as a learned NTK + closed-form GP
regression, and the contribution becomes "the right way to learn the kernel
for amortised SMEFT inference is end-to-end through the ridge solve, not via
GP hyperparameter optimisation per scenario". This framing is more
defensible than "foundation model".

Effort to land: a single derivation paragraph + a finite-width vs infinite-
width comparison on the polynomial-toy oracle (~2 days).

### 6.2 Infinite-context behaviour

Sherman-Morrison gives O(D^2) per-context-update cost, so context size K can
grow arbitrarily large. The K -> infinity limit of the closed-form ridge with
a learned basis is well-defined: the design matrix A converges to the
expected Gram matrix of psi_theta under the kinematic sampling distribution,
and the predictive variance shrinks at rate 1/K along well-conditioned
directions. Reporting an R^2-vs-K curve up to K = 10^4 or 10^5 would be a
genuine differentiator from transformer-based PFNs, which all hit context-
length limits. Effort: <1 day.

### 6.3 Sample-efficiency vs vanilla PFN

A two-axis plot of held-out R^2 vs (N_train scenarios) x (K context size) for
both Intention and a vanilla PFN at matched parameter count would be the
single most informative figure for the foundation-model framing. Even if
Intention only wins in the small-N small-K corner, that is the right framing
of the contribution.

---

## 7. Bottom line

The paper as it stands has:

- A correct and competent PFN implementation specialised to closed-form ridge
  in a learned basis.
- A meta-training pipeline that converges quickly on the analytic SMEFT
  oracle.
- A linear-probe identifiability finding on c_lq^(3) that is real and worth
  reporting.
- A closed-loop chain with drift detection / EPIG / conformal that works.

It is missing:

- A vanilla PFN / attention-equipped baseline (load-bearing).
- A citation and comparison to arXiv:2512.15862, the concurrent SMEFT FM.
- Any scaling analysis at all, despite invoking the rhetoric.
- Probe control tasks (Hewitt & Liang 2019).
- A discussion of where the framing differs from PFN beyond "Equation (3) is
  closed-form".

The fastest path to a sharper paper is (4.1) + (4.3) + (4.5) + (4.6) — four
to five days of work. The most novel research direction is (5), online
Fisher-aligned basis rotation, which on the 1D Drell-Yan problem may show no
effect but is the natural extension to multi-operator / multi-observable
drift events and is the upgrade most likely to push the contribution beyond
"PFN with extra steps".

---

## Sources

- [Mueller et al. 2022, Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510)
- [Nagler 2023, Statistical Foundations of Prior-Data Fitted Networks](https://proceedings.mlr.press/v202/nagler23a/nagler23a.pdf)
- [Helli et al. 2025, NPE-PFN / TabPFN for SBI (arXiv:2504.17660)](https://arxiv.org/abs/2504.17660)
- [Garnelo & Czarnecki 2023, Intention (arXiv:2305.10203)](https://arxiv.org/abs/2305.10203)
- [Akyurek et al. 2022, What learning algorithm is in-context learning? (arXiv:2211.15661)](https://arxiv.org/abs/2211.15661)
- [Mahankali et al. 2023, One step of GD is provably optimal ICL (arXiv:2307.03576)](https://arxiv.org/abs/2307.03576)
- [Katel 2025, HEP-JEPA (arXiv:2502.03933)](https://arxiv.org/abs/2502.03933)
- [Mikuni & Nachman 2025, OmniLearned (arXiv:2510.24066)](https://arxiv.org/abs/2510.24066)
- [Birk et al. 2024, OmniJet-alpha (arXiv:2403.05618)](https://arxiv.org/abs/2403.05618)
- [Heinrich et al. 2024, Masked Particle Modeling (arXiv:2401.13537)](https://arxiv.org/abs/2401.13537)
- [Anonymous 2025, Reusable theory representations: SMEFT foundation model (arXiv:2512.15862)](https://arxiv.org/abs/2512.15862)
- [Spectral Identifiability Principle 2025 (arXiv:2511.16288)](https://arxiv.org/abs/2511.16288)
- [Stuart et al. 2025, Rotated MFVI](https://ww3.math.ucla.edu/wp-content/uploads/2025/10/MFVI_PCA.pdf)
- [Kim et al. 2019, Attentive Neural Processes (arXiv:1901.05761)](https://arxiv.org/abs/1901.05761)
- [Position paper 2025, The Future of Bayesian Prediction is Prior-Fitted (arXiv:2505.23947)](https://arxiv.org/abs/2505.23947)

