# Foundation Model — research summary (agent b)

Companion to `/home/vince/ALETHIA/docs/research/BRIEF.md`. File:line citations refer to `/home/vince/ALETHIA/...` unless noted.

## 1. What a foundation model is (outside HEP)

The phrase "foundation model" is due to Bommasani et al. 2021 (arXiv:2108.07258). The empirical pattern the term names is sharp: a model is pre-trained on a broad data source by a self-supervised (or weakly supervised) objective; the resulting representation transfers, by a relatively cheap downstream adaptation, to many tasks; and the same forward pass is used at both stages. Three properties are operationally distinguishing:

1. **The pre-training objective does not require task labels for the downstream tasks.** Whatever supervision the FM consumes is constructed from the data itself (BERT's masked tokens, GPT's next token, MAE's masked patches, SimCLR's augmentation pairs) or from cheap paired signals (CLIP's image-caption pairs).
2. **The downstream tasks are reached through a probe on the FM's representation**, not by changing the forward pass. Linear probes, kNN probes, prompting, light fine-tuning of a head — these are downstream; they are not the FM.
3. **The representation is the asset.** The reason to call it a "foundation" model is precisely that swapping the probe gives you the next task without re-pretraining. If you cannot change the downstream task without retraining the network, what you have is a regressor.

A tour of methods, with what they see at training:

- **MLM / BERT** (arXiv:1810.04805). Masks tokens, asks the model to fill them. Labels are tokens drawn from the same input.
- **Causal LM / GPT** (arXiv:2005.14165). Predict the next token from previous ones. Self-supervised.
- **MAE** (arXiv:2111.06377). Mask ~75% of image patches; the encoder sees the visible patches; a small decoder reconstructs the masked ones in pixel space.
- **SimCLR** (arXiv:2002.05709), **MoCo** (arXiv:1911.05722). Contrastive: two augmentations of the same image are positives, all other images in the batch are negatives. InfoNCE loss.
- **DINO** (arXiv:2104.14294), **DINOv2** (arXiv:2304.07193). Self-distillation; a student matches a slow EMA teacher on cropped views of the same image. No labels at all.
- **BYOL** (arXiv:2006.07733), **VICReg** (arXiv:2105.04906). Non-contrastive SSL. BYOL relies on an EMA target network and a predictor MLP; VICReg uses variance/invariance/covariance regularisation.
- **JEPA / I-JEPA** (arXiv:2301.08243), **V-JEPA** (arXiv:2404.08471). Predict the *representation* of a masked region from the representation of a visible region. Reconstruction in latent space, not pixel space.
- **CLIP** (arXiv:2103.00020), **ALIGN** (arXiv:2102.05918). Multi-modal contrastive: image and caption embeddings pulled together. The text is a paired signal applied to the loss, not an input to the image encoder.
- **Speech: wav2vec 2.0** (arXiv:2006.11477), **HuBERT** (arXiv:2106.07447). Contrastive over masked latents (wav2vec) or predict clustered pseudo-labels (HuBERT). Self-supervised.
- **Protein: ESM** (arXiv:1907.06576 and successors). MLM on amino-acid sequences; downstream tasks (contact prediction, structure via ESMFold) are probes.
- **Time series and scientific data:** TimesFM (arXiv:2310.10688) trains a decoder transformer with causal forecasting; FourCastNet (arXiv:2202.11214) is on the boundary between FM and forecaster.
- **Particle physics adjacent:** OmniLearn (arXiv:2404.16091) trains a Particle Transformer encoder on jets with a joint diffusion / classification objective; Particle Transformer itself (arXiv:2202.03772) is the canonical event-level architecture. The 2025 SMEFT FM demonstrator (arXiv:2512.15862) trains a distribution→latent encoder via supervised contrastive learning and probes the latent for classification, anomaly detection, retrieval.

The common thread: a representation learned by an objective constructible from the data alone, plus a downstream probe. Where a label is present — contrastive pairing in CLIP, c-tagged samples in arXiv:2512.15862 — it is a *pairing* signal applied to the loss, not an input on the forward pass. SimCLR's positive pairs do not feed augmentation identity into the network; CLIP does not feed the caption into the image encoder; the SMEFT demonstrator does not feed `c` into the encoder.

The label-flow rule, stated precisely:

> A foundation model uses labels (when it uses labels at all) to define *which inputs go together* under the loss. It does not consume labels on the forward pass that produces the representation.

This is exactly the rule the brief enforces for `c`.

A second property worth naming: foundation models pay for capacity with neural scaling laws (Hestness et al. arXiv:1712.00409; Kaplan et al. arXiv:2001.08361; Hoffmann et al. arXiv:2203.15556). Test loss decays as a power of either parameters or data, typically with separate exponents and a knee at a compute-optimal frontier. Ridge regressors over a fixed feature map do not show this behaviour — their asymptote is set by the irreducible noise plus the basis misspecification floor. This is an empirical discriminator between an FM and a regressor.

## 2. What a foundation model is not

Restating the negative space:

- A regressor `σ(c, m) → ℝ` that consumes `c` directly on the forward pass. Polynomial-feature ridge, deep MLP, kernel methods with `c` in the kernel — all the same category.
- A "fast surrogate" or "emulator" that mimics the simulator pointwise. Useful, but not an FM. The cleanest signal: you cannot reuse the model for a different downstream task by just changing the probe.
- A morphing fit that fits `σ(c) ≈ σ_SM + Σ A_i c_i + Σ B_ij c_i c_j`. Mathematically exact at the matrix-element level for dim-6 SMEFT at LO; this is precisely why it is the wrong learning target for an FM. Fitting an exact parametric form yields the parameters of that form, not a representation of the manifold.
- A per-bin polynomial in `c`. Same as above, with extra book-keeping.

Judge-applicable test: can you change the downstream task without retraining? If you have to refit because the target is now `acceptance(c)` instead of `σ(c)`, you do not have an FM.

## 3. Why `modules/surrogate/` is a regressor, not an FM

**`modules/surrogate/features.py` lines 30–40** set up the configuration: `N_WC=4`, `D_C = 1 + N_WC + len(PAIRS) = 15`, `D_X = K_X = 5`, `D_JOINT = 75`. **Lines 43–50** build `phi_c(c) = {1, c_a, c_a c_b}` — the morphing basis written into the code. **Lines 60–65** define `phi_joint(c, m) = phi_c(c) ⊗ phi_x(m)`. The morphing structure is not in the loss as an inductive bias — it is the network's input layer.

**`modules/surrogate/model.py` lines 39–51** implement closed-form ridge:
```
Phi   = phi_joint(C, M)
A     = Phi.T @ Phi + self.lam * np.eye(d)
self.A_inv = np.linalg.inv(A)
self.w     = self.A_inv @ Phi.T @ Y
```
Two violations: (1) `phi_joint(C, M)` puts `c` on the forward pass; (2) `Y = σ(c, m)` is the regression target — the morphing ansatz encoded into the loss. `predict` (lines 54–65), `leverage` (lines 68–70), `update` (lines 73–78) all propagate the violation.

**`modules/surrogate/acquisition.py` lines 121–123** apply EPIG over `phi_joint(C_pool, M_pool)`. The closed-form EPIG derivation in the docstring (lines 25–67) is rigorous (Cauchy-Schwarz on the `A_inv` inner product) and reusable — but the feature map it operates on inherits the violation.

**`modules/surrogate/calibration.py` lines 50–55** use `model.predict` and `model.leverage`, both of which call `phi_joint(c, m)`. The conformal layer is feature-map-agnostic and salvageable; only the stratification variable inherits the violation.

**`tests/test_features.py` lines 22–30** encode the morphing assumption as test expectations (`test_phi_c_quadratic_indices`). **`tests/test_model.py` lines 54–63** (`test_in_distribution_accuracy`) check the regressor against the toy oracle's truth — passes because the *regressor* is correct, not because the design is correct as an FM.

**The tarball `incontext.py`** (`/tmp/smeft_view/smeft_surrogate/smeft_surrogate/incontext.py`) is a half-step toward the correct design. Its docstring lines 6–11 state the intent in the brief's language: "this module does closed-form linear attention with the Wilson coefficients LATENT: c never enters the model". The `predict` method (lines 86–125) takes `M_ctx, Y_ctx, M_query` only — no `C`. The math is identical to `IntentionFM` modulo the absence of `c`. *Plus:* `c` is gone from the forward pass; the task is specified by `(m_i, y_i)` context (in-context learning). *Minus:* `psi(m)` is still hand-engineered (polynomial in `log(m / M_REF)`), no representation is learned, no pre-training. Lines 56–63 of its docstring acknowledge this: "It does not learn psi from data. 'Pretraining' in the proper FM sense would mean learning psi across many SMEFT scenarios. That is a future-work knob." So `incontext.py` demonstrates that the closed-form machinery works without `c` on the forward pass; the missing piece is exactly the FM — learning `psi`.

## 4. Redesign proposal — two candidate architectures

**Architecture A — set/event-level transformer over events.**
*Forward pass.* A Particle-Transformer / DeepSets / Set Transformer style encoder ingests an unordered batch of events `{x_1, …, x_N}` where each `x_i ∈ ℝ^F` is a per-event feature vector (Drell-Yan: `m_ll, p_T^ll, y_ll, cos θ*`). Encoder outputs a single distribution-level embedding `z = Enc({x_i}) ∈ ℝ^d`. Use fixed `N = 4096–16384` for first pass; larger `N` reduces per-batch statistical noise. `d = 64–256`.

*Pre-training objective.* Two compatible options.
- **InfoNCE over c-conditioned batches** (arXiv:1807.03748; arXiv:2002.05709). Two independent batches drawn from the same `c` are positives; batches from different `c` are negatives. `L = -log [exp(sim(z, z+)/τ) / Σ_j exp(sim(z, z_j)/τ)]`. `c` enters only as the pairing signal.
- **VICReg / BYOL** as non-contrastive alternative. Same pairing rule; loss is invariance + variance + covariance (VICReg) or predictor + EMA teacher (BYOL). The SMEFT-FM demonstrator (arXiv:2512.15862) uses a supervised-contrastive variant in this spirit.

*Data-generation procedure.* Sample `c` from a prior over operator space. For each `c`, generate two independent event batches of size `N` from the analytic oracle in `modules/analytic_smeft/smeft.py`. Many batches per `c` is better. The oracle is the data engine.

*Probe.* Freeze encoder. Fit linear (or small MLP) head from `z` to held-out `c` attributes.

**Architecture B — histogram autoencoder.**
*Forward pass.* `Enc: ℝ^B → ℝ^d` mapping a fixed-shape histogram (`B = 50` bins of `m_ll`, normalised to SM) to latent `z`, with `d ≪ B`. Decoder reconstructs `h ≈ Dec(z)`.

*Pre-training objective.* Denoising autoencoder (Vincent et al. 2010): add Poisson/Gaussian event-count noise to `h` before encoding, ask decoder to recover clean `h`. `c` is generator of the data, not input. Optionally add a small InfoNCE term on `z` paired by `c` across replicas.

*Probe.* Same as A.

**Which to pick.**
- *Information content.* Binning throws away event-level correlations. For multi-observable goals A scales; B requires re-binning.
- *Refinement.* The finer the oracle, the more distinct two `c`-conditioned distributions look at event level; A picks this up for free.
- *Closed-form downstream.* Both emit fixed-shape `z`; both compatible. Tie.
- *Cost.* A is more expensive per step; B is cheap. Hackathon-marginal for A; mitigation via DeepSets first (cheaper than Particle Transformer).
- *Hackathon claim.* "The learned representation is the model" reads best on A: the FM ingests the same measurement object an experimentalist sees.

**Recommendation: A as primary, B as ablation.** Pragmatic refinement: implement A with a thin DeepSets encoder first (arXiv:1703.06114) — sum-pool a per-event MLP, then a top MLP — before paying for a Particle Transformer.

**What stays.** `modules/analytic_smeft/smeft.py` (oracle), `modules/surrogate/oracle_smeft.py` (wrapper, with an added `sample_events(c, n_events)` method), and the closed-form machinery in `model.py / calibration.py / acquisition.py` (retained as the *probe*, with feature map replaced).

**What goes.** `phi_c` and `phi_joint`. `IntentionFM.fit(C, M, Y)`'s `C` argument. `tests/test_features.py` is rewritten against `phi_probe(z)`.

## 5. Iterative refinement under the constraint

The user's framing translates to a stagewise training schedule:

- **Stage 0.** Oracle = `k_0 = 4` operators (current `WC_NAMES`), LO, `pdf="analytic"`. FM pre-trained on event batches at this fidelity. Encoder input interface: `(N, F)` event tensors with `F ≈ 4` Drell-Yan kinematics.
- **Stage 1.** Same FM, same interface. Oracle = `k_1 = 8` operators activated; larger `N`. FM continued from stage-0 weights.
- **Stage 2.** `k_2 = 14` operators (all `smeft.py`-supported dim-6 Warsaw operators), still LO.
- **Stage 3 (conceptual scope).** NLO or MadGraph events, same interface.

What changes between stages: the support of the data distribution. What does not change: the FM's input shape. The embedding manifold should *refine* — directions corresponding to newly-activated operators become resolvable. Measurable: per-operator probe `R²` rises (or holds) when an operator is added; previously-resolvable operators are preserved (the catastrophic-forgetting check).

The literature here is continual self-supervised learning (Madaan et al. arXiv:2110.06976). Standard hazards: representation collapse (with non-contrastive losses) and catastrophic forgetting. Both are checkable.

## 6. Probing

The science claim is the probe. A **linear probe** (Alain & Bengio arXiv:1610.01644) is the cleanest representation-quality measurement.

Freeze the encoder. Generate held-out `(z_i, c_i)` pairs. Fit ridge `c_i^a ≈ w_a^T z_i + b_a` per operator on a training portion; report `R²` on held-out. Repeat with a small MLP probe as a sanity check.

The probe rules out trivialities and confirms non-trivialities:
- *Constraint intact.* FM never emits `c`. The probe recovers `c` from `z`.
- *Disclosure.* If a linear probe recovers `c` with high `R²` on a *held-out* region of `c`-space (not seen during pre-training), the FM has learned a generalising representation.
- *Failure modes.* `R² ≈ 0` linear / high MLP = info on a nonlinear manifold (usable but less interpretable); `R² ≈ 0` both = collapse or insufficient loss separation.

Exact protocol:
1. Split `c`-space. Hold out `|c_a| ∈ (0.7, 1.0]` per operator; pre-train on `|c_a| ∈ [0, 0.7]`.
2. Pre-train FM on stage-0 oracle data without held-out `c`.
3. Generate probe data: `c_i` uniform on full space (incl. held-out); fresh event batch; `z_i`.
4. Fit linear probe on training portion; evaluate `R²` on held-out portion.
5. Headline thresholds: `R² > 0.5` per operator = qualitative claim; `R² > 0.8` = "FM has disclosed SMEFT structure".

## 7. Scaling

Foundation models obey neural scaling laws (arXiv:2001.08361, arXiv:2203.15556, arXiv:1712.00409). Ridge regressors over fixed features obey classical `1/√n` + basis-misspecification floor.

**Architecture A scaling.**
- *Parameters.* DeepSets `[F→64→128]` per-event MLP + `[128→64→d]` top MLP ≈ `10^4–10^5` parameters. Particle Transformer ≈ `10^6–10^7`. Hackathon-tractable.
- *Data volume.* Each pre-training example is an event batch of `N ~ 10^4` events. Cost dominated by oracle: analytic LO ≈ milliseconds per batch; MadGraph NLO seconds–minutes.
- *Chinchilla-style.* Compute-optimal ratio ≈ `N_params ~ N_tokens`. 10^5-param encoder over ~10^5 batches is near compute-optimal at this scale. Particle Transformer at 10^7 params needs ~10^7 batches — feasible only with the analytic oracle. This justifies the LO analytic oracle as the pre-training corpus; NLO is held in reserve for evaluation.
- *Irreducible floor.* Set by encoder capacity for the true c-conditional distributional manifold plus the simulator's intrinsic event-level stochasticity.

**Probe scaling.** Test error scales `~ 1/√n` in `n = number of (z, c)` examples, plus the floor set by how well the embedding disentangles `c`. Should *not* show FM-style power-law behaviour.

Side-by-side predictions:

| Quantity | FM-shaped curve | Regressor curve |
|---|---|---|
| Test loss vs n | Power law, non-trivial exponent + floor | `1/√n` + possibly large floor |
| Test loss vs parameters | Power law | Flat past basis size |
| Probe R² vs pre-training compute | Rising | N/A |

The oracle (analytic SMEFT calc) is cheap; the encoder's compute is the bottleneck. This is the data-abundant regime — the easy regime to scale.

## 8. Interaction with the existing closed-form machinery

The closed-form Bayesian ridge + Sherman-Morrison + leverage + EPIG in `modules/surrogate/` is mathematically clean and aligned with Garnelo-Czarnecki arXiv:2305.10203. The EPIG derivation (`acquisition.py` lines 25–67) is rigorous (Cauchy-Schwarz on `A_inv` inner product) — a faithful Gaussian-linear specialisation of arXiv:2304.08151.

The closed-form trick is feature-map-agnostic. It needs only: a feature map `phi: X → ℝ^d`, training labels `Y`, ridge penalty `λ`. The brief forbids `phi_joint(c, m)`; it does not forbid ridge as a probe.

**The fix:** replace `phi_joint(c, m)` with `phi(z)`, where `z` is the FM embedding of a measured distribution.

| Layer | Sees c? | What it does |
|---|---|---|
| Oracle (`oracle_smeft.py`) | Yes (input) | Generates events from `c` |
| FM encoder (new) | **No** | Maps events → `z` |
| Probe (refactored `model.py`) | Yes (label) | Closed-form ridge `z → c_attribute` |
| Calibrator (`calibration.py`) | No directly | Stratifies probe residuals on leverage |
| EPIG (`acquisition.py`) | Yes (label, target set) | Picks next oracle call |

EPIG becomes acquisition over the next measurement to query: pick the `(c_pool, observable_pool)` whose embedding `z` would most reduce predictive variance about a held-out target distribution. `c` is a label on the oracle call; the probe's feature map is `c`-free. Leverage in `phi(z)` space becomes the Phoenix drift signal directly — drift of the embedding relative to the calibration set.

The Garnelo-Czarnecki Intention picture is recovered constraint-respecting: keys / values / queries are `(z_ctx, y_ctx_probe, z_query)`, not `(phi_joint(c_ctx, m_ctx), σ(c_ctx, m_ctx), phi_joint(c*, m*))`. `incontext.py` in the tarball is, modulo a learned `psi`, a working version of this. The hackathon contribution is to *learn the basis* — that is the FM.

## Hackathon claim (one sentence)

The learned representation is the model; probing it for `c` is the science; the closed-form ridge that used to be the regressor is now the probe.

## References (arXiv ids)

arXiv:2108.07258 (Bommasani et al., Foundation Models); 1810.04805 (BERT); 2005.14165 (GPT-3); 2111.06377 (MAE); 2002.05709 (SimCLR); 1911.05722 (MoCo); 2104.14294 (DINO); 2304.07193 (DINOv2); 2006.07733 (BYOL); 2105.04906 (VICReg); 2301.08243 (I-JEPA); 2404.08471 (V-JEPA); 2103.00020 (CLIP); 2102.05918 (ALIGN); 2006.11477 (wav2vec 2.0); 2106.07447 (HuBERT); 1907.06576 (ESM); 2310.10688 (TimesFM); 2202.11214 (FourCastNet); 2202.03772 (Particle Transformer); 2404.16091 (OmniLearn); 2512.15862 (SMEFT FM demonstrator); 1703.06114 (DeepSets); 1807.03748 (InfoNCE / CPC); 2305.10203 (Intention); 2304.08151 (EPIG); 1610.01644 (Linear probes); 2001.08361 (Kaplan scaling); 2203.15556 (Chinchilla); 1712.00409 (Hestness scaling); 2110.06976 (Continual SSL).
