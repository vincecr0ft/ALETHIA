# ALETHIA — Informer-network reframe, span-completeness active learning, and working-point curvature sampling

**Audience:** the coding agent.
**Author of spec:** derivation and task design handed down for implementation.
**Register:** every task states a falsifiable done-when gate. A failed gate is a result to diagnose (wrong hypothesis or wrong implementation, decide which), never a thing to report around. Do not soften a failed numerical check into prose. If a derivation does not reproduce, the derivation in this document is the thing under test, flag it.

A naming caution before anything else. "Informer" here means *a network whose objective is to learn an informative, interpretable representation of the SMEFT model manifold, with the Intention closed-form ridge as its embedding/prediction primitive.* It is **not** the Informer time-series transformer of Zhou et al. (AAAI 2021, arXiv:2012.07436). Do not import that. If you need a non-colliding name in code, use `ManifoldInformer`.

---

## 0. Where this sits, and what changed

We have, already built and tested (do not rebuild, wire into):

- **Oracle**, two tiers. Analytic LO SMEFT Drell-Yan cross-section in `mℓℓ` (and the AFB angular extension), and a MadGraph5 pipeline that cross-validates it at the per-cent level. The analytic decomposition is `σ(c, m) = σ_SM(m) + Σ_i A_i(m) c_i + Σ_{i≤j} B_{ij}(m) c_i c_j` (SMEFiT/morphing form; Greljo-Marzocca EPJC 77:548).
- **Foundation model, three flavours.** `IntentionFM` (explicit `c` in the feature map), `IntentionFMInContext` (latent `c`, context-specified), and the closed-form learned-`ψθ` ridge head used in the current paper. The forward pass is `Ŷ = Q(KᵀK+αI)⁻¹KᵀV` (Garnelo-Czarnecki arXiv:2305.10203). All three currently consume a binned ratio; the binned input is what this document supersedes (see the input mandate below). The closed-form solve is retained as the prediction primitive; the input layer changes.
- **Phoenix tracing/monitoring**, with the three drift evaluators (accuracy DAS-CUSUM, conformal per-region coverage à la Araz-Spannowsky arXiv:2512.17048, condition-number/`κ(A)` context-shape).
- **Fisher information + eigendirections**, computed and tested. The mass-only spectrum is `{19.6, 8.7, 3.0e-4, 1.3e-4}`: two resolved four-fermion directions, two vertex directions at the prior floor.

What changed, and why this document exists. The earlier analysis hit a wall and I mislabelled it. The impossibility result is real: *a response component outside `span(ψ)` cannot be recovered by rotation inside `span(ψ)`* (this is the SMEFT instance of Locatello et al. ICML 2019, arXiv:1811.12359, no disentanglement without inductive bias). I wrongly read that as "active learning cannot help here" and filed feature-map extension under manual intervention. Both are wrong. The wall is the **trigger**, not the terminus. A loop that (i) detects when the current `ψ` has been spanned out, then (ii) extends `ψ`, is exactly the thing the impossibility result fails to forbid, because extension changes the inductive bias rather than rotating within it.

Under the JEPA reading the encoder `ψ` is *the object being learned*, so extension is the loop's native mode, not a workaround. This flips the emphasis: **JEPA philosophy is primary; the Intention closed form is the embedding/prediction primitive inside it.** We deliberately stop leaning on what Intention already does superbly (closed-form ridge regression of a polynomial-in-`c` target it was handed the structure of) and push on what JEPA is *supposed* to deliver: a latent that is a faithful, interpretable image of the physics manifold, capable of telling us its own incompleteness and being grown.

**Input representation is fixed and non-negotiable: event-level point clouds, no histograms.** The model never sees a binned cross-section ratio and never sees a Wilson coefficient. The context at a latent `c` is a variable-length unordered set of per-event feature vectors `{x_1, ..., x_N}`, `x_n` the reconstructed kinematics of event `n` (for neutral-current Drell-Yan: the lepton-pair four-vectors, equivalently `(mℓℓ, cosθ*, y, pT)` per event, log-scaled as appropriate). Binning is an information bottleneck that we are removing on purpose: unbinned multivariate observables are strictly more sensitive because binning discards information (ML4EFT, Chen-Glioti-Marzocca-Nardini-Wulzer, JHEP 03(2023)033, arXiv:2211.02058; Schöfbeck, refinable modeling for unbinned SMEFT, arXiv:2406.19076). The current paper's binned-ratio head is the thing being superseded, not extended. Wherever the prior spec said "binned `σ/σ_SM`" or "kinematic abscissa `m`," read "set of events" and "per-event feature `x`". The encoder must be permutation-invariant over events (Deep Sets / Energy-Flow / Particle-Flow Networks, Zaheer et al. NeurIPS 2017; Komiske-Metodiev-Thaler arXiv:1810.05165), and since self-attention is a strict generalization of Deep Sets, the Intention linear-attention head reads the event set natively with events as the context rows.

The three project goals are unchanged and this document maps every task to them:
**(a)** demonstrate a physics foundation model; **(b)** demonstrate active learning *enabled by Phoenix*; **(c)** interpretable inference via SBI.

---

## 1. The one idea, stated mathematically

### 1.1 The resolvable subspace is a property of the working point, not the operator

The Fisher gradient at working point `c` is

```
∂_i σ(c, m) = A_i(m) + 2 Σ_j B_{ij}(m) c_j .          (1)
```

At the SM point `c = 0` the gradient is `A_i(m)`. The two vertex directions are flat there because their linear template `A_vertex(m) ∝ σ_SM(m)` is degenerate with the overall normalisation; that degeneracy is the `~1e-4` eigenvalue. Move off `c = 0` and the gradient picks up `2 Σ_j B_{ij}(m) c_j`. The cross-template `B_{vertex,ℓq}(m)` carries the four-fermion energy growth, is **not** proportional to `σ_SM(m)`, and is therefore not degenerate with the normalisation. The flat direction lifts. "Some operators shift, others get more constrained" is precisely Eq. (1): moving along the manifold rotates the resolvable subspace through the quadratic morphing.

The single-point Fisher diagonalisation in the current paper (Section 5) linearises at one point and cannot see this. This is the gap.

### 1.2 The manifold picture: tangent vs curvature

Treat the map `c ↦ σ(c, ·)` as a chart of the SMEFT **model manifold** embedded in observable space (Transtrum-Machta-Sethna PRE 83:036701; Transtrum-Qiu PRL 113:098701, the "hyper-ribbon" with a hierarchy of widths and extrinsic curvatures). Then:

- `A_i(m) = ∂_i σ` is the **tangent space** at `c = 0` (first fundamental form / linear response).
- `B_{ij}(m) = ½ ∂_i ∂_j σ` is the **second fundamental form** (extrinsic curvature).

One working point gives you the tangent only. Sampling multiple working points reconstructs the curvature, and the curvature contains directions that are measure-zero in the tangent space at the base point but have extent elsewhere. This is the exact content of "probe beyond interpolation": interpolation inside one chart sees only the tangent; curvature sampling recovers the intrinsic geometry that says which directions are *physically probeable at all*, with a working-point-dependent answer.

This is not a metaphor borrowed from ML. It is the established physics technique of **morphing**: the cross-section is polynomial in `c`, and a set of base points (working points) determines the polynomial templates `{σ_SM, A_i, B_{ij}}` exactly. "Overdetermined morphing" (arXiv:2512.12473) adds base points beyond the minimal set to extend the region of validity, including CP-even/CP-odd base samples, which is *active selection of working points* in all but name. The reference frame for "what does sampling a new working point buy" is morphing-matrix conditioning, and we should use it as the cross-check oracle in Task 2.

### 1.3 Out-of-span detection and the span-completeness criterion

Work in event-feature space, not on a binned curve. The encoder lifts each event to a per-event embedding `ψ(x) ∈ R^D`; the span is `S = span(ψ)` over the event-feature support. The oracle "response" at fixed `c` is the per-event density (equivalently the event-weight function `w_c(x) = p(x|c)/p(x|SM)`, the per-event likelihood ratio that carries the full unbinned information; Schöfbeck arXiv:2406.19076). Decompose the part of the physics the encoder can represent from the part it cannot:

```
w_c(x) = w_∥(x) + w_⊥(x),    w_∥ = Π_S w_c,    w_⊥ ⊥ S .       (2)
```

Three facts (each provable from Eq. (2); reproduce them in Task 1):

1. **Leverage is blind to `w_⊥`.** `lev(q) = ψ(x_q)ᵀ A⁻¹ ψ(x_q)` is a functional of `S` only. An event whose discriminating structure lives in `w_⊥` reads in-distribution.
2. **Accuracy is bounded below by `‖w_⊥‖`.** No set of events fed through `ψ` reduces it.
3. **Calibration breaks.** The predictive variance carries no `w_⊥` term, so the interval is overconfident.

The fingerprint of an out-of-span direction is therefore **coverage clean ∧ accuracy drift ∧ calibration break, simultaneously, and invariant under acquisition.** This is distinct from a Level-2 under-resolved in-span direction, which shows *high* leverage that *drops* with acquisition. This distinction is why all three Phoenix monitors are load-bearing and one is not enough; it is also the established subtlety that "a sample can be statistically more likely w.r.t. the data distribution and less likely w.r.t. the distribution the model has learned" (When Active Learning Fails, arXiv:2511.17760; Forgetful AL for OOD, arXiv:2301.05106).

**The span-completeness detector (new, the core of goal (b)).** Keep the residual as a *function over event-feature space*, not the scalar the DAS-CUSUM standardises. After probing a set of working points `{c_k}`, estimate the per-event residual `r_{c_k}(x) = w_{c_k}(x) − Π_ψ w_{c_k}(x)` (in practice: the residual of the learned per-event weight against its in-span projection, evaluated on the pooled event clouds), and assemble the residual operator `R` with one column per working point over a shared set of probe events. Take the SVD `R = U Σ Vᵀ`. A genuine missing structure appears as a **dominant left singular vector `u_1(x)` (a function on event-feature space) that is coherent across working points**, its weight `V` tracking the known `c`-dependence, because by Eq. (1) the missing piece is a fixed event-feature function modulated by `c`. Sampling noise does not concentrate into a coherent singular vector. The completeness criterion is the negation: across all reachable working points, leverage uniformly low *and* residual SVD consistent with the finite-sample floor, at which point more events at known working points are genuinely useless and `ψ` is complete for this manifold.

`ψ`-extension is then a concrete, runnable operation: append the event-feature function `u_1(x)` to the per-event encoder (a new learned event feature), refit, and the fingerprint must clear on held-out working points.

### 1.4 The acquisition reframe

The acquisition target is no longer "query a high-leverage point on a curve". It is **the working point at which to draw a fresh set of events such that the expected coherent out-of-span residual is maximised**, which by Eq. (1) means draw events at `c` where the quadratic morphing `B_{ij}(x) c_j` is largest: off the SM point, along the cross-couplings, inside the EFT-valid window. Concretely, the score for a candidate working point `c_cand` is the expected energy of `R`'s next singular direction the events drawn there would add. This is curvature probing. It is the design that the isotropic `μ`-space null could never reward, because the anisotropy it exploits lives in the manifold's curvature, not in any 1-D predictive metric. Note the action is "simulate `N` events at `c_cand`," never "evaluate at a coefficient value": the loop chooses *where in theory space to sample events from*, and the events are the only thing that enters the model.

### 1.5 Why this is the JEPA story and why Intention is demoted to a primitive

JEPA (LeCun 2022; I-JEPA Assran et al. CVPR 2023; the world-model framing) predicts the *embedding* of held-out content from visible content, never reconstructing the input, and keeps only the structure that is predictable, discarding the rest. Mapped here:

- The **encoder** is a permutation-invariant per-event map `ψθ: x ↦ R^D` (Energy/Particle-Flow / Deep Sets style; Komiske-Metodiev-Thaler arXiv:1810.05165). The context is the event set; the **manifold coordinate** is the implicit ridge weight `wθ(c) = A⁻¹ Kᵀ V` computed over the event-set features, the point's image in latent space. The Jacobian `∂wθ/∂c` is the learned tangent; drawing event sets at different working points and watching `wθ` move traces the learned curvature.
- The **predictive** task is JEPA-native: predict the embedding of held-out events (or of the event set at a held-out working point) from the embedding of observed events, with agreement measured **in latent space**, never by reconstructing events and never as a `σ`/density MSE. The Intention ridge is the predictor block that does this in closed form. No decoder, no event reconstruction, consistent with JEPA philosophy.
- The **views** of one physical state are re-simulations at fixed `c` (different MC seed, PDF replica, independent event draw), the cleanest physics-aware augmentation (RS3L, Harris et al. arXiv:2403.07066, Phys. Rev. D 111:032010): two independent event sets at the same `c` must map to nearby `wθ`. This is also the natural statistical-bootstrap view, two finite samples of the same distribution.
- **Interpretability (goal c)** is the linear probe `c̃ ≈ W wθ + b` onto the Fisher eigenbasis. A latent that is a faithful manifold image makes this linear; if an MLP probe beats the linear probe out-of-box, work migrated from representation into probe (the disclosure-integrity gate already in the paper).
- **Collapse** is the known JEPA failure (LeWorldModel, arXiv 2026; VICReg, Bardes-Ponce-LeCun ICLR 2022). Guard it at the philosophy level (variance/covariance regularisation or EMA target), but the closed-form ridge predictor already resists trivial collapse because a collapsed `ψ` cannot solve the ridge across scenarios.

The reason Intention is the *primitive* and not the headline: Intention excels at the closed-form ridge of a structure it was handed, which is goal (a)-adjacent but says little about (b) and (c). JEPA's job is to learn the manifold representation *without* being handed `c`, to know when its representation is incomplete, and to grow. That is the thing under test. **Architectural corollary to verify, not assume:** the JEPA reading should be primary precisely because `ψ` must be extensible; the frozen-`ψ` Intention head cannot extend itself and so cannot close goal (b). The active-learning component and the architecture choice are the same decision.

---

## 2. Tasks

Each task: Objective, Inputs, Steps, Cross-checks, Done-when. Do them in order; later tasks gate on earlier done-whens.

### Task 1 — Reproduce the derivations (symbolic + numeric)

**Objective.** Independently re-derive and numerically confirm Eqs. (1)-(2) and the three out-of-span facts, plus the residual-SVD coherence claim. This is the falsification gate for the whole document.

**Inputs.** The analytic oracle templates `{σ_SM, A_i, B_{ij}}`; the existing leverage/`A⁻¹` machinery.

**Steps.**
1. Symbolically (sympy) confirm `∂_i σ(c,m) = A_i(m) + 2 Σ_j B_{ij}(m) c_j` and that the Fisher matrix `F_{ij}(c) = Σ_m ∂_iσ ∂_jσ / s_m²` is `c`-dependent through the `B` term. Confirm `F(0)_{ij} = Σ_m A_i A_j / s_m²`.
2. Numerically diagonalise `F(c)` at `c=0` and at several off-SM points; confirm a vertex eigenvalue rises by orders of magnitude as `|c_{ℓq}|` grows, and identify the responsible `B_{vertex,ℓq}` template.
3. Construct the orthogonal projector `Π_S` over event-feature space for a deliberately deficient `ψ` (omit one true event-feature direction of the per-event likelihood ratio). Verify numerically: leverage of an event whose discriminating structure lives in `w_⊥` is low; accuracy error floors at `‖w_⊥‖`; the nominal interval undercovers.
4. Build `R` over a sweep of working points with the deficient `ψ`, columns being the per-event residuals over a shared probe-event set; verify SVD yields one dominant coherent `u_1(x)` ≈ the omitted event-feature function (cosine similarity to truth > 0.95), and that an independent finite-sample event draw with no missing structure does **not** produce a coherent dominant singular vector (it produces only sampling-noise spectrum).

**Cross-checks.** Compare `∂²σ` against finite differences of the oracle. Compare the recovered `u_1(x)` against the analytic omitted event-feature function directly. Confirm the per-event likelihood ratio `w_c(x)` reconstructs the binned `σ(c,m)/σ_SM` when integrated over events in an `m`-bin (consistency with the old representation, not a dependence on it).

**Done-when.** All four steps pass with the stated tolerances. If step 2 does not show the vertex lift, the hypothesis in Section 1.1 is wrong for this oracle and must be diagnosed before proceeding (likely cause: `B_{vertex,ℓq}` too small inside the EFT-valid `mℓℓ` window; quantify it).

---

### Task 2 — Dummy exercise: known-curvature toy manifold

**Objective.** Demonstrate, on a controlled toy where the manifold and its curvature are known in closed form, that (i) working-point sampling lifts a tangent-flat direction, (ii) residual-SVD recovers a missing template, (iii) the span-completeness criterion fires correctly and only correctly, all *before* touching real SMEFT.

**Inputs.** A toy oracle `μ(c,m) = g_SM(m) + Σ_i a_i(m) c_i + Σ_{ij} b_{ij}(m) c_i c_j` with hand-chosen `a, b` such that one direction is tangent-flat at `c=0` (set its `a_i ∝ g_SM`) but curvature-resolvable (give it nonzero `b_{ij}` with a distinct `m`-shape). Include one genuinely out-of-span shape (e.g. a frequency `ψ` cannot represent) as the Level-3 injection.

**Steps.**
1. Reproduce the morphing logic: show that `N` base points exactly determine `{g_SM, a_i, b_{ij}}` and that the morphing-matrix condition number depends on base-point placement. Use this as the ground-truth oracle for "what a new working point buys."
2. Show the single-point Fisher misses the flat direction and the multi-point (curvature) Fisher recovers it. Plot the eigenvalue of the flat direction vs working-point excursion `|c|`.
3. Run the residual-SVD template recovery on the out-of-span injection; append `u_1` to `ψ`; show accuracy and calibration recover on held-out working points and not on the points used to derive `u_1` (guard against memorising the SVD inputs).
4. Run the completeness criterion: confirm it does **not** fire when only an in-span under-resolved direction is present (that is a Task-4 acquisition case), and **does** fire when an out-of-span template is present.

**Cross-checks.** Morphing-matrix reconstruction of templates must agree with the SVD-recovered template. Held-out working-point validation set, disjoint from derivation set.

**Done-when.** Flat-direction lift is monotone in `|c|`; template recovery cosine > 0.95 on held-out points; completeness criterion has zero false-fire on the in-span-only case across ≥ 20 seeds.

---

### Task 3 — The ManifoldInformer (JEPA philosophy, Intention primitive)

**Objective.** Build the representation-learning network whose latent is an interpretable image of the SMEFT manifold, with Intention as the embedding/prediction primitive. Optimise for goals (a) and (c). **Philosophy over implementation**: the deliverable is a latent with the four geometric properties below, by whatever JEPA-faithful construction achieves them, not a specific transformer recipe.

**Inputs.** The existing closed-form ridge head; a permutation-invariant per-event encoder (Energy/Particle-Flow / Deep Sets, to be built or adapted); the analytic and MadGraph oracles as event generators; re-simulation augmentation via independent event draws / MadGraph seeds / PDF replicas.

**Steps.**
0. **Event-set encoder (the input change, do this first).** Replace the binned-ratio input with a permutation-invariant per-event map `x ↦ ψθ(x) ∈ R^D` whose pooled output forms the Intention context rows. The context at latent `c` is the event set `{x_n}`; the head solves the ridge over those rows. Verify permutation invariance to machine precision and verify the pooled representation is a sufficient statistic by checking it reconstructs the binned ratio on integration (Task 1 cross-check), without ever taking the binned ratio as input.
1. **Latent prediction objective (JEPA-native).** Train `ψθ` so the Intention ridge predicts the *embedding* of held-out events from the embedding of context events, loss in latent space, no event reconstruction, plus a small density/`σ`-anchor term to fix scale. Contrast against the current pure-`σ`-MSE-on-bins objective and report what changes in the latent geometry.
2. **View invariance.** Two independent event sets at the same `c` must map to nearby `wθ`. Add the RS3L-style re-simulation augmentation and an invariance term.
3. **Collapse guard.** Add variance/covariance regularisation (VICReg-style) or an EMA target branch; verify the latent does not collapse (effective rank stays ≳ the number of resolved Fisher directions).
4. **Interpretability readout (goal c).** Fit the linear probe `c̃ ≈ W wθ + b` onto the Fisher eigenbasis. Run the disclosure-integrity gate (linear vs parameter-matched MLP probe); linear must be at or below MLP MSE on held-out working points.

**Required latent properties (the actual deliverable, test each):** (P1) linearity of physical factors, `wθ(c)` traces a near-linear path along each resolved `c̃`; (P2) regime separation, distinct dynamical regimes occupy distinct latent regions; (P3) the Jacobian `∂wθ/∂c` recovers the tangent `A_i` up to the probe map; (P4) curvature, second differences of `wθ` in `c` recover `B_{ij}` structure up to the probe map. P3/P4 are the bridge to Tasks 1-2 and the thing that distinguishes a learned manifold from a fit.

**Cross-checks.** Compare against the three existing FM flavours on the same held-out pools (this is the goal-(a) demonstration: the learned, c-agnostic manifold representation vs the explicit-`c` and fixed-basis variants). Compare linear-probe `R²` and effective rank against the JEPA-FM baseline already in the paper (0.75 vs 0.62 Wilson recoverability is the existing number to beat or explain).

**Done-when.** P1-P4 hold on held-out working points; disclosure-integrity gate passes; the c-agnostic ManifoldInformer matches or explains its gap to the explicit-`c` ceiling without consuming `c`.

---

### Task 4 — Span-completeness active learning, driven by Phoenix (goal b)

**Objective.** Close the loop that distinguishes the two failure kinds, acquires working points by curvature, extends `ψ` when and only when the span is exhausted, and certifies completeness. This is the positive active-learning result the current paper does not have.

**Inputs.** Phoenix tracing and the three existing monitors; the span-completeness detector of Section 1.3; the curvature acquisition score of Section 1.4; the ManifoldInformer of Task 3.

**Steps.**
1. **Add the span-completeness monitor to Phoenix** as a fourth evaluator: it consumes the function-valued residual stream (not the standardised scalar), maintains the working-point residual matrix `R`, and emits (a) the dominant coherent singular value, (b) its coherence score across working points, (c) a boolean out-of-span flag using the Section 1.3 fingerprint (coverage clean ∧ accuracy drift ∧ calibration break). Document the OpenInference span schema for it in `INTEGRATION.md` alongside the existing three.
2. **Aggregator routing by signature.** High leverage that drops with nearby data → in-span acquisition (existing path). Out-of-span fingerprint → `ψ`-extension action: pull `u_1` from `R`, append to `ψ`, refit, validate on held-out working points. Uniformly low leverage + noise-consistent residuals across reachable working points → emit `span-complete` and halt acquisition.
3. **Curvature acquisition backend.** Implement the working-point score of Section 1.4 (expected coherent out-of-span residual energy / next-singular-direction gain). Run it against uniform-random and against the existing in-span EPIG/leverage backends.
4. **Phoenix-as-controller framing.** Phoenix is the oracle-side controller: it changes which region of `(m, c)` space episodes are drawn from, detects the match/exhaustion, and complexifies (extends `ψ` or escalates working-point excursion). Make the trace tell this story end to end.

**Cross-checks.** Every `ψ`-extension must reduce held-out-working-point residual or be rejected and logged. The completeness `span-complete` emission must coincide with morphing-matrix saturation (Task 2 oracle) on the toy, i.e. no further template is recoverable.

**Done-when.** The loop (i) fires `ψ`-extension on an injected out-of-span template and clears the fingerprint, (ii) does *not* fire extension on an in-span under-resolved direction and instead acquires data, (iii) emits `span-complete` correctly, all traced in Phoenix. Curvature acquisition must beat uniform random on a curvature-scored metric at a stressed budget (the metric where anisotropy exists); if it ties random there too, diagnose whether the curvature is below the noise floor.

---

### Task 5 — Real SMEFT example: lift the vertex directions without AFB

**Objective.** The headline physics result. On the existing mass-only Drell-Yan oracle with the four operators `O^{(1,3)}_{ℓq}, O^{(1,3)}_{Hq}`, show the two vertex directions that sit at the prior floor in the paper's Table 4 **lift off it under the working-point-probing loop, on the mass observable alone, with no angular extension**, because `B_{vertex,ℓq}(m)` is energy-dependent and in-span-extendable once exposed.

**Inputs.** Everything above; the analytic oracle; the MadGraph tier for the validation gate.

**Steps.**
1. Run the Task-4 loop with curvature acquisition: draw event sets at working points starting at/near SM and pushing along `c_{ℓq}` inside the EFT-valid window (`ŝ/Λ² < 1`). The action is "simulate `N` events at `c`," never "evaluate at `c`."
2. Track the vertex-direction posterior MSE and 68% coverage vs working-point excursion. The prediction: vertex MSE falls off the `~0.18` prior floor and coverage moves toward nominal as the curvature `B_{vertex,ℓq}` becomes exposed in the event distributions.
3. Compare three routes head to head: (i) working-point curvature route (no angular variable, mass-and-event-level only), (ii) the AFB angular route already in the paper (independent, through chiral structure, available because the events carry `cosθ*`), (iii) baseline single-point Fisher (no lift, control).
4. Validate any `ψ`-extension or recovered event-feature function against MadGraph at the same `(c, m)` cells used for the existing AFB cross-check (median `|Δ|` per-cent level).

**Cross-checks.** The MadGraph tier is the truth gate for every recovered event-feature function. The AFB route and the curvature route should agree on *which* vertex combination lifts and disagree only on efficiency; if they identify different directions, something is wrong, diagnose.

**Done-when (and the sharp null).** Either the vertex directions lift under working-point event-set probing (the positive result), or they do not, in which case the diagnosis is binary and reportable: `B_{vertex,ℓq}` is too small inside the EFT-valid window, **or** the coherent residual sits below the finite-sample floor set by the event count per working point. Both are measurable. Quantify which, and report the event budget at which the floor is crossed. A null here is a measured statement about SMEFT curvature and statistics, not a loop failure, but you must produce the number that settles it.

---

## 3. Dedicated cross-checks (apply across tasks)

1. **Algebraic identities** (Task 1) gate everything. No downstream result is trusted if Eq. (1) does not reproduce against finite differences.
2. **Morphing-matrix oracle** (Task 2) is the ground truth for "what a working point buys"; SVD-recovered templates must match morphing-reconstructed templates.
3. **MadGraph validation** (Task 5) is the truth gate for every recovered template and every `ψ`-extension on real SMEFT.
4. **Held-out working points** everywhere: any template derived from a set of `c_k` must reduce residual on a disjoint set, or it is memorisation.
5. **Disclosure-integrity** (Task 3): linear probe ≤ MLP probe on held-out, else work migrated into the probe.
6. **Coherence-vs-noise** (Tasks 1,4): a recovered singular vector must be coherent across working points; a noise-only control must not produce one.
7. **Two-kinds discrimination** (Task 4): the loop must never extend `ψ` on an in-span under-resolved direction, and never merely acquire on an out-of-span one. Confusion of the two is the single most important failure mode; test it explicitly with both injections.

## 4. Scope boundaries (state these in the writeup, do not overclaim)

- The method recovers **operators the truncation hid** (finite-rank morphing residual, exactly the SMEFT case). It is **not** a discoverer of arbitrary new physics: genuinely non-polynomial structure gives a high-rank or `c`-non-stationary residual and the SVD-append step degrades. Strong claim = "resolve directions a fixed `ψ` could not, by exposing curvature"; not "discover anything."
- EFT validity (`ŝ/Λ² < 1`) caps how far working points can be pushed, hence caps the curvature you can sample. State the window.
- The whole reframe **requires `ψ` extensible**, which is the JEPA head's native mode and the frozen Intention head's wall. The architecture choice and the active-learning capability are one decision; say so.
- "Out-of-span" is defined relative to `ψ`. A richer `ψ` reclassifies a direction as in-span. The completeness criterion is always relative to the current `ψ` and the reachable working-point set.
- **Input is event-level throughout. No histograms, no binned ratios, no Wilson coefficient ever enters the forward pass.** The model consumes sets of events sampled from distributions; `c` is recovered only by the frozen linear probe as a readout. Binned quantities appear solely as a Task-1 consistency cross-check (integrate the per-event likelihood ratio over an `m`-bin and confirm it matches the old binned ratio), never as model input. Event-level is the correct target on its own merits: binning is an information bottleneck (ML4EFT arXiv:2211.02058; refinable unbinned SMEFT arXiv:2406.19076).

## 5. Mapping to the three goals

- **(a) Foundation model in physics:** Task 3 (ManifoldInformer, c-agnostic, learned manifold latent) plus the existing architecture comparison. The deliverable is the latent with properties P1-P4, demonstrated to be a faithful manifold image rather than a fit.
- **(b) Active learning enabled by Phoenix:** Task 4. Phoenix is the controller that discriminates the two failure kinds, drives curvature acquisition, triggers `ψ`-extension, and certifies span-completeness. The positive separation-from-random result lives on the curvature-scored metric at stressed budget.
- **(c) Interpretable via SBI:** the linear-probe readout onto the Fisher eigenbasis (Task 3 step 4), upgraded to per-direction posterior coverage, with the vertex directions moving off the prior floor (Task 5) as the concrete interpretability win.

---

## References

- Garnelo, Czarnecki. Exploring the Space of Key-Value-Query Models with Intention. arXiv:2305.10203.
- LeCun. A Path Towards Autonomous Machine Intelligence (JEPA). 2022. Assran et al., I-JEPA, CVPR 2023. Bardes, Ponce, LeCun, VICReg, ICLR 2022. Thimonier et al., T-JEPA, ICLR 2025.
- Transtrum, Qiu. Model Reduction by Manifold Boundaries. Phys. Rev. Lett. 113:098701 (2014). Transtrum, Machta, Sethna. Phys. Rev. E 83:036701 (2011). Transtrum et al., sloppy models review, J. Chem. Phys. 143:010901 (2015), arXiv:1501.07668. "Effective theory building and manifold learning," Synthese, arXiv:2411.15975 (MBAM ↔ manifold learning ↔ EFT).
- Balasubramanian et al. Effective Lagrangian Morphing. arXiv:2202.13612. Overdetermined Morphing. arXiv:2512.12473.
- Greljo, Marzocca. High-pT dilepton tails and flavour physics. EPJC 77:548 (2017). SMEFiT 3.0 docs (polynomial-in-`c` form). Ellis, Madigan, Mimasu, Sanz, You (Fisher eigenbasis reporting), arXiv:2012.xxxx (paper ref [33]). Das Bakshi, Hobbs, Kriesten, demonstrator SMEFT FM, arXiv:2512.15862.
- Boughezal, Huang, Petriello. High invariant-mass Drell-Yan forward-backward asymmetry and SMEFT, arXiv:2207.01703 / Phys. Rev. D 106:036020. Alioli, Boughezal, Mereghetti, Petriello, dim-8 angular structure, arXiv:2003.11615.
- Locatello et al. Challenging assumptions in disentanglement. ICML 2019, arXiv:1811.12359. Harris et al., RS3L re-simulation, arXiv:2403.07066, Phys. Rev. D 111:032010.
- Smith, Bickford Smith, Rainforth. Prediction-oriented (Expected Predictive Information Gain) Bayesian active learning. arXiv:2304.08151. Kirsch, van Amersfoort, Gal. BatchBALD. NeurIPS 2019. Sabato, Munos. Active Regression by Stratification. NeurIPS 2014, arXiv:1410.5920.
- Thomas, Houssineau. Active learning with a Bayesian representation of epistemic uncertainty. arXiv:2412.08225. When Active Learning Fails (uncalibrated OOD UQ). arXiv:2511.17760. Forgetful Active Learning for OOD. arXiv:2301.05106.
- Araz, Spannowsky. Conformal prediction for calibrated uncertainty in HEP. arXiv:2512.17048. Cui, Martin, Marzouk (likelihood-informed subspace), Inverse Problems 30:114015. Constantine, Active Subspaces, SIAM 2015.
- **Event-level / unbinned input:** Zaheer et al. Deep Sets. NeurIPS 2017. Komiske, Metodiev, Thaler. Energy/Particle Flow Networks: Deep Sets for particle jets. arXiv:1810.05165, JHEP 01(2019)121. Chen, Glioti, Marzocca, Nardini, Wulzer. Unbinned multivariate observables for global SMEFT analyses (ML4EFT). arXiv:2211.02058, JHEP 03(2023)033. Schöfbeck. Refinable modeling for unbinned SMEFT analyses (per-event likelihood ratio, detector-level optimal observables). arXiv:2406.19076. Brehmer, Cranmer, Louppe, Pavez (mining gold / MadMiner per-event likelihood ratio), PNAS 117:5242; MadMiner, Comput. Softw. Big Sci. 4:3.
