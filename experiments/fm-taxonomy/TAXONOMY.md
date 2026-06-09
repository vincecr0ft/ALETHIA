# A taxonomy of foundation modelling, and where ALETHIA sits

This document folds the four branch writeups in this directory into one frame.
Each branch (`01`–`04`) lays out its own mathematics, contributes a set of
distinguishing axes, and ships a runnable demo (all four demos executed and
verified in the main session; numbers are in each `results.md`). The synthesis
below organises those axes and locates ALETHIA's existing studies on them.

Branches:
- `01-fm-surrogate/` — foundation models vs surrogates/emulators/operators/amortized posteriors (the general landscape)
- `02-fm-hep/` — foundation models on collider data specifically
- `03-infogeom-morphing/` — information geometry, Lagrangian/EFT morphing, and manifold learning
- `04-active-learning/` — acquisition functions and the mathematics of the learner

---

## The master axes

The four branches turn out to be four orthogonal questions about the *same*
object: a learned (or constructed) map that approximates something a simulator
produces, and that is reusable across tasks. The axes group as follows.

### I. What is amortized / reused (branch 01)
The single most discriminating column. The variable held fixed at train time and
varied for free at test time:

| Mode | Amortized over | Target | Reuse mechanism |
|---|---|---|---|
| Surrogate | inputs only | forward $\theta\!\mapsto\!x$ | re-fit per task |
| Emulator (GP + discrepancy) | inputs, calibrated | forward + uncertainty | re-fit per system |
| Neural operator | a family parameter | forward in function space | evaluate within family |
| Amortized posterior (NPE/NLE/NRE) | observations | inverse $x\!\mapsto\!p(\theta\mid x)$ | one forward pass |
| **Foundation model** | the data distribution | *neither* until a head is attached | probe / in-context after SSL |

A self-supervised objective is what *licenses* representation reuse — that is the
operational definition of "foundational." (Branch 01 demo: a family-pretrained
representation beats a from-scratch surrogate by up to ~100× MSE at fixed labels.)

### II. The data object and its symmetry budget (branch 02)
Only meaningful once the input is a physical particle set:
- **Granularity:** {jet constituents | subjets/patches | full event | shower}.
- **Symmetry budget:** how much known physics is hard-wired vs learned
  {permutation only → IRC-safe → Lorentz-invariant features → exact Lorentz equivariance}.
- **Augmentation provenance:** {none/generative | hand-built physics | simulator
  re-simulation (RS3L) | latent-only (JEPA)}. Re-simulation has no analogue
  outside the physical sciences.
- **Reconstruction level × supervision:** {parton/truth | detector} ×
  {free simulator labels | SSL}. HEP is the rare regime where supervised
  pretraining at FM scale is even possible, because the simulator emits labels.

### III. Provenance of the prediction manifold (branch 03)
Whether the low-dimensional structure is **constructed from known generative
algebra** or **discovered from data**:
- **Structured/exact (Lagrangian morphing):** $|M|^2$ is polynomial in the Wilson
  coefficients, so $\sigma = T_0 + cT_1 + c^2T_2$; a minimal basis of
  $N=\binom{n+2}{2}$ exact samples spans the *entire* family with no fit (demo:
  reconstruction error $1.2\times10^{-14}$, template recovery $4.4\times10^{-16}$).
- **Learned/empirical (PCA, diffusion maps, autoencoder, FM latent):** the same
  rank is *discovered* from sampled curves (demo: PCA recovers rank 3 = degree+1).
- **Unifying claim:** morphing is the analytic limit of manifold learning when the
  generative polynomial structure is known. The Fisher metric
  $F_{ij}=\sum_x \partial_i\sigma\,\partial_j\sigma/\sigma_y^2$ is then available in
  closed form, and it *collapses* at working points where SM and BSM gradients
  align to cancel — the same mechanism as ALETHIA's flat-vertex-direction finding.

### IV. The mathematics of the learner (branch 04)
How the next sample/label is chosen:
- **Axis A — information about what:** parameters (BALD/Fisher/D-/A-optimal,
  target-agnostic, maximises posterior contraction) vs predictions (EPIG,
  target-weighted, maximises test-time accuracy). In the Gaussian-linear regime
  BALD $=\tfrac12\log(1+\psi^\top A^{-1}\psi)$ — *literally* ALETHIA's `leverage`.
- **Axis B — where $\psi(x)$ comes from:** task-fit vs frozen/fine-tuned FM
  backbone. With a frozen backbone the binding constraint moves from the data to
  the embedding geometry.
- **Axis C — greedy vs joint batch:** only bites once the representation makes
  high-score candidates collinear (`cos_Ainv → 1`).

---

## Where ALETHIA sits on this frame

| Axis | ALETHIA's coordinate |
|---|---|
| I. Amortized over | the data distribution → **foundation encoder ∘ amortized-posterior head** (SSL pretraining, then a regression/inference head) |
| II. Data object | **event-level, parton/analytic-level, SSL-JEPA** (VICReg + RS3L-style re-simulation views) on a permutation-invariant Deep Sets / EFN set encoder — a corner no published *jet-level* FM occupies |
| III. Manifold provenance | the physics target is the **structured/exact morphing manifold** (known EFT polynomial); the ManifoldInformer tries to **learn** that same manifold in latent space — the central tension of the project is structured-vs-learned on the *same* object |
| IV. Learner | closed-form acquisition: `leverage` = BALD/D-optimal, `param_epig_a/d` = directional A-/D-optimal, `epig` = EPIG; monitored diagnostic `cos_Ainv` is the BatchBALD redundancy signature |

The **contraction-vs-MLE gap** documented in `al-phoenix-studies` is, on this
frame, Axis IV-A inverted by weak identifiability: in a well-conditioned
representation (branch 04 demo E1) parameter-information acquisition contracts the
posterior *and* improves the estimate together (z ≈ −4.5 on both); on an
ALETHIA-shaped collinear ridge (E2, `cos_Ainv` 0.533 → 0.851) the contraction
signal survives but the estimator improvement collapses (zVar shrinks toward 0,
zErr → 0). The first-class monitored quantity should therefore be the *gap*
between contraction and estimator error, not contraction alone — a greedy
information-gain learner on a collinear ridge contracts hardest precisely while it
is fooling itself. This is the operational restatement of the project finding that
the operative gate is mean `|cos_Ainv|` top-decile, not CV(IG).

---

## How this frames the three Phase-2 studies

The taxonomy above makes the Phase-2 tasks precise:
1. **What each FM does/represents** — fill the branch-01/02 tables exhaustively
   against axes I–II, one row per model, stating the objective and what the
   representation encodes.
2. **An AL demonstration with a monitoring program** — instrument axis IV: track
   contraction, estimator error, and `cos_Ainv` live, showing where E1-like and
   E2-like regimes separate (the math of the learner is the subject).
3. **An EFT/SMEFT oracle survey** — axis III: how well simple learners recover the
   structured morphing manifold (the exact polynomial oracle) that a full MC
   program represents, and where the learned approximation departs from the exact
   construction.
