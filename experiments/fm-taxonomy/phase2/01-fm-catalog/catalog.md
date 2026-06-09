# An exhaustive catalog of foundation models: what each one represents

Phase-2 study 1 of the ALETHIA FM taxonomy. Builds on the Phase-1 frame in
`../../TAXONOMY.md` and the two branch writeups `../../01-fm-surrogate/writeup.md`
(general landscape) and `../../02-fm-hep/writeup.md` (HEP). This document is not a
literature list: every entry answers five fields in the same order, so the entries
are directly comparable.

The five fields, fixed once:

1. **Objective (formal loss).** The pretraining risk, written out.
2. **Data object.** What one example *is* (token sequence, image patches, particle
   set, function pair, simulator draw), and the variable cardinality / symmetry.
3. **What the representation encodes.** The operative field. What is made invariant,
   what is linearly predictable from the latent, and what is discarded.
4. **Reuse mechanism.** How a trained artefact is applied to a new task: re-fit,
   evaluate-in-family, single forward pass, linear probe, fine-tune, in-context.
5. **Axis coordinates (I, II).** Axis I = what is amortized / reused (the master
   discriminator from branch 01); axis II = data-object + symmetry budget
   (the HEP axes A1–A4 from branch 02). General-ML models occupy axis I fully and
   only the granularity sub-axis of II; HEP models occupy both.

Every arXiv id below was web-verified against the paper's title and authors in this
session (June 2026). Where the Phase-1 writeups left a citation incomplete it is
corrected here and flagged.

---

## Master comparison table (axes I–II)

Axis I — *amortized over* / *forward-vs-inverse* / *reuse*. Axis II — *granularity*
(A1) / *symmetry budget* (A2) / *augmentation provenance* (A3) / *level × supervision*
(A4). "—" means the axis does not bind for that model (general-ML rows have no
particle-set symmetry budget).

| Model (arXiv) | Objective family | I: amortized over | I: target | I: reuse | II-A1 granularity | II-A2 symmetry | II-A3 augmentation | II-A4 level × supervision |
|---|---|---|---|---|---|---|---|---|
| BERT (1810.04805) | masked prediction | data distribution | representation | fine-tune / probe | token sequence | — | none | text · SSL |
| MAE (2111.06377) | masked reconstruction | data distribution | representation | fine-tune / probe | image patches | — | none | image · SSL |
| I-JEPA (2301.08243) | JEPA latent-predictive | data distribution | representation | linear probe | image patches | — | latent-only | image · SSL |
| VICReg (2105.04906) | joint-embedding, var-cov | data distribution | representation | linear probe | image | — | hand-built views | image · SSL |
| Conditional Neural Process (1807.01613) | predictive log-lik | datasets / context sets | fwd (fn posterior) | condition on context | function samples | — | none | generic · supervised-context |
| FNO (2010.08895) | supervised MSE in fn space | a PDE family | forward operator | evaluate on new $a$ | discretized field | — | none | PDE · supervised |
| DeepONet (1910.03193) | supervised MSE (branch/trunk) | a function family | forward operator | evaluate on new input fn | fn samples | — | none | operator · supervised |
| NPE / APT (1605.06376, 1705.07057, 1905.07488) | conditional density MLE | observations $x$ | inverse $p(\theta\mid x)$ | one forward pass | simulator output | — | none | simulator · supervised(joint) |
| NRE (1903.04057) | ratio classifier (BCE) | observations $x$ | inverse (ratio) | MCMC w/ amortized ratio | simulator output | — | none | simulator · supervised(joint) |
| ParT (2202.03772) | supervised classification | — (label-supervised) | discriminative | fine-tune backbone | jet constituents | Lorentz-inv pair features | none | detector · supervised |
| JetCLR (2108.04253) | NT-Xent contrastive | data distribution | representation | linear probe | jet constituents | perm + IRC-safe (learned) | hand-built physics | detector · SSL |
| RS3L (2403.07066) | NT-Xent contrastive | data distribution | representation | fine-tune / probe | jet constituents | perm; shower-robust | re-simulation | detector · SSL |
| MPM (2401.13537) | masked token cross-entropy | data distribution | representation | fine-tune / probe / in-context | particle set | perm-inv ($p_T$-ordered head) | none (tokenizer) | detector · SSL |
| OmniJet-α (2403.05618) | autoregressive NLL | data distribution | gen → disc | fine-tune / transfer | jet constituents | perm via $p_T$ order | none (tokenizer) | detector · SSL |
| Aspen Open Jets (2412.10504) | autoregressive NLL | data distribution | generative | fine-tune / transfer | jet constituents (CMS real) | perm via $p_T$ order | none | detector(real) · SSL |
| OmniLearn (2404.16091) | classification + generation | — + data dist. | both | fine-tune / transfer | jet constituents | perm-inv PET | hand-built / none | detector · supervised+gen |
| OmniLearned (2510.24066) | classification + generation, >1B | — + data dist. | both | fine-tune / transfer | jet constituents | perm-inv PET | hand-built / none | detector · supervised+gen |
| L-GATr (2405.14806) | supervised (also gen via flow) | — / a family | both | fine-tune / evaluate | four-vectors | exact Lorentz equivariance | none | detector · supervised |
| J-JEPA (2412.05333) | JEPA latent-predictive | data distribution | representation | linear probe / fine-tune | subjets | perm; augmentation-free | latent-only | detector · SSL |
| HEP-JEPA (2502.03933) | JEPA latent-predictive | data distribution | representation | linear probe / fine-tune | jet patches | perm; latent masking | latent-only | detector · SSL |
| SMEFT-FM (2512.15862) | supervised contrastive | Wilson configs (100 discrete) | representation | clustering / probe | binned $d\sigma$ (NC DY) | none (binned obs) | none | parton · supervised(c-label) |
| **ALETHIA / ManifoldInformer** (this repo) | **JEPA latent + VICReg + RS3L-view + density anchor** | **data distribution → +amortized-posterior head** | **representation → inverse $p(c\mid\{x\})$** | **linear probe + closed-form in-context ridge** | **full event** | **perm-inv (Deep Sets/EFN)** | **re-simulation (RS3L-style) + latent** | **parton/analytic · SSL** |

Reading the table: axis I sorts the rows into four blocks — *forward-operator* (FNO,
DeepONet, CNP), *inverse-posterior* (NPE/NRE), *supervised-discriminative*
(ParT, L-GATr, SMEFT-FM), and *self-supervised representation* (the JEPA / masked /
contrastive / autoregressive cluster). ALETHIA is the only row that spans two blocks:
a self-supervised encoder feeding an inverse-posterior head.

---

# Part A — General-ML foundation models and their neighbours

The four non-FM modes (surrogate, emulator, operator, amortized posterior) are
included because axis I is only legible by contrast: a foundation model is the row
whose *objective* targets a representation rather than a task output, and that is
what licenses reuse. Branch 01 gives the formal setup; this part fills the per-model
fields.

## A.1 Masked / denoising prediction

### BERT — arXiv:1810.04805 (Devlin, Chang, Lee, Toutanova; NAACL 2019)
1. **Objective.** Masked-language-model cross-entropy: mask ~15% of WordPiece tokens,
   predict each from the bidirectional context,
   $\min_\phi \mathbb{E}\big[-\sum_{i\in\mathcal M}\log p_\phi(t_i\mid t_{\setminus\mathcal M})\big]$,
   plus a next-sentence-prediction term (later shown largely removable).
2. **Data object.** A token sequence of fixed max length; cardinality fixed by padding;
   order carried by positional encodings.
3. **Represents.** The conditional distribution of a token given its two-sided context.
   The pooled `[CLS]` / per-token vectors make syntactic role, coreference, and
   sense linearly decodable; surface form (exact wordpiece, casing) is partly discarded.
4. **Reuse.** Fine-tune the whole stack or probe `[CLS]`. The canonical "pretrain once,
   adapt to many" model that defines axis I's foundation row.
5. **Axes.** I: data distribution / representation / fine-tune. II: token-sequence
   granularity; no symmetry budget.

### MAE — arXiv:2111.06377 (He, Chen, Xie, Li, Dollár, Girshick)
1. **Objective.** Pixel reconstruction of masked patches: mask ~75% of image patches,
   reconstruct the missing pixels with an asymmetric encoder–decoder,
   $\min_\phi \mathbb{E}\,\|x_{\mathcal M}-g_\phi(x_{\setminus\mathcal M})\|^2$ on masked tokens only.
2. **Data object.** A grid of image patches; fixed cardinality; 2-D position encoded.
3. **Represents.** Whatever supports pixel infilling — coarse semantic layout and
   texture statistics. The high masking ratio forces holistic structure over local
   copy; fine high-frequency detail is partly discarded into the decoder.
4. **Reuse.** Fine-tune the encoder; the lightweight decoder is dropped.
5. **Axes.** I: data distribution / representation / fine-tune. II: image-patch granularity.

## A.2 Joint-embedding / latent-predictive (the JEPA line ALETHIA follows)

### I-JEPA — arXiv:2301.08243 (Assran et al.; CVPR 2023)
1. **Objective.** Predict the *embeddings* (not pixels) of target image blocks from a
   single context block, in latent space, against an EMA target encoder:
   $\min_\phi \sum \|\mathrm{pred}_\phi(h_\phi(\text{ctx}))-\mathrm{sg}[h_{\bar\phi}(\text{tgt})]\|^2$.
2. **Data object.** Image patches; multi-block masking in latent space.
3. **Represents.** The latent that makes held-out regions linearly predictable from
   context — empirically more semantic and lower-dimensional than MAE's pixel target,
   because there is no pressure to model nuisance detail. Pixel-level appearance is
   discarded by construction (the loss never touches pixels).
4. **Reuse.** Linear probe on frozen features; no fine-tune needed for the headline.
5. **Axes.** I: data distribution / representation / linear-probe. II: image-patch
   granularity; augmentation provenance = latent-only.

### VICReg — arXiv:2105.04906 (Bardes, Ponce, LeCun; ICLR 2022)
1. **Objective.** Two augmented views $z,z'$; minimise invariance (MSE between views)
   plus a variance hinge $\sum_d\max(0,\gamma-\mathrm{std}(z_{\cdot d}))$ and a
   covariance penalty $\sum_{i\neq j}\mathrm{Cov}(z)_{ij}^2$ that together prevent
   collapse without negatives, stop-gradient, or a memory bank.
2. **Data object.** Image + hand-built augmentations.
3. **Represents.** A space where each coordinate carries independent variance —
   the anti-collapse mechanism ALETHIA reuses directly (`vicreg_loss`,
   `manifold_informer.py:315-325`). What is discarded is whatever the augmentations
   declare nuisance.
4. **Reuse.** Linear probe.
5. **Axes.** I: data distribution / representation / probe. II: image granularity;
   hand-built augmentation provenance.

## A.3 Neural processes (the amortized function-posterior)

### Conditional Neural Process — arXiv:1807.01613 (Garnelo et al.; ICML 2018)
1. **Objective.** Maximise the predictive log-likelihood of target points given a
   context set: $\max_\phi \mathbb{E}\big[\sum_t \log p_\phi(y_t\mid x_t, r_C)\big]$,
   $r_C=\frac1{|C|}\sum_{c}h_\phi(x_c,y_c)$ a permutation-invariant context summary.
2. **Data object.** A context set of $(x,y)$ pairs plus query inputs; variable cardinality.
3. **Represents.** A posterior over functions given the context — the context is
   encoded into $r_C$, from which the target conditional is linearly read out. It
   amortizes the *map from datasets to predictive distributions*.
4. **Reuse.** Condition on a new context set at inference; no weight update.
5. **Axes.** I: datasets/context sets / forward (function posterior) / condition.
   The closest classical analogue to ALETHIA's in-context ridge head.

## A.4 Neural operators (the family-amortized forward map)

### FNO — arXiv:2010.08895 (Li et al.; ICLR 2021)
1. **Objective.** $\min_\phi\sum_i\|\mathcal G_\phi(a_i)-u_i\|^2$, $\mathcal G_\phi$ a
   stack of Fourier kernel-integral layers, over samples of a PDE family.
2. **Data object.** Discretized coefficient field $a$ → solution field $u$.
3. **Represents.** The solution operator of a *family*; the latent encodes the
   spectral response, discretization-invariant within the family. Anything outside
   the trained family is not represented.
4. **Reuse.** Evaluate on a new $a$, no re-fit.
5. **Axes.** I: a PDE family / forward operator / evaluate-in-family.

### DeepONet — arXiv:1910.03193 (Lu et al.; Nat. Mach. Intell. 2021)
1. **Objective.** Same supervised MSE, with a branch network (input function samples)
   and trunk network (query location) whose dot product approximates the operator,
   licensed by the operator universal-approximation theorem.
2–5. Same role as FNO: family-amortized forward operator, evaluate-in-family reuse.

## A.5 Amortized posteriors (simulation-based inference) — the mode ALETHIA's head occupies

### NPE / APT — arXiv:1605.06376, 1705.07057, 1905.07488
1. **Objective.** Fit a conditional density $q_\phi(\theta\mid x)$ by MLE over joint
   draws $\theta\sim p(\theta), x\sim p(x\mid\theta)$:
   $\max_\phi\mathbb E[\log q_\phi(\theta\mid x)]$; APT (Greenberg 2019) is the
   proposal-corrected sequential variant; density estimators are MAF (Papamakarios 2017).
2. **Data object.** Simulator output $x$ (any structure), paired with $\theta$.
3. **Represents.** The inverse map: a distribution over parameters given data. The
   summary network's latent encodes the sufficient statistics of $x$ for $\theta$;
   everything orthogonal to the likelihood gradient is discarded.
4. **Reuse.** One forward pass per new $x_{\rm obs}$ — amortized over observations.
5. **Axes.** I: observations / inverse posterior / one forward pass.

### NRE — arXiv:1903.04057 (Hermans, Begy, Louppe; ICML 2020)
1. **Objective.** Train a classifier to estimate the likelihood-to-evidence ratio
   $r(x,\theta)=p(x\mid\theta)/p(x)$ via binary cross-entropy on joint-vs-marginal
   pairs; use $r$ inside MCMC.
2–5. Same axis-I coordinate as NPE; the representation encodes the log-ratio surface.

---

# Part B — HEP foundation models, grouped by objective family

The HEP rows fix the three things general FMs leave free (branch 02 §1): the data
object is a variable-length particle set; the symmetries (permutation, IRC, Lorentz)
are exact and known; the corpus can be a simulator that emits labels and re-simulation
augmentations. Axis II (A1–A4) becomes load-bearing here.

## B.1 Supervised pretraining (HEP-only option: the simulator emits labels)

### ParT — arXiv:2202.03772 (Qu, Li, Qian; ICML 2022)
1. **Objective.** Supervised cross-entropy on JetClass (100M jets, 10 classes); not
   self-supervised. The architectural contribution is a Lorentz-invariant pairwise
   interaction matrix $U_{ij}$ from $(\ln\Delta,\ln k_T,\ln z,\ln m^2)$ injected as an
   attention bias.
2. **Data object.** Jet constituents; variable $N$; permutation-invariant (no positional
   encoding); pair features are boost/rotation invariant along the jet axis.
3. **Represents.** A discriminative latent separating the 10 truth jet flavours; the
   pair-feature bias bakes in the IRC-aware kinematic relations, so the latent encodes
   substructure relevant to flavour. Verified anchors: fine-tuned top-tagging acc 0.944 /
   AUC 0.9877; q/g acc 0.852 / AUC 0.9230; pretrained ParT 0.850 vs ParticleNet 0.837 at 10% JetClass.
4. **Reuse.** Fine-tune the backbone on a downstream tagger.
5. **Axes.** I: label-supervised / discriminative / fine-tune. II: constituents · Lorentz-inv
   features · no augmentation · detector × supervised.

### L-GATr — arXiv:2405.14806 (Spinner, Brehmer et al.; NeurIPS 2024)
1. **Objective.** Task losses (regression / classification; also a Lorentz-equivariant
   generative model via Riemannian flow matching). The defining property is in the
   *architecture*, not the loss: exact $SO^+(1,3)$ equivariance through spacetime
   geometric (Clifford) algebra in the attention.
2. **Data object.** Four-vectors in geometric-algebra multivector form.
3. **Represents.** Quantities that transform covariantly under Lorentz; invariants are
   read off equivariant features. The symmetry is hard-wired, not learned, so nothing
   Lorentz-violating can be represented — the strongest point on axis A2.
4. **Reuse.** Fine-tune / evaluate; competitive with or above ParT on tagging.
5. **Axes.** I: supervised (or a family, for the flow) / both / fine-tune. II: four-vectors ·
   exact Lorentz equivariance · none · detector × supervised.

## B.2 Contrastive (SimCLR-style NT-Xent)

NT-Xent loss for all three:
$\mathcal L=-\log\frac{\exp(\mathrm{sim}(z,z')/\tau)}{\sum_j\exp(\mathrm{sim}(z,z_j)/\tau)}$.
What differs is *where the two views come from* — the HEP content.

### JetCLR — arXiv:2108.04253 ("Symmetries, Safety, and Self-Supervision"; Dillon et al.; SciPost 2022)
1. **Objective.** NT-Xent over a permutation-invariant transformer encoder.
2. **Data object.** Jet constituents in $(\eta,\phi,p_T)$.
3. **Represents.** A space quotiented by the augmentation group: rotations/translations
   in $(\eta,\phi)$ around the $p_T$-weighted centroid plus soft and collinear smearing.
   The latent is therefore approximately invariant to exactly the IRC-safe directions —
   by construction it *discards* soft-junk and orientation, and *encodes* what survives
   IRC-safe deformation. Truth labels never enter.
4. **Reuse.** Linear classifier test on frozen features.
5. **Axes.** I: data distribution / representation / probe. II: constituents · perm + learned
   IRC-safety · hand-built physics augmentation · detector × SSL.

### RS3L — arXiv:2403.07066 (Harris, Kagan, Krupa, Maier, Woodward; PRD 111 032010)
1. **Objective.** NT-Xent over a DynamicEdgeConv GNN.
2. **Data object.** Jet constituents, with views produced by **re-simulation**: fix the
   hard process, re-run the parton shower with a different seed / FSR scale
   $\times\sqrt2,1/\sqrt2$ / alternative shower (Herwig vs Pythia).
3. **Represents.** A space invariant to the simulator's own physics-modelling
   variations — i.e. *robust to a class of systematic uncertainties*. This is the
   augmentation provenance with no analogue outside the physical sciences (A3). It
   discards shower-model-dependent detail and encodes the physics common to all
   re-simulations. Verified: fine-tuned QCD rejection 135±1 at 70% Higgs eff. vs 115±1
   fully-supervised on the same 3M; up to 10% gain on unseen W-vs-QCD.
4. **Reuse.** Fine-tune / probe.
5. **Axes.** I: data distribution / representation / fine-tune. II: constituents · perm,
   shower-robust · **re-simulation** · detector × SSL.

### SMEFT-FM — arXiv:2512.15862 (Das Bakshi, Hobbs, Kriesten; Dec 2025)
> **Correction to Phase-1.** Branch 02 cited this as "Das Bakshi, Hobbs, Kriesten 2025"
> with no arXiv id. The verified paper is arXiv:2512.15862, *"Reusable theory
> representations for colliders: a demonstrator SMEFT foundation model."* It is
> ALETHIA's nearest published neighbour and shares the physics (NC Drell-Yan SMEFT),
> so the id is recorded here.
1. **Objective.** Supervised contrastive loss over a controlled sampling of the
   Warsaw-basis dim-6 Wilson-coefficient space; the label is the Wilson configuration
   (a discrete corpus of configurations), pulling together cross sections from the same
   configuration and pushing apart others.
2. **Data object.** High-resolution *binned* differential cross sections $d\sigma$ for
   neutral-current Drell-Yan (theory-simulated), one per Wilson configuration.
3. **Represents.** A low-dimensional latent on which SMEFT deformations acquire a
   geometric structure: latent directions correlate with characteristic shape
   distortions (energy-growing four-fermion contributions; electroweak vertex
   corrections), and clusters correspond to families of Wilson configurations with
   similar phenomenological impact. It encodes the *shape* of the deformation; the
   overall normalisation and bin-level noise are quotiented.
4. **Reuse.** Inspect / cluster the latent; probe for downstream SMEFT tasks.
5. **Axes.** I: Wilson configs (discrete) / representation / clustering+probe. II: binned
   $d\sigma$ (not a particle set) · no particle symmetry · none · parton × supervised(c-label).

## B.3 Masked particle modeling (BERT-style on sets)

### MPM — arXiv:2401.13537 ("Masked particle modeling on sets …"; Golling et al.; MLST 2024)
1. **Objective.** Two stages. (a) A VQ-VAE tokenizer maps each continuous particle to a
   codebook id ($K=512$, latent dim 16; reconstruction + codebook + commitment terms,
   ~80% codebook utilisation). (b) Masked cross-entropy: replace a mask subset
   $\mathcal M$ by a learnable embedding, predict the discrete token $t_i$ from the
   visible context, $\mathcal L=\frac1{|\mathcal M|}\sum_{i\in\mathcal M}-\log\mathrm{softmax}(\ell_{\theta,i})_{t_i}$.
2. **Data object.** A particle set; permutation-invariant backbone; the masked-query
   head is ordered by $p_T$ to break the target-assignment ambiguity that identical mask
   embeddings create (the HEP-specific subtlety general masked modeling never confronts).
3. **Represents.** The conditional density of a constituent given its neighbours — a
   learned model of how partons hadronise (the codebook *is* a discretised fragmentation
   model). Verified: at 10k labels a fixed pretrained backbone reaches ~90%+ acc vs ~75%
   from-scratch; advantage persists out-of-context and cross-domain (RODEM).
4. **Reuse.** Fine-tune / linear probe / in-context.
5. **Axes.** I: data distribution / representation / fine-tune+in-context. II: particle set ·
   perm-inv ($p_T$-ordered head) · none (tokenizer) · detector × SSL.

## B.4 Generative / autoregressive (GPT-style)

### OmniJet-α — arXiv:2403.05618 (Birk, Hallin, Kasieczka; 2024)
1. **Objective.** VQ-VAE tokenize constituents, then next-token NLL
   $\mathcal L=-\sum_i\log p_\theta(t_i\mid t_{<i})$ with $p_T$-ordering supplying the
   autoregressive order.
2. **Data object.** Jet constituents as integer-token sequences.
3. **Represents.** The autoregressive factorisation of the constituent sequence — a
   learned generative model of jets. Reported as the first model to transfer between an
   unsupervised task (generation) and a supervised one (tagging), the cross-task
   demonstration that defines "foundation model" for this field.
4. **Reuse.** Transfer the backbone to a tagger; fine-tune.
5. **Axes.** I: data distribution / gen→disc / transfer. II: constituents · perm via $p_T$
   token order · none · detector × SSL.

### Aspen Open Jets — arXiv:2412.10504 (Amram et al.; MLST 2025)
1. **Objective.** Same autoregressive OmniJet-α objective, but **pretrained on real CMS
   2016 Open Data** (~178M high-$p_T$ jets), not simulation.
2. **Data object.** Real-detector jet constituents.
3. **Represents.** The generative statistics of *real* LHC jets; transfer improves
   generation under domain shift to simulated JetClass top/QCD. The contribution is the
   corpus + the real-data-pretraining demonstration.
4. **Reuse.** Fine-tune / transfer to simulated downstream tasks.
5. **Axes.** I: data distribution / generative / transfer. II: constituents (real) · perm via
   $p_T$ order · none · detector(real) × SSL.

## B.5 Multi-task supervised + generative (one backbone, both heads)

### OmniLearn — arXiv:2404.16091 (Mikuni, Nachman; 2024); OmniLearned — arXiv:2510.24066
1. **Objective.** A composite loss combining multiclass classification and a
   generation/diffusion term, on a point-edge transformer (PET) backbone. OmniLearned
   scales the same recipe past 1B jets.
2. **Data object.** Jet constituents; permutation-invariant PET.
3. **Represents.** A shared latent serving classification, generation, likelihood-ratio
   estimation, and anomaly detection, transferable across detector simulations and
   collision systems (pp vs ep). It encodes a general-purpose jet representation graded
   by the union of those tasks.
4. **Reuse.** Fine-tune / transfer across tasks and datasets.
5. **Axes.** I: label + data distribution / both / transfer. II: constituents · perm-inv PET ·
   hand-built/none · detector × supervised+gen.

## B.6 JEPA (latent-predictive, no reconstruction, no negatives)

JEPA loss for both:
$\mathcal L=\|\mathrm{pred}_\theta(h_\theta(\text{ctx}))-\mathrm{sg}[h_{\bar\theta}(\text{tgt})]\|^2+\lambda\mathcal R_{\rm VICReg}$,
EMA target encoder.

### J-JEPA — arXiv:2412.05333 (Katel, Li, Zhang et al.; 2024)
1. **Objective.** JEPA latent prediction; recluster a jet into subjets, mask some as
   targets, predict target-subjet embeddings from context-subjet embeddings conditioned
   on target positions. Augmentation-free (masks the encoder *output*, so both encoders
   see full semantic content).
2. **Data object.** Subjets as tokens.
3. **Represents.** The latent that makes target subjets linearly predictable from context
   subjets — a symmetry-independent jet representation, no hand-built augmentation.
4. **Reuse.** Linear probe / fine-tune (top tagging, q/g).
5. **Axes.** I: data distribution / representation / probe. II: subjets · perm, augmentation-free ·
   latent-only · detector × SSL.

### HEP-JEPA — arXiv:2502.03933 (Bardhan et al.; 2025)
1. **Objective.** JEPA latent prediction; patch the jet ViT-style with a particle jet
   tokeniser, mask patch embeddings, predict the masked ones from context patches.
   Pretrained on JetClass (100M).
2. **Data object.** Jet patches (groups of particles).
3. **Represents.** The latent that fills masked patch embeddings; tested on top tagging
   and q/g. Reported ~half the training cost of OmniJet-α / MPMv2.
4. **Reuse.** Linear probe / fine-tune; transfers to other datasets.
5. **Axes.** I: data distribution / representation / probe. II: jet patches · perm, latent
   masking · latent-only · detector × SSL.

---

# Part C — Where ALETHIA lands, with file:line references

ALETHIA's ManifoldInformer is a **composition**: a self-supervised event-level encoder
(foundation-model side) feeding a closed-form amortized-posterior head (SBI side). On
the master table it is the only row spanning two axis-I blocks. The construction is in
`experiments/manifold-informer/manifold_informer.py`; the geometric grading is in
`manifold_informer_gates.py`; the inverse-posterior loop is in
`experiments/full-chain-run/run.py`; the paper claims are in `paper/alethia.tex:45`.

## C.1 The encoder objective (formal)

`ManifoldInformer.forward(..., return_losses=True)` (`manifold_informer.py:329-400`)
sums four terms — this is the full pretraining risk:

- **JEPA latent prediction** (`:365`):
  `jepa_loss = ((Z_pred - V_q_target.detach())**2).mean()`,
  i.e. $\|Z_{\rm pred}(X_q)-\mathrm{sg}[h_{\bar\theta}(X_q)]\|^2$ — predict the EMA
  target encoder's embeddings of held-out query events. The predictor is *not* an MLP
  head: `Z_pred = ridge(K, V_ctx, K_q)` (`:359`), so the prediction is made by the
  closed-form Intention ridge $w_\theta=(K^\top K+\alpha I)^{-1}K^\top V$
  (`IntentionRidgeHead.w_implicit`, `:124-139`). This is JEPA with a *Bayesian-linear-
  regression in-context predictor* in place of the usual learned predictor — the CNP
  amortization (Part A.3) realised in closed form.
- **VICReg anti-collapse** (`:368`, `vicreg_loss` `:315-325`): variance hinge + squared
  off-diagonal covariance on `Z_pred`, the Bardes-Ponce-LeCun term (Part A.2), weight 0.04.
- **RS3L view-invariance** (`:371-378`): `((w1 - w2)**2).mean()` between the per-scenario
  summary vectors of two independent event sets `X_ctx`, `X_ctx_view2` at the *same* $c$ —
  the re-simulation augmentation (Part B.3 RS3L) applied at the event level, weight 0.1.
- **Optional density anchor** (`:381-384`): MSE of a tiny decoder against the per-event
  log-likelihood ratio $\log w_c(x)$, weight 0.05 — sets the *scale* of the basis the
  JEPA term shapes; can be set to 0 for pure JEPA.

So the objective family is **JEPA + VICReg + RS3L-view (+ density anchor)**, label-free
in $c$ (the Wilson coefficient never enters the forward pass; `manifold_informer.py:6`).
At 1,969 trainable parameters (`paper/alethia.tex:45`) it is ~2 orders smaller than its
own transformer-JEPA cross-check arm.

## C.2 What the representation is trained to encode — probes P1–P4

The representation is the deliverable, graded by four pre-registered geometric properties
(`manifold_informer_gates.py`), reported as medians over 15 retrained seeds
(`paper/alethia.tex:45`):

- **P1 — probe-map linearity / disclosure integrity** (`gate_P1_probe_linearity`, `:299-334`):
  the readout $\tilde c\approx W w_\theta+b$ to the Fisher-rotated leading Wilson direction
  is *linear*, with linear-probe MSE at or below a parameter-matched MLP. Measured
  $0.077 \le 0.081$ (passes on 13/15 seeds at a thin margin). The original P1
  ("$w_\theta(c)$ linear in $c$") was **withdrawn 2026-05-31** as self-contradictory with
  P4 — the morphing is quadratic in $c$, so $w_\theta$ must be nonlinear in $c$
  (`manifold_informer_gates.py:1-28`). This is a documented Phase-1→2 correction.
- **P2 — regime separation** (`gate_P2`, `:102-163`): SM-like, four-fermion-dominant, and
  vertex-dominant scenarios occupy distinct latent regions; inter/intra centroid ratio
  $4.41 > 2$.
- **P3 — tangent recovery** (`gate_P3_P4`, `:169`): $\partial w_\theta/\partial c$ at SM,
  via fixed-seed central differences, matches the analytic morphing $A_i$ up to a linear
  probe map, $R^2=0.99999$.
- **P4 — curvature recovery** (`:222-287`): second differences of $w_\theta$ in $c$ recover
  the $B_{ij}$ structure up to the probe map, $R^2=0.954$.

P3/P4 are the *manifold-identity* gates: they certify the learned latent **is** the
analytic SMEFT morphing geometry $\mu(c,m)=\mu_{\rm SM}+A_i c_i+B_{ij}c_ic_j$, not a
free-form regression. This is the field-3-of-the-five answer (what the representation
encodes): the latent is trained — via JEPA view-invariance on event sets drawn at fixed
$c$ — to recover, linearly, the tangent and curvature of the known generative polynomial.

## C.3 The amortized-posterior head and the active-learning loop

`full-chain-run/run.py` runs the inverse mode. `pretrain_intention` (`run.py:409-463`)
trains an `IntentionFM` closed-form ridge as an in-context predictor over event sets; the
loop (`run_loop`, `:479-933`) holds one target Wilson scenario in a withheld band
$|c_{\ell q}^{(3)}|\in[0.6,1.0]$, watches drift detectors fire (CUSUM, BH coverage,
$\kappa$/projection), and selects oracle queries by an acquisition rule — `epig`,
`leverage` (= BALD/D-optimal), `param_epig_d/a` (directional A-/D-optimal), or `random`
(`:733-777`). The monitored $c$̃-space trace (`_ctilde_metrics`, `:353-404`) is the
posterior entropy on the resolved subspace plus the per-direction MLE error — the
contraction-vs-MLE gap the Phase-1 memo flags. This is axis I's amortized-posterior row,
amortized over event sets / observations.

`working_point_fisher.py` and `task4_span_completeness_loop.py` add the axis-III/IV
machinery: the working-point Fisher $F_{ij}(c)=\sum_m\partial_i\mu\,\partial_j\mu/\sigma_y^2$
with $\partial_i\mu=A_i+2B_{ij}c_j$ (`working_point_fisher.py:100-128`) shows the flat
$c=0$ vertex direction lifting ~84.5× as $|c_{\ell q}^{(3)}|\to1$
(`output_task1_step2/summary.json`, `final_lift_tracked`), and the span-completeness loop
(`task4_span_completeness_loop.py`) detects an out-of-span residual direction via the
SVD fingerprint $\sigma_1>$ floor and $\sigma_1/\sigma_2>$ threshold, then **extends the
encoder basis** by appending the leading residual singular vector $u_1$ — a first-class
`psi_extend` action that drops $\sigma_1$ ×~57 at the next cycle
(`output_task4_loop/summary.json`: 5.51 → 0.096, arm `def_psi_with_extension`).

## C.4 How ALETHIA differs from every cataloged model

| Dimension | The catalog | ALETHIA |
|---|---|---|
| **Granularity (A1)** | jet constituents / subjets / patches (all HEP FMs) | **full event** — the event analogue of the constituent set; no published jet FM is event-level |
| **Objective** | one family per model (masked, contrastive, AR, JEPA, supervised) | **JEPA + VICReg + RS3L-view + density anchor combined**, with a *closed-form ridge as the JEPA predictor* (no learned predictor head) |
| **Reuse** | probe OR fine-tune OR transfer | **linear probe (P1–P4) AND a closed-form in-context amortized-posterior head** in one model — spans two axis-I blocks |
| **Level / supervision (A4)** | detector-level, SSL or simulator-supervised | **parton/analytic level, SSL**; the only parton-level SSL-JEPA |
| **Augmentation (A3)** | none / hand-built / re-simulation / latent | **re-simulation-style event views + latent** at event level |
| **What it grades** | linear-classifier-test accuracy/AUC | **geometric identity** with the known morphing manifold (P3 $R^2{=}0.99999$, P4 $R^2{=}0.954$) — certifies the latent *is* the analytic $A_i,B_{ij}$ geometry, not just separable |
| **Extensibility** | fixed backbone | **$\psi$-extension as a first-class AL action** (residual-SVD direction appended when a span-completeness fingerprint fires) |
| **Nearest neighbour** | SMEFT-FM (2512.15862): supervised contrastive, 100 discrete Wilson universes, binned $d\sigma$ | ALETHIA: **continuous** Wilson prior, **JEPA** objective, **event-set** input, **regression-posterior** head, **extensibility** |

Net placement: on the jet-centric HEP-FM map ALETHIA is the **event-level, parton-level,
SSL-JEPA, amortized-posterior-head** corner — a cell no verified jet FM occupies — and on
the general axis I it is the unique self-supervised-encoder ∘ inverse-posterior-head
composition, graded not by an LCT but by linear recovery of a known generative manifold's
tangent and curvature.

---

## References (web-verified this session, June 2026)

- ParT — Qu, Li, Qian, "Particle Transformer for Jet Tagging," arXiv:2202.03772 (ICML 2022).
- JetCLR — Dillon, Kasieczka, Olischläger, Plehn, Sorrenson, Vogel, "Symmetries, Safety, and Self-Supervision," arXiv:2108.04253 (SciPost Phys. 2022).
- RS3L — Harris, Kagan, Krupa, Maier, Woodward, "Re-Simulation-based Self-Supervised Learning for Pre-Training Foundation Models," arXiv:2403.07066 (Phys. Rev. D 111, 032010).
- MPM — Golling, Heinrich, Kagan, Klein, Leigh, Osadchy, Raine, "Masked particle modeling on sets: towards self-supervised high energy physics foundation models," arXiv:2401.13537 (Mach. Learn. Sci. Technol. 2024).
- OmniJet-α — Birk, Hallin, Kasieczka, "OmniJet-α: The first cross-task foundation model for particle physics," arXiv:2403.05618 (2024).
- Aspen Open Jets — Amram et al., "Aspen Open Jets: Unlocking LHC Data for Foundation Models in Particle Physics," arXiv:2412.10504 (Mach. Learn. Sci. Technol. 2025).
- OmniLearn — Mikuni, Nachman, "OmniLearn: A Method to Simultaneously Facilitate All Jet Physics Tasks," arXiv:2404.16091 (2024).
- OmniLearned — "OmniLearned: A Foundation Model Framework for All Tasks Involving Jet Physics," arXiv:2510.24066 (2025).
- L-GATr — Spinner, Bresó, de Haan, Plehn, Thaler, Brehmer, "Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics," arXiv:2405.14806 (NeurIPS 2024).
- J-JEPA — Katel, Li, Zhang et al., "Learning Symmetry-Independent Jet Representations via Jet-Based Joint Embedding Predictive Architecture," arXiv:2412.05333 (2024).
- HEP-JEPA — Bardhan et al., "HEP-JEPA: A foundation model for collider physics using joint embedding predictive architecture," arXiv:2502.03933 (2025).
- SMEFT-FM — Das Bakshi, Hobbs, Kriesten, "Reusable theory representations for colliders: a demonstrator SMEFT foundation model," arXiv:2512.15862 (2025). [Phase-1 arXiv id was missing; supplied here.]
- BERT — Devlin, Chang, Lee, Toutanova, arXiv:1810.04805 (NAACL 2019).
- MAE — He, Chen, Xie, Li, Dollár, Girshick, arXiv:2111.06377 (2021).
- I-JEPA — Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," arXiv:2301.08243 (CVPR 2023).
- VICReg — Bardes, Ponce, LeCun, arXiv:2105.04906 (ICLR 2022).
- CNP — Garnelo et al., "Conditional Neural Processes," arXiv:1807.01613 (ICML 2018).
- FNO — Li et al., arXiv:2010.08895 (ICLR 2021).
- DeepONet — Lu et al., arXiv:1910.03193 (Nat. Mach. Intell. 2021).
- NPE/MAF/APT — Papamakarios-Murray arXiv:1605.06376; Papamakarios-Pavlakou-Murray arXiv:1705.07057; Greenberg-Nonnenmacher-Macke arXiv:1905.07488.
- NRE — Hermans, Begy, Louppe, arXiv:1903.04057 (ICML 2020).
- Bommasani et al., "On the Opportunities and Risks of Foundation Models," arXiv:2108.07258 (2021).
- Cranmer, Brehmer, Louppe, "The frontier of simulation-based inference," arXiv:1911.01429 (PNAS 2020).
- Deep Sets — Zaheer et al., arXiv:1703.06114 (NeurIPS 2017). EFN/PFN — Komiske, Metodiev, Thaler, arXiv:1810.05165 (2019).
