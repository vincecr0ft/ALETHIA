# Foundation models in high-energy physics

Branch 02 of the ALETHIA FM taxonomy. Siblings: 01 (FM/surrogate, general), 03
(information geometry & Lagrangian morphing), 04 (active learning). This branch
isolates what changes when "foundation model" is instantiated on **collider
data** rather than on text, images, or generic simulator surrogates.

The organising claim is that HEP fixes three things the general FM literature
leaves free, and these three choices generate the design axes below:

1. **The data object is a variable-length set (point cloud) of particles**, not a
   dense grid or a fixed-length token sequence.
2. **Physical symmetries are exact and known a priori**: permutation invariance
   of the constituent set, and (channel-dependent) Lorentz invariance and
   infrared-and-collinear (IRC) safety.
3. **The "unlabeled corpus" can be a high-fidelity stochastic simulator**, so
   labels (parton truth) and physically meaningful augmentations (re-simulation)
   are available in a way they are not for natural data.

All papers cited below were located and verified by title + authors + arXiv id;
the verified inventory is in the table in §5 and the reference list in §7.

---

## 1. The data object and its symmetries

### 1.1 Jets and events as sets

A jet (or an event) is a set of reconstructed particles

$$
\mathcal{X} \;=\; \{\, x_i \,\}_{i=1}^{N}, \qquad x_i \in \mathbb{R}^{F},
\qquad N \text{ variable across examples.}
$$

Each particle carries kinematics and (optionally) identity. The two standard
feature parametrisations are:

- **kinematic**: transverse momentum and angular coordinates relative to the jet
  axis, $(\,\log p_T,\ \Delta\eta = \eta_i - \eta_{\text{jet}},\ \Delta\phi,\
  \log E,\ \log(p_T/p_T^{\text{jet}}),\ \Delta R\,)$;
- **identity / trajectory**: charge, particle-ID one-hot, impact parameters
  $d_0, d_z$ and their significances (the inputs that make $b$-tagging possible).

The cardinality $N$ ranges from a handful to a few hundred constituents per jet,
and into the thousands for full events. A model on $\mathcal{X}$ must therefore
accept variable-length input and respect that the *set* carries no intrinsic
order.

### 1.2 Permutation invariance (exact, always)

There is no canonical ordering of the constituents. A representation function
$f$ and a per-set summary $g$ must satisfy, for any permutation $\pi \in S_N$,

$$
g\big(\{x_{\pi(1)}, \dots, x_{\pi(N)}\}\big) \;=\; g\big(\{x_1, \dots, x_N\}\big).
$$

Two constructions dominate. **Deep Sets** (Zaheer et al. 2017, arXiv:1703.06114)
proves that any permutation-invariant continuous set function admits the
decomposition

$$
g(\mathcal{X}) \;=\; \rho\!\Big(\, \textstyle\sum_{i=1}^{N} \phi(x_i)\,\Big),
$$

with $\phi$ a per-particle map and $\rho$ acting on the pooled sum. **Energy Flow
Networks / Particle Flow Networks** (Komiske, Metodiev, Thaler 2019,
arXiv:1810.05165) is the HEP specialisation: the per-particle map is weighted by
energy, $g(\mathcal{X}) = \rho(\sum_i z_i\, \phi(\hat p_i))$ with $z_i$ the energy
fraction, which makes IRC safety expressible (see §1.4). Transformer encoders
(ParT, MPM, the JEPA models) achieve permutation invariance instead by using
**no positional encoding**, so self-attention is permutation-equivariant and a
pooling token or mean restores invariance.

### 1.3 Lorentz structure

Constituent four-momenta live in Minkowski space; many observables are invariant
under the proper orthochronous Lorentz group $SO^+(1,3)$ (boosts + rotations).
Architectures differ in how much of this they hard-wire. The strongest is
**L-GATr** (Lorentz-Equivariant Geometric Algebra Transformer, Spinner et al.
2024, arXiv:2405.14806; Brehmer et al.), which builds equivariance into the
attention via spacetime geometric algebra and is competitive with or above ParT
on tagging. Most jet FMs instead present already-Lorentz-*invariant* features
(the $(\Delta, k_T, z, m^2)$ pair features of ParT are boost/rotation invariant
along the jet axis) and learn the rest.

### 1.4 IRC safety

An observable is **IRC safe** if it is unchanged under (i) an infinitely soft
emission $z \to 0$ and (ii) a collinear splitting of one particle into two with
the same total momentum. Energy-weighting (EFN form, §1.2) and the soft/collinear
data augmentations in JetCLR (§2.2) are the two ways HEP FMs build in or learn
this safety. It is the property that separates a physically sensible jet
representation from one that overfits to soft junk.

---

## 2. Pretraining objectives on point-cloud event data

Let $h_\theta$ be the (permutation-respecting) backbone producing per-particle
tokens $h_i = h_\theta(\mathcal{X})_i$ and a pooled summary $z = \text{pool}(h)$.
The HEP-FM landscape uses four objective families.

### 2.1 Masked particle modeling (BERT-style) — formal

This branch's worked-out objective. **Masked Particle Modeling (MPM)**, Golling,
Heinrich, Kagan, Klein, Leigh, Osadchy, Raine 2024 (arXiv:2401.13537,
*Mach. Learn. Sci. Technol.* 2024). Stages:

**(a) Tokenizer.** A vector-quantised VAE is trained first to map each continuous
particle to a discrete codebook id. With encoder $e$, codebook
$\mathcal{C} = \{c_k\}_{k=1}^{K}$ ($K = 512$, latent dim 16 in the paper), the
token of particle $x_i$ is

$$
t_i \;=\; \arg\min_{k}\, \big\| e(x_i) - c_k \big\|_2,
\qquad t_i \in \{1,\dots,K\}.
$$

The VQ-VAE is trained with reconstruction + codebook + commitment terms
$\mathcal{L}_{\text{VQ}} = \|x - D(c_{t})\|^2 + \|\,\text{sg}[e(x)] - c_t\|^2
+ \beta \,\|e(x) - \text{sg}[c_t]\|^2$, $\text{sg}$ the stop-gradient; the paper
reports $\sim 80\%$ codebook utilisation.

**(b) Masking.** Sample a mask subset $\mathcal{M} \subset \{1,\dots,N\}$
(uniformly, fraction $\sim 0.4$–0.7). Replace masked particles by a learnable
mask embedding; keep the unmasked set $\mathcal{U} = \mathcal{X}\setminus\mathcal{M}$.

**(c) Objective.** Predict the token of each masked particle from the visible
context. With backbone $h_\theta$ and a $K$-way classification head producing
logits $\ell_{\theta,i}\in\mathbb{R}^K$, minimise the masked cross-entropy

$$
\mathcal{L}_{\text{MPM}}(\theta)
\;=\;
\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}}
\Big[\,-\log \mathrm{softmax}\big(\ell_{\theta,i}\big)_{\,t_i}\,\Big],
\qquad
\ell_{\theta,i} = W\, h_\theta\!\big(\mathcal{U}, \mathcal{M}\big)_i .
$$

**(d) The permutation subtlety.** Because identical mask embeddings are inserted,
a strictly permutation-*equivariant* backbone cannot tell two masked slots apart,
so the *targets* become ambiguous (any assignment of the multiset of masked
tokens is equally consistent). MPM resolves this by **ordering masked queries by
$p_T$ at the prediction head only**, leaving the backbone permutation invariant.
This is the cleanest illustration of a HEP-specific subtlety the general
masked-modeling literature never confronts.

The demo (`demo.py`) implements exactly (a)–(c) on toy jets, with $p_T$ ordering
of the masked head per (d).

### 2.2 Contrastive (SimCLR-style)

Map two augmented views $\mathcal{X}, \mathcal{X}'$ of the same jet to embeddings
$z, z'$ and pull them together against a batch of negatives with the InfoNCE /
NT-Xent loss

$$
\mathcal{L}_{\text{NT-Xent}}
= -\log
\frac{\exp\!\big(\mathrm{sim}(z, z')/\tau\big)}
     {\sum_{j}\exp\!\big(\mathrm{sim}(z, z_j)/\tau\big)},
\qquad
\mathrm{sim}(a,b) = \frac{a^\top b}{\|a\|\,\|b\|}.
$$

The HEP content is **where the views come from**:
- **JetCLR** (Dillon, Kasieczka, Olischläger, Plehn, Sorrenson, Vogel 2021,
  arXiv:2108.04253, *SciPost*) uses hand-built physics augmentations:
  rotations and translations in $(\eta,\phi)$, plus **soft** and **collinear**
  smearings, so the learned space is invariant exactly to the IRC-safe directions.
- **RS3L** (Harris, Kagan, Krupa, Maier, Woodward 2024, arXiv:2403.07066,
  *Phys. Rev. D* 111 032010) replaces hand-built augmentations by
  **re-simulation**: fix the upstream generated event, re-run the parton shower
  with a different seed, FSR scale $\times\sqrt 2,\,1/\sqrt2$, or an alternative
  shower (Herwig vs. Pythia). The augmentations are then the simulator's own
  physics uncertainties, which is unavailable outside the physical sciences.

### 2.3 Generative / autoregressive (GPT-style)

**OmniJet-α** (Birk, Hallin, Kasieczka 2024, arXiv:2403.05618) tokenizes jet
constituents and trains a next-token model

$$
\mathcal{L}_{\text{AR}} = -\sum_{i=1}^{N}\log p_\theta\big(t_i \,\big|\, t_{<i}\big),
$$

with $p_T$-ordering supplying the autoregressive order. It is reported as the
first model to transfer between an unsupervised task (generation) and a
supervised task (tagging) — the cross-task demonstration that defines "foundation
model" for this field.

### 2.4 JEPA (latent-predictive, no reconstruction, no negatives)

Predict the *embedding* of a masked/target region from a context region, in
latent space, with an exponential-moving-average target encoder $h_{\bar\theta}$:

$$
\mathcal{L}_{\text{JEPA}}
= \big\|\, \mathrm{pred}_\theta\big(h_\theta(\text{context})\big)
        - \mathrm{sg}\big[\,h_{\bar\theta}(\text{target})\,\big]\big\|^2
\;+\; \lambda\,\mathcal{R}_{\text{VICReg}},
$$

with a variance-covariance regulariser $\mathcal{R}_{\text{VICReg}}$ (Bardes,
Ponce, LeCun 2022, arXiv:2105.04906) to stop collapse. HEP instances:
**J-JEPA** (Katel, Li, Zhang et al. 2024, arXiv:2412.05333) reclusters a jet into
subjets and predicts target-subjet embeddings from context subjets; **HEP-JEPA**
(Bardhan et al. 2025, arXiv:2502.03933) patches the jet ViT-style and predicts
target-patch embeddings, reporting roughly half the training cost of OmniJet-α
and MPMv2. ALETHIA's own ManifoldInformer (§6) is a JEPA at the **event** level.

---

## 3. What the representation represents

The pretraining objective fixes the *content* of the learned space:

- **Masked / generative** objectives learn the **conditional density of a
  constituent given its neighbours** — i.e. the fragmentation/shower statistics.
  The token codebook (MPM) or the autoregressive factorisation (OmniJet-α)
  literally is a learned model of how partons hadronise.
- **Contrastive** objectives learn a space **quotiented by the augmentation
  group**: JetCLR's space is invariant to rotations/translations and IRC-safe
  smearing by construction; RS3L's is invariant to shower-model and FSR
  variations, which is exactly a representation **robust to a class of systematic
  uncertainties**.
- **JEPA** objectives learn whatever latent makes target regions linearly
  predictable from context — empirically a smoother, lower-dimensional manifold
  than reconstruction, validated by linear probes rather than by sample fidelity.

The downstream test in every case is the **linear classifier test (LCT)**: freeze
the backbone, fit a linear head on a small labeled set, and read accuracy/AUC. A
good representation makes the physics task linearly separable with few labels.
The demo measures exactly the LCT gap.

---

## 4. Two orthogonal level distinctions

**Detector-level vs parton-level.** The input can be reconstructed
detector objects (tracks, calorimeter clusters, particle-flow candidates — what an
experiment measures, what OmniLearned's CMS/ATLAS-simulation arms use) or
parton/truth-level objects (what the theory predicts, used by analytic and
generator-truth studies). A detector-level FM must additionally absorb the
detector response; a parton-level FM is closer to the Lagrangian. ALETHIA
operates at parton/analytic level (§6).

**Supervised-pretrain vs SSL.** ParT and OmniLearn(ed) pretrain on **labeled**
classification (truth jet flavour) and transfer the backbone; MPM, OmniJet-α,
JetCLR, RS3L, J-JEPA, HEP-JEPA pretrain **self-supervised** with no truth labels.
HEP is unusual in that supervised pretraining is even an option at FM scale,
because the simulator hands out free labels. This is the axis where HEP and the
general FM literature most sharply diverge.

**Generative vs discriminative.** Orthogonal again: OmniJet-α and the generative
arm of OmniLearn produce samples; ParT, JetCLR, MPM-as-classifier are
discriminative. A single backbone can serve both (the OmniJet-α / OmniLearn
cross-task claim).

---

## 5. Taxonomy table of verified HEP-FM design choices

| Model (arXiv) | Data object | Backbone / invariance | Pretraining objective | Supervision | Gen/Disc | Symmetry handling |
|---|---|---|---|---|---|---|
| **ParT** (2202.03772) | jet constituents | transformer, no pos-enc → perm-inv | supervised classification on JetClass (100M) | supervised | disc | Lorentz-inv pair features $(\ln\Delta,\ln k_T,\ln z,\ln m^2)$ as attention bias |
| **JetCLR** (2108.04253) | jet constituents | transformer encoder | contrastive (NT-Xent) | SSL | disc | hand-built rot/trans + soft/collinear (IRC) augmentations |
| **RS3L** (2403.07066) | jet constituents | DynamicEdgeConv GNN | contrastive (NT-Xent) | SSL | disc | re-simulation augmentations (shower/FSR variations) |
| **MPM** (2401.13537) | particle set | Normformer transformer (40M) | masked token cross-entropy over VQ-VAE codebook | SSL | disc | perm-inv backbone, $p_T$-ordered mask head |
| **OmniJet-α** (2403.05618) | jet constituents | transformer, no pos-enc | autoregressive next-token | SSL | gen→disc | perm handled by $p_T$ token order |
| **OmniLearn / OmniLearned** (2404.16091 / 2510.24066) | jet constituents | point-edge transformer (PET) | hybrid: classification + generation/diffusion | supervised + gen | both | perm-inv PET; >1B jets in OmniLearned |
| **L-GATr** (2405.14806) | four-vectors | geometric-algebra transformer | supervised (also generative variants) | supervised | both | exact Lorentz equivariance in architecture |
| **J-JEPA** (2412.05333) | subjets | transformer | JEPA latent prediction | SSL | disc | subjet tokens, no hand-built augmentation |
| **HEP-JEPA** (2502.03933) | jet patches | ViT-style transformer | JEPA latent prediction | SSL | disc | patch tokens, latent masking |
| **Aspen Open Jets** (2412.10504) | jet constituents (CMS Open Data, 180M) | OmniJet-α backbone | autoregressive (pretrain on real data) | SSL | gen | dataset contribution; real-data pretraining |
| **ALETHIA / ManifoldInformer** (this repo) | **event-level** kinematics $\{x_n\}$ | Deep Sets / EFN pool → closed-form ridge | JEPA latent + VICReg + RS3L views | SSL | disc (regression posterior) | perm-inv event encoder; parton/analytic level |

Verified quantitative anchors used elsewhere in this branch:
- ParT fine-tuned top-tagging accuracy 0.944, AUC 0.9877; q/g accuracy 0.852,
  AUC 0.9230; with 10% of JetClass, pretrained ParT 0.850 acc vs 0.837 ParticleNet.
- MPM: at 10k labeled samples a fixed pretrained backbone reaches ~90%+ accuracy
  vs ~75% from-scratch (in-context); advantage persists out-of-context and
  cross-domain (RODEM).
- RS3L: fine-tuned on 3M labeled, QCD rejection 135±1 at 70% Higgs efficiency vs
  115±1 fully-supervised on the same 3M; up to 10% gain on unseen W-vs-QCD.

---

## 6. Where ALETHIA sits on this map

ALETHIA (paper `paper/alethia.tex`; experiments `experiments/manifold-informer/`,
`full-chain-run/`, `al-phoenix-studies/`) is a HEP foundation model, and the map
above locates it precisely:

- **Data object: event-level, not jet-level.** The input is an unordered set of
  per-event kinematic vectors $\{x_n\}$ — $(\log m_{\ell\ell}, \cos\theta^\star)$
  for neutral-current Drell-Yan — pooled permutation-invariantly. It is the
  *event* analogue of the constituent-set object that ParT/MPM use at *jet* level.
  The "point cloud" the encoder receives is rendered in
  `paper/figures/event_pointclouds.png`.
- **Symmetry: permutation invariance via Deep Sets / EFN** (Zaheer 2017; Komiske
  2019) — the same two constructions as §1.2. No Lorentz layer; the relevant
  structure is the SMEFT morphing in $(m_{\ell\ell}, \cos\theta^\star)$.
- **Objective: JEPA latent prediction** (LeCun 2022; Assran I-JEPA 2023) with a
  **VICReg** anti-collapse regulariser (Bardes 2022) and **RS3L** re-simulation
  views (Harris 2024). This places it in §2.4 alongside J-JEPA and HEP-JEPA,
  but at the event level and feeding a closed-form ridge head rather than a
  classifier.
- **Supervision: SSL**; **level: parton/analytic** (LO morphing + MadGraph
  cross-check), not detector-level.
- **What it adds beyond the jet-FM cluster:** the representation is the
  deliverable, validated by *geometric* probes (P1 linear-probe recoverability of
  the leading Wilson direction; P2 regime separation; P3/P4 tangent and curvature
  recovery against the morphing coefficients $A_i, B_{ij}$ at $R^2 = 0.99999$ /
  $0.954$), and the encoder is **extensible inside an active-learning loop**
  ($\psi$-extension appends a residual-SVD direction $\mathbf{u}_1$ when a
  span-completeness fingerprint fires; leading singular value drops ×78 at the
  next cycle). Its nearest published neighbour is the SMEFT contrastive model of
  Das Bakshi, Hobbs, Kriesten 2025 (supervised contrastive over 100 discrete
  Wilson universes); ALETHIA differs by a continuous Wilson prior, a JEPA
  objective, a regression posterior, and the extensibility action.

So on the jet-centric HEP-FM map, ALETHIA is the **event-level, parton-level,
SSL-JEPA, regression-head** corner — a corner none of the verified jet FMs
occupy.

---

## 7. Taxonomy contribution: the axes this branch adds

The general FM/surrogate branch (01) supplies axes like {pretrain objective,
scale, fine-tune vs frozen}. This HEP branch adds **four axes that only exist
once the data object is a physical particle set**:

- **A1 — Data object granularity.** {jet constituents | subjets/patches |
  full event | calorimeter shower}. Determines $N$, the symmetry group, and what
  "a token" is.
- **A2 — Symmetry budget.** How much known physical symmetry is *hard-wired*
  vs *learned*: {permutation only | + IRC-safe augmentation | + Lorentz-invariant
  features | + exact Lorentz equivariance (L-GATr)}. This is a HEP-specific axis
  because the symmetries are exact and known a priori.
- **A3 — Augmentation provenance.** {none/generative | hand-built physics
  (JetCLR) | simulator re-simulation (RS3L) | latent-only (JEPA)}. The
  re-simulation option — augmentations drawn from the simulator's own physics
  uncertainties — has no analogue outside the physical sciences and directly buys
  systematic-uncertainty robustness.
- **A4 — Reconstruction level & supervision source.** {parton/truth vs
  detector-level} × {free simulator labels (supervised pretrain) vs SSL}. HEP is
  the regime where supervised pretraining at FM scale is even possible, because
  the simulator emits labels for free.

These four axes, crossed with the objective family (§2), are sufficient to place
every verified model in the §5 table and to locate ALETHIA as the event-level
parton-level SSL-JEPA regression corner.
