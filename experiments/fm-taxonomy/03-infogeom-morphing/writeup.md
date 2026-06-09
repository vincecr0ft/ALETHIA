# Information geometry and Lagrangian morphing, and their relationship to manifold learning

Branch 03 of the ALETHIA foundation-model taxonomy. Sibling branches:
01 (FM/surrogate, general), 02 (FM in HEP), 04 (active learning).

The question this branch isolates: a Lagrangian-morphing model gives an
**exact, structured, low-dimensional** parametrization of the manifold of
physics predictions, derived from a *known* polynomial structure in the
couplings; a learned manifold / foundation model must **discover** that
structure from data. Below: the morphing algebra formally, the Fisher-metric
geometry of the resulting coupling manifold, the manifold-learning connection,
and the taxonomy contribution. A runnable demo (`demo.py`) and its output
(`results.md`) instantiate every claim.

---

## 1. Lagrangian / EFT morphing: the algebra

### 1.1 Where the polynomial comes from

An EFT (or any "characterisation") Lagrangian is the SM plus a finite set of
operators with free couplings,
`L = L_SM + sum_a (c_a / Lambda^{d-4}) O_a`. Each amplitude is linear in the
couplings,

```
A(g) = sum_v g_v A_v ,            g = (g_SM, c_1, ..., c_n) ,
```

where the sum runs over the interaction vertices `v` that the process can use.
The observable — a cross section, a differential rate, or a single histogram
bin — is the squared amplitude integrated over phase space, hence a
**homogeneous quadratic form** in the couplings:

```
sigma(g) = sum_{v,v'} g_v g_{v'} * I_{v v'} ,    I_{v v'} = integral A_v* A_{v'} .
```

For a single coefficient `c` added to the SM this collapses to the familiar
three-term decomposition

```
sigma(c) = T0 + c T1 + c^2 T2          (SM-only, interference, BSM-squared),     (1)
```

which is exactly the `sm_only / interference / bsm_squared` triple that
ALETHIA's analytic SMEFT calculator returns
(`modules/analytic_smeft/smeft.py`). The key physical content: when an operator
carries energy growth (the four-fermion contact term `~ M^2/Lambda^2`), `T1`
and `T2` are **not** proportional to `T0` — the basis is genuinely
multi-dimensional, which is what makes morphing informative rather than a
rescaling.

### 1.2 The morphing matrix and the exact interpolation

The polynomial coefficients `T_i` are unknown a priori; what is cheap is to
*sample* `sigma` at a set of benchmark couplings `{g^{(1)}, ..., g^{(N)}}` (one
Monte-Carlo campaign per benchmark). Collect the coupling monomials of the
quadratic form into a row vector `v(g)`; the **morphing matrix** stacks these
rows over the benchmarks:

```
M_{i j} = v_j(g^{(i)}) ,          v(g) = ( monomials g_v g_{v'} ) .
```

A new prediction is the polynomial re-evaluated at the query `g`, expressed
through the samples by inverting `M`:

```
sigma(g) = v(g) M^{-1} ( sigma(g^{(1)}), ..., sigma(g^{(N)}) )^T
         = sum_i w_i(g) sigma(g^{(i)}) ,     w(g) = v(g) M^{-1} .                 (2)
```

The morphing weights `w_i(g)` are themselves polynomials in `g`; the morphed
prediction reproduces `sigma(g)` **exactly at every `g`** (interpolating or
extrapolating), provided `M` is invertible and the sample count meets the
minimum. This is moment morphing (Baak, Gadatsch, Harrington & Verkerke,
arXiv:1410.7388) and its EFT generalisation (Balasubramanian, Brenner, Burgard
& Verkerke, arXiv:2202.13612). Equivalently, `M^{-1}` applied to the sampled
predictions *recovers the physical templates* `T_i` themselves — the demo shows
this to `~1e-14`.

### 1.3 Minimal sample count

`M` must be square and invertible, so `N` equals the number of monomials in the
quadratic form. For `n` BSM couplings (arXiv:2202.13612, Eq. 21):

```
linear (interference only):   N = n + 1 ,
linear + quadratic:           N = (n^2 + 3 n + 2) / 2 = C(n+2, 2) .                (3)
```

At `n = 1`, quadratic order, `N = 3` — the minimal basis the demo uses. The
count is the dimension of the space `Sym^2` of the coupling vector; with
separate production and decay vertices it multiplies combinatorially, which is
the practical reason morphing matters (it replaces an exponential grid of MC
campaigns with the minimal polynomial basis).

### 1.4 What morphing is, geometrically

The family `{sigma(g, .)}` traced out as `g` varies is a **Veronese variety**:
the image of the coupling space under the degree-2 monomial map `g -> v(g)`.
It is an algebraic manifold of dimension `n` embedded in the `N`-dimensional
template space, and morphing is the statement that the prediction lives in the
linear span of `N` samples — an *exact, finite, structured* low-rank
representation with no learning.

---

## 2. Information geometry of the coupling manifold

### 2.1 Fisher metric

Equip the coupling space with a likelihood `p(data | g)`. The Fisher
information matrix

```
F_{ij}(g) = E[ (d_i log p)(d_j log p) ]                                            (4)
```

is the natural Riemannian metric on the statistical manifold of distributions
(Rao 1945, Bull. Calcutta Math. Soc. 37:81–91; Amari, *Information Geometry and
Its Applications*). For the additive-Gaussian observation model `Y = sigma(g) +
N(0, sigma_y^2)` used throughout ALETHIA's Fisher studies
(`working_point_fisher.py`), it reduces to a Gram matrix of prediction
gradients,

```
F_{ij}(g) = sum_x (d_i sigma)(d_j sigma) / sigma_y^2 ,                             (5)
```

and the gradient is **read straight off the morphing polynomial**:

```
d_i sigma(g) = A_i(x) + 2 sum_j B_{ij}(x) c_j ,                                    (6)
```

with `A_i` the interference templates and `B_{ij}` the quadratic templates.
This is *the same equation* ALETHIA documents as "Eq. 1" of its work-point
handoff. The morphing algebra and the Fisher geometry are not two topics — the
templates that define the morphing *are* the tangent (`A_i`) and curvature
(`B_{ij}`) data of the metric.

### 2.2 Cramér–Rao, natural gradient, degenerate working points

Three geometric consequences, all exercised in the demo:

- **Cramér–Rao** (Rao 1945; Cramér, *Mathematical Methods of Statistics*,
  1946): any unbiased estimator obeys `Cov(ĝ) >= F^{-1}`. The metric *is* the
  attainable-precision floor; flat directions of `F` are unmeasurable
  directions of `g`.
- **Natural gradient** (Amari, *Natural Gradient Works Efficiently in
  Learning*, Neural Comput. 10(2):251–276, 1998): the reparametrization
  invariant steepest-descent direction is `F^{-1} grad`, and the Fisher arc
  length `s(g) = integral sqrt(g^T F g)` is the information-distance coordinate.
  In 1-D the demo's `s(c) = integral sqrt(F) dc` makes equal-information steps
  equal-length.
- **Degenerate working points.** Where `d sigma/dg` shrinks, `F` collapses and
  the Cramér–Rao bound diverges. In the demo this is the closed-form
  least-informative coupling `c* = -sum(a_BSM^3 a_SM)/sum(a_BSM^4)`. This is the
  geometric origin of ALETHIA's *flat-vertex-direction* finding: at `c = 0` the
  vertex operators only rescale the SM curve, so their Fisher eigenvalues sit at
  the prior floor; moving the working point along an energy-growing direction
  lifts them. The flatness is a property of *where you sit in `g`*, dictated by
  the morphing templates — not something a model discovers.

---

## 3. Relationship to manifold learning

The prediction family `{sigma(g, .)}` is a low-dimensional manifold in
function/observable space. Two epistemically opposite ways to obtain it:

| | structured (morphing) | empirical (learned) |
|---|---|---|
| **dimension** | known a priori: `dim = n`, span `= N = C(n+2,2)` | estimated from the data's singular spectrum (PCA), diffusion-operator spectrum, or bottleneck width |
| **coordinates** | the couplings `g`; physically interpretable templates `T_i` | abstract latent axes, identified only up to rotation/nonlinearity |
| **fit** | none — exact polynomial interpolation, `M^{-1}` | optimisation over a sampled dataset |
| **off-support** | exact (it is a polynomial) | extrapolation error grows |
| **method refs** | moment morphing 1410.7388; EFT morphing 2202.13612 | PCA (Hotelling 1933); Isomap (Tenenbaum, de Silva & Langford, Science 2000); diffusion maps (Coifman & Lafon, Appl. Comput. Harmon. Anal. 21:5–30, 2006); autoencoders |

The demo's part [7] makes the connection concrete: PCA/SVD of the sampled
rate-curves returns numerical rank 3 (= `N` = degree + 1), and rank 2 after
mean-centering. The **learned** manifold *recovers* exactly the dimension the
**morphing** algebra *asserts*. They are the same manifold; morphing knows its
equation, learning measures its shadow.

This is the bridge to ALETHIA's ManifoldInformer. That model is a JEPA-trained,
permutation-invariant per-event encoder whose closed-form ridge head produces a
per-scenario manifold coordinate `w_theta(c)` (see
`experiments/manifold-informer/manifold_informer.py`). It never sees the Wilson
coefficient; it must discover the prediction manifold from event sets. The
paper's pre-registered gates are precisely tests that the *learned* latent
geometry matches the *morphing* geometry derived here: P3 recovers the tangent
`A_i` (linear-probe `R^2 = 0.99999`) and P4 the curvature `B_{ij}`
(`R^2 = 0.954`). Morphing is the analytic answer key; the FM is graded against
it.

A neural surrogate also adds information geometry *of its own parameters* —
Amari's natural-gradient analysis of deep networks (Amari, Karakida et al.,
arXiv:1808.07172) lives on the *model's* parameter manifold, a different
statistical manifold from the *physics* coupling manifold of §2. Keeping these
two Fisher metrics distinct (parameters-of-the-physics vs
parameters-of-the-network) is a recurring source of confusion the taxonomy
should flag.

A complementary HEP line makes the gradient `d_i log p` itself the learning
target: the score `t(x|g) = grad_g log p(x|g)` is the locally sufficient
statistic, and regressing it ("mining gold") yields the Fisher information
directly from simulated events (Brehmer, Cranmer, Louppe & Pavez, *A Guide to
Constraining Effective Field Theories with Machine Learning*, arXiv:1805.00020;
MadMiner, Brehmer, Kling, Espejo & Cranmer, arXiv:1907.10621). Score regression
is the learned counterpart of reading `d_i sigma` off the morphing polynomial —
the same geometric object, obtained by fit rather than by algebra.

---

## 4. Taxonomy contribution

This branch adds the following classification axes to the ALETHIA FM taxonomy.

**Axis A — provenance of the prediction manifold.**
*Structured/exact* (morphing: dimension and span fixed by the Lagrangian's
polynomial degree; `M^{-1}` interpolation; zero fit) ↔ *empirical/learned*
(PCA, diffusion maps, autoencoders, JEPA FM: dimension estimated, coordinates
non-interpretable, error off-support). ALETHIA's ManifoldInformer is the
learned pole; the analytic SMEFT oracle is the structured pole.

**Axis B — basis cost and exactness.**
Morphing needs the *provably minimal* `N = C(n+2, 2)` exact samples and
reproduces the family at every `g`. A learned model needs many more samples,
amortises across scenarios, and degrades away from support. The trade is
exactness-for-cheap-in-`n` (morphing, but `N` grows polynomially in operators)
vs. amortisation-for-data (FM).

**Axis C — geometry: where the metric comes from.**
The Fisher metric (and its degenerate, low-sensitivity working points) is, for
morphing, *computed in closed form* from the same templates `(A_i, B_{ij})`
that define the interpolation. For a learned model it must be *probed* from the
latent representation (ALETHIA P3/P4). Two distinct statistical manifolds live
here and must not be conflated: the **physics coupling manifold** (Fisher of
`g`) and the **network parameter manifold** (Fisher of `theta`, natural-gradient
training).

**Axis D — interpretability of coordinates.**
Morphing coordinates are the physical couplings and amplitude-interference
templates; learned coordinates are identified only up to symmetry and require a
probe (linear or otherwise) to map back to physics. This is the axis on which a
foundation model "pays" for not knowing the Lagrangian.

The unifying statement the branch contributes: **morphing is the analytic limit
of manifold learning when the generative polynomial structure is known.** A
physics FM is worth building precisely in the regime where that structure is
*not* fully known, *not* cheap to sample over all operators, or must be inferred
from event-level data rather than cross-section templates — and its latent
geometry should be validated against the morphing/Fisher answer key wherever
that key exists.

---

## References (verified: title, authors, arXiv/journal id)

- M. Baak, S. Gadatsch, R. Harrington, W. Verkerke, *Interpolation between
  multi-dimensional histograms using a new non-linear moment morphing method*,
  Nucl. Instrum. Meth. A771 (2015) 39–48, **arXiv:1410.7388**.
- R. Balasubramanian, L. Brenner, C. Burgard, W. Verkerke, *Effective Lagrangian
  Morphing*, **arXiv:2202.13612** (2022).
- P. Artoisenet, P. de Aquino, F. Demartin, R. Frederix, S. Frixione,
  F. Maltoni, M. K. Mandal, P. Mathews, K. Mawatari, V. Ravindran, S. Seth,
  P. Torrielli, M. Zaro, *A framework for Higgs characterisation*,
  JHEP 11 (2013) 043, **arXiv:1306.6464**.
- C. R. Rao, *Information and the accuracy attainable in the estimation of
  statistical parameters*, Bull. Calcutta Math. Soc. 37 (1945) 81–91.
- H. Cramér, *Mathematical Methods of Statistics*, Princeton Univ. Press, 1946.
- S. Amari, *Natural Gradient Works Efficiently in Learning*, Neural Computation
  10(2) (1998) 251–276, doi:10.1162/089976698300017746.
- S. Amari, R. Karakida, M. Oizumi, *Fisher Information and Natural Gradient
  Learning of Random Deep Networks*, **arXiv:1808.07172** (2018; AISTATS 2019).
- J. B. Tenenbaum, V. de Silva, J. C. Langford, *A global geometric framework
  for nonlinear dimensionality reduction*, Science 290 (2000) 2319–2323.
- R. R. Coifman, S. Lafon, *Diffusion maps*, Appl. Comput. Harmon. Anal. 21
  (2006) 5–30.
- J. Brehmer, K. Cranmer, G. Louppe, J. Pavez, *A Guide to Constraining
  Effective Field Theories with Machine Learning*, Phys. Rev. D 98 (2018)
  052004, **arXiv:1805.00020**.
- J. Brehmer, F. Kling, I. Espejo, K. Cranmer, *MadMiner: Machine learning-based
  inference for particle physics*, Comput. Softw. Big Sci. 4 (2020) 3,
  **arXiv:1907.10621**.
