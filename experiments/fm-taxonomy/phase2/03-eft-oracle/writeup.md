# The EFT/SMEFT quadratic oracle, and when a simple learner can stand in for the MC program

Phase-2, axis III of the ALETHIA foundation-model taxonomy (provenance of the
prediction manifold: *structured/exact* vs *learned/empirical*). The companion
script `oracle_survey.py` instantiates every claim below; `results.md` holds the
verified run numbers.

The question this study isolates: a dimension-6 SMEFT differential rate is an
*exact* polynomial in the Wilson coefficients — a known, structured object.
Lagrangian morphing reconstructs that polynomial exactly from a provably minimal
basis of noiseless samples. A full Monte-Carlo generator
(MadGraph/SMEFT@NLO-style reweighting) produces the *same* rate with statistical
noise. Between these two extremes sits a panel of generic learners. We measure
how well each recovers the oracle, and where the structure-aware learner wins.

---

## 1. The oracle: a quadratic cross section

### 1.1 Where the polynomial comes from

Each SMEFT amplitude is linear in the Wilson coefficients,

```
A(c, x) = a_SM(x) + sum_i c_i a_i(x) ,        c = (c_1, ..., c_n) ,
```

so the differential rate — the squared amplitude integrated over the angular
phase space — is a homogeneous quadratic form:

```
sigma(c, x) = |A|^2 = T_0(x)
            + sum_i c_i T_i(x)                          # interference
            + sum_{i<=j} c_i c_j T_{ij}(x) ,            # BSM-squared + cross
```

with `T_0 = a_SM^2`, `T_i = 2 a_SM a_i` (interference), `T_{ii} = a_i^2`
(BSM-squared), and `T_{ij} = 2 a_i a_j` for `i != j` (the genuine *interference
between two operators*). This is exactly the `sm_only / interference /
bsm_squared` decomposition that ALETHIA's analytic SMEFT calculator returns
(`modules/analytic_smeft/smeft.py`, `_partonic_xs`): the SM piece, the
`O(1/Lambda^2)` interference, and the `O(1/Lambda^4)` quadratic.

The physical content that makes morphing informative rather than a rescaling:
the four-fermion contact term carries energy growth (`~ hat-s / Lambda^2`),
so the contact amplitude is constant in `hat-s` while the SM amplitude falls as
`1/hat-s`; the interference/SM ratio grows as `(M_ll/Lambda)^2` and the
BSM^2/SM ratio as `(M_ll/Lambda)^4` (Greljo & Marzocca, arXiv:1704.09015). The
templates `T_i, T_{ij}` are therefore *not* proportional to `T_0`, and the
morphing basis is genuinely `(n+1)(n+2)/2`-dimensional. The script's oracle
encodes this with BSM amplitudes `a_i(x) ~ x` (linear energy growth), one with a
distinct kinematic shape per operator.

The scalar observable used for the learner panel is the integrated rate
`sigma(c) = sum_x sigma(c, x)` — a cross section, inheriting the same quadratic
structure `sigma(c) = theta_0 + sum theta_i c_i + sum theta_{ij} c_i c_j`.

### 1.2 The morphing reconstruction

The templates are unknown a priori; what is cheap is to *sample* `sigma` at a
set of benchmark couplings. Collect the coupling monomials into the row vector
`v(c) = (1, c_1, ..., c_i c_j, ...)`; the morphing matrix stacks these rows over
the benchmarks, `M_{ki} = v_i(c^{(k)})`. A new prediction is the polynomial
re-evaluated at the query through the inverse,

```
sigma(c) = v(c) M^{-1} (sigma(c^{(1)}), ..., sigma(c^{(N)}))^T = sum_k w_k(c) sigma(c^{(k)}) ,
```

with `w(c) = v(c) M^{-1}` the (polynomial) morphing weights. `M^{-1}` applied to
the samples also *recovers the physical templates* `theta` directly. This is
moment morphing (Baak, Gadatsch, Harrington & Verkerke, arXiv:1410.7388) and its
EFT generalisation (Balasubramanian, Brenner, Burgard & Verkerke,
arXiv:2202.13612). The reconstruction is *exact at any `c`*, interpolating or
extrapolating, because a polynomial is its own continuation.

### 1.3 Minimal basis count

`M` must be square and invertible, so `N` is the number of monomials of the
quadratic form. For `n` coefficients at quadratic order (arXiv:2202.13612):

```
N = (n^2 + 3n + 2)/2 = C(n+2, 2):     n=1 -> 3,  n=2 -> 6,  n=3 -> 10.
```

The script verifies this count in block `[0]` and uses a well-conditioned basis
(SM point + axis excursions `±e_i` + the cross points `e_i + e_j`).

### 1.4 The MC generator as a noisy oracle

A real generator estimates `sigma(c)` from `n_events` draws, with relative
statistical error `~ 1/sqrt(n_events)`. We model the returned rate as Gaussian
about the truth with that relative width (`mc_sample`), the additive-Gaussian
observation model of ALETHIA's Fisher studies (`working_point_fisher.py`,
`sigma_y`) with a coupling-dependent scale. Morphing assumes its basis campaigns
are run to negligible MC error (one expensive high-stats campaign per basis
point — standard practice); the generic learners see the noisy rate. That
asymmetry is the realistic comparison.

---

## 2. The learner panel

| learner | provenance | what it assumes | samples |
|---|---|---|---|
| `morphing(exact)` | structured | the polynomial degree *and* the minimal basis | `N = C(n+2,2)` noiseless |
| `poly_deg2(correct)` | structured (degree) | the correct total degree 2 | any `>= N`, noisy |
| `poly_deg1(under)` | wrong structure | degree 1 (no BSM^2 curvature) | noisy |
| `poly_deg3(over)` | over-structured | degree 3 (spurious freedom) | noisy |
| `gaussian_process` | learned | smoothness only (RBF + white noise) | noisy |
| `mlp_64x64` | learned | universal approximation | noisy |

The design matrix `v(c)` is the *same* monomial map for morphing and for
polynomial regression; the only differences are (i) `N` square vs
overdetermined and (ii) noiseless basis vs noisy training set. Morphing and the
correct-degree polynomial are the two structure-aware poles; the GP and MLP are
the learned pole; `poly_deg1` is the misspecified control.

The script scores each on four axes:

- **(a) interpolation** — relative RMSE on `400` random couplings inside the
  training hull `|c| <= 1`;
- **(b) extrapolation** — relative RMSE on `400` couplings in the shell
  `1 < ||c||_inf <= 2` (genuinely outside the box);
- **(c) flat-Fisher behaviour** — error along a built-in near-degenerate
  direction (§4);
- **(d) sample efficiency** — interpolation error vs number of MC training
  points, against the fixed morphing basis size.

---

## 3. What the panel shows (structure)

The qualitative result the construction guarantees, to be confirmed by the run
numbers in `results.md`:

- `morphing(exact)` reaches machine precision both inside and outside the hull;
  its template-recovery error (`theta_err`) is `~1e-13`. It is an exact-arithmetic
  identity realised at floating point. This is the *structured* pole: dimension
  and span fixed by the Lagrangian's polynomial degree, zero fit, minimal
  samples, exact extrapolation.
- `poly_deg2(correct)` reaches the MC noise floor (`~1/sqrt(n_events)`) in *and*
  out of the hull, with `N` plus a few points: knowing the degree buys
  extrapolation that the generic learners cannot match, because the fitted
  polynomial *is* the oracle up to noise on the coefficients.
- `poly_deg1(under)` is biased everywhere the quadratic term contributes; its
  error does not fall with more data (misspecification floor), and it diverges
  fastest off-support.
- `poly_deg3(over)` matches deg-2 inside the hull but pays extrapolation
  variance from the unconstrained cubic terms.
- `gaussian_process` / `mlp_64x64` interpolate acceptably *inside* the hull but
  degrade sharply off-support (no polynomial continuation; the GP reverts to its
  mean, the MLP to whatever its activations extrapolate to) and need many more
  points to match the structure-aware learners inside the hull.

The sample-efficiency block makes the data cost explicit: morphing is exact at
`N = 6` (n=2) noiseless points; the GP and MLP need an order of magnitude more
noisy points to approach the same interpolation error, and never close the
extrapolation gap.

---

## 4. The Fisher geometry and the flat direction

The Fisher information of the couplings, for the differential observation
`Y(x) = sigma(c, x) + N(0, sigma_y^2)`, is read straight off the morphing
polynomial:

```
F_ij(c) = sum_x (d_i sigma)(d_j sigma) / sigma_y^2 ,
d_i sigma(c, x) = T_i(x) + 2 c_i T_{ii}(x) + sum_{j!=i} c_j T_{ij}(x) .
```

This is Eq. (1) of ALETHIA's work-point handoff (`A_i + 2 B_{ij} c_j`): the
templates that define the morphing *are* the tangent (`T_i`) and curvature
(`T_{ij}`) data of the metric. `fisher_scalar` computes it in closed form.

The degeneracy probe (`degeneracy_probe`, n=2) makes `a_2` approach a multiple
of `a_1`, so the `c_1 - c_2` combination becomes nearly unmeasurable: `F(0)` has
a small eigenvalue, large condition number. Two consequences the script reports:

1. **The structured learner stays exact along the flat direction.** Along
   `c_1 = -c_2`, where the generic learner has little signal to fit, the
   correct-degree polynomial is still the same polynomial — its error on the
   flat line is the noise floor, while the GP's grows. Flatness is a property of
   *where you sit in `c`*, dictated by the algebra; the structured learner does
   not have to "discover" it.
2. **A working point lifts the flat eigenvalue.** Evaluating `F` at
   `c = (0.6, 0.6)` along the energy-growing axis multiplies the smallest Fisher
   eigenvalue by `lift_min_eig >> 1`, because the `O(1/Lambda^4)` curvature
   templates `T_{ij}` carry energy growth and inject sensitivity that the `c=0`
   anchor lacks. This is the closed-form version of ALETHIA's
   flat-vertex-direction finding (`working_point_fisher.py`): the metric
   collapses at `c=0` and lifts off-SM, by the morphing algebra, not by anything
   a model reads off the events.

---

## 5. What this says for axis III and ALETHIA

The survey quantifies the taxonomy's central distinction on a single object.
The prediction manifold `{sigma(c, .)}` is a degree-2 Veronese variety in
observable space. Morphing parametrizes it *exactly* from the known polynomial
structure with the provably minimal `N = C(n+2,2)` samples; the GP/MLP recover
the *same* low-dimensional structure but only approximately and only within
their sampled support. A simple learner can stand in for the MC program **when,
and only when, it carries the right structure**: the correct-degree polynomial
is a faithful, cheap, extrapolating surrogate; a degree-agnostic GP/MLP is a
local interpolator that must be re-trained, fed more data, and never trusted
off-support.

This is the structured pole of ALETHIA's ManifoldInformer tension. The
ManifoldInformer is the *learned* counterpart: a JEPA-trained, permutation-
invariant per-event encoder that never sees the Wilson coefficient and must
discover the prediction manifold from event sets, with a closed-form ridge head
producing the per-scenario coordinate `w_theta(c)`
(`experiments/manifold-informer/manifold_informer.py`). The paper's
pre-registered probes grade that learned geometry against the morphing answer
key derived here: the linear probe recovers the tangent templates `T_i` and the
curvature templates `T_{ij}`. The complementary HEP line makes the gradient the
learning target directly — score regression / "mining gold" (Brehmer, Cranmer,
Louppe & Pavez, arXiv:1805.00020; MadMiner, Brehmer, Kling, Espejo & Cranmer,
arXiv:1907.10621) regresses `t(x|c) = grad_c log p(x|c)`, the same `d_i sigma`
that morphing reads off the polynomial, obtained by fit rather than by algebra.

The unifying statement, restated for the oracle survey: **morphing is the
analytic limit of a learned surrogate when the generative polynomial structure
is known.** A physics FM earns its keep precisely where that structure is *not*
fully known, *not* cheap to sample over all operators (`N` grows polynomially in
operator count, combinatorially with separate production/decay vertices), or
must be inferred from event-level data rather than cross-section templates — and
its latent geometry should be validated against the morphing/Fisher answer key
wherever that key exists.

---

## References (verified: title, authors, arXiv/journal id)

- M. Baak, S. Gadatsch, R. Harrington, W. Verkerke, *Interpolation between
  multi-dimensional histograms using a new non-linear moment morphing method*,
  Nucl. Instrum. Meth. A771 (2015) 39–48, **arXiv:1410.7388**.
  (verified via arXiv: DOI 10.1016/j.nima.2014.10.033)
- R. Balasubramanian, L. Brenner, C. Burgard, W. Verkerke, *Effective Lagrangian
  Morphing*, **arXiv:2202.13612** (2022). (verified via arXiv)
- A. Greljo, D. Marzocca, *High-p_T dilepton tails and flavour physics*,
  Eur. Phys. J. C 77 (2017) 548, **arXiv:1704.09015**. (verified via arXiv;
  energy-growing four-fermion contact structure, basis for the oracle's
  amplitudes)
- J. Brehmer, K. Cranmer, G. Louppe, J. Pavez, *A Guide to Constraining
  Effective Field Theories with Machine Learning*, Phys. Rev. D 98 (2018)
  052004, **arXiv:1805.00020**. (verified via arXiv)
- J. Brehmer, F. Kling, I. Espejo, K. Cranmer, *MadMiner: Machine learning-based
  inference for particle physics*, Comput. Softw. Big Sci. 4 (2020) 3,
  **arXiv:1907.10621**. (verified via arXiv)
- C. R. Rao, *Information and the accuracy attainable in the estimation of
  statistical parameters*, Bull. Calcutta Math. Soc. 37 (1945) 81–91.
