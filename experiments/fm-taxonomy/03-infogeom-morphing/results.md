# Results — analytic morphing + Fisher geometry demo

`demo.py` is self-contained (numpy only). Run:

```
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
uv run python experiments/fm-taxonomy/03-infogeom-morphing/demo.py
```

Executed and verified in the main session (numpy 2.4.4): reconstruction error
`1.24e-14`, template recovery `4.44e-16`, `cond(M) = 3.961`, PCA rank 3/2 — all
matching the construction below. The reconstruction and template-recovery errors
are exact-arithmetic identities realised at floating-point precision; the Fisher
figures follow in closed form from the templates `T1 = 2 a_SM a_BSM`,
`T2 = a_BSM^2` on the 64-point grid `x = linspace(0,1,64)` with `sigma_y = 0.05`.

## What the demo establishes

**[1–2] Minimal basis and morphing matrix.** One coefficient (`n = 1`),
quadratic order. The minimal sample count is
`N_samp = (n^2 + 3n + 2)/2 = 3` (Balasubramanian et al. arXiv:2202.13612,
Eq. 21). Three distinct couplings `c_i = (-1.0, 0.0, 1.5)` give a 3×3
Vandermonde morphing matrix `M` with rows `v(c_i) = (1, c_i, c_i^2)`,
`cond(M) ≈ 4` — well conditioned.

**[3] Exact reconstruction, no new simulation.** For eight new couplings
spanning `c ∈ [-0.7, 5.0]` (including extrapolation well outside the basis at
`c = 5`), the morphed prediction `sum_i w_i(c) sigma(c_i)` with
`w(c) = v(c) M^{-1}` matches the true `sigma(c, x)` to
`max|morphed − true| ~ 1e-13` at every query. The 3-sample basis spans the
*entire* one-coefficient family because the family is genuinely degree-2 — the
morphing is interpolation of a polynomial, exact at any `c`, interpolating or
extrapolating alike. This is the load-bearing contrast with a learned model,
whose error grows away from the training support.

**[4] The morphing inverse extracts the physical templates.** `M^{-1}` applied
to the three mixed observations returns the SM-only, interference and
BSM-squared curves `(T0, T1, T2)` to `~1e-14`. The structured prior is not just
"a low-dimensional fit": the coordinates are the physically meaningful
amplitude-interference components.

**[5] Fisher geometry of the coupling.** The 1-D Fisher metric is
`F(c) = sum_x (T1 + 2 c T2)^2 / sigma_y^2`, with
`dsigma/dc = 2 a_BSM (a_SM + c a_BSM)`. `F(c=0)` and `F(c=1)` are both
O(1e4–1e5) given `sigma_y = 0.05`, and the Cramér–Rao bound on any unbiased
estimator is `sigma_c >= 1/sqrt(F(c))`. The least-informative coupling is the
closed-form `c* = -sum(a_BSM^3 a_SM)/sum(a_BSM^4) ≈ -3` (it sits at/near the
left edge of the scanned grid): there `dsigma/dc` is smallest in norm and the
metric collapses. This is exactly the mechanism behind ALETHIA's
flat-vertex-direction result — the sensitivity is a property of the *working
point* `c`, set by the algebra, not something a model reads off the events.

**[6] Natural-gradient coordinate.** `s(c) = integral_0^c sqrt(F) dc'` is the
Fisher arc length (Amari 1998). Equal steps in `s` are equal-information steps;
`s(c)` is monotone and increases fastest where `F` is largest. This is the
coordinate a natural-gradient optimiser implicitly works in, preconditioning by
`F^{-1}`.

**[7] Learned-manifold contrast.** SVD of 40 sampled rate-curves gives raw
numerical rank 3 (= degree + 1 = number of templates) and mean-centred rank 2
(the constant offset removed, leaving the `c` and `c^2` variation). A
PCA/autoencoder *discovers* this dimension from data; morphing *asserts* it a
priori and needs only 3 exact samples and no fit. Same manifold, opposite
epistemics.

## Interpretation

The demo isolates the taxonomy's central distinction in a single runnable
object. The prediction manifold `{sigma(c, .)}` is a degree-2 Veronese curve in
function space. Morphing parametrizes it *exactly* from the known polynomial
structure with the provably minimal number of samples; PCA/diffusion-maps/
autoencoders recover the *same* low dimension but only approximately and only
within their sampled support. The Fisher information turns the coefficient axis
into a Riemannian manifold whose metric — and whose degenerate, low-sensitivity
working points — is fixed by the morphing algebra, which is precisely the
geometry ALETHIA's ManifoldInformer is trained to reproduce in a learned latent
space (paper P3/P4: tangent `R^2 = 0.99999` against `A_i`, curvature
`R^2 = 0.954` against `B_ij`).
