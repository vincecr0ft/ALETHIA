# Architecture

The four pieces of this package, and why they look the way they look.

## 1. The Intention foundation model

The model is a Bayesian linear regression on a deliberately chosen feature
map. The fit is the closed-form ridge

    w     = (Phi^T Phi + lam I)^{-1} Phi^T Y
    A_inv = (Phi^T Phi + lam I)^{-1}

and the predictive mean and variance at a query `(c*, m*)` are

    mu*  = phi(c*, m*) @ w
    var* = sigma_aleo(c*, m*)^2 * (1 + leverage(c*, m*))
    leverage(c*, m*) = phi(c*, m*) A_inv phi(c*, m*)^T

Heteroscedastic noise comes in only at predict time: `sigma_aleo =
noise_frac * |mu*|`. The ridge fit itself assumes homoscedasticity, which
is what makes the closed form clean and what makes the EPIG derivation
below (where sigma cancels) work exactly.

The numbers: with `N_WC = 4` Wilson coefficients and `K_X = 5` modes in
`log(m/M_ref)`, the joint feature dimension is

    D_C = 1 + N_WC + N_WC*(N_WC+1)/2  =  1 + 4 + 10  =  15
    D_X = K_X                          =  5
    D_JOINT = D_C * D_X                =  75

so the closed-form `A_inv` is a 75×75 matrix. Fit, predict, leverage,
update, and one-step Sherman-Morrison acquisition are all sub-millisecond
on a laptop for the size of training and pool sets the orchestrator will
ever build up in this loop.

This is the Intention closed-form
`Q (K^T K + alpha I)^{-1} K^T V` written in regression notation:
`Q = phi(x*)`, `K = Phi`, `V = Y`.

## 2. Why these feature maps

`phi_c` carries the structure of SMEFT cross-sections at dimension 6:
the cross-section is exactly linear and quadratic in the Wilson
coefficients, and the closed-form `{1, c_a, c_a c_b}` basis spans that
space exactly. So the model has the correct inductive bias by
construction: the surrogate cannot misrepresent the polynomial structure
of the physics. Whatever residual is left for the regression to fit lives
entirely in the `m` direction.

`phi_x` is a polynomial in `log(m / M_ref)`. This is a smooth
approximation to soft spectra like Drell-Yan in `m_ll`. It's the right
default for a tail that grows like a small power of `m^2 / Lambda^2`.
For spectra with resonances or kinematic edges, swap for a B-spline
basis in `m`. The change is one feature function.

`phi_joint` is the tensor product `phi_c ⊗ phi_x`. Concretely
`phi_joint[n, c*D_X + x] = phi_c[n, c] * phi_x[n, x]`. This means
"the SMEFT coefficient of each polynomial mode in `m`" is a separate
basis function, which is the right factorisation when SMEFT operators
modify the differential cross-section's shape in `m`.

For multi-observable (jointly differential in `m_ll` and `p_T`, for
instance), extend to `phi_c ⊗ phi_m ⊗ phi_pT`. The dimensionality grows
multiplicatively; the closed form is the same.

## 3. Stratified split-conformal calibration

Raw Gaussian predictive intervals from a ridge regression under-cover
when the noise model is misspecified or when the query point is far from
the training support. The marginal failure isn't large (you can be 5%
off and look fine on average), but the conditional failure on
high-leverage points is severe: the highest-leverage stratum in the
included demo covers around 0.65 at the 0.95 target with the raw
intervals.

Stratified split-conformal fixes this. Procedure:

1. On a held-out calibration set, compute the conformity score
   `s_i = |y_i - mu_i| / sigma_i` at each point. Under the model's
   noise assumption, `s_i` would be a half-Gaussian and the empirical
   `1.96`-quantile would land at 1.96.
2. Stratify the calibration points by their `leverage` quantile (the
   default uses five strata).
3. Within each stratum, take the empirical `coverage`-quantile of `s_i`
   using the exchangeability-corrected rank `ceil((n+1) * coverage) / n`.
4. At query time, use that per-stratum factor to rescale `sigma`.

This gives approximate CONDITIONAL coverage at the stratum level:
each leverage band hits its nominal rate. Marginal coverage is also
near-nominal by construction.

The one trap to avoid: the calibration set must be drawn from the same
support the model will be queried on. If the orchestrator probes
`|c| <= 2` in production but the calibration set was drawn from
`|c| <= 1`, the high-leverage stratum's calibration factor is
extrapolated and you get the same conditional under-coverage you
started with. Both demo scripts and the integration brief assume the
broader probe region for the calibration draw.

## 4. Acquisition strategies

Three functions with the same return type (indices into the candidate
pool). The orchestrator chooses which to call.

### random_acquire

Uniform sample, no replacement. Baseline.

### leverage_acquire

Sherman-Morrison sequential greedy on leverage. At each step, pick the
pool candidate with the largest current `leverage = phi A_inv phi^T`,
then update `A_inv` hypothetically (rank-1) so the next pick sees the
model as if the previous pick had already been observed. This is
equivalent to one step of D-optimal experimental design at each pick and
prevents the obvious failure of selecting `k` near-identical
high-leverage points clustered on one hot spot.

The information being maximised is about the model parameters
globally. There is no preferred target region.

### epig_acquire

Sherman-Morrison sequential greedy on Expected Predictive Information
Gain about a user-supplied target set. Closed-form derivation (the full
version is in the `acquisition.py` module docstring):

For Bayesian linear regression with `Sigma_w = sigma^2 A_inv`, the
latent-function predictive variance at a target point `x_T` is
`sigma^2 lev_T`, where `lev_T = phi_T A_inv phi_T^T`. Conditioning on
a new observation at `x_p` gives, via Sherman-Morrison on `Sigma_w`,

    Var(f_T | data, y_p) = sigma^2 * [ lev_T  -  k_Tp^2 / (lev_p + 1) ]

with `k_Tp = phi_T A_inv phi_p^T`. The sigma^2 cancels in the
information gain:

    IG_T(p) = 0.5 log( lev_T / [lev_T - k_Tp^2 / (lev_p + 1)] )

EPIG averages `IG_T` over the supplied target set. Non-negativity
follows from Cauchy-Schwarz on the `A_inv` inner product:
`k_Tp^2 <= lev_T lev_p`, so the log argument is always in `(0, 1]`.

When the target set is uniform over the same support as the candidate
pool, EPIG and leverage acquisition pick similar points: both go after
the highest-leverage regions. When the target is focused (a tight
cloud at a specific Wilson-and-mass region the physics question lives
in), EPIG steers acquisitions toward points that reduce variance AT
the target. The included `demo_epig.py` shows the two strategies
having zero pick overlap under a focused target, and EPIG reducing
target predictive variance 1.3–4.9× faster per oracle call.

The price of EPIG over leverage is one matrix-vector product per pool
candidate per pick, scaled by the target-set size. For pool ~ 500 and
target ~ 200 this is microseconds and the orchestrator can call it
freely.

## What's deliberately not here

Multi-observable joint differential (`m_ll` AND `p_T`): one feature
function extension. The package is structured so this is a local change.

B-spline basis in `m_ll`: same. Swap `phi_x` for a B-spline basis
defined on the relevant kinematic range.

Heteroscedastic noise in the EPIG denominator: the closed form assumes
homoscedastic noise in the fit (which is what's actually used in the
ridge). Adding heteroscedasticity inflates `lev_p + 1` to
`lev_p + (sigma_p / sigma_avg)^2`. Five-line change in
`acquisition.py` if the orchestrator ever wants it.

Non-Gaussian likelihoods: this is a linear regression. For Poisson or
Bernoulli observations, the closed-form ridge becomes Laplace-approximated
IRLS and the rest of the architecture still works. Out of scope for the
current package.
