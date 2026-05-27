r"""
Acquisition strategies for the Intention FM.

Three functions, identical interface (return indices into the candidate pool):

- ``random_acquire(rng, pool_size, k)``    : baseline
- ``leverage_acquire(model, C_pool, M_pool, k)`` : sequential greedy on
  leverage. Equivalent to one step of D-optimal experimental design at each
  pick. Maximises information about the model parameters globally; no
  preferred prediction target.
- ``epig_acquire(model, C_pool, M_pool, C_target, M_target, k)`` : sequential
  greedy Expected Predictive Information Gain. Maximises information about
  the latent function on a user-supplied TARGET set. Reduces to leverage
  when the target set is uniform over the same support as the candidate
  pool; differs sharply when the target is focused on a specific region.

EPIG reference: Smith, Bickford Smith, Rainforth (2023),
"Prediction-Oriented Bayesian Active Learning". The closed-form below is
the Gaussian-linear specialisation.

==============================================================
EPIG closed-form derivation (under our Bayesian linear model)
==============================================================

Bayesian linear regression on the joint feature space::

    y    = Phi w + eps,  eps ~ N(0, sigma^2 I),  w ~ N(0, alpha^{-1} I)
    A    = Phi^T Phi + lam I,           lam = alpha sigma^2
    A_inv = (Phi^T Phi + lam I)^{-1}
    Sigma_w = sigma^2 A_inv             (posterior covariance over w)

For a query point with feature row ``phi_x``, define the leverage
``lev_x = phi_x A_inv phi_x^T``. Latent-function predictive variance::

    Var( f(x) | data ) = phi_x Sigma_w phi_x^T = sigma^2 lev_x

When we condition on a new observation ``(x_p, y_p)``, the posterior over
``w`` updates by Sherman-Morrison::

    Sigma_w' = Sigma_w  -  (Sigma_w phi_p)(phi_p^T Sigma_w) / (phi_p^T Sigma_w phi_p + sigma^2)

Plugging in and simplifying (sigma^2 cancels everywhere) gives the new
latent variance at a target point ``x_T``::

    Var( f(x_T) | data, y_p ) = sigma^2 * [ lev_T  -  k_Tp^2 / (lev_p + 1) ]

where ``k_Tp = phi_T A_inv phi_p^T``. Information gain on f(x_T) from one
observation::

    IG_T(p) = 0.5 log( Var(f_T|D) / Var(f_T|D, y_p) )
            = 0.5 log( lev_T / [ lev_T  -  k_Tp^2 / (lev_p + 1) ] )

Cauchy-Schwarz on the A_inv inner product guarantees ``k_Tp^2 <= lev_T lev_p``,
so ``k_Tp^2 / (lev_p + 1) <= lev_T lev_p / (lev_p + 1) < lev_T``: the log
argument is always in (0, 1] and information gain is non-negative.

EPIG averages IG_T over the supplied target set::

    EPIG(p | T) = mean over i in T of IG_T_i(p)

Sequential greedy: pick p* maximising EPIG, Sherman-Morrison-update A_inv,
recompute, pick the next, and so on.

The sigma^2 cancellation is exact only under the homoscedastic noise
assumption used in the ridge fit. Our model also returns heteroscedastic
predictive sigmas at predict time; for EPIG we use the homoscedastic
ridge posterior, which is the right object for design optimality.
"""
from __future__ import annotations

import numpy as np

from .features import phi_joint


def random_acquire(rng: np.random.Generator, pool_size: int, k: int) -> np.ndarray:
    """Uniform random selection of ``k`` indices, no replacement."""
    return rng.choice(pool_size, k, replace=False)


def leverage_acquire(model, C_pool: np.ndarray, M_pool: np.ndarray, k: int) -> np.ndarray:
    """Sherman-Morrison sequential greedy on leverage.

    At each step, pick the available pool candidate with maximal current
    leverage, then update A_inv hypothetically (rank-1) so subsequent picks
    see the model as if the previous pick had already been folded in.
    Prevents clustering on a single hot spot.
    """
    Phi   = phi_joint(C_pool, M_pool)
    A_inv = model.A_inv.copy()
    avail = np.ones(len(Phi), dtype=bool)
    chosen: list[int] = []
    for _ in range(k):
        lev = np.einsum("nd,de,ne->n", Phi, A_inv, Phi)
        lev = np.where(avail, lev, -np.inf)
        i   = int(np.argmax(lev))
        chosen.append(i)
        avail[i] = False
        p  = Phi[i]
        Ap = A_inv @ p
        A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
    return np.array(chosen)


def epig_acquire(
    model,
    C_pool: np.ndarray,
    M_pool: np.ndarray,
    C_target: np.ndarray,
    M_target: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Sequential greedy EPIG acquisition.

    Picks ``k`` indices into ``(C_pool, M_pool)`` that maximally reduce the
    expected predictive variance of the latent function on the target set
    ``(C_target, M_target)``. See module docstring for the closed-form
    derivation.
    """
    Phi_pool = phi_joint(C_pool, M_pool)
    Phi_T    = phi_joint(C_target, M_target)
    A_inv    = model.A_inv.copy()
    avail    = np.ones(len(Phi_pool), dtype=bool)
    chosen: list[int] = []

    for _ in range(k):
        lev_T  = np.einsum("id,de,ie->i", Phi_T,    A_inv, Phi_T)     # (n_T,)
        lev_P  = np.einsum("pd,de,pe->p", Phi_pool, A_inv, Phi_pool)  # (n_P,)
        K_TP   = Phi_T @ A_inv @ Phi_pool.T                            # (n_T, n_P)
        var_red = K_TP ** 2 / (lev_P[None, :] + 1.0)                   # (n_T, n_P)

        # Guard against tiny negative residuals from finite precision.
        denom = lev_T[:, None] - var_red
        denom = np.maximum(denom, 1e-12 * np.maximum(lev_T[:, None], 1e-12))
        log_ratio = np.log(np.maximum(lev_T[:, None], 1e-12) / denom)
        scores    = 0.5 * log_ratio.mean(axis=0)                       # (n_P,)
        scores    = np.where(avail, scores, -np.inf)

        i = int(np.argmax(scores))
        chosen.append(i)
        avail[i] = False
        p  = Phi_pool[i]
        Ap = A_inv @ p
        A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
    return np.array(chosen)
