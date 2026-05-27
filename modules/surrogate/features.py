r"""
Feature maps for the SMEFT surrogate.

Three pure functions:

- ``phi_c(c)``  : SMEFT polynomial structure :math:`\{1, c_a, c_a c_b\}`,
  dim ``D_C = 1 + N_WC + N_PAIRS``. This is exact for dimension-6 SMEFT
  cross-sections to :math:`O(\Lambda^{-4})`, so the model has the correct
  inductive bias for the regression task by construction.

- ``phi_x(m)``  : polynomial basis in :math:`\log(m / M_{\text{ref}})`,
  dim ``D_X = K_X``. Smooth approximation appropriate for soft spectra
  like Drell-Yan in :math:`m_{\ell\ell}`. Swap for a B-spline basis if
  you need to capture resonances or kinematic edges.

- ``phi_joint(c, m)`` : tensor product :math:`\phi_c(c) \otimes \phi_x(m)`,
  dim ``D_C * D_X``. This is the feature map the IntentionFM uses for
  closed-form ridge regression.

Module-level constants ``N_WC``, ``WC_NAMES``, ``K_X``, ``M_REF`` are the
configuration knobs. The defaults give ``D_JOINT = 75``, which is
well-conditioned with a few hundred training points.
"""
from __future__ import annotations
from itertools import combinations_with_replacement

import numpy as np


# ---- configuration (override before module import if needed) ----
N_WC      = 4
WC_NAMES  = ("cHq3", "cHq1", "clq3", "clq1")
PAIRS     = list(combinations_with_replacement(range(N_WC), 2))

K_X       = 5
M_REF     = 1.0   # TeV

D_C       = 1 + N_WC + len(PAIRS)     # 15
D_X       = K_X                       # 5
D_JOINT   = D_C * D_X                 # 75


def phi_c(c: np.ndarray) -> np.ndarray:
    """SMEFT-structure features. Output shape ``(n, D_C)``."""
    c = np.atleast_2d(c)
    n = c.shape[0]
    feats = [np.ones((n, 1)), c]
    quad  = np.stack([c[:, a] * c[:, b] for a, b in PAIRS], axis=1)
    feats.append(quad)
    return np.concatenate(feats, axis=1)


def phi_x(m: np.ndarray) -> np.ndarray:
    """Polynomial-in-log(m/M_REF) features. Output shape ``(n, D_X)``."""
    m = np.atleast_1d(m).astype(float)
    log_m = np.log(m / M_REF)
    return np.stack([log_m ** k for k in range(K_X)], axis=1)


def phi_joint(c: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Tensor-product feature map. c and m must have the same number of
    samples. Output shape ``(n, D_C * D_X)``."""
    fc = phi_c(c)
    fx = phi_x(m)
    return np.einsum("nc,nx->ncx", fc, fx).reshape(fc.shape[0], -1)
