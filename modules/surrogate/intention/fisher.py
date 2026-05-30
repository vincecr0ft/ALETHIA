"""Empirical Fisher information of an oracle on a c-prior, in Wilson space.

Disclosure-side companion to `eigen.py`. Where `eigen.py` rotates the
acquisition geometry into the eigenbasis of the *model's* design matrix
A on M-context, this module rotates the disclosure target into the
eigenbasis of the *oracle's* Fisher information on c. They are
independent rotations serving different upgrades (eigen for U5,
Fisher for U7) and are kept in separate modules to keep that
distinction explicit.

The construction:

    F_ij = E_{c, m} [ d_i log mu(c, m) * d_j log mu(c, m) ]

estimated by central finite differences in c against a callable oracle
that returns mu(c, m) (the SMEFT cross-section ratio). The Fisher
eigendecomposition F = V D V^T defines a rotated Wilson basis
c_tilde = V^T c on which:

- the leading eigendirection is the kinematic-observable's single most
  informative combination of operators;
- the trailing eigendirections are (by construction) the directions
  that mu(c, m) cannot resolve — which on m_ll alone include
  c_phi_q^(1) (rate-only) and the c_lq^(1) / c_lq^(3) isospin
  partner combination.

Pure numpy + a Callable. No torch, no dependency on the surrogate
package beyond the oracle protocol.

References:
- Cui-Martin-Marzouk 2014 (likelihood-informed subspace)
- Constantine 2015 (active subspaces)
- Costa-Marzocca-Mimasu-Salko (fitmaker, Fisher eigen-basis in SMEFT)
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class OracleProtocol(Protocol):
    """The minimum interface we need for Fisher estimation: oracle.truth(c, m).

    Matches `modules.surrogate.oracle_smeft.AnalyticSMEFTOracle.truth`
    and `modules.surrogate.oracle_smeft.AnalyticSMEFTOracle.__call__(..., noise=False)`.
    """

    def truth(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:  # noqa: D401
        ...


def empirical_fisher_c(
    oracle: OracleProtocol,
    c_samples: np.ndarray,
    m_samples: np.ndarray,
    *,
    fd_step: float = 1e-3,
    eps_floor: float = 1e-12,
) -> np.ndarray:
    """Estimate F = E[ grad_c log mu(c, m) (grad_c log mu(c, m))^T ].

    Args:
        oracle: provides `truth(c, m) -> mu`.
        c_samples: (N, n_wc) Wilson coefficient samples drawn from the
            prior over which the expectation is taken.
        m_samples: (N,) kinematic samples paired one-to-one with c_samples.
            Caller is responsible for the joint distribution.
        fd_step: central finite-difference step in each c-direction.
            1e-3 is a safe default for c in [-1, 1]; reduce if mu has
            large second derivatives in c (it does not for SMEFT
            because mu is a quadratic polynomial in c, so the
            central-difference truncation error is identically zero).
        eps_floor: numerical floor on |mu| to stop log-grad from
            exploding near the Standard-Model line where mu ~ 1.

    Returns:
        F: (n_wc, n_wc) symmetric PSD matrix.
    """
    c_samples = np.atleast_2d(np.asarray(c_samples, dtype=float))
    m_samples = np.atleast_1d(np.asarray(m_samples, dtype=float))
    N, n_wc = c_samples.shape
    if m_samples.shape[0] != N:
        raise ValueError(
            f"c_samples N={N} != m_samples N={m_samples.shape[0]}"
        )

    mu0 = oracle.truth(c_samples, m_samples)                   # (N,)
    inv_mu = 1.0 / np.where(np.abs(mu0) > eps_floor,
                            mu0, np.sign(mu0) * eps_floor + eps_floor)

    # Central differences in each c-direction. SMEFT mu is quadratic in
    # c, so central differences are exact up to floating-point error
    # for any fd_step; we use a moderate fd_step to avoid catastrophic
    # cancellation rather than to manage truncation.
    grad_log_mu = np.empty((N, n_wc), dtype=float)
    for i in range(n_wc):
        c_plus = c_samples.copy(); c_plus[:, i] += fd_step
        c_minus = c_samples.copy(); c_minus[:, i] -= fd_step
        mu_plus = oracle.truth(c_plus, m_samples)
        mu_minus = oracle.truth(c_minus, m_samples)
        # d_i log mu = (1/mu) d_i mu
        grad_log_mu[:, i] = inv_mu * (mu_plus - mu_minus) / (2.0 * fd_step)

    # F = (1/N) sum_n g_n g_n^T, symmetrised for numerical safety.
    F = grad_log_mu.T @ grad_log_mu / float(N)
    F = 0.5 * (F + F.T)
    return F


def fisher_basis(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose F into descending eigenpairs.

    Returns:
        D_desc: (n_wc,) eigenvalues, descending. D_desc[0] is the
            leading Fisher direction's sensitivity.
        V_desc: (n_wc, n_wc) eigenvectors, columns descending — the
            rotation matrix to apply as c_tilde = V_desc.T @ c.
    """
    F = 0.5 * (np.asarray(F, dtype=float) + np.asarray(F, dtype=float).T)
    lam, V = np.linalg.eigh(F)            # ascending
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


def rotate_c(c: np.ndarray, V_desc: np.ndarray) -> np.ndarray:
    """c_tilde = V_desc.T @ c, broadcasting over a (N, n_wc) batch.

    The first column of `c_tilde` is the leading Fisher direction —
    the linear combination of (c_phi_q^(3), c_phi_q^(1), c_lq^(3),
    c_lq^(1)) the kinematic observable resolves best.
    """
    c = np.atleast_2d(np.asarray(c, dtype=float))
    return c @ V_desc                     # (N, n_wc)


def sample_c_prior_inbox(
    n: int,
    n_wc: int,
    box: float,
    rng: np.random.Generator,
    *,
    withhold_dim: int | None = None,
    withhold_band: tuple[float, float] | None = None,
) -> np.ndarray:
    """U([-box, box]^n_wc), optionally excluding a magnitude band on one
    dimension. Mirrors the convention in
    `experiments/full-chain-run/identifiability_probe.py` so the Fisher
    matrix is estimated on the same prior the disclosure probe is
    trained against.
    """
    if withhold_dim is None:
        return rng.uniform(-box, box, size=(n, n_wc))
    if withhold_band is None:
        raise ValueError("withhold_band required when withhold_dim is given")
    out = np.empty((n, n_wc))
    i = 0
    while i < n:
        c = rng.uniform(-box, box, size=n_wc)
        if abs(c[withhold_dim]) < withhold_band[0]:
            out[i] = c
            i += 1
    return out
