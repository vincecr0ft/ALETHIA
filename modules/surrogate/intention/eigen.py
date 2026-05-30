"""Eigendecomposition utilities for the Intention ridge design matrix.

Spine of the upgrade-architecture (docs/research/upgrade-architecture.md
§1). The Intention closed-form head reduces to a ridge solve against

    A = psi(M_ctx).T @ psi(M_ctx) + alpha * I    in R^{d x d}

where psi is the learned MLP basis. The full eigendecomposition of A is
already computed inside `kappa_drift` (modules/surrogate/intention/drift.py)
and immediately discarded after extracting the smallest eigenvector and
the condition number. This module surfaces the rest of it.

Three downstream consumers:

- Phoenix drift spans embed the spectrum and the top/low eigenvectors
  (`experiments/full-chain-run/run.py`, `chain.drift.eigen` span).
- Eigen-redirected acquisition resamples the EPIG candidate pool from
  the leading directions of the low-eigenvalue subspace
  (`modules/surrogate/intention/acquisition.py::epig_acquire_m_eigen`).
- The disclosure probe rotates into the Fisher eigenbasis on c, an
  independent rotation companion to this one
  (`modules/surrogate/intention/fisher.py`).

Pure numpy. No torch, no Phoenix imports — cheap to call from anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenState:
    """Eigendecomposition of the ridge design matrix A on a fixed context.

    Eigenpairs are returned ascending by eigenvalue: lam[0] is the
    smallest, lam[-1] is the largest, and U[:, j] is the eigenvector
    paired with lam[j].
    """

    lam: np.ndarray      # (d,) eigenvalues, ascending
    U: np.ndarray        # (d, d) eigenvectors as columns
    alpha: float
    d: int

    @property
    def kappa(self) -> float:
        return float(self.lam[-1] / max(self.lam[0], 1e-30))

    def low_subspace(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k smallest-eigenvalue directions: (U_low (d,k), lam_low (k,))."""
        k = int(min(k, self.d))
        return self.U[:, :k], self.lam[:k]

    def high_subspace(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k largest-eigenvalue directions: (U_high (d,k), lam_high (k,))."""
        k = int(min(k, self.d))
        return self.U[:, -k:], self.lam[-k:]

    def lis_rank(self, tau: float = 1e-2) -> int:
        """# eigenvalues with lam_j > tau * lam_max — the likelihood-informed
        rank under the threshold tau. Matches the Cui-Martin-Marzouk LIS
        rank when alpha-regularised."""
        return int(np.sum(self.lam > tau * self.lam[-1]))


def eigen_state(Psi: np.ndarray, alpha: float) -> EigenState:
    """Eigendecompose A = Psi.T @ Psi + alpha I.

    Psi is the (K, d) basis evaluated on a context of size K. The
    returned EigenState carries A's spectrum and eigenvectors.
    """
    d = int(Psi.shape[1])
    A = Psi.T @ Psi + float(alpha) * np.eye(d)
    lam, U = np.linalg.eigh(A)
    return EigenState(lam=lam, U=U, alpha=float(alpha), d=d)


def eigen_resample_weights(
    Psi_pool: np.ndarray, U_low: np.ndarray, lam_low: np.ndarray
) -> np.ndarray:
    """Importance weights r(m) = sum_j (psi(m) . u_j)^2 / lam_j over a
    low-eigenvalue subspace.

    Heavier weight on candidates whose basis embedding projects onto
    eigen-directions of A that are currently *under-resolved* by the
    context — i.e. the directions the next acquisition should target.

    Args:
        Psi_pool: (P, d) basis evaluated on a candidate pool of P points.
        U_low: (d, k) low-eigenvalue eigenvectors (columns).
        lam_low: (k,) eigenvalues paired with U_low.

    Returns:
        (P,) non-negative weights. NOT normalised — caller decides
        whether to use as probabilities or as multiplicative score
        weights.
    """
    proj = Psi_pool @ U_low                          # (P, k)
    w = (proj ** 2) / np.clip(lam_low, 1e-30, None)
    return w.sum(axis=1)


def eigenvector_stability(
    U_prev: np.ndarray, U_curr: np.ndarray
) -> np.ndarray:
    """|cos(u_prev,j, u_curr,j)| per column, sign-flip invariant.

    A diagnostic for "did the eigen-direction collapse between cycles?".
    Values near 1 indicate a stable eigenvector; values near 0 indicate
    rotation (which on the low-subspace is the signature of a drift
    event). Returns shape (k,).
    """
    if U_prev.shape != U_curr.shape:
        raise ValueError(
            f"U_prev {U_prev.shape} != U_curr {U_curr.shape}"
        )
    cos = np.einsum("dj,dj->j", U_prev, U_curr)
    return np.abs(cos)


def union_subspace(Us: list[np.ndarray], rcond: float = 1e-10) -> np.ndarray:
    """Column-orthonormalised union of a list of (d, k_i) eigenvector blocks.

    Used by the Phoenix MCP rubric (experiments/full-chain-run/eigen_query.py)
    to assemble the "current LIS gap subspace" across drift firings.
    Returns (d, r) where r is the rank of the stacked union.
    """
    if not Us:
        raise ValueError("union_subspace requires at least one block")
    stacked = np.concatenate(Us, axis=1)
    Q, _ = np.linalg.qr(stacked)
    # Drop near-zero columns via SVD on the QR factor.
    U_svd, s, _ = np.linalg.svd(stacked, full_matrices=False)
    r = int(np.sum(s > rcond * s[0]))
    return U_svd[:, :r]
