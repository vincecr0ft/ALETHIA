"""EPIG closed-form acquisition over the Intention feature space.

Direct translation of the closed-form derivation in
modules/surrogate/acquisition.py:25-67, with phi_joint(c, m) replaced by
the Intention head's psi_theta(m). Sherman-Morrison sequential greedy.

The candidate space is m-values to query the oracle at; the target set is
held-out m-values inside the drifted region. The result of the call is a
list of m's to ask the oracle about; the orchestrator does the oracle
query and folds the result into the context.

Also implements parameter-space EPIG (INV-3 of ALETHEIA_investigations.md):
``param_epig_d_acquire`` (D-optimal joint entropy of the resolved Wilson
subspace) and ``param_epig_a_acquire`` (A-optimal single-direction
reduction). Both reuse the Sherman-Morrison cache and act on a projection
matrix ``P = V^T W`` supplied by the caller. ``P`` is fit once in INV-2
and frozen; the helpers here only consume it.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def epig_acquire_m(model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                   M_pool: np.ndarray, M_target: np.ndarray,
                   k: int) -> np.ndarray:
    """Sequential-greedy EPIG over m-values, in the Intention feature space.

    Returns indices into M_pool (length k).
    """
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)  # (d, d)
    Psi_pool = model.psi_np(M_pool)                 # (n_P, d)
    Psi_T = model.psi_np(M_target)                  # (n_T, d)
    avail = np.ones(len(Psi_pool), dtype=bool)
    chosen: list[int] = []
    for _ in range(k):
        lev_T = np.einsum("id,de,ie->i", Psi_T, A_inv, Psi_T)
        lev_P = np.einsum("pd,de,pe->p", Psi_pool, A_inv, Psi_pool)
        K_TP = Psi_T @ A_inv @ Psi_pool.T               # (n_T, n_P)
        var_red = K_TP ** 2 / (lev_P[None, :] + 1.0)
        denom = lev_T[:, None] - var_red
        denom = np.maximum(denom, 1e-12 * np.maximum(lev_T[:, None], 1e-12))
        log_ratio = np.log(np.maximum(lev_T[:, None], 1e-12) / denom)
        scores = 0.5 * log_ratio.mean(axis=0)
        scores = np.where(avail, scores, -np.inf)
        i = int(np.argmax(scores))
        chosen.append(i)
        avail[i] = False
        p = Psi_pool[i]
        Ap = A_inv @ p
        A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
    return np.array(chosen)


def target_set_entropy(model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                       M_target: np.ndarray, sigma: float = 1.0) -> float:
    """H_T = 0.5 sum_i log(2 pi e sigma^2 lev_T_i). The headline span attribute."""
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    Psi_T = model.psi_np(M_target)
    lev_T = np.einsum("id,de,ie->i", Psi_T, A_inv, Psi_T)
    lev_T = np.maximum(lev_T, 1e-15)
    return float(0.5 * np.sum(np.log(2 * np.pi * np.e * (sigma ** 2) * lev_T)))


def epig_acquire_m_eigen(model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                          M_pool: np.ndarray, M_target: np.ndarray,
                          *, eigen=None, k_low: int = 3, k_pick: int = 5,
                          resample_size: int | None = None,
                          rng: np.random.Generator | None = None,
                          ) -> np.ndarray:
    """Eigen-redirected EPIG: importance-resample the candidate pool from
    the design-matrix's low-eigenvalue subspace, then run sequential
    greedy EPIG on the resampled pool.

    The importance weights r(m) = sum_{j in low} (psi(m).u_j)^2 / lam_j
    concentrate mass on m-values whose basis embedding projects onto
    eigen-directions currently under-resolved by the context. After
    resampling, EPIG ranks the redrawn pool with the usual closed-form
    information gain. See docs/research/upgrade-architecture.md §3.

    Args:
        model: an IntentionFM.
        M_ctx, Y_ctx: current context (K,), (K,).
        M_pool: candidate pool (P,).
        M_target: target set (T,).
        eigen: an EigenState on the current context. If None, computed
            from `model.A_eigen(M_ctx)`.
        k_low: how many low-eigenvalue directions to sample from.
        k_pick: how many m-values to pick in total.
        resample_size: size of the importance-resampled pool. Defaults
            to len(M_pool) so the EPIG call sees the same shape.
        rng: numpy Generator for the resampling step.

    Returns:
        m_picked: (k_pick,) array of selected m-values from the
            (resampled) pool. Returned as the *values*, not indices,
            because the resampled pool is no longer the same as
            `M_pool`. The caller folds these into the context directly.
    """
    if rng is None:
        rng = np.random.default_rng()
    if eigen is None:
        eigen = model.A_eigen(M_ctx)
    if resample_size is None:
        resample_size = len(M_pool)

    from .eigen import eigen_resample_weights
    U_low, lam_low = eigen.low_subspace(k_low)
    Psi_pool = model.psi_np(M_pool)
    w = eigen_resample_weights(Psi_pool, U_low, lam_low)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        # Degenerate: fall back to uniform.
        probs = np.full(len(M_pool), 1.0 / len(M_pool))
    else:
        probs = w / s
    # With-replacement resample; downstream EPIG deduplicates picks
    # via its sequential-greedy mask.
    idx = rng.choice(len(M_pool), size=resample_size, replace=True, p=probs)
    M_pool_resampled = M_pool[idx]
    picked_idx = epig_acquire_m(
        model, M_ctx, Y_ctx, M_pool_resampled, M_target, k=k_pick)
    return M_pool_resampled[picked_idx]


# ---------------------------------------------------------------------------
# Parameter-space EPIG (INV-3)
# ---------------------------------------------------------------------------

def _resolve_P(P: np.ndarray, resolved_dim: int | None) -> np.ndarray:
    """Optionally restrict P to its first ``resolved_dim`` rows.

    The closed-form `ΔH_D` requires ``det Σ`` to be safely positive. INV-2
    of ALETHEIA_investigations.md identifies the data-dominated rows; the
    caller passes them as the leading rows of P (for the mass-only setup
    these are the two four-fermion directions, r = 2).
    """
    if resolved_dim is None:
        return P
    if resolved_dim <= 0 or resolved_dim > P.shape[0]:
        raise ValueError(
            f"resolved_dim={resolved_dim} out of range for P with {P.shape[0]} rows")
    return P[:resolved_dim]


def _sigma_chol(Sigma: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """Cholesky factor of Σ with a small ridge for numerical safety (Pitfall 3).

    Returns L lower-triangular such that L L^T = Σ + ridge·I. The ridge is
    far below the data-dominated eigenvalues so it does not shift the
    score appreciably; it only prevents underflow when ``r`` is set too
    aggressively. A correct call with ``resolved_dim`` restricted to the
    data-dominated subspace will have a healthy Σ.
    """
    r = Sigma.shape[0]
    return np.linalg.cholesky(Sigma + ridge * np.eye(r))


def _quad_sigma_inv(L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Per-row quadratic form u_i^T Σ^{-1} u_i for U of shape (n, r), via
    one triangular solve. Avoids forming Σ^{-1} explicitly (Pitfall 3).
    """
    # Solve L X = U^T  ⇒  X_i^T X_i = u_i^T (L L^T)^{-1} u_i = u_i^T Σ^{-1} u_i.
    try:
        from scipy.linalg import solve_triangular
        X = solve_triangular(L, U.T, lower=True)
    except ImportError:
        X = np.linalg.solve(L, U.T)
    return np.einsum("ri,ri->i", X, X)


def param_epig_d_acquire(model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                          M_pool: np.ndarray, P: np.ndarray, k: int,
                          *, sigma_y: float,
                          resolved_dim: int | None = None) -> np.ndarray:
    """Sequential-greedy D-optimal parameter-EPIG over m-values.

    Implements the closed-form ΔH_D from INV-3:

        ΔH_D(p) = -½ log( 1 - σ_y² (u^T Σ^{-1} u) / (1 + lev_p) )

    where Σ = σ_y² P A^{-1} P^T is the posterior covariance of the resolved
    c̃-subspace, u = P A^{-1} ψ_p, and lev_p = ψ_p^T A^{-1} ψ_p. Picks are
    made greedily; A^{-1} is maintained via Sherman-Morrison after each
    pick (matching ``epig_acquire_m``).

    Args:
        model: an ``IntentionFM`` (or anything providing ``A_inv_and_w``
            and ``psi_np``).
        M_ctx, Y_ctx: current context, shapes (K,), (K,).
        M_pool: candidate pool of m-values, shape (n_P,).
        P: projection matrix, shape (r, D). For INV-3, ``P = V^T W`` with
            W the frozen disclosure probe (INV-2) and V the Fisher
            rotation. The first ``resolved_dim`` rows should be the
            data-dominated directions.
        k: number of picks (with replacement avoided via masking).
        sigma_y: noise stddev on Y (the ratio). Σ scales as σ_y², so the
            score is independent of an overall σ_y rescaling — but the
            log-argument numerator does pick up the factor.
        resolved_dim: if given, restrict P to its first ``resolved_dim``
            rows. INV-2 identifies these as the data-dominated directions
            (mass-only: r=2, the two four-fermion directions).

    Returns:
        Array of length ``k`` with indices into ``M_pool`` of the picks
        (in selection order).
    """
    P_use = np.asarray(_resolve_P(P, resolved_dim), dtype=np.float64)
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)        # (D, D)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    Psi_pool = np.asarray(model.psi_np(M_pool), dtype=np.float64)  # (n_P, D)
    avail = np.ones(len(Psi_pool), dtype=bool)
    chosen: list[int] = []
    sy2 = float(sigma_y) ** 2

    for _ in range(k):
        # Σ in the resolved subspace, current A_inv.
        Sigma = sy2 * (P_use @ A_inv @ P_use.T)           # (r, r)
        L = _sigma_chol(Sigma)
        # u_p = P A^{-1} ψ_p for every candidate p.
        AP = Psi_pool @ A_inv                              # (n_P, D)
        U  = AP @ P_use.T                                  # (n_P, r)
        lev_P = np.einsum("pd,pd->p", AP, Psi_pool)        # (n_P,)
        quad = _quad_sigma_inv(L, U)                       # (n_P,)
        arg = 1.0 - sy2 * quad / (1.0 + lev_P)
        # Pitfall 3: clamp away from zero; correct usage should not trigger this.
        arg_clamped = np.maximum(arg, 1e-12)
        if np.any(arg < 1e-12):
            warnings.warn(
                "param_epig_d_acquire: log argument clamped — check resolved_dim "
                "or σ_y; some Σ may be near-singular.",
                RuntimeWarning, stacklevel=2)
        scores = -0.5 * np.log(arg_clamped)
        scores = np.where(avail, scores, -np.inf)
        i = int(np.argmax(scores))
        chosen.append(i)
        avail[i] = False
        # Sherman-Morrison update of A_inv.
        p = Psi_pool[i]
        Ap = A_inv @ p
        A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
    return np.array(chosen)


def param_epig_a_acquire(model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                          M_pool: np.ndarray, P: np.ndarray, k: int,
                          target_direction: int,
                          *, sigma_y: float) -> np.ndarray:
    """Sequential-greedy A-optimal parameter-EPIG on a single c̃ direction.

    Score (INV-3):

        ΔH_a(p) = -½ log( 1 - σ_y² u_a² / ((1 + lev_p) Σ_{aa}) )

    where ``a = target_direction`` indexes the row of P. This is the same
    machinery as ΔH_D restricted to one direction; useful when the
    weakly-resolved direction (e.g. c̃_2) is the publication target.

    Args:
        model, M_ctx, Y_ctx, M_pool, k, sigma_y: as in
            ``param_epig_d_acquire``.
        P: projection matrix, shape (r, D).
        target_direction: index in ``[0, r)`` of the c̃ direction to
            target.
    """
    if target_direction < 0 or target_direction >= P.shape[0]:
        raise ValueError(
            f"target_direction={target_direction} out of range for P with "
            f"{P.shape[0]} rows")
    P = np.asarray(P, dtype=np.float64)
    p_a = P[target_direction]                              # (D,)
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    Psi_pool = np.asarray(model.psi_np(M_pool), dtype=np.float64)
    avail = np.ones(len(Psi_pool), dtype=bool)
    chosen: list[int] = []
    sy2 = float(sigma_y) ** 2

    for _ in range(k):
        Sigma_aa = sy2 * float(p_a @ A_inv @ p_a)
        AP = Psi_pool @ A_inv                              # (n_P, D)
        u_a = AP @ p_a                                     # (n_P,)
        lev_P = np.einsum("pd,pd->p", AP, Psi_pool)
        arg = 1.0 - sy2 * (u_a ** 2) / ((1.0 + lev_P) * max(Sigma_aa, 1e-30))
        arg_clamped = np.maximum(arg, 1e-12)
        if np.any(arg < 1e-12):
            warnings.warn(
                "param_epig_a_acquire: log argument clamped — check σ_y or P.",
                RuntimeWarning, stacklevel=2)
        scores = -0.5 * np.log(arg_clamped)
        scores = np.where(avail, scores, -np.inf)
        i = int(np.argmax(scores))
        chosen.append(i)
        avail[i] = False
        p = Psi_pool[i]
        Ap = A_inv @ p
        A_inv = A_inv - np.outer(Ap, Ap) / (1.0 + p @ Ap)
    return np.array(chosen)


def _load_probe_artifact(path: str) -> dict[str, Any]:
    """Load the INV-2 disclosure-probe artifact ``probe_W_mass_only.npz``.

    Expected keys: ``W`` (4×D probe), ``V`` (4×4 Fisher rotation),
    ``sigma_y`` (scalar) and optionally ``resolved_dim`` (int) and
    ``P`` (precomputed V^T W). ``W`` is fit and frozen by INV-2 (see
    Pitfall 6 of ALETHEIA_investigations.md); this loader does not re-fit
    anything. Raises FileNotFoundError with an explicit pointer to INV-2
    if the file is absent.
    """
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"probe artifact not found at {path}. "
            "This file is emitted by INV-2 (disclosure-probe efficiency); "
            "run that investigation first to fit and freeze W. "
            "See ALETHEIA_investigations.md §INV-2.")
    npz = np.load(path)
    out: dict[str, Any] = {}
    for key in ("W", "V", "sigma_y", "P", "resolved_dim"):
        if key in npz.files:
            arr = npz[key]
            out[key] = arr.item() if arr.ndim == 0 else arr
    if "P" not in out and "W" in out and "V" in out:
        out["P"] = out["V"].T @ out["W"]
    return out
