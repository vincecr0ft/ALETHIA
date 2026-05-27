"""EPIG closed-form acquisition over the Intention feature space.

Direct translation of the closed-form derivation in
modules/surrogate/acquisition.py:25-67, with phi_joint(c, m) replaced by
the Intention head's psi_theta(m). Sherman-Morrison sequential greedy.

The candidate space is m-values to query the oracle at; the target set is
held-out m-values inside the drifted region. The result of the call is a
list of m's to ask the oracle about; the orchestrator does the oracle
query and folds the result into the context.
"""
from __future__ import annotations

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
