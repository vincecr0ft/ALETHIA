"""Three drift detectors for the full-chain run.

Implemented inline (no modules/drift/ yet). Standalone numpy functions plus
small dataclasses for streaming state. Theoretical references:

- DAS-CUSUM: Ahad, Davenport, Xie, arXiv:2210.17353.
- BH FDR: Benjamini-Hochberg 1995.
- Eckart-Young condition-number bound: Eckart-Young 1936.

Threshold h=6.0 for DAS-CUSUM gives ARL_0 ~ 1000 empirically on this
implementation (validated in docs/research/03-drift/empirical-results.md).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.stats import binomtest


@dataclass
class DASCUSUMState:
    """Streaming DAS-CUSUM state. Two-sided (pos + neg)."""
    S_pos: float = 0.0
    S_neg: float = 0.0
    n_seen: int = 0
    buf: deque = field(default_factory=lambda: deque(maxlen=30))


def das_cusum_update(state: DASCUSUMState, z: float, *,
                     w: int = 30, h: float = 6.0, k: float = 0.5) -> tuple:
    """Single-step streaming update.

    Returns (state, fired, S_t) where S_t = max(S_pos, S_neg).
    Warmup is 2w samples; during warmup we accumulate the buffer but emit
    fired=False unconditionally.
    """
    state.n_seen += 1
    state.buf.append(z)
    if state.n_seen < 2 * w:
        return state, False, 0.0
    # Use buffered window to estimate variance adaptively.
    arr = np.fromiter(state.buf, dtype=float)
    mu_w = arr.mean()
    sd_w = max(arr.std(ddof=1), 1e-6)
    z_s = (z - mu_w) / sd_w   # standardised against current window
    # Two-sided CUSUM with reference k.
    state.S_pos = max(0.0, state.S_pos + z_s - k)
    state.S_neg = max(0.0, state.S_neg - z_s - k)
    S_t = max(state.S_pos, state.S_neg)
    fired = S_t > h
    if fired:
        # Reset after a firing so we can detect subsequent shifts.
        state.S_pos, state.S_neg = 0.0, 0.0
    return state, fired, float(S_t)


def coverage_bh_test(counts_covered: np.ndarray, counts_total: np.ndarray,
                     target_coverage: float = 0.683,
                     alpha: float = 0.05) -> tuple:
    """Per-region two-sided binomial test on empirical coverage, BH-corrected.

    Returns (fired, pvalues, failing_mask, threshold).
    """
    S = len(counts_covered)
    pvalues = np.ones(S)
    for s in range(S):
        n = int(counts_total[s])
        if n < 8:
            pvalues[s] = 1.0
            continue
        k = int(counts_covered[s])
        # Two-sided exact binomial.
        result = binomtest(k, n, target_coverage, alternative="two-sided")
        pvalues[s] = float(result.pvalue)
    # BH correction.
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    m = S
    bh_threshold = alpha * (np.arange(1, m + 1) / m)
    rejected_sorted = sorted_p <= bh_threshold
    if rejected_sorted.any():
        # All p <= largest passing threshold get rejected (the BH step-up).
        max_idx = np.where(rejected_sorted)[0].max()
        threshold = float(bh_threshold[max_idx])
        failing_mask = np.zeros(S, dtype=bool)
        failing_mask[order[: max_idx + 1]] = True
    else:
        threshold = float(bh_threshold[0])
        failing_mask = np.zeros(S, dtype=bool)
    fired = bool(failing_mask.any())
    return fired, pvalues, failing_mask, threshold


def kappa_drift(model, M_ctx: np.ndarray, recent_M: np.ndarray,
                train_proj_var: float, *,
                kappa_threshold: float = 1e4,
                proj_ratio_threshold: float = 3.0) -> tuple:
    """Coverage drift via condition number and v_min projection ratio.

    model: an IntentionFM (provides psi_np and kappa_A).
    M_ctx: current context m-values.
    recent_M: recent probe m-values.
    train_proj_var: baseline Var_train(<psi(m), v_min>) computed once.

    Returns (fired, kappa, proj_ratio).
    """
    # Compute kappa(A) on current context.
    kappa = model.kappa_A(M_ctx)
    # Eigendecompose A to get v_min.
    Psi = model.psi_np(M_ctx)
    d = Psi.shape[1]
    A = Psi.T @ Psi + model.alpha * np.eye(d)
    eigs, vecs = np.linalg.eigh(A)
    v_min = vecs[:, 0]  # column for smallest eigenvalue
    # Recent projection variance.
    Psi_recent = model.psi_np(recent_M)
    proj = Psi_recent @ v_min
    recent_proj_var = float(np.var(proj))
    proj_ratio = recent_proj_var / max(train_proj_var, 1e-30)
    fired = (kappa > kappa_threshold) or (proj_ratio > proj_ratio_threshold)
    return bool(fired), float(kappa), float(proj_ratio)


def aggregate_action(acc: bool, cal: bool, cov: bool,
                     history: list, persistence_N: int = 3) -> tuple:
    """8-row decision table from docs/research/03-drift/eval.md section 5.

    Returns (action, target_signal). "-> recal" tail is always implicit
    after a retrain (caller handles).
    """
    # Persistence for the (acc=1, cal=0, cov=0) row: require N consecutive
    # to escalate beyond watch.
    if acc and not cal and not cov:
        # Look back N-1 cycles for the same pattern.
        recent_acc_only = sum(1 for h in history[-(persistence_N - 1):]
                              if h == (True, False, False))
        if recent_acc_only >= persistence_N - 1:
            return "local_retrain", "acc"
        return "watch", None
    if not acc and cal and not cov:
        return "recal", "cal"
    if not acc and not cal and cov:
        return "local_retrain", "cov"
    if acc and cal and not cov:
        return "recal_then_check", "cal"
    if acc and not cal and cov:
        return "local_retrain", "cov"
    if not acc and cal and cov:
        return "local_retrain", "cov"
    if acc and cal and cov:
        return "global_retrain", "cov"
    return "noop", None
