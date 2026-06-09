r"""Shared, capacity-controlled evaluation metrics for the FM capability matrix.

These are the scoring primitives the cells use. The design directly repairs the
failure mode of the prior ALETHIA single-probe survey (see
ALETHEIA_manifold_informer_landscape_practice.md §3): its P3/P4 "manifold
identity" gates fit 16 latent dims against ≤10 targets *in-sample*, so R²≈1
measured probe capacity, not representation quality.

The fixes encoded here:
  * `held_out_probe` always fits on a train split and scores on a disjoint
    test split — never in-sample.
  * The probe is a *ridge* with the regulariser chosen on an inner validation
    split, so an over-wide representation cannot interpolate the targets.
  * `random_feature_floor` runs the identical probe on a random projection of
    the raw events with the *same* output dimension, giving the capacity floor
    a representation must beat to count as informative.
  * Every probe reports `d_z` (representation width) and `n_train` so the
    capacity regime is auditable.

Other metrics (coverage/ECE for inference, ROC/significance for anomaly,
Wasserstein/energy distance for generation) are standard and pure-numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import wasserstein_distance


# --------------------------------------------------------------------------
# Regression / representation quality.
# --------------------------------------------------------------------------


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, pooled over all targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _ridge_fit(X, Y, alpha):
    """Closed-form ridge with an explicit intercept. Returns (W, b)."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    d = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    W = np.linalg.solve(A, Xc.T @ Yc)
    b = Y.mean(0) - X.mean(0) @ W
    return W, b


@dataclass
class ProbeResult:
    r2: float                      # held-out pooled R²
    r2_per_target: list            # held-out R² per target column
    d_z: int                       # representation width
    n_train: int
    alpha: float                   # selected ridge regulariser
    floor_r2: float = float("nan")  # random-feature capacity floor (same d_z)
    margin: float = float("nan")    # r2 - floor_r2 (informative if > 0)

    def to_dict(self):
        return {
            "r2": self.r2, "r2_per_target": self.r2_per_target,
            "d_z": self.d_z, "n_train": self.n_train, "alpha": self.alpha,
            "floor_r2": self.floor_r2, "margin": self.margin,
        }


def held_out_probe(
    Z_train: np.ndarray, Y_train: np.ndarray,
    Z_test: np.ndarray, Y_test: np.ndarray,
    *,
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    val_frac: float = 0.25,
    seed: int = 0,
) -> ProbeResult:
    """Fit a ridge Z->Y on train (alpha picked on an inner val split), score on
    the disjoint test split. This is the capacity-controlled representation
    probe — linear, regularised, never in-sample.
    """
    Z_train = np.asarray(Z_train, float)
    Y_train = np.atleast_2d(np.asarray(Y_train, float))
    if Y_train.shape[0] != Z_train.shape[0]:
        Y_train = Y_train.T
    Z_test = np.asarray(Z_test, float)
    Y_test = np.atleast_2d(np.asarray(Y_test, float))
    if Y_test.shape[0] != Z_test.shape[0]:
        Y_test = Y_test.T

    rng = np.random.default_rng(seed)
    n = Z_train.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    vi, ti = idx[:n_val], idx[n_val:]

    best_alpha, best_val = alphas[0], -np.inf
    for a in alphas:
        W, b = _ridge_fit(Z_train[ti], Y_train[ti], a)
        val_r2 = r2_score(Y_train[vi], Z_train[vi] @ W + b)
        if val_r2 > best_val:
            best_val, best_alpha = val_r2, a

    W, b = _ridge_fit(Z_train, Y_train, best_alpha)
    pred = Z_test @ W + b
    per = [r2_score(Y_test[:, j:j + 1], pred[:, j:j + 1])
           for j in range(Y_test.shape[1])]
    return ProbeResult(
        r2=r2_score(Y_test, pred), r2_per_target=per,
        d_z=Z_train.shape[1], n_train=n, alpha=best_alpha,
    )


def random_feature_floor(
    raw_train: np.ndarray, Y_train: np.ndarray,
    raw_test: np.ndarray, Y_test: np.ndarray,
    d_z: int, *, seed: int = 0,
) -> float:
    """Capacity floor: a random nonlinear projection of the raw per-scenario
    summary to width ``d_z``, scored through the identical held-out probe.

    raw_* are (n, d_raw) per-scenario summaries (e.g. mean/quantiles of raw
    events) — NOT the learned representation. A learned representation that
    fails to beat this floor at the same width is not adding information beyond
    what a random projection of the raw data already provides.
    """
    rng = np.random.default_rng(seed)
    d_raw = raw_train.shape[1]
    Wp = rng.standard_normal((d_raw, d_z)) / np.sqrt(d_raw)
    bp = rng.standard_normal(d_z)
    Zr_tr = np.tanh(raw_train @ Wp + bp)
    Zr_te = np.tanh(raw_test @ Wp + bp)
    return held_out_probe(Zr_tr, Y_train, Zr_te, Y_test, seed=seed).r2


def probe_with_floor(
    Z_train, Y_train, Z_test, Y_test,
    raw_train, raw_test, *, seed: int = 0,
) -> ProbeResult:
    """held_out_probe plus the matched random-feature floor and margin."""
    res = held_out_probe(Z_train, Y_train, Z_test, Y_test, seed=seed)
    res.floor_r2 = random_feature_floor(
        raw_train, Y_train, raw_test, Y_test, res.d_z, seed=seed)
    res.margin = res.r2 - res.floor_r2
    return res


# --------------------------------------------------------------------------
# Probabilistic inference: coverage & calibration.
# --------------------------------------------------------------------------


def gaussian_coverage(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray,
    levels=(0.5, 0.68, 0.8, 0.9, 0.95),
) -> dict:
    """Empirical central-interval coverage of a Gaussian predictive, per level.

    Returns nominal levels, empirical coverage, and the mean |coverage-nominal|
    calibration error (lower is better). Pooled over all target dims.
    """
    from scipy.stats import norm
    y_true = np.asarray(y_true, float).ravel()
    mean = np.asarray(mean, float).ravel()
    std = np.clip(np.asarray(std, float).ravel(), 1e-9, None)
    z = np.abs(y_true - mean) / std
    emp = []
    for lv in levels:
        half = norm.ppf(0.5 + lv / 2.0)
        emp.append(float(np.mean(z <= half)))
    cal_err = float(np.mean([abs(e - lv) for e, lv in zip(emp, levels)]))
    return {"levels": list(levels), "coverage": emp, "calibration_error": cal_err}


def interval_coverage_from_quantiles(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, nominal: float,
) -> float:
    """Empirical coverage of a [lo, hi] predictive interval at a nominal level."""
    y_true = np.asarray(y_true, float).ravel()
    lo = np.asarray(lo, float).ravel()
    hi = np.asarray(hi, float).ravel()
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


# --------------------------------------------------------------------------
# Anomaly detection: ROC AUC & significance.
# --------------------------------------------------------------------------


def roc_auc(scores_bg: np.ndarray, scores_sig: np.ndarray) -> float:
    """AUC for "higher score = more anomalous". Rank-based (Mann-Whitney)."""
    s_bg = np.asarray(scores_bg, float)
    s_sig = np.asarray(scores_sig, float)
    all_s = np.concatenate([s_bg, s_sig])
    ranks = all_s.argsort().argsort().astype(float) + 1.0
    n_bg, n_sig = len(s_bg), len(s_sig)
    r_sig = ranks[n_bg:].sum()
    auc = (r_sig - n_sig * (n_sig + 1) / 2.0) / (n_bg * n_sig)
    return float(auc)


def significance_at_background(
    scores_bg: np.ndarray, scores_sig: np.ndarray, bg_eff: float = 0.01,
) -> dict:
    """Signal efficiency and a Poisson significance at a fixed background
    working point. The threshold is the (1 - bg_eff) quantile of background
    scores; eps_s = fraction of signal above it; Z = eps_s/sqrt(eps_b) is the
    standard fixed-background figure of merit (relative S/sqrt(B) at equal
    pre-cut yields).
    """
    s_bg = np.asarray(scores_bg, float)
    s_sig = np.asarray(scores_sig, float)
    thr = np.quantile(s_bg, 1.0 - bg_eff)
    eps_s = float(np.mean(s_sig >= thr))
    eps_b = float(np.mean(s_bg >= thr))
    z = eps_s / np.sqrt(max(eps_b, 1e-6))
    return {"bg_eff": bg_eff, "threshold": float(thr),
            "sig_eff": eps_s, "bg_eff_emp": eps_b, "significance": z}


# --------------------------------------------------------------------------
# Generative fidelity: marginal Wasserstein & 2D energy distance.
# --------------------------------------------------------------------------


def marginal_wasserstein(real: np.ndarray, gen: np.ndarray) -> dict:
    """W1 distance per feature column between real and generated event clouds."""
    real = np.asarray(real, float)
    gen = np.asarray(gen, float)
    d = real.shape[1]
    w = [float(wasserstein_distance(real[:, j], gen[:, j])) for j in range(d)]
    return {"w1_per_feature": w, "w1_mean": float(np.mean(w))}


def energy_distance_2d(real: np.ndarray, gen: np.ndarray,
                       *, max_n: int = 1500, seed: int = 0) -> float:
    """Multivariate energy distance D^2 = 2 E|X-Y| - E|X-X'| - E|Y-Y'|.

    Subsamples to max_n per cloud for an O(n^2) pairwise compute. >=0, zero iff
    distributions match. Lower is better.
    """
    rng = np.random.default_rng(seed)
    real = np.asarray(real, float)
    gen = np.asarray(gen, float)
    if len(real) > max_n:
        real = real[rng.choice(len(real), max_n, replace=False)]
    if len(gen) > max_n:
        gen = gen[rng.choice(len(gen), max_n, replace=False)]

    def _mean_pdist(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return float(np.sqrt(np.maximum(d2, 0.0)).mean())

    xy = _mean_pdist(real, gen)
    xx = _mean_pdist(real, real)
    yy = _mean_pdist(gen, gen)
    return float(max(2.0 * xy - xx - yy, 0.0))


# --------------------------------------------------------------------------
# Raw per-scenario summary (for floors and simple baselines).
# --------------------------------------------------------------------------


def raw_event_summary(X: np.ndarray) -> np.ndarray:
    """Permutation-invariant hand summary of an event set, for capacity floors.

    X: (n_scenarios, n_events, 2) -> (n_scenarios, d_raw). Uses per-feature
    mean, std, and a few quantiles of (log m, cos θ*) — the kind of summary a
    physicist would write down without any learned representation.
    """
    X = np.asarray(X, float)
    qs = [0.1, 0.5, 0.9]
    feats = [X.mean(1), X.std(1)]
    for q in qs:
        feats.append(np.quantile(X, q, axis=1))
    return np.concatenate(feats, axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # quick self-test
    Z = rng.standard_normal((200, 8))
    Wtrue = rng.standard_normal((8, 4))
    Y = Z @ Wtrue + 0.1 * rng.standard_normal((200, 4))
    res = held_out_probe(Z[:150], Y[:150], Z[150:], Y[150:])
    print("probe r2", round(res.r2, 3), "alpha", res.alpha, "d_z", res.d_z)
    print("coverage", gaussian_coverage(Y[150:, 0], (Z @ Wtrue)[150:, 0],
                                         np.full(50, 0.1))["calibration_error"])
    print("auc", round(roc_auc(rng.standard_normal(500),
                                rng.standard_normal(500) + 1.0), 3))
    a = rng.standard_normal((400, 2)); b = rng.standard_normal((400, 2)) + 0.5
    print("energy", round(energy_distance_2d(a, b), 3),
          "w1", round(marginal_wasserstein(a, b)["w1_mean"], 3))
