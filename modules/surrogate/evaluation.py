r"""
Evaluation diagnostics: coverage, drift-signal validation, per-stratum
metrics. Used by demos and by Phoenix online evaluators (see
docs/INTEGRATION.md).
"""
from __future__ import annotations

import numpy as np


def empirical_coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Fraction of points where ``|y - mu| < sigma``."""
    return float(np.mean(np.abs(y - mu) < sigma))


def decile_calibration(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-decile (predicted sigma, empirical RMS error). Validates that the
    drift signal sigma is a quantitatively correct error predictor.

    If sigma is calibrated, RMS_emp / sigma_pred ~ 1 in each decile.
    """
    err     = np.abs(y - mu)
    order   = np.argsort(sigma)
    deciles = np.array_split(np.arange(len(sigma)), n_bins)
    pred, emp = [], []
    for ix in deciles:
        pred.append(float(sigma[order][ix].mean()))
        emp.append(float(np.sqrt(np.mean(err[order][ix] ** 2))))
    return np.array(pred), np.array(emp)


def stratified_coverage(
    model,
    calibrator,
    C: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    coverage: float = 0.683,
) -> list[dict]:
    """Per-stratum coverage on a held-out test set, raw vs conformal.

    Returns a list of dicts (one per leverage stratum) suitable for printing
    or feeding into a Phoenix evaluator. Each dict carries the empirical
    coverage of the raw 1σ-equivalent (or 2σ-equivalent at 0.954) and the
    conformal-rescaled coverage.
    """
    mu, sd_raw = model.predict(C, M)
    sd_cc      = calibrator.coverage_sigma(model, C, M, coverage=coverage)
    lev        = model.leverage(C, M)
    strata     = calibrator.stratum_index(lev)
    sigma_z    = 1.0 if coverage < 0.7 else 1.96

    rows: list[dict] = []
    for s in range(calibrator.n_strata):
        m = strata == s
        if m.sum() == 0:
            continue
        rows.append({
            "stratum":            s,
            "n":                  int(m.sum()),
            "mean_leverage":      float(lev[m].mean()),
            "raw_coverage":       empirical_coverage(Y[m], mu[m], sigma_z * sd_raw[m]),
            "conformal_coverage": empirical_coverage(Y[m], mu[m], sd_cc[m]),
            "target_coverage":    coverage,
        })
    return rows
