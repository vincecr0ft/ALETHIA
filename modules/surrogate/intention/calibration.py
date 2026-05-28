"""Leverage-stratified split-conformal calibrator for the Intention head.

Mirrors modules/surrogate/calibration.py:ConformalCalibrator with the
predict / leverage interface swapped to the Intention model. The model
fingerprint guards against the bug-class refit-after-update issue documented
in docs/research/04-epig-conformal/empirical-results.md section 2.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _model_fingerprint(M_ctx: np.ndarray, Y_ctx: np.ndarray) -> str:
    """Hash the current context. If the FM rebinds context, the calibrator
    must be refit."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(M_ctx).tobytes())
    h.update(np.ascontiguousarray(Y_ctx).tobytes())
    return h.hexdigest()


class IntentionConformal:
    """Leverage-stratified split conformal for the Intention head.

    Predictive standard deviation follows the homoscedastic Bayesian-linear
    form of paper Eq. (5):

        sd_raw(q) = sigma_y * sqrt(1 + lev(q))

    with ``sigma_y`` estimated from the residuals on the calibration set.
    Per-stratum conformal multipliers absorb any residual heteroscedasticity.
    """

    def __init__(self, n_strata: int = 5, noise_frac: float = 0.05):
        self.n_strata = n_strata
        self.noise_frac = noise_frac  # retained for backwards-compatible callers
        self.edges = None
        self.factors = {}
        self.fingerprint = None
        self.sigma_y = None

    def _sd_raw(self, mu: np.ndarray, lev: np.ndarray) -> np.ndarray:
        return self.sigma_y * np.sqrt(1.0 + lev)

    def fit(self, model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
            M_cal: np.ndarray, Y_cal: np.ndarray,
            coverages: tuple = (0.683, 0.954)) -> "IntentionConformal":
        """Fit per-stratum multipliers on calibration set drawn from the
        broader probe region (invariant: NOT only training distribution)."""
        mu_cal = model.predict_np(M_ctx, Y_ctx, M_cal)
        lev_cal = model.leverage(M_ctx, M_cal)
        # Whitened-residual estimator of sigma_y: removes the (1+lev) scaling
        # before taking the standard deviation, so leverage-induced variance
        # does not inflate the homoscedastic constant.
        whitened = (Y_cal - mu_cal) / np.sqrt(1.0 + lev_cal)
        self.sigma_y = float(np.std(whitened, ddof=1))
        if self.sigma_y <= 0.0:
            self.sigma_y = 1e-6
        sd_raw = self._sd_raw(mu_cal, lev_cal)
        scores = np.abs(Y_cal - mu_cal) / np.maximum(sd_raw, 1e-12)
        self.edges = np.quantile(lev_cal, np.linspace(0, 1, self.n_strata + 1))
        self.edges[0] = -np.inf
        self.edges[-1] = np.inf
        for cov in coverages:
            default = 1.0 if cov < 0.7 else 1.96
            f = np.full(self.n_strata, default, dtype=float)
            for s in range(self.n_strata):
                mask = (lev_cal >= self.edges[s]) & (lev_cal < self.edges[s + 1])
                n = int(mask.sum())
                if n >= 8:
                    q = min(1.0, float(np.ceil((n + 1) * cov)) / n)
                    f[s] = float(np.quantile(scores[mask], q))
            self.factors[cov] = f
        self.fingerprint = _model_fingerprint(M_ctx, Y_ctx)
        return self

    def stratum(self, lev: np.ndarray) -> np.ndarray:
        return np.clip(
            np.searchsorted(self.edges[1:-1], lev), 0, self.n_strata - 1)

    def coverage_sigma(self, model, M_ctx: np.ndarray, Y_ctx: np.ndarray,
                       M_q: np.ndarray, coverage: float = 0.683) -> np.ndarray:
        # The runtime check: refuse to use stale calibrator.
        if _model_fingerprint(M_ctx, Y_ctx) != self.fingerprint:
            raise RuntimeError(
                "OutdatedCalibratorError: context has changed since this "
                "calibrator was fit. Call .fit(model, M_ctx, Y_ctx, ...) "
                "on the current context.")
        mu_q = model.predict_np(M_ctx, Y_ctx, M_q)
        lev_q = model.leverage(M_ctx, M_q)
        sd_raw = self._sd_raw(mu_q, lev_q)
        strat = self.stratum(lev_q)
        return self.factors[coverage][strat] * sd_raw
