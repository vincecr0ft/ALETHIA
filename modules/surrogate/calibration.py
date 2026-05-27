r"""
Leverage-stratified split-conformal calibration.

Standard split-conformal gives MARGINAL coverage: a (1-alpha) interval
covers (1-alpha) of test points on average. Stratifying by predictive
uncertainty (we use leverage quantiles) gives approximate CONDITIONAL
coverage: each stratum hits its nominal rate. This is what fixes the
high-leverage under-coverage you see if you trust raw Gaussian predictive
intervals on extrapolation points.

The conformal layer rescales the raw model's predicted ``sigma`` by a
per-stratum factor learned on a held-out calibration set. The interface
returns coverage-specific sigmas: ``coverage_sigma(model, C, M, 0.683)``
gives ``sigma`` such that ``|y - mu| < sigma`` covers ~68.3% of points
within each leverage stratum.
"""
from __future__ import annotations

import numpy as np


class ConformalCalibrator:
    """Stratified split-conformal for leverage-grouped Gaussian predictives."""

    def __init__(self, n_strata: int = 5):
        self.n_strata = n_strata
        self.edges:   np.ndarray | None  = None        # (n_strata + 1,)
        self.factors: dict[float, np.ndarray] = {}     # coverage -> (n_strata,)

    @staticmethod
    def _finite_sample_quantile(n: int, coverage: float) -> float:
        """Exchangeability-aware quantile rank: ceil((n+1) coverage) / n."""
        return min(1.0, float(np.ceil((n + 1) * coverage)) / n)

    def fit(
        self,
        model,
        C: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        coverages: tuple[float, ...] = (0.683, 0.954),
    ) -> "ConformalCalibrator":
        """
        Fit per-stratum conformal multipliers from a held-out calibration set.

        For honest conditional coverage on the broader probe region, the
        calibration set should cover the same input distribution the model
        will be queried on (NOT just the training distribution).
        """
        mu_cal  = model.predict(C, M, return_std=False)
        sd_cal  = model.predict(C, M)[1]
        lev_cal = model.leverage(C, M)
        scores  = np.abs(Y - mu_cal) / np.maximum(sd_cal, 1e-12)

        self.edges = np.quantile(lev_cal, np.linspace(0, 1, self.n_strata + 1))
        self.edges[0]  = -np.inf
        self.edges[-1] =  np.inf

        for cov in coverages:
            # Defaults if a stratum is too thin to estimate from.
            default = 1.0 if cov < 0.7 else 1.96
            f = np.full(self.n_strata, default, dtype=float)
            for s in range(self.n_strata):
                m = (lev_cal >= self.edges[s]) & (lev_cal < self.edges[s + 1])
                n = int(m.sum())
                if n >= 8:
                    q = self._finite_sample_quantile(n, cov)
                    f[s] = float(np.quantile(scores[m], q))
            self.factors[cov] = f
        return self

    def stratum_index(self, lev: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges[1:-1], lev), 0, self.n_strata - 1)

    def coverage_sigma(
        self,
        model,
        C: np.ndarray,
        M: np.ndarray,
        coverage: float = 0.683,
    ) -> np.ndarray:
        """Return sigma such that |y - mu| < sigma covers ``coverage`` fraction
        within the leverage stratum of each query point."""
        if coverage not in self.factors:
            raise KeyError(f"Calibrator was not fit for coverage {coverage}. "
                           f"Available: {sorted(self.factors)}.")
        sd_raw = model.predict(C, M)[1]
        lev    = model.leverage(C, M)
        return sd_raw * self.factors[coverage][self.stratum_index(lev)]

    # ---- serialization ----
    def state_dict(self) -> dict:
        return {
            "n_strata": self.n_strata,
            "edges":    self.edges,
            "factors":  {float(k): v for k, v in self.factors.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ConformalCalibrator":
        cc = cls(n_strata=state["n_strata"])
        cc.edges   = state["edges"]
        cc.factors = state["factors"]
        return cc
