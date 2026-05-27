r"""
Ground-truth oracle protocol and a placeholder analytic implementation.

The Oracle interface is intentionally minimal so it can be replaced by a real
analytic SMEFT calculator (closed-form Drell-Yan from Boughezal-Petriello or
similar) or a MadGraph subprocess wrapper without touching the rest of the
package. Implementations return a (n,) array of mu values, optionally with
multiplicative noise that reflects Monte Carlo statistical uncertainty.
"""
from __future__ import annotations
from typing import Protocol

import numpy as np

from .features import phi_joint, D_C, D_X, N_WC, PAIRS, K_X


class Oracle(Protocol):
    """Callable interface: oracle(c, m, noise=True) -> (n,) array of mu."""

    def __call__(
        self,
        c: np.ndarray,
        m: np.ndarray,
        *,
        noise: bool = True,
    ) -> np.ndarray: ...

    def truth(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Noiseless oracle, for evaluation."""
        ...


class DummyAnalyticOracle:
    """
    Deterministic SMEFT-like oracle for tests and demos.

    The truth ``mu(c, m)`` is exactly polynomial in (c, log m) by construction:
    ``mu = phi_joint(c, m) @ W_true.flatten()`` with ``W_true`` shape
    ``(D_C, D_X)``. ``W_true[0]`` encodes the SM piece (mu = 1 at c = 0),
    the next ``N_WC`` rows encode linear interference (energy-growing,
    vanishing at log m = 0), and the remaining rows encode the quadratic
    SMEFT term with stronger energy growth.

    Noise is multiplicative Gaussian at fractional level ``noise_frac``,
    a reasonable proxy for sqrt(N) MC noise normalised by event yield.

    Replace with the real analytic calculator when ready. The signature is
    fixed; nothing else needs to change.
    """

    def __init__(self, seed: int = 0, noise_frac: float = 0.05):
        self.seed = seed
        self.noise_frac = float(noise_frac)
        rng = np.random.default_rng(seed)
        self.W = np.zeros((D_C, D_X))
        # SM = 1
        self.W[0, 0] = 1.0
        # Linear interference per Wilson coefficient: smooth higher modes only
        for a in range(N_WC):
            self.W[1 + a, 1:] = rng.normal(0, 0.18, K_X - 1)
        # Quadratic in c, faster energy growth: only m^2, m^3, m^4 modes
        for k in range(len(PAIRS)):
            self.W[1 + N_WC + k, 2:] = rng.normal(0, 0.10, K_X - 2)
        self._W_flat = self.W.flatten()
        self._noise_rng = np.random.default_rng(seed + 100003)

    def __call__(self, c, m, *, noise: bool = True) -> np.ndarray:
        mu = phi_joint(c, m) @ self._W_flat
        if noise:
            mu = mu + self._noise_rng.normal(0.0, self.noise_frac * np.abs(mu))
        return mu

    def truth(self, c, m) -> np.ndarray:
        return phi_joint(c, m) @ self._W_flat
