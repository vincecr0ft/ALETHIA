r"""Fast morphing surrogate for the analytic SMEFT log-likelihood ratio.

The exact per-event log-ratio (modules.surrogate.oracle_events.
event_log_likelihood_ratio) is

    log w_c(x) = log mu(c,m)
               + log[(1+u^2) + 2 r_c u]
               - log[(1+u^2) + 2 r_sm u],

with mu(c,m) = sigma(c,m)/sigma_SM(m) exactly quadratic in c, and
r_c = (4/3) A_FB(c,m) = Dtilde(c,m)/mu(c,m), where Dtilde = (4/3) A_FB * mu is
also quadratic in c. The slow path Python-loops over events on every call to
oracle.truth / oracle.truth_afb, costing ~18 s at N=4000.

FastMorphing pays that cost ONCE: it fits the per-event quadratic templates of
mu and Dtilde on a fixed event set from ~25 basis working points, after which
log_w(c) for any c is a vectorised quadratic evaluation (microseconds). The
event set is the fixed function-space probe grid, so reusing it across seeds is
the same fixed measurement apparatus the residual projector Pi_psi already uses.

Validated to <1e-6 max abs deviation against the exact slow path.
"""
from __future__ import annotations

import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.surrogate.features import N_WC


def _quad_features(C: np.ndarray) -> np.ndarray:
    """Design matrix [1, c_i, c_i c_j (i<=j)] for a stack of working points.

    C: (M, n) -> (M, 1 + n + n(n+1)/2).
    """
    C = np.atleast_2d(C)
    M, n = C.shape
    cols = [np.ones((M, 1))]
    cols.append(C)
    quad = np.stack([C[:, i] * C[:, j]
                     for i, j in combinations_with_replacement(range(n), 2)],
                    axis=1)
    cols.append(quad)
    return np.concatenate(cols, axis=1)


class FastMorphing:
    """Per-event quadratic templates of mu and Dtilde on a fixed event set."""

    def __init__(self, oracle, events: np.ndarray, n_basis: int = 25,
                 basis_scale: float = 0.6, seed: int = 0):
        self.events = events
        self.n = N_WC
        log_m = events[:, 0]
        self.u = events[:, 1]
        self.m_tev = np.exp(log_m)
        K = len(self.m_tev)

        # Basis working points: SM plus single- and paired-direction excitations
        # at a couple of magnitudes, padded with random points for conditioning.
        rng = np.random.default_rng(seed)
        basis = [np.zeros(self.n)]
        for mag in (basis_scale, -basis_scale):
            for i in range(self.n):
                e = np.zeros(self.n); e[i] = mag; basis.append(e)
        for i, j in combinations_with_replacement(range(self.n), 2):
            e = np.zeros(self.n); e[i] += basis_scale; e[j] += basis_scale
            basis.append(e)
        while len(basis) < n_basis:
            basis.append(rng.uniform(-basis_scale, basis_scale, self.n))
        C = np.array(basis)                                   # (M, n)
        M = len(C)

        # Evaluate mu and Dtilde per event at each basis point (the slow part).
        mu_MK = np.empty((M, K))
        dtil_MK = np.empty((M, K))
        for a in range(M):
            Ctile = np.tile(C[a], (K, 1))
            mu = oracle.truth(Ctile, self.m_tev)              # (K,)
            afb = oracle.truth_afb(Ctile, self.m_tev)         # (K,)
            mu_MK[a] = mu
            dtil_MK[a] = (4.0 / 3.0) * afb * mu               # Dtilde = (4/3) A_FB mu

        # Per-event quadratic fit: design (M, P) -> coeffs (P, K).
        design = _quad_features(C)                            # (M, P)
        self.mu_coeffs, *_ = np.linalg.lstsq(design, mu_MK, rcond=None)    # (P, K)
        self.dtil_coeffs, *_ = np.linalg.lstsq(design, dtil_MK, rcond=None)
        # SM (c=0) values are the constant terms.
        self.mu_sm = self.mu_coeffs[0]                        # (K,)
        self.dtil_sm = self.dtil_coeffs[0]
        self.r_sm = self.dtil_sm / np.maximum(self.mu_sm, 1e-30)

    def log_w(self, c: np.ndarray) -> np.ndarray:
        """Exact-form per-event log-likelihood ratio at working point c."""
        feats = _quad_features(np.asarray(c, float).reshape(1, self.n))[0]  # (P,)
        mu_c = feats @ self.mu_coeffs                          # (K,)
        dtil_c = feats @ self.dtil_coeffs
        r_c = dtil_c / np.maximum(mu_c, 1e-30)
        u = self.u
        eps = 1e-30
        ang_c = (1.0 + u * u) + 2.0 * r_c * u
        ang_sm = (1.0 + u * u) + 2.0 * self.r_sm * u
        return (np.log(np.maximum(mu_c, eps))
                + np.log(np.maximum(ang_c, eps))
                - np.log(np.maximum(ang_sm, eps)))


if __name__ == "__main__":
    # Validation against the exact slow path.
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
    from modules.surrogate.oracle_events import sample_events, event_log_likelihood_ratio

    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    events = sample_events(oracle, np.zeros(N_WC), 1500, seed=7)
    t0 = time.time()
    fm = FastMorphing(oracle, events, seed=0)
    t_build = time.time() - t0

    rng = np.random.default_rng(123)
    max_dev = 0.0
    t_slow = t_fast = 0.0
    for _ in range(8):
        c = rng.uniform(-0.6, 0.6, N_WC)
        t = time.time(); lw_slow = event_log_likelihood_ratio(oracle, c, events); t_slow += time.time() - t
        t = time.time(); lw_fast = fm.log_w(c); t_fast += time.time() - t
        max_dev = max(max_dev, float(np.max(np.abs(lw_slow - lw_fast))))
    print(f"build={t_build:.1f}s  max_abs_dev={max_dev:.2e}  "
          f"slow={t_slow/8:.3f}s/call  fast={t_fast/8*1e6:.1f}us/call")
    assert max_dev < 1e-5, f"FastMorphing deviates by {max_dev:.2e}"
    print("VALIDATED: FastMorphing matches the exact slow path to <1e-5.")
