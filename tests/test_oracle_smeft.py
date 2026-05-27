"""Smoke tests for the analytic-SMEFT oracle adapter.

Verifies the bridge satisfies the surrogate's :class:`Oracle` protocol,
that ``mu(c=0, m) == 1`` (SM-only), that turning on a four-fermion operator
makes the BSM tail rise above SM, and that the noise model behaves.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.surrogate import N_WC, WC_NAMES, AnalyticSMEFTOracle, IntentionFM


@pytest.fixture(scope="module")
def oracle() -> AnalyticSMEFTOracle:
    return AnalyticSMEFTOracle(
        sqrt_s_gev=13000.0,
        lambda_scale_gev=3000.0,
        order="quadratic",
        pdf="analytic",
        noise_frac=0.0,
        seed=0,
    )


def test_truth_sm_limit(oracle: AnalyticSMEFTOracle) -> None:
    """At all Wilson coefficients zero, mu(c, m) == 1 exactly."""
    c = np.zeros((4, N_WC))
    m = np.array([0.3, 0.8, 1.5, 2.5])
    mu = oracle.truth(c, m)
    assert mu.shape == (4,)
    assert np.allclose(mu, 1.0, atol=1e-12)


def test_truth_eft_energy_growth(oracle: AnalyticSMEFTOracle) -> None:
    """Activating a four-fermion operator drives the high-mass tail above SM."""
    c = np.zeros((1, N_WC))
    c[0, WC_NAMES.index("clq1")] = 0.3
    # Low m: mu ~ 1; high m: mu > 1 (EFT energy growth).
    mu_low = oracle.truth(c, np.array([0.3]))[0]
    mu_high = oracle.truth(c, np.array([2.5]))[0]
    assert 0.85 < mu_low < 1.15
    assert mu_high > 1.5


def test_call_matches_truth_when_noiseless(oracle: AnalyticSMEFTOracle) -> None:
    """noise_frac=0 means ``__call__`` and ``truth`` agree exactly."""
    rng = np.random.default_rng(0)
    c = rng.uniform(-0.3, 0.3, (5, N_WC))
    m = rng.uniform(0.5, 2.0, 5)
    assert np.allclose(oracle(c, m, noise=True), oracle.truth(c, m))


def test_shape_mismatch_raises() -> None:
    oracle = AnalyticSMEFTOracle(noise_frac=0.0, pdf="analytic")
    with pytest.raises(ValueError, match="rows"):
        oracle.truth(np.zeros((3, N_WC)), np.array([1.0, 2.0]))


def test_intention_fm_fits_oracle() -> None:
    """End-to-end: closed-form fit recovers the SMEFT response with low error."""
    oracle = AnalyticSMEFTOracle(
        lambda_scale_gev=3000.0,
        order="quadratic",
        pdf="analytic",
        noise_frac=0.03,
        seed=0,
    )
    rng = np.random.default_rng(42)
    C = rng.uniform(-0.5, 0.5, (200, N_WC))
    M = rng.uniform(0.3, 2.5, 200)
    Y = oracle(C, M, noise=True)

    fm = IntentionFM(lam=1e-3).fit(C, M, Y)
    # MAD-estimated noise should land near the truth.
    assert 0.01 < fm.noise_frac < 0.06

    # Evaluate on a held-out slice; median rel err should be small.
    c_slice = np.zeros(N_WC)
    c_slice[WC_NAMES.index("clq1")] = 0.2
    m_grid = np.linspace(0.5, 2.0, 30)
    C_grid = np.tile(c_slice, (m_grid.size, 1))
    mu_pred = fm.predict(C_grid, m_grid, return_std=False)
    mu_true = oracle.truth(C_grid, m_grid)
    rel = np.abs(mu_pred - mu_true) / np.maximum(np.abs(mu_true), 1e-6)
    assert float(np.median(rel)) < 0.05
