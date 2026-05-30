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


def test_truth_channels_sum_to_mu(oracle: AnalyticSMEFTOracle) -> None:
    """sm_only + interference + bsm_squared == mu, per the morphing identity.
    Surfaces the named channels for the joint-score aux loss (U9).
    """
    rng = np.random.default_rng(7)
    c = rng.uniform(-0.5, 0.5, size=(20, N_WC))
    m = rng.uniform(0.4, 2.0, size=20)
    channels = oracle.truth_channels(c, m)
    mu = oracle.truth(c, m)
    reconstructed = (
        channels["sm_only"] + channels["interference"]
        + channels["bsm_squared"]
    )
    assert np.allclose(channels["mu"], mu, rtol=1e-6, atol=1e-9)
    assert np.allclose(reconstructed, mu, rtol=1e-6, atol=1e-9)


def test_truth_channels_sm_limit(oracle: AnalyticSMEFTOracle) -> None:
    """At c=0: interference and bsm_squared vanish; sm_only normalised to 1."""
    c = np.zeros((5, N_WC))
    m = np.linspace(0.4, 2.0, 5)
    ch = oracle.truth_channels(c, m)
    assert np.allclose(ch["sm_only"], 1.0)
    assert np.allclose(ch["interference"], 0.0, atol=1e-10)
    assert np.allclose(ch["bsm_squared"], 0.0, atol=1e-10)
    assert np.allclose(ch["mu"], 1.0)


def test_angular_costheta_bin_partition_reproduces_rate() -> None:
    """Gate 1 of INV-1: summing the cos-theta* binned cross section over a
    partition of [-1, 1] reproduces the angle-integrated rate to GL-floor.

    The target is 1e-10 (closed-form S piece, GL quadrature on the parton
    luminosities); we accept 1e-8 to leave headroom for the symmetric
    Gauss-Legendre node placement when the asymmetric luminosity also runs.
    The forward and backward bins must cancel their D contributions
    *exactly* in arithmetic so the residual measures S and the angular
    weights, not the D dilution.

    EFT-valid configuration per the spec: ``|c_i| <= 0.3`` and
    ``Lambda = 2 TeV``. Uses the analytic toy PDF (the ratio is PDF-
    independent at LO; the toy is faster and removes the LHAPDF dependency
    from the unit-test path).
    """
    from modules.analytic_smeft import (
        differential_xs,
        differential_xs_costheta_bin,
    )

    rng = np.random.default_rng(13)
    n_configs = 5
    m_grid_gev = np.logspace(np.log10(300.0), np.log10(2300.0), 10)
    edges = np.linspace(-1.0, 1.0, 5)  # 4 equal-width bins
    bins = list(zip(edges[:-1], edges[1:]))

    common = dict(
        sqrt_s=13000.0,
        lambda_scale=2000.0,  # Lambda = 2 TeV, EFT-valid for |c| <= 0.3.
        order="quadratic",
        pdf="analytic",
    )

    worst_rel_err = 0.0
    for _ in range(n_configs):
        c_vec = rng.uniform(-0.3, 0.3, size=4)
        wc = {
            "c_phi_q^(3)": float(c_vec[0]),
            "c_phi_q^(1)": float(c_vec[1]),
            "c_lq^(3)": float(c_vec[2]),
            "c_lq^(1)": float(c_vec[3]),
        }
        rate = differential_xs(wc, m_grid_gev, **common)["differential_xs"]
        binned_sum = np.zeros_like(rate)
        for bin_edges in bins:
            res = differential_xs_costheta_bin(
                wc, m_grid_gev, bin_edges, **common,
            )
            binned_sum += res["differential_xs"]
        rel_err = np.max(
            np.abs(binned_sum - rate) / np.maximum(np.abs(rate), 1e-30)
        )
        worst_rel_err = max(worst_rel_err, float(rel_err))

    # GL quadrature floor: 1e-10 target, accept 1e-8.
    assert worst_rel_err < 1e-8, (
        f"angular partition residual {worst_rel_err:.3e} exceeds 1e-8 floor"
    )


def test_angular_costheta_bin_sm_limit() -> None:
    """At c=0, full cos-theta* integral of the binned cross section
    reproduces the SM differential cross section.

    A complementary check on top of the random-c configurations: the SM
    rate must be reproduced exactly (no operator-induced cancellations).
    """
    from modules.analytic_smeft import (
        differential_xs,
        differential_xs_costheta_bin,
    )

    common = dict(
        sqrt_s=13000.0,
        lambda_scale=2000.0,
        order="quadratic",
        pdf="analytic",
    )
    m_grid_gev = np.array([300.0, 800.0, 1500.0, 2300.0])
    rate_sm = differential_xs({}, m_grid_gev, **common)["differential_xs"]
    edges = np.linspace(-1.0, 1.0, 5)
    binned_sum = np.zeros_like(rate_sm)
    for lo, hi in zip(edges[:-1], edges[1:]):
        res = differential_xs_costheta_bin({}, m_grid_gev, (lo, hi), **common)
        binned_sum += res["differential_xs"]
    rel = np.max(np.abs(binned_sum - rate_sm) / np.abs(rate_sm))
    assert rel < 1e-10


def test_truth_costheta_bin_sm_limit() -> None:
    """At c=0 the bin-integrated ratio mu_bin == 1 exactly."""
    oracle = AnalyticSMEFTOracle(
        sqrt_s_gev=13000.0,
        lambda_scale_gev=2000.0,
        order="quadratic",
        pdf="analytic",
        noise_frac=0.0,
        seed=0,
    )
    c = np.zeros((4, N_WC))
    m_tev = np.array([0.4, 0.8, 1.5, 2.3])
    mu_bin = oracle.truth_costheta_bin(c, m_tev, (0.0, 0.5))
    assert mu_bin.shape == (4,)
    assert np.allclose(mu_bin, 1.0, atol=1e-12)
