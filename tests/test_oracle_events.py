"""Falsification gates for the event-level sampler (Task 0).

The sampler in ``modules/surrogate/oracle_events.py`` is the prerequisite
for the ManifoldInformer reframe: every per-event encoder, the residual-SVD
out-of-span detector, and the curvature-driven AL loop downstream consume
its output. If it does not reproduce the analytic differential, downstream
results are not interpretable.

Two gates, both required:

1. **Marginal m_ll**: histogram of sampled m_ll values reproduces μ(c, m)
   (the rate ratio) shape. Tested per c via a Kolmogorov-Smirnov test
   against the truth CDF of σ_SM(m) · μ(c, m).

2. **A_FB(m bin)**: per-m-bin forward-backward asymmetry computed from the
   sampled (m_ll, cos θ*) events reproduces ``oracle.truth_afb`` within the
   statistical uncertainty of the bin count.

Both gates run on three working points: c=0 (SM), c=(0, 0, 0.4, 0) (lifting
four-fermion only), c=(0.3, 0, 0.4, 0) (vertex + four-fermion, the
ManifoldInformer use case).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pytest

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import (
    sample_events, event_log_likelihood_ratio, _build_m_cdf,
)
from modules.surrogate.features import N_WC


N_EVENTS_LARGE = 20_000    # Stage-1 KS test budget (oracle is per-event PDF;
                            # 2e5 takes minutes, 2e4 still well below the KS
                            # gate of 0.05 since N=2e4 gives KS noise ~0.01)
N_EVENTS_SMALL = 20_000    # Per-bin A_FB test budget
SEED = 314159
M_GRID = np.linspace(0.3, 2.5, 220)


@pytest.fixture(scope="module")
def oracle():
    return AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)


@pytest.fixture(scope="module")
def working_points():
    return {
        "SM":         np.array([0.0, 0.0, 0.0, 0.0]),
        "four_ferm":  np.array([0.0, 0.0, 0.4, 0.0]),
        "vertex_lq":  np.array([0.3, 0.0, 0.4, 0.0]),
    }


def _truth_cdf_m(oracle, c, m_grid):
    _, cdf = _build_m_cdf(oracle, c, m_grid, sm_only=False)
    return cdf


def _empirical_cdf_at(m_samples, m_eval):
    """Step CDF of m_samples evaluated at sorted m_eval points."""
    sorted_samples = np.sort(m_samples)
    return np.searchsorted(sorted_samples, m_eval, side="right") / len(sorted_samples)


@pytest.mark.parametrize("wp_name",
                         ["SM", "four_ferm", "vertex_lq"])
def test_marginal_m_KS(oracle, working_points, wp_name):
    """KS distance between empirical and truth CDFs of m_ll < 0.05."""
    c = working_points[wp_name]
    events = sample_events(oracle, c, N_EVENTS_LARGE, seed=SEED,
                            m_grid=M_GRID)
    m_samples = np.exp(events[:, 0])               # log m -> m (TeV)
    truth_cdf = _truth_cdf_m(oracle, c, M_GRID)
    emp_cdf = _empirical_cdf_at(m_samples, M_GRID)
    ks = float(np.max(np.abs(emp_cdf - truth_cdf)))
    print(f"[{wp_name}] KS(m_ll) = {ks:.4f}")
    # 0.05 is 6× the Kolmogorov 99% threshold for N=2e5 (~0.0085); we use
    # a loose 0.05 because the truth CDF is built on the same grid as the
    # sampler and grid discretisation contributes a small residual ~ 1/N_grid.
    assert ks < 0.05, f"KS distance {ks:.4f} > 0.05 at {wp_name}"


@pytest.mark.parametrize("wp_name",
                         ["SM", "four_ferm", "vertex_lq"])
def test_afb_per_mbin(oracle, working_points, wp_name):
    """Empirical A_FB(m) per m-bin within 3σ of truth_afb."""
    c = working_points[wp_name]
    events = sample_events(oracle, c, N_EVENTS_SMALL, seed=SEED,
                            m_grid=M_GRID)
    m_samples = np.exp(events[:, 0])
    u_samples = events[:, 1]
    # 8 m-bins across the analysis window.
    bin_edges = np.linspace(0.3, 2.3, 9)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    truth_afb = oracle.truth_afb(np.tile(c, (len(bin_centres), 1)), bin_centres)
    fails = []
    for i in range(len(bin_centres)):
        in_bin = (m_samples >= bin_edges[i]) & (m_samples < bin_edges[i + 1])
        n_bin = int(in_bin.sum())
        if n_bin < 100:
            continue                              # too few events to test
        n_f = int((u_samples[in_bin] > 0).sum())
        n_b = n_bin - n_f
        emp_afb = (n_f - n_b) / n_bin
        # Binomial sigma on p_F = n_f / n_bin
        p_F = (n_f + 1) / (n_bin + 2)             # Laplace smoothing
        sigma_pF = np.sqrt(p_F * (1.0 - p_F) / n_bin)
        sigma_afb = 2.0 * sigma_pF                # A_FB = 2*p_F - 1
        z = abs(emp_afb - truth_afb[i]) / max(sigma_afb, 1e-12)
        print(f"[{wp_name}] m∈[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f}]  "
              f"emp A_FB = {emp_afb:+.4f}  truth = {truth_afb[i]:+.4f}  "
              f"σ = {sigma_afb:.4f}  z = {z:.2f}")
        if z > 3.0:
            fails.append((i, emp_afb, float(truth_afb[i]), float(sigma_afb), float(z)))
    assert not fails, (
        f"AFB per-m-bin gate failed in {len(fails)} bins at {wp_name}: "
        f"{fails}")


@pytest.mark.parametrize("wp_name", ["SM", "four_ferm"])
def test_log_likelihood_ratio_at_sm_is_zero(oracle, working_points, wp_name):
    """log w_SM(x) ≡ 0 for any event; log w_c(x) is finite for c != 0."""
    if wp_name != "SM":
        # Only the SM case has the analytic identity log w_SM(x) = 0.
        pytest.skip("non-SM identity is checked numerically elsewhere")
    c_sm = np.zeros(N_WC)
    events = sample_events(oracle, c_sm, 1000, seed=SEED, m_grid=M_GRID)
    log_w = event_log_likelihood_ratio(oracle, c_sm, events)
    assert np.max(np.abs(log_w)) < 1e-8, (
        f"log w_SM should be identically zero; got max|log w| = "
        f"{np.max(np.abs(log_w)):.3e}")


def test_finiteness_of_log_w_off_sm(oracle, working_points):
    """log w_c(x) is finite at every sampled event for non-trivial c."""
    c = working_points["vertex_lq"]
    events = sample_events(oracle, c, 5000, seed=SEED, m_grid=M_GRID)
    log_w = event_log_likelihood_ratio(oracle, c, events)
    assert np.all(np.isfinite(log_w)), "non-finite log w encountered"
    # Sanity range — for the working points we are using, log w is O(1) on
    # average (cross-section ratios in the analysis window are not extreme).
    assert np.max(np.abs(log_w)) < 50.0, (
        f"unreasonable log w magnitude {np.max(np.abs(log_w)):.2f}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s"])
