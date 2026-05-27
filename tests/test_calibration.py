import numpy as np
import pytest

from modules.surrogate import (
    IntentionFM, ConformalCalibrator, DummyAnalyticOracle, N_WC,
)


def _fit_with_cal(seed=2, n_tr=200, n_cal=300):
    oracle = DummyAnalyticOracle(seed=seed)
    rng = np.random.default_rng(seed + 10)
    C_tr = rng.uniform(-1, 1, (n_tr, N_WC));   M_tr = rng.uniform(0.2, 2.5, n_tr)
    Y_tr = oracle(C_tr, M_tr, noise=True)
    # Cal set on the broader probe region, not just the train box
    C_cal = rng.uniform(-2, 2, (n_cal, N_WC)); M_cal = rng.uniform(0.2, 2.5, n_cal)
    Y_cal = oracle(C_cal, M_cal, noise=True)
    fm = IntentionFM().fit(C_tr, M_tr, Y_tr)
    cc = ConformalCalibrator(n_strata=5).fit(fm, C_cal, M_cal, Y_cal)
    return oracle, fm, cc, rng


def test_factors_populated_at_both_coverages():
    _, _, cc, _ = _fit_with_cal()
    assert 0.683 in cc.factors
    assert 0.954 in cc.factors
    assert cc.factors[0.683].shape == (5,)
    assert cc.factors[0.954].shape == (5,)


def test_marginal_coverage_near_target():
    oracle, fm, cc, rng = _fit_with_cal()
    C_te = rng.uniform(-2, 2, (1500, N_WC))
    M_te = rng.uniform(0.2, 2.5, 1500)
    Y_te = oracle(C_te, M_te, noise=True)
    mu   = fm.predict(C_te, M_te, return_std=False)
    sd   = cc.coverage_sigma(fm, C_te, M_te, coverage=0.683)
    cov  = float(np.mean(np.abs(Y_te - mu) < sd))
    assert 0.60 < cov < 0.78, f"Got marginal coverage {cov:.3f}, expected ~0.683"


def test_high_leverage_factor_at_least_as_large():
    _, _, cc, _ = _fit_with_cal()
    # Raw model under-covers at high leverage, so the conformal factor there
    # should be at least the low-leverage factor (typically much larger).
    assert cc.factors[0.954][-1] >= cc.factors[0.954][0]


def test_state_dict_roundtrip():
    _, _, cc, _ = _fit_with_cal()
    s   = cc.state_dict()
    cc2 = ConformalCalibrator.from_state_dict(s)
    assert np.allclose(cc.edges, cc2.edges)
    for k in cc.factors:
        assert np.allclose(cc.factors[k], cc2.factors[k])
