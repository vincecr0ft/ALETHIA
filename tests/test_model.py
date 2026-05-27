import numpy as np
import pytest

from modules.surrogate import IntentionFM, DummyAnalyticOracle, N_WC


def _make_data(seed=0, n=200):
    oracle = DummyAnalyticOracle(seed=seed + 1)
    rng    = np.random.default_rng(seed)
    C = rng.uniform(-1, 1, (n, N_WC))
    M = rng.uniform(0.2, 2.5, n)
    Y = oracle(C, M, noise=True)
    return oracle, C, M, Y


def test_fit_predict_shapes():
    _, C, M, Y = _make_data()
    fm = IntentionFM().fit(C, M, Y)
    mu, sd = fm.predict(C[:10], M[:10])
    assert mu.shape == (10,)
    assert sd.shape == (10,)


def test_predict_without_std():
    _, C, M, Y = _make_data()
    fm = IntentionFM().fit(C, M, Y)
    mu = fm.predict(C[:10], M[:10], return_std=False)
    assert mu.shape == (10,)


def test_leverage_nonneg():
    _, C, M, Y = _make_data()
    fm = IntentionFM().fit(C, M, Y)
    lev = fm.leverage(C, M)
    assert np.all(lev >= 0)


def test_update_equals_refit():
    oracle, C, M, Y = _make_data()
    rng = np.random.default_rng(7)
    C_new = rng.uniform(-1, 1, (10, N_WC))
    M_new = rng.uniform(0.2, 2.5, 10)
    Y_new = oracle(C_new, M_new, noise=True)

    fm_a = IntentionFM().fit(C, M, Y).update(C_new, M_new, Y_new)
    fm_b = IntentionFM().fit(
        np.vstack([C, C_new]),
        np.concatenate([M, M_new]),
        np.concatenate([Y, Y_new]),
    )
    assert np.allclose(fm_a.w, fm_b.w)


def test_in_distribution_accuracy():
    oracle, C, M, Y = _make_data()
    fm  = IntentionFM().fit(C, M, Y)
    rng = np.random.default_rng(99)
    C_te = rng.uniform(-1, 1, (300, N_WC))
    M_te = rng.uniform(0.2, 2.5, 300)
    mu   = fm.predict(C_te, M_te, return_std=False)
    Y_te = oracle.truth(C_te, M_te)
    rel  = np.abs(mu - Y_te) / np.maximum(np.abs(Y_te), 1e-2)
    assert np.median(rel) < 0.05


def test_state_dict_roundtrip():
    _, C, M, Y = _make_data()
    fm  = IntentionFM(lam=2e-3).fit(C, M, Y)
    s   = fm.state_dict()
    fm2 = IntentionFM.from_state_dict(s)
    assert np.allclose(fm.w, fm2.w)
    assert np.allclose(fm.A_inv, fm2.A_inv)
    assert fm.noise_frac == fm2.noise_frac
