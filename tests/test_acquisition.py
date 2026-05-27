import numpy as np
import pytest

from modules.surrogate import (
    IntentionFM, DummyAnalyticOracle,
    random_acquire, leverage_acquire, epig_acquire,
    N_WC,
)


def _setup(seed=4):
    oracle = DummyAnalyticOracle(seed=seed)
    rng = np.random.default_rng(seed + 10)
    C = rng.uniform(-1, 1, (200, N_WC));  M = rng.uniform(0.2, 2.5, 200)
    Y = oracle(C, M, noise=True)
    fm = IntentionFM().fit(C, M, Y)
    pool_rng = np.random.default_rng(seed + 20)
    C_pool = pool_rng.uniform(-2, 2, (300, N_WC))
    M_pool = pool_rng.uniform(0.2, 2.5, 300)
    return oracle, fm, C_pool, M_pool


# --------- random ----------
def test_random_returns_unique_indices():
    rng = np.random.default_rng(0)
    idx = random_acquire(rng, 100, 30)
    assert len(set(idx.tolist())) == 30
    assert idx.min() >= 0 and idx.max() < 100


# --------- leverage ----------
def test_leverage_picks_above_median():
    _, fm, C_pool, M_pool = _setup()
    idx        = leverage_acquire(fm, C_pool, M_pool, k=10)
    lev_picks  = fm.leverage(C_pool[idx], M_pool[idx])
    lev_pool   = fm.leverage(C_pool, M_pool)
    assert lev_picks.mean() > np.median(lev_pool)


def test_leverage_no_duplicates():
    _, fm, C_pool, M_pool = _setup()
    idx = leverage_acquire(fm, C_pool, M_pool, k=30)
    assert len(set(idx.tolist())) == 30


# --------- EPIG ----------
def test_epig_returns_unique_indices():
    _, fm, C_pool, M_pool = _setup()
    rng  = np.random.default_rng(0)
    C_T  = rng.uniform(-1, 1, (50, N_WC));  M_T  = rng.uniform(0.2, 2.5, 50)
    idx  = epig_acquire(fm, C_pool, M_pool, C_T, M_T, k=20)
    assert len(set(idx.tolist())) == 20


def test_epig_focused_target_picks_toward_target():
    """EPIG with a focused target should pull picks toward that region in
    feature space, relative to leverage acquisition."""
    _, fm, C_pool, M_pool = _setup()
    # Tightly focused target: small cloud near c_0 = +1.8, others = 0, m = 2.0
    n_T = 80
    rng = np.random.default_rng(11)
    C_T = np.tile([1.8, 0.0, 0.0, 0.0], (n_T, 1)) + rng.normal(0, 0.05, (n_T, N_WC))
    M_T = np.full(n_T, 2.0) + rng.normal(0, 0.02, n_T)

    idx_lev  = leverage_acquire(fm, C_pool, M_pool, k=15)
    idx_epig = epig_acquire(fm, C_pool, M_pool, C_T, M_T, k=15)

    # The two strategies should not produce identical sets.
    assert set(idx_lev.tolist()) != set(idx_epig.tolist())
    # EPIG picks' mean c_0 should be closer to the target c_0 (1.8) than
    # leverage picks' mean c_0.
    c0_lev_dist  = abs(C_pool[idx_lev,  0].mean() - 1.8)
    c0_epig_dist = abs(C_pool[idx_epig, 0].mean() - 1.8)
    assert c0_epig_dist < c0_lev_dist


def test_epig_scores_nonnegative_information():
    """The closed-form EPIG score is non-negative by construction
    (Cauchy-Schwarz). Verify against a direct computation."""
    from modules.surrogate import phi_joint as pj
    _, fm, C_pool, M_pool = _setup()
    rng  = np.random.default_rng(0)
    C_T  = rng.uniform(-2, 2, (40, N_WC));  M_T  = rng.uniform(0.2, 2.5, 40)
    Phi_T   = pj(C_T, M_T)
    Phi_P   = pj(C_pool[:50], M_pool[:50])
    A_inv   = fm.A_inv
    lev_T   = np.einsum("id,de,ie->i", Phi_T, A_inv, Phi_T)
    lev_P   = np.einsum("pd,de,pe->p", Phi_P, A_inv, Phi_P)
    K_TP    = Phi_T @ A_inv @ Phi_P.T
    var_red = K_TP ** 2 / (lev_P[None, :] + 1.0)
    assert np.all(var_red <= lev_T[:, None] + 1e-9), \
        "Cauchy-Schwarz: variance reduction must not exceed prior latent variance"
