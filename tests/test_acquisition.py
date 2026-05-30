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


# ----------------------------------------------------------------------
# Parameter-space EPIG (INV-3 of ALETHEIA_investigations.md)
# ----------------------------------------------------------------------

from modules.surrogate.intention import (
    IntentionFM as _IntentionFM,
    param_epig_d_acquire, param_epig_a_acquire,
)


def _param_setup(seed=0, K=8, n_pool=50, D=16, r=2):
    """Build a small fresh IntentionFM and a Gaussian-random P.

    Uses default-initialised psi_theta — sufficient for the algebraic
    tests below, which check the closed-form score against a brute-force
    log-det computation, not any property of the trained representation.
    The synthetic P is what a frozen INV-2 probe would supply, restricted
    to the r data-dominated directions.
    """
    torch_seed = seed + 7
    import torch
    torch.manual_seed(torch_seed)
    model = _IntentionFM(d_psi=D, hidden=24, alpha=1e-3)
    rng = np.random.default_rng(seed)
    M_ctx = rng.uniform(0.3, 2.3, size=K)
    Y_ctx = rng.normal(size=K)
    M_pool = rng.uniform(0.3, 2.3, size=n_pool)
    P = rng.normal(size=(r, D)) / np.sqrt(D)
    sigma_y = 0.1
    return model, M_ctx, Y_ctx, M_pool, P, sigma_y


def test_param_epig_d_nonnegative():
    """ΔH_D ≥ 0 by Cauchy-Schwarz. Verify by recomputing the score on a
    50-point pool from a K=8 context, r=2, and checking positivity."""
    model, M_ctx, Y_ctx, M_pool, P, sigma_y = _param_setup(
        seed=0, K=8, n_pool=50, D=16, r=2)
    # Single-pick budget — internal score for every candidate.
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    A_inv = A_inv.astype(np.float64)
    Psi_pool = model.psi_np(M_pool).astype(np.float64)
    Sigma = (sigma_y ** 2) * (P @ A_inv @ P.T)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(P.shape[0]))
    AP = Psi_pool @ A_inv
    U = AP @ P.T
    from scipy.linalg import solve_triangular
    X = solve_triangular(L, U.T, lower=True)
    quad = np.einsum("ri,ri->i", X, X)
    lev_P = np.einsum("pd,pd->p", AP, Psi_pool)
    arg = 1.0 - (sigma_y ** 2) * quad / (1.0 + lev_P)
    scores = -0.5 * np.log(arg)
    assert np.all(scores >= -1e-12), \
        f"ΔH_D must be non-negative; min={scores.min():.3e}"
    # K=8 with well-conditioned A: expect comfortably positive scores.
    assert scores.max() > 1e-6, \
        f"ΔH_D should be appreciable for some candidate; max={scores.max():.3e}"
    # Sanity: the actual acquire call returns k unique indices.
    idx = param_epig_d_acquire(
        model, M_ctx, Y_ctx, M_pool, P, k=5,
        sigma_y=sigma_y, resolved_dim=2)
    assert len(set(idx.tolist())) == 5


def test_param_epig_d_matches_brute_force():
    """For each candidate, compare the closed-form ΔH_D against a direct
    log-det Σ before/after computation that inverts A by hand."""
    model, M_ctx, Y_ctx, M_pool, P, sigma_y = _param_setup(
        seed=1, K=5, n_pool=5, D=8, r=2)
    sy2 = sigma_y ** 2
    Psi_ctx = model.psi_np(M_ctx).astype(np.float64)
    Psi_pool = model.psi_np(M_pool).astype(np.float64)
    D = Psi_ctx.shape[1]
    A = Psi_ctx.T @ Psi_ctx + model.alpha * np.eye(D)
    A_inv = np.linalg.inv(A)
    Sigma = sy2 * (P @ A_inv @ P.T)
    sign0, logdet0 = np.linalg.slogdet(Sigma)
    assert sign0 > 0
    # Closed-form scores on every candidate.
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(P.shape[0]))
    AP = Psi_pool @ A_inv
    U = AP @ P.T
    from scipy.linalg import solve_triangular
    X = solve_triangular(L, U.T, lower=True)
    quad = np.einsum("ri,ri->i", X, X)
    lev_P = np.einsum("pd,pd->p", AP, Psi_pool)
    arg = 1.0 - sy2 * quad / (1.0 + lev_P)
    scores_cf = -0.5 * np.log(arg)
    # Brute force: explicitly form A_new^{-1} per candidate, recompute Σ_new.
    rel_errs = []
    for i, p in enumerate(Psi_pool):
        A_new = A + np.outer(p, p)
        A_new_inv = np.linalg.inv(A_new)
        Sigma_new = sy2 * (P @ A_new_inv @ P.T)
        _, logdet_new = np.linalg.slogdet(Sigma_new)
        score_bf = 0.5 * (logdet0 - logdet_new)
        rel = abs(scores_cf[i] - score_bf) / max(abs(score_bf), 1e-12)
        rel_errs.append(rel)
    rel_errs = np.array(rel_errs)
    assert rel_errs.max() < 1e-8, \
        f"closed-form vs brute-force ΔH_D rel-err too large: {rel_errs}"
    # Emit one sample line so the reporter can quote it.
    print(f"[param_epig_d brute-force] cand0: cf={scores_cf[0]:.6e} "
          f"bf-rel-err={rel_errs[0]:.2e}; max rel-err over 5 cands = "
          f"{rel_errs.max():.2e}")


def test_param_epig_d_monotone_picks():
    """Sequential-greedy picks must strictly decrease log det Σ each step
    (resolved-subspace entropy is monotone non-increasing under data
    acquisition; with non-zero u it strictly decreases)."""
    model, M_ctx, Y_ctx, M_pool, P, sigma_y = _param_setup(
        seed=2, K=8, n_pool=40, D=16, r=2)
    sy2 = sigma_y ** 2
    idx = param_epig_d_acquire(
        model, M_ctx, Y_ctx, M_pool, P, k=5,
        sigma_y=sigma_y, resolved_dim=2)
    # Replay the log-det at each step independently via direct inverse.
    Psi_ctx = model.psi_np(M_ctx).astype(np.float64)
    D = Psi_ctx.shape[1]
    A = Psi_ctx.T @ Psi_ctx + model.alpha * np.eye(D)
    Psi_pool = model.psi_np(M_pool).astype(np.float64)
    logdets = []
    A_curr = A.copy()
    Sigma = sy2 * (P @ np.linalg.inv(A_curr) @ P.T)
    _, ld = np.linalg.slogdet(Sigma)
    logdets.append(ld)
    for i in idx:
        p = Psi_pool[i]
        A_curr = A_curr + np.outer(p, p)
        Sigma = sy2 * (P @ np.linalg.inv(A_curr) @ P.T)
        _, ld = np.linalg.slogdet(Sigma)
        logdets.append(ld)
    diffs = np.diff(logdets)
    assert np.all(diffs < 0), \
        f"log det Σ must strictly decrease each pick; diffs={diffs}"


def test_param_epig_a_single_direction():
    """ΔH_a on direction 0 must equal ΔH_D evaluated on the 1-row P."""
    model, M_ctx, Y_ctx, M_pool, P, sigma_y = _param_setup(
        seed=3, K=6, n_pool=10, D=8, r=2)
    sy2 = sigma_y ** 2
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    A_inv = A_inv.astype(np.float64)
    Psi_pool = model.psi_np(M_pool).astype(np.float64)
    # ΔH_a closed form for direction 0:
    p_a = P[0]
    Sigma_aa = sy2 * float(p_a @ A_inv @ p_a)
    AP = Psi_pool @ A_inv
    u_a = AP @ p_a
    lev_P = np.einsum("pd,pd->p", AP, Psi_pool)
    arg_a = 1.0 - sy2 * (u_a ** 2) / ((1.0 + lev_P) * Sigma_aa)
    scores_a = -0.5 * np.log(arg_a)
    # ΔH_D on the 1-row P.
    P_one = P[:1]
    Sigma1 = sy2 * (P_one @ A_inv @ P_one.T)
    L = np.linalg.cholesky(Sigma1 + 1e-12 * np.eye(1))
    U = AP @ P_one.T
    from scipy.linalg import solve_triangular
    X = solve_triangular(L, U.T, lower=True)
    quad = np.einsum("ri,ri->i", X, X)
    arg_d = 1.0 - sy2 * quad / (1.0 + lev_P)
    scores_d = -0.5 * np.log(arg_d)
    rel = np.abs(scores_a - scores_d) / np.maximum(np.abs(scores_d), 1e-12)
    assert rel.max() < 1e-10, \
        f"ΔH_a should equal ΔH_D on 1-row P; max rel-err = {rel.max():.2e}"
    # And the two acquire helpers agree on the same picks for k=3.
    idx_a = param_epig_a_acquire(
        model, M_ctx, Y_ctx, M_pool, P, k=3,
        target_direction=0, sigma_y=sigma_y)
    idx_d = param_epig_d_acquire(
        model, M_ctx, Y_ctx, M_pool, P[:1], k=3,
        sigma_y=sigma_y)
    assert list(idx_a) == list(idx_d), \
        f"single-direction picks must match: a={idx_a} d={idx_d}"
