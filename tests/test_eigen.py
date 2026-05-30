"""Tests for modules.surrogate.intention.eigen.

Spine of the upgrade-architecture (docs/research/upgrade-architecture.md
§1). Verifies the eigendecomposition utilities behave consistently with
how the drift span, eigen-redirected acquisition, and the MCP rubric
will consume them.
"""
import numpy as np
import pytest

from modules.surrogate.intention import (
    IntentionFM,
    EigenState, eigen_state, eigen_resample_weights,
    eigenvector_stability, union_subspace,
)


def test_eigen_state_round_trip_reconstructs_A():
    rng = np.random.default_rng(0)
    Psi = rng.normal(size=(20, 8))
    alpha = 1e-3
    es = eigen_state(Psi, alpha)
    A = Psi.T @ Psi + alpha * np.eye(8)
    # U diag(lam) U^T = A
    A_reco = es.U @ np.diag(es.lam) @ es.U.T
    assert np.allclose(A_reco, A, atol=1e-9)


def test_eigen_state_ascending_order():
    rng = np.random.default_rng(1)
    Psi = rng.normal(size=(15, 5))
    es = eigen_state(Psi, alpha=1e-2)
    assert np.all(np.diff(es.lam) >= 0), (
        "eigenvalues must be ascending in EigenState.lam")


def test_eigen_state_kappa_matches_manual():
    rng = np.random.default_rng(2)
    Psi = rng.normal(size=(30, 6))
    alpha = 1e-3
    es = eigen_state(Psi, alpha)
    expected = es.lam[-1] / max(es.lam[0], 1e-30)
    assert np.isclose(es.kappa, expected)


def test_low_and_high_subspace_shapes_and_orthogonality():
    rng = np.random.default_rng(3)
    Psi = rng.normal(size=(50, 16))
    es = eigen_state(Psi, alpha=1e-3)
    U_low, lam_low = es.low_subspace(3)
    U_high, lam_high = es.high_subspace(3)
    assert U_low.shape == (16, 3) and lam_low.shape == (3,)
    assert U_high.shape == (16, 3) and lam_high.shape == (3,)
    # orthogonal blocks within themselves
    assert np.allclose(U_low.T @ U_low, np.eye(3), atol=1e-9)
    assert np.allclose(U_high.T @ U_high, np.eye(3), atol=1e-9)
    # low vs high orthogonal
    assert np.allclose(U_low.T @ U_high, np.zeros((3, 3)), atol=1e-9)


def test_low_subspace_clamps_to_d():
    rng = np.random.default_rng(4)
    es = eigen_state(rng.normal(size=(10, 4)), alpha=1e-3)
    U_low, _ = es.low_subspace(99)
    assert U_low.shape == (4, 4)


def test_lis_rank_monotone_in_tau():
    rng = np.random.default_rng(5)
    es = eigen_state(rng.normal(size=(100, 8)), alpha=1e-3)
    r_tight = es.lis_rank(tau=1e-1)
    r_loose = es.lis_rank(tau=1e-6)
    assert r_loose >= r_tight


def test_eigen_resample_weights_non_negative():
    rng = np.random.default_rng(6)
    Psi_ctx = rng.normal(size=(30, 6))
    es = eigen_state(Psi_ctx, alpha=1e-3)
    U_low, lam_low = es.low_subspace(3)
    Psi_pool = rng.normal(size=(200, 6))
    w = eigen_resample_weights(Psi_pool, U_low, lam_low)
    assert w.shape == (200,)
    assert np.all(w >= 0)
    # Some positive mass somewhere.
    assert w.sum() > 0


def test_eigen_resample_weights_concentrate_on_aligned_directions():
    """A pool point aligned exactly with u_low,0 should outscore one
    aligned with u_high,0 by ~ lam_high / lam_low (an O(kappa) margin
    for a well-conditioned A)."""
    rng = np.random.default_rng(7)
    Psi_ctx = rng.normal(size=(50, 8))
    es = eigen_state(Psi_ctx, alpha=1e-3)
    U_low, lam_low = es.low_subspace(1)
    U_high, _ = es.high_subspace(1)
    Psi_pool = np.vstack([U_low[:, 0], U_high[:, 0]])         # (2, d)
    w = eigen_resample_weights(Psi_pool, U_low, lam_low)
    assert w[0] > w[1] * 10.0      # low direction far heavier than high


def test_eigenvector_stability_identity_is_one():
    rng = np.random.default_rng(8)
    es = eigen_state(rng.normal(size=(20, 6)), alpha=1e-3)
    U = es.low_subspace(3)[0]
    s = eigenvector_stability(U, U)
    assert np.allclose(s, 1.0)


def test_eigenvector_stability_orthogonal_is_zero():
    U_prev = np.eye(4)[:, :2]
    U_curr = np.eye(4)[:, 2:4]                 # orthogonal columns
    s = eigenvector_stability(U_prev, U_curr)
    assert np.allclose(s, 0.0, atol=1e-12)


def test_eigenvector_stability_sign_flip_invariant():
    """A sign-flipped eigenvector should still register as stable."""
    rng = np.random.default_rng(9)
    es = eigen_state(rng.normal(size=(30, 5)), alpha=1e-3)
    U = es.low_subspace(2)[0]
    s = eigenvector_stability(U, -U)
    assert np.allclose(s, 1.0)


def test_eigenvector_stability_shape_mismatch_raises():
    with pytest.raises(ValueError):
        eigenvector_stability(np.eye(4)[:, :2], np.eye(4)[:, :3])


def test_union_subspace_recovers_span_of_one_block():
    rng = np.random.default_rng(10)
    U = np.linalg.qr(rng.normal(size=(6, 3)))[0]   # (6, 3) orthonormal
    out = union_subspace([U])
    # rank 3
    assert out.shape == (6, 3)
    # spans the same column space as U: project U onto out and recover it
    proj = out @ (out.T @ U)
    assert np.allclose(proj, U, atol=1e-9)


def test_union_subspace_deduplicates_overlapping_blocks():
    rng = np.random.default_rng(11)
    U = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    out = union_subspace([U, U])
    # Same span, not 4 columns of orthonormal basis.
    assert out.shape[1] == 2


# ----- IntentionFM.A_eigen integration -----

def test_intention_A_eigen_returns_EigenState():
    rng = np.random.default_rng(12)
    model = IntentionFM(d_psi=16, hidden=32, alpha=1e-3)
    model.eval()
    M_ctx = rng.uniform(0.3, 2.3, size=20)
    es = model.A_eigen(M_ctx)
    assert isinstance(es, EigenState)
    assert es.d == 16
    assert es.lam.shape == (16,)
    assert es.U.shape == (16, 16)
    # Spectrum is non-negative because A = Psi^T Psi + alpha I is PSD.
    assert np.all(es.lam > 0)
    # Smallest eigenvalue is at least alpha (within ~ float tolerance).
    assert es.lam[0] >= model.alpha - 1e-6


def test_intention_A_eigen_kappa_matches_kappa_A():
    rng = np.random.default_rng(13)
    model = IntentionFM(d_psi=12, hidden=24, alpha=1e-3)
    model.eval()
    M_ctx = rng.uniform(0.3, 2.3, size=15)
    es = model.A_eigen(M_ctx)
    assert np.isclose(es.kappa, model.kappa_A(M_ctx), rtol=1e-6)
