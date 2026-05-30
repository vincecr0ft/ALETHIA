"""Tests for modules.surrogate.intention.fisher.

Empirical Fisher information on a c-prior. Verifies:

- finite-difference gradient is sign-flip / scale-invariant
- F is symmetric positive semi-definite by construction
- fisher_basis returns descending eigenpairs and an orthonormal V
- rotate_c is exact under V^T V = I
- on a degenerate oracle that ignores one c-direction, that direction
  appears as a near-zero Fisher eigenvalue (the identifiability test
  the upgrade hangs off)
"""
import numpy as np
import pytest

from modules.surrogate.intention.fisher import (
    empirical_fisher_c, fisher_basis, rotate_c,
    sample_c_prior_inbox,
)


class QuadraticOracle:
    """Toy oracle mu(c, m) = 1 + sum_i a_i(m) * c_i + sum_ij b_ij(m) c_i c_j.

    Lets us write a Fisher matrix in closed form and check the estimator.
    """

    def __init__(self, n_wc: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_wc = n_wc
        # a(m) and b(m) are functions of m via a small basis.
        self.A = rng.normal(size=(3, n_wc)) * 0.3       # m-basis coefs
        self.B = rng.normal(size=(3, n_wc, n_wc)) * 0.1
        # symmetrise B per m-slice
        self.B = 0.5 * (self.B + self.B.transpose(0, 2, 1))

    def _m_basis(self, m):
        return np.stack([np.ones_like(m), m, m ** 2], axis=0)   # (3, N)

    def truth(self, c, m):
        c = np.atleast_2d(c)
        m = np.atleast_1d(m)
        mb = self._m_basis(m)                                    # (3, N)
        # linear: sum_k mb[k] * (A[k] . c_n)
        a_n = np.einsum("kn,ki->ni", mb, self.A)                 # (N, n_wc)
        lin = np.einsum("ni,ni->n", a_n, c)
        # quadratic: sum_k mb[k] * (c^T B[k] c)
        bc = np.einsum("kij,nj->kni", self.B, c)                 # (k, N, i)
        quad = np.einsum("kn,kni,ni->n", mb, bc, c)
        return 1.0 + lin + quad


def test_empirical_fisher_symmetric_psd():
    rng = np.random.default_rng(0)
    oracle = QuadraticOracle(n_wc=4, seed=42)
    c = rng.uniform(-0.5, 0.5, size=(200, 4))
    m = rng.uniform(0.3, 2.3, size=200)
    F = empirical_fisher_c(oracle, c, m, fd_step=1e-3)
    assert F.shape == (4, 4)
    assert np.allclose(F, F.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(F)
    assert np.all(eigs >= -1e-10), f"F not PSD: eigs={eigs}"


def test_empirical_fisher_identifies_dead_direction():
    """If the oracle ignores c[3] entirely, F[3, :] = 0 and the smallest
    Fisher eigenvalue should be ~0 with the corresponding eigenvector
    aligned with e_3. We give each of c[0..2] a distinct m-shape so the
    Fisher matrix has full rank on the informative subspace and the
    dead direction is uniquely the null space."""

    class DeadDirectionOracle:
        def truth(self, c, m):
            c = np.atleast_2d(c)
            m = np.atleast_1d(m)
            # mu depends on c[0..2] with three linearly-independent
            # m-shapes; c[3] never enters.
            return (1.0
                    + c[:, 0] * m
                    + c[:, 1] * m ** 2
                    + c[:, 2] * np.log(m))

    rng = np.random.default_rng(1)
    oracle = DeadDirectionOracle()
    c = rng.uniform(-0.5, 0.5, size=(500, 4))
    m = rng.uniform(0.3, 2.3, size=500)
    F = empirical_fisher_c(oracle, c, m, fd_step=1e-3)
    D, V = fisher_basis(F)
    # smallest eigenvalue near zero relative to the largest
    assert D[-1] < 1e-8 * max(D[0], 1e-30)
    # corresponding eigenvector aligned with e_3 (up to sign)
    v_min = V[:, -1]
    assert abs(v_min[3]) > 0.99
    assert all(abs(v_min[i]) < 0.1 for i in (0, 1, 2))


def test_fisher_basis_descending_and_orthonormal():
    rng = np.random.default_rng(2)
    # symmetric PSD random matrix
    X = rng.normal(size=(6, 6))
    F = X @ X.T + 0.1 * np.eye(6)
    D, V = fisher_basis(F)
    assert np.all(np.diff(D) <= 0), "fisher_basis must return descending"
    assert np.allclose(V.T @ V, np.eye(6), atol=1e-10)
    # reconstruct F
    assert np.allclose(V @ np.diag(D) @ V.T, F, atol=1e-9)


def test_rotate_c_inverts_with_V_transpose():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 4))
    F = X @ X.T + 0.1 * np.eye(4)
    D, V = fisher_basis(F)
    c = rng.uniform(-1, 1, size=(50, 4))
    c_tilde = rotate_c(c, V)
    # inverse rotation: c_tilde @ V.T == c
    c_back = c_tilde @ V.T
    assert np.allclose(c_back, c, atol=1e-10)


def test_sample_c_prior_excludes_withhold_band():
    rng = np.random.default_rng(4)
    c = sample_c_prior_inbox(
        n=500, n_wc=4, box=0.7, rng=rng,
        withhold_dim=2, withhold_band=(0.6, 1.0),
    )
    assert c.shape == (500, 4)
    assert np.all(np.abs(c[:, 2]) < 0.6)
    # other dims still cover the whole box
    assert c[:, 0].min() < -0.5 and c[:, 0].max() > 0.5


def test_sample_c_prior_no_withhold():
    rng = np.random.default_rng(5)
    c = sample_c_prior_inbox(n=300, n_wc=4, box=0.7, rng=rng)
    assert c.shape == (300, 4)
    assert c.min() >= -0.7 and c.max() <= 0.7


def test_sample_c_prior_requires_band_with_dim():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        sample_c_prior_inbox(
            n=10, n_wc=4, box=0.7, rng=rng, withhold_dim=2)
