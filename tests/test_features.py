import numpy as np
import pytest

from modules.surrogate import (
    phi_c, phi_x, phi_joint,
    D_C, D_X, D_JOINT, N_WC, M_REF,
)


def test_phi_c_shape():
    c = np.random.uniform(-1, 1, (10, N_WC))
    assert phi_c(c).shape == (10, D_C)


def test_phi_c_at_zero():
    """phi_c(0) should be [1, 0, ..., 0]: constant term only."""
    out = phi_c(np.zeros((5, N_WC)))
    assert np.allclose(out[:, 0], 1.0)
    assert np.allclose(out[:, 1:], 0.0)


def test_phi_c_quadratic_indices():
    """At c = (1, 0, 0, 0): const = 1, linear c0 = 1, quadratic c0^2 = 1."""
    c = np.zeros((1, N_WC))
    c[0, 0] = 1.0
    out = phi_c(c)[0]
    assert out[0] == 1.0                # constant
    assert out[1] == 1.0                # linear c0
    assert out[2] == 0.0                # linear c1
    assert out[1 + N_WC] == 1.0         # quadratic (0, 0) = c0^2


def test_phi_x_shape():
    m = np.linspace(0.2, 2.5, 20)
    assert phi_x(m).shape == (20, D_X)


def test_phi_x_at_reference_mass():
    """log(M_REF / M_REF) = 0, so phi_x = [1, 0, 0, ...]."""
    out = phi_x(np.array([M_REF]))[0]
    assert out[0] == 1.0
    assert np.allclose(out[1:], 0.0)


def test_phi_joint_shape():
    c = np.random.uniform(-1, 1, (10, N_WC))
    m = np.random.uniform(0.2, 2.5, 10)
    assert phi_joint(c, m).shape == (10, D_JOINT)


def test_phi_joint_is_tensor_product():
    c  = np.random.uniform(-1, 1, (5, N_WC))
    m  = np.random.uniform(0.2, 2.5, 5)
    fj = phi_joint(c, m)
    fc = phi_c(c)
    fx = phi_x(m)
    for n in range(5):
        for ci in range(D_C):
            for xi in range(D_X):
                assert np.isclose(fj[n, ci * D_X + xi], fc[n, ci] * fx[n, xi])
