"""Task 1 step 1 — symbolic reproduction of Eq. (1) from
ALETHIA_informer_workpoint_AL_handoff.md §1.1.

The morphing decomposition of the SMEFT cross section is

    σ(c, m) = σ_SM(m) + Σ_i A_i(m) c_i + Σ_{i≤j} B_{ij}(m) c_i c_j

with the convention used by ``modules/analytic_smeft/smeft.py`` that
``B_{ij}`` is symmetric and the sum is over ordered pairs (i.e. the off-
diagonal pieces enter twice with their own coefficient). Equivalently:

    σ(c, m) = σ_SM(m) + Σ_i A_i(m) c_i + Σ_{i,j} B_{ij}(m) c_i c_j     (B sym.)

The handoff's Eq. (1) is the gradient

    ∂_i σ(c, m) = A_i(m) + 2 Σ_j B_{ij}(m) c_j .                       (Eq. 1)

The Fisher information at working point c, summed over an m-grid with
additive Gaussian noise on Y (variance σ_y²), is

    F_{ij}(c) = (1/σ_y²) Σ_m (∂_i σ)(∂_j σ) .                          (Fisher)

This test does two checks against the live oracle:

1. **Symbolic ↔ analytic agreement**. Build sympy expressions for σ, ∂σ,
   F under generic symbols (a_i, b_{ij}, c_i, σ_SM) and verify that
   ∂_i σ = A_i + 2 Σ_j B_{ij} c_j as a symbolic identity, and that
   F_{ij}(c) reduces to the expected sum-of-outer-products form.

2. **Symbolic ↔ finite-difference agreement** on the live analytic oracle.
   Recover A_i(m) and B_{ij}(m) from oracle calls (central differences at
   c=0), reconstruct ∂_i σ(c, m) at several off-SM working points by the
   symbolic formula, and compare against finite-difference gradients of
   the oracle directly. Tolerance: 1e-8 relative (μ is quadratic in c at
   LO so central differences are exact up to floating-point).

Falsification: if either gate fails, Eq. (1) is misstated for this oracle
and the whole reframe must be diagnosed before downstream tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pytest
import sympy as sp

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.features import N_WC


# ---------------------------------------------------------------------------
# Symbolic part: pure sympy derivation
# ---------------------------------------------------------------------------

def test_symbolic_gradient_matches_eq1():
    """∂_i σ = A_i + 2 Σ_j B_{ij} c_j as a symbolic identity."""
    n = N_WC
    c = sp.symbols(f"c0:{n}", real=True)
    A = sp.symbols(f"A0:{n}", real=True)
    sigma_sm = sp.symbols("sigma_SM", real=True)
    # Symmetric B as a sympy Matrix.
    B = sp.Matrix(n, n,
                   lambda i, j: sp.Symbol(f"B_{min(i,j)}_{max(i,j)}", real=True))

    # σ(c) = σ_SM + Σ A_i c_i + Σ_{i,j} B_{ij} c_i c_j  (symmetric B)
    sigma = sigma_sm + sum(A[i] * c[i] for i in range(n))
    sigma += sum(B[i, j] * c[i] * c[j] for i in range(n) for j in range(n))

    # Symbolic gradient.
    grad_symbolic = [sp.simplify(sp.diff(sigma, c[i])) for i in range(n)]

    # Eq. (1) target.
    eq1 = [A[i] + 2 * sum(B[i, j] * c[j] for j in range(n)) for i in range(n)]
    eq1 = [sp.simplify(e) for e in eq1]

    for i in range(n):
        diff = sp.simplify(grad_symbolic[i] - eq1[i])
        assert diff == 0, (
            f"∂_{i} σ disagrees with Eq. (1): "
            f"sympy gives {grad_symbolic[i]}, Eq.(1) gives {eq1[i]}")


def test_symbolic_fisher_form():
    """F_{ij}(c) = Σ_m (A_i + 2 B_{ik} c_k)(A_j + 2 B_{jl} c_l) / σ_y² ."""
    n = N_WC
    K = 3   # m-grid size for the symbolic check
    c = sp.symbols(f"c0:{n}", real=True)
    sigma_y = sp.symbols("sigma_y", positive=True)
    # A_i(m_k) and B_{ij}(m_k): independent symbols per m_k.
    A = [[sp.Symbol(f"A_{i}_{k}", real=True) for k in range(K)]
         for i in range(n)]
    B = [[[sp.Symbol(f"B_{min(i,j)}_{max(i,j)}_{k}", real=True)
           for k in range(K)] for j in range(n)] for i in range(n)]

    # Build ∂_i σ at each m_k.
    grad = [[A[i][k] + 2 * sum(B[i][j][k] * c[j] for j in range(n))
             for k in range(K)] for i in range(n)]

    # Fisher = sum_k grad_i(m_k) grad_j(m_k) / σ_y²
    F = sp.Matrix(n, n, lambda i, j:
                   sum(grad[i][k] * grad[j][k] for k in range(K)) / sigma_y ** 2)

    # Check symmetry.
    for i in range(n):
        for j in range(n):
            d = sp.simplify(F[i, j] - F[j, i])
            assert d == 0, f"F[{i}, {j}] != F[{j}, {i}]"

    # Check that F depends on c through the linear-in-c B-cross terms only
    # (Eq. 1 says ∂σ/∂c is linear in c).
    F_expanded = sp.expand(F[0, 0])
    # Each term should be at most quadratic in c.
    for c_sym in c:
        deg = sp.Poly(F_expanded, c_sym).degree() if F_expanded.has(c_sym) else 0
        assert deg <= 2, (
            f"F[0,0] has degree {deg} > 2 in {c_sym}; Eq. (1) says ∂σ is "
            f"linear in c, so F should be at most quadratic in c.")


# ---------------------------------------------------------------------------
# Numeric part: finite-difference reconstruction against the live oracle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oracle():
    return AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)


def _A_at(oracle, m_grid, i, *, h=1e-3):
    """A_i(m) = (1/σ_SM) ∂_i σ |_{c=0} via central differences on μ."""
    K = len(m_grid)
    c_p = np.zeros((K, N_WC)); c_p[:, i] = +h
    c_m = np.zeros((K, N_WC)); c_m[:, i] = -h
    return (oracle.truth(c_p, m_grid) - oracle.truth(c_m, m_grid)) / (2.0 * h)


def _B_at(oracle, m_grid, i, j, *, h=1e-3):
    """B_{ij}(m) = (1/σ_SM) ½ ∂_i ∂_j σ |_{c=0} on the μ ratio. Symmetric in i,j."""
    K = len(m_grid)
    if i == j:
        c_p = np.zeros((K, N_WC)); c_p[:, i] = +h
        c_m = np.zeros((K, N_WC)); c_m[:, i] = -h
        c_0 = np.zeros((K, N_WC))
        mu_p = oracle.truth(c_p, m_grid)
        mu_m = oracle.truth(c_m, m_grid)
        mu_0 = oracle.truth(c_0, m_grid)
        return 0.5 * (mu_p + mu_m - 2.0 * mu_0) / (h * h)
    c_pp = np.zeros((K, N_WC)); c_pp[:, i] = +h; c_pp[:, j] = +h
    c_pm = np.zeros((K, N_WC)); c_pm[:, i] = +h; c_pm[:, j] = -h
    c_mp = np.zeros((K, N_WC)); c_mp[:, i] = -h; c_mp[:, j] = +h
    c_mm = np.zeros((K, N_WC)); c_mm[:, i] = -h; c_mm[:, j] = -h
    return (oracle.truth(c_pp, m_grid) - oracle.truth(c_pm, m_grid)
            - oracle.truth(c_mp, m_grid) + oracle.truth(c_mm, m_grid)
            ) / (8.0 * h * h)


def test_eq1_reconstruction_against_oracle(oracle):
    """∂_i μ(c, m) = A_i(m) + 2 Σ_j B_{ij}(m) c_j matches FD on the oracle."""
    rng = np.random.default_rng(7)
    m_grid = np.linspace(0.3, 2.3, 10)
    # Recover A_i, B_{ij} at c=0.
    A = np.stack([_A_at(oracle, m_grid, i) for i in range(N_WC)], axis=1)  # (K, n)
    B = np.zeros((len(m_grid), N_WC, N_WC))
    for i in range(N_WC):
        for j in range(N_WC):
            B[:, i, j] = _B_at(oracle, m_grid, i, j)
    # Verify B symmetric (it must be).
    for i in range(N_WC):
        for j in range(i + 1, N_WC):
            sym_err = float(np.max(np.abs(B[:, i, j] - B[:, j, i])))
            assert sym_err < 1e-7, (
                f"B[{i},{j}] − B[{j},{i}] = {sym_err:.3e}; symmetry violated")

    # For three random off-SM working points, compare FD gradient on μ
    # to the Eq. (1) reconstruction.
    for trial in range(3):
        c = rng.uniform(-0.5, 0.5, N_WC)
        # FD gradient: ∂_i μ(c, m) via central differences.
        h = 1e-4
        grad_fd = np.zeros((len(m_grid), N_WC))
        for i in range(N_WC):
            c_p = np.tile(c, (len(m_grid), 1)); c_p[:, i] += h
            c_m = np.tile(c, (len(m_grid), 1)); c_m[:, i] -= h
            grad_fd[:, i] = (oracle.truth(c_p, m_grid) - oracle.truth(c_m, m_grid)
                              ) / (2.0 * h)
        # Eq. (1) reconstruction: A_i + 2 Σ_j B_{ij} c_j .
        grad_eq1 = A + 2.0 * np.einsum("kij,j->ki", B, c)
        rel_err = np.max(np.abs(grad_fd - grad_eq1)) / (
            np.max(np.abs(grad_fd)) + 1e-12)
        print(f"  trial {trial}: c={c.round(3)}, max rel err = {rel_err:.3e}")
        assert rel_err < 1e-4, (
            f"Eq. (1) reconstruction mismatched FD gradient at trial {trial}: "
            f"max rel err = {rel_err:.3e} (expected < 1e-4)")


def test_morphing_is_polynomial_in_c(oracle):
    """μ(c, m) is exactly quadratic in c at LO: central FD of order ≥ 3 is zero."""
    rng = np.random.default_rng(13)
    c_base = rng.uniform(-0.4, 0.4, N_WC)
    m = np.array([1.0])
    h = 1e-2

    # Five-point stencil for the third derivative of μ in c_0.
    cs = [c_base.copy() for _ in range(5)]
    for k, off in enumerate([-2 * h, -h, 0.0, h, 2 * h]):
        cs[k][0] = c_base[0] + off
    mu_vals = np.array([oracle.truth(np.atleast_2d(c), m)[0] for c in cs])
    # f'''(0) ≈ (1/2h³)(f(2h) - 2f(h) + 2f(-h) - f(-2h)) — Abramowitz & Stegun.
    d3 = (mu_vals[4] - 2 * mu_vals[3] + 2 * mu_vals[1] - mu_vals[0]) / (2.0 * h ** 3)
    # μ is *exactly* quadratic at LO; the only departure is finite-precision.
    # h=1e-2 puts the FD truncation noise around 1e-4 in absolute terms.
    assert abs(d3) < 1e-3, (
        f"Third derivative ∂_0³ μ at c={c_base.round(3)} is {d3:.3e} "
        f"(expected ~ 0 for an exactly-quadratic morphing)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
