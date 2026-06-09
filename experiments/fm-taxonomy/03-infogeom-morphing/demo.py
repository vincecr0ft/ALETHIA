r"""Analytic Lagrangian morphing + Fisher geometry — self-contained demo.

Branch 03 of the ALETHIA FM taxonomy: information geometry and Lagrangian
morphing, and their relationship to manifold learning.

Thesis under test
------------------
Morphing gives an EXACT, structured, low-dimensional parametrization of the
prediction manifold from a *known* polynomial structure. A learned manifold /
foundation model must *discover* that structure from data. We isolate the
distinction by:

  (1) Building a genuinely polynomial observable. A single BSM amplitude added
      to the SM amplitude,

          A(c) = a_SM + c * a_BSM,                                     (toy ME)

      gives a squared matrix element / differential rate that is exactly
      quadratic in the coupling c:

          sigma(c, x) = |a_SM(x)|^2 + c * 2 Re[a_SM(x)* a_BSM(x)] + c^2 |a_BSM(x)|^2
                      = T0(x) + c * T1(x) + c^2 * T2(x).

      The three "templates" T0 (SM-only), T1 (interference), T2 (BSM-squared)
      span the whole one-coefficient family. This mirrors the
      sm_only / interference / bsm_squared decomposition that the ALETHIA
      analytic SMEFT oracle (modules/analytic_smeft/smeft.py) already returns,
      and the linear+quadratic morphing of Balasubramanian et al.
      (arXiv:2202.13612), built on the moment-morphing method of Baak,
      Gadatsch, Harrington & Verkerke (arXiv:1410.7388).

  (2) Sampling sigma at N_samp = 3 distinct couplings (the minimal basis for a
      degree-2 polynomial in one coefficient: N = (n^2 + 3n + 2)/2 = 3 at
      n = 1). Inverting the 3x3 Vandermonde-type morphing matrix M recovers the
      morphing weights w(c) = v(c) . M^{-1}, where v(c) = (1, c, c^2). The
      morphed prediction sum_i w_i(c) sigma(c_i) reproduces sigma(c) EXACTLY at
      ANY new c (to machine precision), with NO new simulation.

  (3) Computing the Fisher information for c from the morphing polynomial and
      reading off its information geometry: a 1-D Riemannian metric g(c) =
      F(c), the Cramer-Rao bound 1/F(c), and the natural-gradient /
      arc-length reparametrization s(c) = integral sqrt(F) dc (Amari 1998,
      arXiv form of natural gradient; Rao 1945 / Cramer 1946).

We print numbers that prove (a) exact reconstruction and (b) the Fisher
geometry, contrasting the closed polynomial against a PCA "learned manifold"
fit to the same samples.

Run:
    python experiments/fm-taxonomy/03-infogeom-morphing/demo.py
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. A genuinely polynomial observable: |a_SM + c a_BSM|^2 over a kinematic grid
# ---------------------------------------------------------------------------
def amplitudes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Toy SM and BSM amplitudes over a kinematic variable x in [0, 1].

    The BSM amplitude carries an explicit energy growth ~ x (the EFT contact
    operator's M^2/Lambda^2 behaviour in modules/analytic_smeft), so the
    interference and BSM-squared templates are NOT proportional to the SM
    template — the morphing basis is genuinely 3-dimensional, not degenerate.
    """
    a_sm = 1.0 + 0.3 * np.cos(2.0 * np.pi * x)          # SM amplitude (real toy)
    a_bsm = 0.4 * x + 0.1                                # BSM: linear energy growth
    return a_sm, a_bsm


def sigma_true(c: float, x: np.ndarray) -> np.ndarray:
    r"""Exact differential rate sigma(c, x) = |a_SM + c a_BSM|^2.

    This is the ground-truth "simulator". Quadratic in c by construction.
    """
    a_sm, a_bsm = amplitudes(x)
    return (a_sm + c * a_bsm) ** 2


def templates(x: np.ndarray) -> np.ndarray:
    r"""The three physical templates (T0, T1, T2) on the grid x.

    sigma(c, x) = T0 + c T1 + c^2 T2 with
        T0 = a_SM^2                 (SM-only)
        T1 = 2 a_SM a_BSM           (interference)
        T2 = a_BSM^2                (BSM-squared)
    These are exactly the components the morphing exposes; a real analysis
    never has them separately and must reconstruct them from samples.
    """
    a_sm, a_bsm = amplitudes(x)
    return np.stack([a_sm ** 2, 2.0 * a_sm * a_bsm, a_bsm ** 2], axis=0)


# ---------------------------------------------------------------------------
# 2. Morphing algebra: basis of samples -> coefficient matrix -> exact interp
# ---------------------------------------------------------------------------
def morphing_matrix(c_basis: np.ndarray, degree: int = 2) -> np.ndarray:
    r"""Morphing matrix M: rows are the coupling polynomials v(c_i) at samples.

    For one coefficient at degree d, v(c) = (1, c, c^2, ..., c^d) and M is the
    (N_samp x (d+1)) Vandermonde matrix. Following Balasubramanian et al.
    (arXiv:2202.13612 Eq. 9-10): the morphed prediction at a new c is

        sigma(c) = v(c) . M^{-1} . (sigma(c_1), ..., sigma(c_N))^T
                 = sum_i w_i(c) sigma(c_i),     w(c) = v(c) M^{-1}.
    """
    powers = np.arange(degree + 1)
    return c_basis[:, None] ** powers[None, :]          # (N_samp, degree+1)


def morphing_weights(c_query: float, M_inv: np.ndarray, degree: int = 2) -> np.ndarray:
    r"""w(c) = v(c) . M^{-1}; the per-sample weights for a query coupling."""
    v = c_query ** np.arange(degree + 1)                # (degree+1,)
    return v @ M_inv                                    # (N_samp,)


# ---------------------------------------------------------------------------
# 3. Fisher information geometry of the coupling c
# ---------------------------------------------------------------------------
def fisher_information(c: float, x: np.ndarray, sigma_y: float = 0.05) -> float:
    r"""Additive-Gaussian-noise Fisher information for the coupling c.

    With Y(x) = sigma(c, x) + N(0, sigma_y^2) observed on the grid x,

        F(c) = sum_x (d sigma / d c)^2 / sigma_y^2,
        d sigma / d c = T1(x) + 2 c T2(x).

    This is exactly the gradient structure A_i + 2 sum_j B_ij c_j used in
    ALETHIA's working_point_fisher.py (here A=T1, B=T2 for one coefficient).
    F(c) is the 1-D Fisher metric g(c) on the parameter manifold; 1/F(c) is the
    Cramer-Rao variance bound on any unbiased estimator of c.
    """
    T = templates(x)
    dsigma_dc = T[1] + 2.0 * c * T[2]                   # (len(x),)
    return float(np.sum(dsigma_dc ** 2) / sigma_y ** 2)


def fisher_arclength(c_grid: np.ndarray, x: np.ndarray, sigma_y: float = 0.05) -> np.ndarray:
    r"""Natural (Fisher) arc length s(c) = integral_0^c sqrt(F(c')) dc'.

    The reparametrization c -> s flattens the Fisher metric to the identity
    (Amari, Natural Gradient Works Efficiently in Learning, Neural Comput.
    1998). Equal steps in s are equal-information steps — the natural-gradient
    coordinate on this 1-D statistical manifold.
    """
    sqrtF = np.array([np.sqrt(fisher_information(c, x, sigma_y)) for c in c_grid])
    s = np.concatenate([[0.0], np.cumsum(0.5 * (sqrtF[1:] + sqrtF[:-1]) * np.diff(c_grid))])
    return s


# ---------------------------------------------------------------------------
# 4. Learned-manifold contrast: PCA on the sample rate-curves
# ---------------------------------------------------------------------------
def pca_rank(c_samples: np.ndarray, x: np.ndarray, tol: float = 1e-12) -> tuple[int, int, np.ndarray]:
    r"""Empirical rank of the family {sigma(c, .)} via SVD (a 'learned manifold').

    A foundation model / autoencoder / PCA does NOT know the polynomial degree;
    it must read the dimensionality off the data. The raw stacked rate-curves
    sigma(c, .) = T0 + c T1 + c^2 T2 live in a 3-D space (basis T0, T1, T2), so
    the uncentered SVD has 3 non-negligible singular values. Mean-centering
    removes the constant offset (one affine direction), leaving 2 — the c and
    c^2 *variation*. Either way the learned manifold *discovers* a dimension
    the morphing algebra *asserts* a priori (degree+1 = 3).
    """
    rows = np.stack([sigma_true(c, x) for c in c_samples], axis=0)   # (S, len(x))
    sv_raw = np.linalg.svd(rows, compute_uv=False)
    rank_raw = int(np.sum(sv_raw > tol * sv_raw[0]))
    centered = rows - rows.mean(axis=0, keepdims=True)
    sv_c = np.linalg.svd(centered, compute_uv=False)
    rank_c = int(np.sum(sv_c > tol * sv_c[0]))
    return rank_raw, rank_c, sv_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 64)                        # kinematic grid
    degree = 1 * 2 // 2 + 1                               # = 2 (one coeff, quadratic)
    sigma_y = 0.05

    print("=" * 72)
    print("ANALYTIC LAGRANGIAN MORPHING + FISHER GEOMETRY")
    print("=" * 72)

    # --- Minimal basis: N_samp = (n^2 + 3n + 2)/2 = 3 at n=1 (arXiv:2202.13612 Eq.21)
    n = 1
    N_samp = (n ** 2 + 3 * n + 2) // 2
    c_basis = np.array([-1.0, 0.0, 1.5])                 # 3 arbitrary distinct couplings
    assert len(c_basis) == N_samp
    print(f"\n[1] Minimal morphing basis: n={n} coefficient(s), degree={degree}")
    print(f"    N_samp = (n^2+3n+2)/2 = {N_samp}  (linear+quadratic count)")
    print(f"    sampled couplings c_i = {c_basis}")

    # --- Build morphing matrix and invert
    M = morphing_matrix(c_basis, degree)
    M_inv = np.linalg.inv(M)
    print(f"\n[2] Morphing matrix M (rows = v(c_i) = (1, c, c^2)):")
    for i, row in enumerate(M):
        print(f"      c={c_basis[i]:+.2f} -> {row}")
    print(f"    cond(M) = {np.linalg.cond(M):.3f}")

    # --- The simulator is queried ONLY at the basis points
    sigma_samples = np.stack([sigma_true(c, x) for c in c_basis], axis=0)   # (3, 64)

    # --- Exact reconstruction at many NEW couplings, no new simulation
    c_queries = np.array([-0.7, -0.25, 0.13, 0.6, 0.95, 1.3, 2.4, 5.0])
    print(f"\n[3] EXACT reconstruction at new couplings (no new simulation):")
    print(f"    {'c_query':>9}  {'max|morphed-true|':>18}  {'sum w_i':>10}")
    max_err_overall = 0.0
    for cq in c_queries:
        w = morphing_weights(cq, M_inv, degree)
        morphed = w @ sigma_samples                      # (64,)
        truth = sigma_true(cq, x)
        err = float(np.max(np.abs(morphed - truth)))
        max_err_overall = max(max_err_overall, err)
        # sum_i w_i = v(c) M^{-1} 1 is NOT a partition of unity for polynomial
        # morphing (the weights can be negative / exceed 1); printed for context.
        print(f"    {cq:>9.3f}  {err:>18.3e}  {w.sum():>10.4f}")
    print(f"    --> worst-case reconstruction error over all queries: {max_err_overall:.3e}")
    print(f"    (machine precision => the 3-sample basis spans the WHOLE family)")

    # --- Recover the hidden templates from the basis samples (morphing inverts to T_i)
    T_recovered = M_inv @ sigma_samples                  # (3, 64): rows are T0,T1,T2
    T_true = templates(x)
    t_err = float(np.max(np.abs(T_recovered - T_true)))
    print(f"\n[4] Template recovery (M^{{-1}} . samples = (T0, T1, T2)):")
    print(f"    max|T_recovered - T_true| = {t_err:.3e}")
    print(f"    The morphing inverse literally extracts SM-only / interference /")
    print(f"    BSM-squared from 3 mixed observations — exact, by construction.")

    # --- Fisher information geometry of c
    print(f"\n[5] Fisher information geometry of the coupling c (sigma_y={sigma_y}):")
    c_grid = np.linspace(-3.0, 2.0, 501)
    F = np.array([fisher_information(c, x, sigma_y) for c in c_grid])
    c_min = c_grid[int(np.argmin(F))]
    # Closed-form least-informative coupling: dF/dc = 0
    # => sum a_bsm^3 (a_sm + c a_bsm) = 0  =>  c* = -sum(a_bsm^3 a_sm)/sum(a_bsm^4)
    a_sm, a_bsm = amplitudes(x)
    c_star = -float(np.sum(a_bsm ** 3 * a_sm) / np.sum(a_bsm ** 4))
    print(f"    F(c) = sum_x (T1 + 2 c T2)^2 / sigma_y^2   (1-D Fisher metric g(c))")
    print(f"    F(c=0)   = {fisher_information(0.0, x, sigma_y):.4e}   "
          f"CR bound sigma_c >= {1/np.sqrt(fisher_information(0.0,x,sigma_y)):.4e}")
    print(f"    F(c=1)   = {fisher_information(1.0, x, sigma_y):.4e}   "
          f"CR bound sigma_c >= {1/np.sqrt(fisher_information(1.0,x,sigma_y)):.4e}")
    print(f"    grid min at c={c_min:+.3f}; closed-form least-informative c* = {c_star:+.3f}")
    print(f"    where dsigma/dc = 2 a_bsm (a_sm + c a_bsm) is smallest in norm —")
    print(f"    a working point of vanishing sensitivity. This is the same")
    print(f"    mechanism behind ALETHIA's flat-vertex-direction finding: the")
    print(f"    Fisher metric collapses where the SM and BSM gradients align to")
    print(f"    cancel, and that is a property of the working point c, not the data.")

    # --- Natural (Fisher) arc length: equal-information coordinate
    s = fisher_arclength(c_grid, x, sigma_y)
    print(f"\n[6] Natural-gradient coordinate s(c) = integral sqrt(F) dc (Amari 1998):")
    for cc in (-1.0, 0.0, 1.0, 2.0):
        j = int(np.argmin(np.abs(c_grid - cc)))
        print(f"      c={cc:+.1f} -> s={s[j]:8.2f}  (cumulative information distance)")
    print(f"    Equal steps in s are equal-information steps; this is the metric a")
    print(f"    natural-gradient optimiser would precondition by F^{{-1}}.")

    # --- Learned-manifold contrast: PCA discovers the dimension morphing asserts
    c_dense = np.linspace(-1.0, 2.0, 40)
    rank_raw, rank_c, sv = pca_rank(c_dense, x)
    print(f"\n[7] Learned-manifold contrast (PCA/SVD on 40 sampled rate-curves):")
    print(f"    singular values (top 6): {np.array2string(sv[:6], precision=3)}")
    print(f"    raw numerical rank      = {rank_raw}  (= degree+1 = #templates T0,T1,T2)")
    print(f"    mean-centred rank       = {rank_c}  (constant offset removed -> c, c^2 vary)")
    print(f"    A PCA/autoencoder RECOVERS this dim from data; morphing KNOWS it is")
    print(f"    degree-2 a priori and needs only {N_samp} exact samples, no fit.")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  exact morphing reconstruction error      : {max_err_overall:.2e}")
    print(f"  template (T0,T1,T2) recovery error       : {t_err:.2e}")
    print(f"  basis samples used                       : {N_samp}")
    print(f"  Fisher metric range over c in [-1,2]     : "
          f"[{F.min():.2e}, {F.max():.2e}]")
    print(f"  least-informative working point c*       : {c_min:+.3f}")
    print(f"  PCA-discovered rank (raw / centred)      : {rank_raw} / {rank_c}")
    print("=" * 72)


if __name__ == "__main__":
    main()
