r"""Task 2 — known-curvature toy manifold oracle.

A controlled morphing oracle whose templates ``{g_SM, a_i, b_{ij}}`` are
hand-chosen so that:

- One direction (``c_0``) is **tangent-flat at SM** (``a_0(m) ∝ g_SM(m)``)
  but **curvature-resolvable** (``b_{0,1}(m)`` has a distinct m-shape and
  ``b_{0,0}(m)`` is small): the vertex-analogue of the §1.1 mechanism on
  a controlled toy.
- One direction (``c_1``) is **tangent-resolved** (``a_1(m)`` has strong
  m-shape) and **curvature-quadratic** (``b_{1,1}(m)`` carries energy
  growth): the four-fermion analogue.
- One synthetic **out-of-span injection channel**, used to inject a
  feature ψ cannot represent (a high-frequency event-feature function),
  for the residual-SVD recovery experiment of Task 1 step 4.

This file defines only the closed-form oracle. The studies that use it
(morphing recovery, multi-point Fisher lift, residual SVD, completeness
criterion) live in ``run_toy_curvature.py``.

The construction follows ALETHIA_informer_workpoint_AL_handoff.md §2.

Conventions:
    µ(c, m) = g_SM(m) + sum_i a_i(m) c_i + sum_{i,j} b_{ij}(m) c_i c_j
    (B symmetric, no factor-2 convention for off-diagonals)

The injected out-of-span shape sits in a "hidden" event-feature axis
``x_aux`` ∈ [-1, +1] that the µ-marginal oracle returns *integrated over*
— so the binned rate hides it, but per-event w_c(x) values along x_aux
expose it. This mimics how cos θ* hides angular structure when only the
rate is observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def _g_SM(m: NDArray) -> NDArray:
    """SM baseline; positive, monotone-decreasing in m for plotting realism."""
    return 1.0 / (1.0 + (m / 1.0) ** 2)


def _a_vertex_like(m: NDArray) -> NDArray:
    """a_0(m) = 0.5 · g_SM(m). Tangent-flat at c=0 in the sense that
    a_0/g_SM is constant; rate-only shift."""
    return 0.5 * _g_SM(m)


def _a_four_fermion_like(m: NDArray) -> NDArray:
    """a_1(m) = 0.3 · m². Strong m-shape; four-fermion-like energy growth."""
    return 0.3 * (m ** 2)


def _b_diag_vertex(m: NDArray) -> NDArray:
    """b_{0,0}(m) = 0.02 · g_SM(m). Small self-curvature on the vertex axis
    (matches the SMEFT analogue where vertex^2 is tiny)."""
    return 0.02 * _g_SM(m)


def _b_diag_four_fermion(m: NDArray) -> NDArray:
    """b_{1,1}(m) = 0.5 · m⁴. Strong four-fermion self-curvature."""
    return 0.5 * (m ** 4)


def _b_cross(m: NDArray) -> NDArray:
    """b_{0,1}(m) = b_{1,0}(m) = 0.4 · m² · g_SM(m) — distinct m-shape from
    a_0 (which is ∝ g_SM): this is the curvature-resolvable signature that
    lifts the c_0 direction when c_1 ≠ 0."""
    return 0.4 * (m ** 2) * _g_SM(m)


# Out-of-span injection: a function of an auxiliary event-feature x_aux
# that ψ cannot represent. We pick a high-frequency sinusoid in x_aux.
def _injection_shape(x_aux: NDArray) -> NDArray:
    """High-frequency event-feature function, sin(6 x_aux). The per-event
    encoder ψ for the toy lives in (m, x_aux); a deliberately deficient ψ
    omits cos(6 x_aux), sin(6 x_aux) from its basis so this shape is
    genuinely out-of-span."""
    return np.sin(6.0 * x_aux)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

N_OPS = 2   # toy uses 2 Wilson directions


@dataclass
class ToyCurvatureOracle:
    """Closed-form morphing oracle for Task 2.

    The differential rate over (m, x_aux) is

        dN(c, m, x_aux)/(dm dx_aux) = g_SM(m) · h_SM(x_aux)
            · [ 1 + Σ_i a_i(m)/g_SM(m) c_i
                  + Σ_{i,j} b_{ij}(m)/g_SM(m) c_i c_j
                  + ε_inj · η(c) · I(x_aux) ]

    where h_SM(x_aux) = 1/2 (uniform on [-1, +1]) and I(x_aux) is the
    injected out-of-span shape. ε_inj toggles the injection; η(c)
    modulates it with c (the injection has its own c-dependence so the
    residual-SVD will see a coherent c-dependent left singular vector).

    When ``injection_eps == 0`` the oracle is exactly the morphing toy
    with the chosen (a, b) templates — used as the in-span control.
    When ``injection_eps > 0`` and ``injection_coupling`` defines the
    c-dependence of the injected channel — used for Task 1 step 4 and
    the §1.3 fingerprint experiment.

    Attributes:
        injection_eps: scalar amplitude of the out-of-span injection.
        injection_coupling: (N_OPS,) coupling η_i so that the injection
            amplitude at working point c is ε_inj · (Σ_i η_i c_i).
            Defaults to (0, 1) — the injection is excited by c_1 only.
    """

    injection_eps: float = 0.0
    injection_coupling: tuple = (0.0, 1.0)
    name: str = "ToyCurvatureOracle"

    # Marginal-over-x_aux interface (for templates / Fisher / "the binned ratio")

    def marginal_mu(self, c: NDArray, m: NDArray) -> NDArray:
        """µ(c, m) = ∫ dx_aux · differential / g_SM(m) — the binned-ratio
        observable analogue. The x_aux-injection integrates to zero (the
        injection sin(6x) has zero mean over [-1, +1]) so the marginal
        rate is independent of injection_eps. This is the §1.3
        observation: 'the binned curve cannot see the out-of-span piece'.
        """
        m = np.asarray(m, dtype=float).ravel()
        c = np.asarray(c, dtype=float).reshape(-1, N_OPS)
        if c.shape[0] != m.shape[0]:
            raise ValueError(
                f"c rows {c.shape[0]} != m len {m.shape[0]}")
        a0 = _a_vertex_like(m); a1 = _a_four_fermion_like(m)
        b00 = _b_diag_vertex(m); b11 = _b_diag_four_fermion(m)
        b01 = _b_cross(m)
        g = _g_SM(m)
        mu = (g
              + a0 * c[:, 0] + a1 * c[:, 1]
              + b00 * c[:, 0] ** 2 + b11 * c[:, 1] ** 2
              + 2.0 * b01 * c[:, 0] * c[:, 1])
        return mu / g                                            # ratio µ(c,m)

    # ---- analytical morphing pieces (the Task 2 ground truth) ----

    def template_g_sm(self, m: NDArray) -> NDArray:
        return _g_SM(np.asarray(m, dtype=float))

    def template_a(self, m: NDArray) -> tuple[NDArray, NDArray]:
        m = np.asarray(m, dtype=float)
        return _a_vertex_like(m), _a_four_fermion_like(m)

    def template_b(self, m: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        m = np.asarray(m, dtype=float)
        return _b_diag_vertex(m), _b_diag_four_fermion(m), _b_cross(m)

    # ---- per-event differential (with injection) ----

    def joint_density(self, c: NDArray, m: NDArray, x_aux: NDArray) -> NDArray:
        """dN/(dm dx_aux) at one working point c, broadcast over (m, x_aux).

        c: (N_OPS,)
        m, x_aux: (N,) arrays, paired per event.

        Returns rate per (m, x_aux) point.
        """
        c = np.asarray(c, dtype=float).reshape(N_OPS)
        m = np.asarray(m, dtype=float).ravel()
        x_aux = np.asarray(x_aux, dtype=float).ravel()
        if m.shape != x_aux.shape:
            raise ValueError("m and x_aux must have the same shape")
        # Replicate c per event.
        C = np.tile(c, (len(m), 1))
        mu_marg = self.marginal_mu(C, m)                          # (N,)
        h_sm = 0.5 * np.ones_like(x_aux)                          # uniform
        # The injection: ε · (η·c) · I(x_aux); rate at SM is h_sm = 1/2,
        # and the marginal mu factor is multiplied through.
        eta_dot_c = float(np.dot(np.asarray(self.injection_coupling), c))
        injection = self.injection_eps * eta_dot_c * _injection_shape(x_aux)
        # Total: g_SM(m) · mu_marg · h_sm · (1 + injection)
        return _g_SM(m) * mu_marg * h_sm * (1.0 + injection)

    def sample_events(self, c: NDArray, N: int, *, seed: int | None = None,
                       m_range: tuple = (0.3, 2.5),
                       n_m_grid: int = 200) -> NDArray:
        r"""Sample N events (m, x_aux) from the joint density at c.

        Two-stage:
            (1) inverse-CDF on the marginal m, p(m|c) ∝ g_SM(m) · µ(c, m).
                The injection integrates to zero in x_aux so the marginal
                m is the same as the no-injection marginal — but we keep
                the injection-aware code path for future generalisation.
            (2) accept-reject on x_aux per sampled m:
                p(x_aux|m, c) ∝ ½ (1 + ε·(η·c) · I(x_aux))
                envelope = ½(1 + ε|η·c|).

        Returns:
            events: (N, 2) array. Columns: (log_m_over_ref=log(m/1), x_aux).
        """
        rng = np.random.default_rng(seed)
        c = np.asarray(c, dtype=float).reshape(N_OPS)
        m_grid = np.linspace(m_range[0], m_range[1], n_m_grid)
        # Marginal in m via the binned-ratio analogue.
        mu_grid = self.marginal_mu(np.tile(c, (n_m_grid, 1)), m_grid)
        rate = _g_SM(m_grid) * mu_grid
        rate = np.clip(rate, 0.0, None)
        # Trapezoidal CDF.
        cdf = np.zeros_like(rate)
        cdf[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) *
                              (m_grid[1:] - m_grid[:-1]))
        total = cdf[-1]
        if total <= 0:
            raise ValueError(f"degenerate rate at c={c}; integral = {total}")
        cdf = cdf / total
        # Inverse-CDF sample.
        u_m = rng.uniform(0.0, 1.0, size=N)
        m_sampled = np.interp(u_m, cdf, m_grid)
        # Accept-reject on x_aux.
        eta_dot_c = float(np.dot(np.asarray(self.injection_coupling), c))
        envelope_height = 0.5 * (1.0 + self.injection_eps * abs(eta_dot_c))
        out_xa = np.empty(N, dtype=np.float64)
        pending = np.ones(N, dtype=bool)
        max_iter = 200
        for _ in range(max_iter):
            n_p = int(pending.sum())
            if n_p == 0:
                break
            xa = rng.uniform(-1.0, 1.0, size=n_p)
            v = rng.uniform(0.0, envelope_height, size=n_p)
            density = 0.5 * (1.0 + self.injection_eps * eta_dot_c
                              * _injection_shape(xa))
            accept = v <= density
            idx_global = np.flatnonzero(pending)[accept]
            out_xa[idx_global] = xa[accept]
            pending[idx_global] = False
        if pending.any():
            out_xa[pending] = rng.uniform(-1.0, 1.0, size=int(pending.sum()))
        events = np.empty((N, 2), dtype=np.float64)
        events[:, 0] = np.log(m_sampled / 1.0)
        events[:, 1] = out_xa
        return events

    def log_likelihood_ratio_per_event(self, c: NDArray,
                                         events: NDArray) -> NDArray:
        """log w_c(x) = log p(x|c) − log p(x|SM) per event."""
        c = np.asarray(c, dtype=float).reshape(N_OPS)
        log_m = events[:, 0]
        x_aux = events[:, 1]
        m = np.exp(log_m)
        # p(x|c) ∝ g_SM(m) · µ(c, m) · ½ · (1 + ε(η·c) I(x))
        # p(x|SM) ∝ g_SM(m) · 1 · ½ · 1 (µ(SM,m)=1, injection vanishes)
        mu_c = self.marginal_mu(np.tile(c, (len(m), 1)), m)
        eta_dot_c = float(np.dot(np.asarray(self.injection_coupling), c))
        ang = 1.0 + self.injection_eps * eta_dot_c * _injection_shape(x_aux)
        # Numerical safety:
        ang = np.maximum(ang, 1e-30)
        mu_c = np.maximum(mu_c, 1e-30)
        return np.log(mu_c) + np.log(ang)

    def density_ratio_per_event(self, c: NDArray, events: NDArray) -> NDArray:
        """w_c(x) = p(x|c) / p(x|SM) per event (not log).

        For the toy:

            w_c(x) = µ(c, m) · [1 + ε (η·c) · sin(6 x_aux)]

        which is *exactly* representable in the joint event-feature basis

            ψ_in_span(x) = {1, m², m⁴, m⁶, x_aux, x_aux², x_aux³,
                            m²·x_aux, m⁴·x_aux, ...}

        because µ(c, m) is at-most-quartic in m (from the b_{11} ∝ m⁴ /
        g_SM ∝ m⁴ (1+m²) term, max power m⁶) and the angular factor is
        linear in sin(6 x_aux). The injection adds a sin(6 x_aux) channel
        which is *not* spanned by polynomial x_aux up to any finite
        degree. That makes this object the natural test for the
        residual-SVD detector: pick ψ to span the polynomial-in-(m, x_aux)
        in-span structure, and the only thing left in the residual is
        ε(η·c) µ(c, m) sin(6 x_aux), a clean coherent signal across
        working points.
        """
        c = np.asarray(c, dtype=float).reshape(N_OPS)
        log_m = events[:, 0]
        x_aux = events[:, 1]
        m = np.exp(log_m)
        mu_c = self.marginal_mu(np.tile(c, (len(m), 1)), m)
        eta_dot_c = float(np.dot(np.asarray(self.injection_coupling), c))
        ang = 1.0 + self.injection_eps * eta_dot_c * _injection_shape(x_aux)
        return mu_c * ang


# ---------------------------------------------------------------------------
# Morphing-matrix ground-truth oracle (Task 2 step 1)
# ---------------------------------------------------------------------------

def morphing_basis_eval(c_base: NDArray, m_grid: NDArray,
                          oracle: ToyCurvatureOracle) -> NDArray:
    """Evaluate µ(c, m) at each base point on the m-grid.

    Returns (N_base, K) array of µ values; the morphing-matrix system to
    invert for templates is then

        µ_obs(b, m) = 1·g_norm(m) + Σ_i a_i(m) c_b^i / g_SM(m)
                    + Σ_{i,j} b_{ij}(m) c_b^i c_b^j / g_SM(m)

    where the c-dependence per base point lives in a fixed feature row
    [1, c_0, c_1, c_0², c_1², 2 c_0 c_1].
    """
    N_base = c_base.shape[0]
    out = np.empty((N_base, len(m_grid)), dtype=float)
    for b in range(N_base):
        out[b] = oracle.marginal_mu(np.tile(c_base[b], (len(m_grid), 1)),
                                      m_grid)
    return out


def morphing_feature_row(c: NDArray) -> NDArray:
    """Row [1, c_0, c_1, c_0², c_1², 2 c_0 c_1] for a single c."""
    return np.array([1.0, c[0], c[1], c[0] ** 2, c[1] ** 2, 2.0 * c[0] * c[1]])


def morphing_design_matrix(c_base: NDArray) -> NDArray:
    """(N_base, 6) design matrix of morphing features per base point.

    cond(M) is the morphing-matrix conditioning Task 2 step 1 watches; it
    depends on base-point placement (handoff §1.2 / Belyaev et al.).
    """
    return np.stack([morphing_feature_row(c) for c in c_base])


def solve_morphing_templates(c_base: NDArray, m_grid: NDArray,
                                oracle: ToyCurvatureOracle
                                ) -> dict:
    """Recover (g_norm, a_0, a_1, b_{00}, b_{11}, b_{01}) at each m on the
    grid by solving the morphing linear system at every m independently:

        Y_b(m) = µ(c_b, m), one row per base point
        Y(m)   = M · θ(m), with M the (N_base, 6) design matrix
        θ(m)   = least-squares solution

    Reports condition number of M and per-template max relative error vs
    the analytic templates as the Task 2 ground-truth oracle.
    """
    Y = morphing_basis_eval(c_base, m_grid, oracle)              # (N_base, K)
    M = morphing_design_matrix(c_base)                            # (N_base, 6)
    cond = float(np.linalg.cond(M))
    # Per-m least-squares: θ(m) = (M^T M)^{-1} M^T Y(:, m)
    theta, _residuals, _rank, _sv = np.linalg.lstsq(M, Y, rcond=None)
    # theta: (6, K). Rows are [g_norm, ã_0, ã_1, b̃_{00}, b̃_{11}, b̃_{01}]
    # where the tilded quantities are normalised by g_SM(m).
    g = _g_SM(m_grid)
    g_norm_rec = theta[0]                          # should be ≈ 1 (we already
                                                     # normalised by g_SM)
    a_0_rec = theta[1] * g
    a_1_rec = theta[2] * g
    b_00_rec = theta[3] * g
    b_11_rec = theta[4] * g
    b_01_rec = theta[5] * g
    truth_a_0, truth_a_1 = oracle.template_a(m_grid)
    truth_b_00, truth_b_11, truth_b_01 = oracle.template_b(m_grid)
    return {
        "cond_morphing_matrix": cond,
        "g_sm_recovered": g * g_norm_rec,        # the reconstructed σ_SM·1
        "a_0_recovered": a_0_rec, "a_1_recovered": a_1_rec,
        "b_00_recovered": b_00_rec, "b_11_recovered": b_11_rec,
        "b_01_recovered": b_01_rec,
        "a_0_truth": truth_a_0, "a_1_truth": truth_a_1,
        "b_00_truth": truth_b_00, "b_11_truth": truth_b_11,
        "b_01_truth": truth_b_01,
        "rel_err_a_0": float(np.max(np.abs(a_0_rec - truth_a_0))
                              / (np.max(np.abs(truth_a_0)) + 1e-12)),
        "rel_err_a_1": float(np.max(np.abs(a_1_rec - truth_a_1))
                              / (np.max(np.abs(truth_a_1)) + 1e-12)),
        "rel_err_b_00": float(np.max(np.abs(b_00_rec - truth_b_00))
                               / (np.max(np.abs(truth_b_00)) + 1e-12)),
        "rel_err_b_11": float(np.max(np.abs(b_11_rec - truth_b_11))
                               / (np.max(np.abs(truth_b_11)) + 1e-12)),
        "rel_err_b_01": float(np.max(np.abs(b_01_rec - truth_b_01))
                               / (np.max(np.abs(truth_b_01)) + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Working-point Fisher on the toy
# ---------------------------------------------------------------------------

def working_point_fisher_toy(oracle: ToyCurvatureOracle, c_wp: NDArray,
                                m_grid: NDArray, sigma_y: float = 0.05,
                                fd_step: float = 1e-3) -> NDArray:
    """F_{ij}(c) = Σ_m (∂_i µ)(∂_j µ) / σ_y² on the toy marginal observable."""
    K = len(m_grid)
    C0 = np.tile(c_wp, (K, 1))
    grad = np.empty((K, N_OPS))
    for i in range(N_OPS):
        cp = C0.copy(); cp[:, i] += fd_step
        cm = C0.copy(); cm[:, i] -= fd_step
        grad[:, i] = (oracle.marginal_mu(cp, m_grid)
                       - oracle.marginal_mu(cm, m_grid)) / (2.0 * fd_step)
    F = grad.T @ grad / (sigma_y ** 2)
    return 0.5 * (F + F.T)


if __name__ == "__main__":
    oracle = ToyCurvatureOracle(injection_eps=0.0)
    m = np.array([0.5, 1.0, 1.5, 2.0])
    print(f"# template eval at m = {m}")
    print(f"  g_SM(m)        = {oracle.template_g_sm(m).round(4)}")
    a0, a1 = oracle.template_a(m)
    print(f"  a_0(m) vertex  = {a0.round(4)}  (should be 0.5·g_SM)")
    print(f"  a_1(m) 4-ferm  = {a1.round(4)}  (should be 0.3·m²)")
    b00, b11, b01 = oracle.template_b(m)
    print(f"  b_{{0,0}}(m) self = {b00.round(4)}  (small, 0.02·g_SM)")
    print(f"  b_{{1,1}}(m) self = {b11.round(4)}  (large, 0.5·m^4)")
    print(f"  b_{{0,1}}(m) cross= {b01.round(4)}  (m²·g_SM, the lift driver)")

    c = np.array([0.4, 0.5])
    print(f"\n# µ(c={c.tolist()}, m={m.tolist()}) sample:")
    print(f"  {oracle.marginal_mu(np.tile(c, (len(m), 1)), m).round(4)}")

    # Morphing recovery
    rng = np.random.default_rng(0)
    c_base = rng.uniform(-0.5, 0.5, size=(6, N_OPS))               # 6 base points
    m_grid = np.linspace(0.3, 2.5, 30)
    res = solve_morphing_templates(c_base, m_grid, oracle)
    print(f"\n# Morphing recovery at {len(c_base)} base points:")
    print(f"  cond(M) = {res['cond_morphing_matrix']:.3e}")
    print(f"  rel err  a_0={res['rel_err_a_0']:.3e}  a_1={res['rel_err_a_1']:.3e}")
    print(f"           b_00={res['rel_err_b_00']:.3e}  b_11={res['rel_err_b_11']:.3e}")
    print(f"           b_01={res['rel_err_b_01']:.3e}")
