r"""Event-level sampler for the analytic SMEFT Drell-Yan oracle.

The current :class:`AnalyticSMEFTOracle` returns *integrated* rate ratios at
single ``(c, m_ll)`` cells (``truth``, ``truth_mu_fb``, ``truth_costheta_bin``,
``truth_afb``). The ManifoldInformer reframe
(``ALETHIA_informer_workpoint_AL_handoff.md`` §1) requires per-event
kinematic samples ``x = (m_ℓℓ, cos θ*_CS)`` distributed according to the
analytic differential cross section at a fixed Wilson working point ``c``.
This is Task 0 of the research plan.

We do *not* re-derive the differential cross section. We compose the
existing pieces. The angular structure factorises as

.. math::

    \frac{d\sigma(c, m_{\ell\ell}, \cos\theta^*)}{dm_{\ell\ell}\, d\cos\theta^*}
        \propto S(c, m) \, (1 + \cos^2\theta^*)  +  2 D(c, m) \cdot \cos\theta^*

where ``S`` is the symmetric piece (which integrates to the rate ratio
``truth(c, m)``) and ``D`` is the antisymmetric piece (which integrates
to the FB-asymmetry numerator ``truth_mu_fb(c, m)``).

The sampling proceeds in two stages:

1. **Marginal in m_ll**: build the CDF of ``p(m_ll | c) ∝ σ_SM(m_ll) ·
   μ(c, m_ll)`` on a dense grid, inverse-CDF sample ``N`` values of m_ll.
2. **Conditional in cos θ***: per sampled ``m_ll``, accept-reject sample
   ``u ∈ [-1, +1]`` from ``p(u | m_ll, c) ∝ S(c, m_ll)(1 + u^2) + 2D(c, m_ll) u``
   using a constant envelope ``2(S + |D|)``.

This produces unweighted events at the analytic LO. The implementation
keeps c-independent quantities (σ_SM, A_FB_SM) cached because the
oracle re-derives them per call.

Falsification gates (``tests/test_oracle_events.py``):

- Empirical p(m_ll) from N=1e5 samples reproduces ``μ(c, m_ll)`` from
  ``truth(c, m_ll)`` to within Kolmogorov-Smirnov 1%.
- Empirical A_FB(m_ll bin) reproduces ``truth_afb(c, m_bin)`` to within
  statistical uncertainty over 10 m-bins, ≥ 1e5 events.

References
----------
- ``modules/analytic_smeft/smeft.py``: differential cross section and the
  S/D angular split (Greljo-Marzocca arXiv:1704.09015).
- ALETHIA_informer_workpoint_AL_handoff.md §1.3 and §3 step 0.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .oracle_smeft import AnalyticSMEFTOracle
from .features import N_WC


# m_ll grid for the marginal CDF. Wide enough to cover the analysis window
# the rest of the codebase uses; dense enough that inverse-CDF is smooth.
_DEFAULT_M_TEV = np.linspace(0.3, 2.5, 220)


def _angular_S_DoverS(oracle: AnalyticSMEFTOracle, c: NDArray, m_tev: NDArray
                       ) -> tuple[NDArray, NDArray]:
    r"""Return ``(S(c, m), (D/S)(c, m))`` for the conditional sampler.

    The angular density is ``dσ/du ∝ S(1 + u²) + 2 D u``. Integrating over
    ``u ∈ [-1, +1]``:

        ∫ S(1+u²) du = (8/3) S  →  this is the angle-integrated cross section
        ∫ 2 D u du   = 0
        σ_F − σ_B    = 2 D
        A_FB(c, m)   = (σ_F − σ_B) / (σ_F + σ_B) = 2 D / (8 S / 3) = 3 D / (4 S)
        ⇒  D / S    = (4/3) A_FB(c, m)

    So ``D/S`` is recoverable directly from ``oracle.truth_afb`` without
    needing the absolute ``D`` (which ``truth_mu_fb`` normalises away by
    dividing by the SM FB numerator — that is why an earlier construction
    here mis-extracted ``D`` and the sampler over-produced forward events).

    The absolute ``S`` is computed from the rate ratio:
        truth(c, m) = (8/3) S(c, m) / [(8/3) S(SM, m)] = S(c, m) / S(SM, m)
    So ``S(c, m)`` in the same units as ``S(SM, m)`` is just ``truth(c, m)``
    times a positive constant; the constant cancels in the conditional
    (only the ratio ``D/S`` enters ``p(u | m, c)`` after factoring ``S``
    out of ``S(1 + u²) + 2 D u``).

    Args:
        oracle: AnalyticSMEFTOracle.
        c: (n_wc,) working point.
        m_tev: (K,) array of m_ll values in TeV.

    Returns:
        S_rel: (K,) array; positive. ``S(c, m)`` in S(SM, m)-units; equals
            ``oracle.truth(c, m)``.
        DoverS: (K,) array; signed. ``D(c, m) / S(c, m) = (4/3) A_FB(c, m)``.
    """
    K = len(m_tev)
    C = np.tile(c, (K, 1))
    S_rel = oracle.truth(C, m_tev)                                  # = µ(c, m)
    A_FB = oracle.truth_afb(C, m_tev)                                # ∈ [-1, +1]
    DoverS = (4.0 / 3.0) * A_FB
    return S_rel, DoverS


def _build_m_cdf(oracle: AnalyticSMEFTOracle, c: NDArray,
                  m_grid: NDArray, sm_only: bool = False
                  ) -> tuple[NDArray, NDArray]:
    r"""Build the marginal m_ll CDF for a working point.

    The differential rate at LO is
        dσ/dm_ll(c, m) = σ_SM(m) · μ(c, m)
    where σ_SM(m) is the SM differential and μ(c, m) = truth(c, m) is the
    ratio. To get σ_SM(m) we call truth at c=0 and recover the absolute
    normalisation via ``differential_xs`` for c=0... but at LO and with the
    ``pdf="analytic"`` toy, the *shape* of σ_SM(m) is what matters for the
    marginal CDF — the absolute scale just rescales the CDF unit interval.

    To keep this fast (no per-event PDF calls), we precompute σ_SM(m) once
    via the existing ``differential_xs`` of the underlying calculator on
    the m_grid, then scale by μ(c, m).

    Args:
        oracle: AnalyticSMEFTOracle.
        c: working point (n_wc,).
        m_grid: m-grid in TeV (K,).
        sm_only: if True, return the SM-only CDF; useful for asymptotic checks.

    Returns:
        (m_grid, cdf) — cdf has shape (K,), monotone in [0, 1].
    """
    from modules.analytic_smeft import differential_xs
    # σ_SM(m) at c = 0: differential_xs with empty wc and the same oracle settings.
    res_sm = differential_xs(
        {},                                          # empty Wilson dict ⇒ SM
        m_grid * 1000.0,                              # TeV → GeV
        sqrt_s=oracle.sqrt_s,
        lambda_scale=oracle.lam,
        order=oracle.order,
        pdf=oracle.pdf,
    )
    sigma_sm = np.asarray(res_sm["sm_only"], dtype=float)        # (K,)

    if sm_only:
        rate = sigma_sm
    else:
        C = np.tile(c, (len(m_grid), 1))
        mu = oracle.truth(C, m_grid)                              # (K,) = σ_BSM / σ_SM
        rate = sigma_sm * mu                                       # absolute differential

    rate = np.clip(rate, 0.0, None)                                # numerical safety
    # Trapezoidal CDF over m_grid.
    cdf = np.zeros_like(rate)
    cdf[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1])
                          * (m_grid[1:] - m_grid[:-1]))
    total = cdf[-1]
    if total <= 0.0:
        # Degenerate case (numerical) — return uniform fallback to avoid NaN.
        cdf = np.linspace(0.0, 1.0, len(m_grid))
    else:
        cdf = cdf / total
    return m_grid, cdf


def _sample_m_from_cdf(m_grid: NDArray, cdf: NDArray, N: int,
                        rng: np.random.Generator) -> NDArray:
    """Inverse-CDF sample N m-values. Linear interpolation between grid points."""
    u = rng.uniform(0.0, 1.0, size=N)
    # numpy.interp inverts a monotone function; the (x, y) here is (cdf, m).
    # It clips outside the range, which matches our boundary convention.
    return np.interp(u, cdf, m_grid)


def _sample_costheta(DoverS_per_event: NDArray,
                      rng: np.random.Generator, *,
                      max_iter: int = 100) -> NDArray:
    r"""Per-event accept-reject sampling of cos θ*_CS from
        p(u | D/S) ∝ (1 + u²) + 2 (D/S) u,   u ∈ [-1, +1].

    ``S`` factors out of the conditional, so only the ratio ``D/S = (4/3)
    A_FB(c, m)`` is needed (cf. ``_angular_S_DoverS``). For ``|D/S| < 1``
    — equivalent to ``|A_FB| < 3/4``, which holds physically across the
    Drell-Yan analysis window above the Z pole — the density is positive
    everywhere on [-1, +1].

    Envelope: ``f(u) = (1 + u²) + 2 r u`` where ``r = D/S``. Endpoints
    give ``f(+1) = 2(1 + r)``, ``f(-1) = 2(1 - r)``, so the constant
    envelope is ``2(1 + |r|)``.

    Args:
        DoverS_per_event: (N,) signed ``D/S = (4/3) A_FB`` per event.
        rng: numpy Generator.

    Returns:
        u: (N,) cos θ*_CS values in [-1, +1].
    """
    N = len(DoverS_per_event)
    out = np.empty(N, dtype=np.float64)
    pending = np.ones(N, dtype=bool)
    r = DoverS_per_event
    envelope = 2.0 * (1.0 + np.abs(r))                              # (N,) > 0
    for _ in range(max_iter):
        n_pending = int(pending.sum())
        if n_pending == 0:
            break
        u = rng.uniform(-1.0, 1.0, size=n_pending)
        v = rng.uniform(0.0, 1.0, size=n_pending)
        f = (1.0 + u * u) + 2.0 * r[pending] * u
        env = envelope[pending]
        accepted = v * env <= f
        idx_global = np.flatnonzero(pending)[accepted]
        out[idx_global] = u[accepted]
        pending[idx_global] = False
    if pending.any():
        out[pending] = rng.uniform(-1.0, 1.0, size=int(pending.sum()))
    return out


def sample_events(oracle: AnalyticSMEFTOracle, c: NDArray, N: int,
                   *, seed: int | None = None,
                   m_grid: NDArray | None = None) -> NDArray:
    r"""Sample N events from the analytic SMEFT Drell-Yan differential
    cross section at working point ``c``.

    Per event, returns two features:

        x = (log(m_ll / 1 TeV),  cos θ*_CS)   ∈ R^2

    log-scaling on m_ll matches the convention in ``IntentionFM.PsiMLP``
    and lets the per-event encoder consume the same scale as the rest of
    the FM stack.

    Args:
        oracle: AnalyticSMEFTOracle (must support truth, truth_mu_fb).
        c: (n_wc,) working point in Wilson space, canonical order.
        N: number of events to draw.
        seed: optional RNG seed (else fresh entropy).
        m_grid: optional m-grid in TeV for the CDF construction. Defaults
            to a 220-point linspace over [0.3, 2.5] TeV.

    Returns:
        events: (N, 2) array. Columns are (log_m_over_ref, cos θ*_CS).

    Notes:
        - This samples from p(m, cosθ* | c) only; the longitudinal rapidity
          ``y`` is marginalised away. At LO the dilepton p_T is zero (back-
          to-back leptons in the dilepton rest frame), so p_T is *not* an
          independent kinematic axis at this approximation and is omitted.
          NLO event-level samples (with non-zero p_T_ℓℓ) come from MadGraph
          via a separate LHE adapter.
        - The marginal m_ll CDF uses the analytic-PDF σ_SM(m_ll) shape; if
          the calling code rebuilds the oracle with a different PDF it is
          the caller's responsibility to pass a compatible ``m_grid`` and
          accept the recomputed CDF.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(c, dtype=float).reshape(N_WC)
    if m_grid is None:
        m_grid = _DEFAULT_M_TEV

    # Stage 1: marginal m_ll.
    _, cdf = _build_m_cdf(oracle, c, m_grid, sm_only=False)
    m_sampled = _sample_m_from_cdf(m_grid, cdf, N, rng)          # (N,)

    # Stage 2: conditional cos θ*_CS per sampled m_ll.
    _, DoverS_per = _angular_S_DoverS(oracle, c, m_sampled)       # (N,)
    u_sampled = _sample_costheta(DoverS_per, rng)                  # (N,)

    out = np.empty((N, 2), dtype=np.float64)
    out[:, 0] = np.log(m_sampled / 1.0)                            # log(m/M_ref=1 TeV)
    out[:, 1] = u_sampled
    return out


def event_log_likelihood_ratio(
        oracle: AnalyticSMEFTOracle, c: NDArray, events: NDArray
        ) -> NDArray:
    r"""Per-event log-likelihood-ratio at working point ``c``:

        log w_c(x) = log p(x | c) - log p(x | SM)

    The marginal-m factor σ_SM(m) cancels and we are left with the rate
    ratio µ(c, m) and the angular-shape ratio. Writing the conditional
    in (1 + u²) + 2 r u form (where r = (4/3) A_FB), p(x | c) factorises as

        p(x | c) ∝ σ_SM(m) · µ(c, m) ·
                   [(1 + u²) + 2 r(c, m) u] / [(8/3) · 1]

    so the log-likelihood ratio against SM is

        log w_c(x) = log µ(c, m)
                   + log[(1 + u²) + 2 r(c, m) u]
                   - log[(1 + u²) + 2 r(SM, m) u].
    """
    c = np.asarray(c, dtype=float).reshape(N_WC)
    log_m = events[:, 0]
    u = events[:, 1]
    m_tev = np.exp(log_m)
    C = np.tile(c, (len(m_tev), 1))
    mu_c = oracle.truth(C, m_tev)
    _, r_c = _angular_S_DoverS(oracle, c, m_tev)
    _, r_sm = _angular_S_DoverS(oracle, np.zeros(N_WC), m_tev)
    ang_c = (1.0 + u * u) + 2.0 * r_c * u
    ang_sm = (1.0 + u * u) + 2.0 * r_sm * u
    eps = 1e-30
    return (np.log(np.maximum(mu_c, eps))
            + np.log(np.maximum(ang_c, eps))
            - np.log(np.maximum(ang_sm, eps)))
