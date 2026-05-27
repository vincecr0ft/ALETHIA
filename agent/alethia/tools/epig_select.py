"""epig_select: closed-form EPIG acquisition over m-values."""
from __future__ import annotations

import numpy as np
from opentelemetry import trace

from alethia.state import STATE
from modules.surrogate.intention import epig_acquire_m

_tracer = trace.get_tracer("alethia.tools")


def epig_select(k: int, m_lo_tev: float, m_hi_tev: float,
                pool_size: int = 200) -> dict:
    """Choose k m-values in [m_lo, m_hi] that maximally reduce predictive
    variance on a fixed target set covering the full operating range.

    Uses the closed-form EPIG over the Intention head's psi_theta features:
    sequential-greedy with Sherman-Morrison updates. The picked m-values
    should be passed to oracle_query then fm_update.

    Args:
        k: number of m-values to pick.
        m_lo_tev: lower bound of the candidate pool, in TeV.
        m_hi_tev: upper bound.
        pool_size: candidate pool size (random uniform in [m_lo, m_hi]).

    Returns:
        Dict with the picked m-values and the candidate-pool size.
    """
    STATE.ensure_initialised()
    rng = np.random.default_rng()
    M_pool = rng.uniform(m_lo_tev, m_hi_tev, size=pool_size)
    M_target = np.linspace(0.4, 2.2, 50)
    with _tracer.start_as_current_span("tool.epig.select") as sp:
        idx = epig_acquire_m(STATE.model, STATE.M_ctx, STATE.Y_ctx,
                             M_pool, M_target, k=k)
        picked = M_pool[idx]
        sp.set_attribute("aletheia.epig.k", int(k))
        sp.set_attribute("aletheia.epig.pool_size", int(pool_size))
        sp.set_attribute("aletheia.epig.m_lo_tev", float(m_lo_tev))
        sp.set_attribute("aletheia.epig.m_hi_tev", float(m_hi_tev))
        sp.set_attribute("aletheia.epig.picked_m_values",
                         [float(m) for m in picked])
    return {
        "picked_m_values": [float(m) for m in picked],
        "k": int(k),
        "pool_size": int(pool_size),
        "m_range_tev": [float(m_lo_tev), float(m_hi_tev)],
    }
