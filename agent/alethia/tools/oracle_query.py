"""oracle_query: evaluate the SMEFT oracle at given m-values, current target c."""
from __future__ import annotations

import time

import numpy as np
from opentelemetry import trace

from alethia.state import STATE

_tracer = trace.get_tracer("alethia.tools")


def oracle_query(m_values: list[float], fidelity: str = "T1") -> dict:
    """Call the SMEFT oracle at the requested m-values for the agent's
    current target Wilson scenario.

    Args:
        m_values: m_ll in TeV. Positive.
        fidelity: "T1" (analytic LO, ~ms/call) or "T2" (MadGraph LO,
            ~30 s/call with nevents=500).

    Returns:
        Dict with mu_values (list), fidelity_tier, n_points, wall_seconds.
        Wilson coefficients are NOT returned in the response — they are an
        internal property of the agent state, per the BRIEF.
    """
    STATE.ensure_initialised()
    m = np.asarray(m_values, dtype=float)
    with _tracer.start_as_current_span("tool.oracle.query") as sp:
        t0 = time.time()
        mu = STATE._oracle_truth(m, fidelity=fidelity)
        wall = time.time() - t0
        STATE.oracle_calls += len(m)
        wnorm = float(np.linalg.norm(STATE.target_c))
        sp.set_attribute("aletheia.oracle.fidelity_tier", fidelity)
        sp.set_attribute("aletheia.oracle.n_points", len(m))
        sp.set_attribute("aletheia.oracle.cost_seconds", float(wall))
        sp.set_attribute("aletheia.oracle.wilson_norm", wnorm)
        sp.set_attribute("aletheia.oracle.cumulative_calls",
                         int(STATE.oracle_calls))
    return {
        "mu_values": [float(v) for v in mu],
        "fidelity_tier": fidelity,
        "n_points": int(len(m)),
        "wall_seconds": float(wall),
        "cumulative_oracle_calls": int(STATE.oracle_calls),
    }
