"""Code-owned ALETHIA ADK agent.

Gemini orchestrates five deterministic physics tools (`fm_predict`,
`check_drift`, `epig_select`, `oracle_query`, `fm_update`) on top of a
singleton state that owns the trained Intention foundation model, the
conformal calibrator, the current (M_ctx, Y_ctx) context, and a
target Wilson scenario. Trace stream lands in Phoenix project
``alethia``.
"""
from .agent import root_agent

__all__ = ["root_agent"]
