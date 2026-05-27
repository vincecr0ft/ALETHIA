"""ALETHIA ADK root agent.

Gemini orchestrates five deterministic physics tools that act on a
singleton state owning the trained Intention FM, the conformal
calibrator, and the running context. The agent never receives a Wilson
coefficient; the target scenario is configured at session start and
lives in `alethia.state.STATE.target_c`.
"""
from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from dotenv import load_dotenv

from instrumentation import setup_tracing
from alethia.prompt import ALETHIA_AGENT_INSTRUCTION
from alethia.tools import (
    fm_predict, check_drift, epig_select, oracle_query, fm_update,
    recover_from_drift,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
setup_tracing()

_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

root_agent = Agent(
    model=_model,
    name="alethia_physics_agent",
    instruction=ALETHIA_AGENT_INSTRUCTION,
    tools=[
        FunctionTool(func=fm_predict),
        FunctionTool(func=check_drift),
        FunctionTool(func=recover_from_drift),  # preferred for drift recovery
        FunctionTool(func=epig_select),         # fine-grained primitives
        FunctionTool(func=oracle_query),
        FunctionTool(func=fm_update),
    ],
)
