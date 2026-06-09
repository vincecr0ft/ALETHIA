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

# The agent's own physics tools. Each emits an OpenInference span to Phoenix.
_tools: list = [
    FunctionTool(func=fm_predict),
    FunctionTool(func=check_drift),
    FunctionTool(func=recover_from_drift),  # preferred for drift recovery
    FunctionTool(func=epig_select),         # fine-grained primitives
    FunctionTool(func=oracle_query),
    FunctionTool(func=fm_update),
]

_instruction = ALETHIA_AGENT_INSTRUCTION


def _phoenix_mcp_toolset():
    """Arize Phoenix MCP server as an agent-level toolset, or ``None``.

    Enabling this (``ALETHIA_PHOENIX_MCP=1``, set in the Cloud Run image) lets
    the agent introspect *its own* observability data: list trace projects,
    pull spans, read experiment results. The agent that recovers from physics
    drift can then also answer "how did the last recovery go?" by reading the
    Phoenix traces it just emitted — closing the observe→reason loop on a
    partner (Arize) MCP server.

    Guarded behind an env flag and a try/except so local dev (no Node) and the
    cold-start path are never broken by an unreachable MCP server.
    """
    if (os.environ.get("ALETHIA_PHOENIX_MCP") or "").strip().lower() not in {"1", "true", "on"}:
        return None
    base_url = (os.environ.get("PHOENIX_MCP_BASE_URL")
                or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
                or "http://localhost:6006").rstrip("/")
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    try:
        from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        from mcp import StdioServerParameters

        args = ["-y", "@arizeai/phoenix-mcp@latest", "--baseUrl", base_url]
        if api_key:
            args += ["--apiKey", api_key]
        return McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(command="npx", args=args),
                timeout=60,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — never let MCP wiring break boot
        print(f"[alethia] Phoenix MCP toolset disabled: {exc}")
        return None


_mcp = _phoenix_mcp_toolset()
if _mcp is not None:
    _tools.append(_mcp)
    _instruction += (
        "\n\nYou also have Arize Phoenix MCP tools that read your own "
        "observability data (trace projects, spans, experiments). When the "
        "user asks how a prediction, drift check, or recovery actually went, "
        "query Phoenix for the relevant traces and summarise what you find — "
        f"the active project is '{os.environ.get('PHOENIX_PROJECT_NAME', 'alethia')}'."
    )

root_agent = Agent(
    model=_model,
    name="alethia_physics_agent",
    instruction=_instruction,
    tools=_tools,
)
