# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phoenix tracing for ALETHIA — ``register(auto_instrument=True)`` per the ADK doc.

https://arize.com/docs/phoenix/integrations/python/google-adk/google-adk-tracing

By default the agent traces into a **local self-hosted Phoenix** (the docker-compose
service at ``http://localhost:6006``), which requires no API key. To use Phoenix Cloud
instead, set ``PHOENIX_COLLECTOR_ENDPOINT`` to your space hostname and ``PHOENIX_API_KEY``.

Environment:
  ``PHOENIX_COLLECTOR_ENDPOINT``  collector base URL (default ``http://localhost:6006``)
  ``PHOENIX_API_KEY``             only needed for Phoenix Cloud
  ``PHOENIX_PROJECT_NAME``        project shown in the Phoenix UI (default ``alethia``)
  ``PHOENIX_TRACING``             set to ``0``/``off``/``false`` to disable tracing
"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen

from phoenix.otel import register

DEFAULT_LOCAL_ENDPOINT = "http://localhost:6006"

_provider: Optional[Any] = None


def _resolve_endpoint() -> str:
    """Collector base URL: an explicit env var wins, else the local docker Phoenix."""
    endpoint = (os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or "").strip()
    return endpoint.rstrip("/") if endpoint else DEFAULT_LOCAL_ENDPOINT


def _warn_if_unreachable(endpoint: str) -> None:
    """Print a hint (don't fail) when nothing is listening at ``endpoint``."""
    try:
        urlopen(endpoint, timeout=2)
    except (URLError, OSError):
        print(
            f"[alethia] Phoenix not reachable at {endpoint} — "
            "run `make phoenix-up` to start the local container. "
            "The agent still runs; traces are dropped until Phoenix is up."
        )


def setup_tracing() -> Optional[Any]:
    """Register the Phoenix tracer provider once.

    Returns the provider, or ``None`` when tracing is disabled via ``PHOENIX_TRACING``.
    """
    global _provider
    if _provider is not None:
        return _provider

    if (os.environ.get("PHOENIX_TRACING") or "").strip().lower() in {"0", "off", "false"}:
        return None

    endpoint = _resolve_endpoint()
    # Make the resolved endpoint explicit so phoenix.otel.register picks it up.
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint
    _warn_if_unreachable(endpoint)

    _provider = register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "alethia"),
        batch=False,
        auto_instrument=True,
        verbose=False,
    )
    return _provider
