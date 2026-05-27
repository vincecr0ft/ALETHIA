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

"""Offline smoke check for the ALETHIA local dev setup.

Runs without Gemini credentials. Verifies:
  1. the webshop tool environment behaves correctly,
  2. Phoenix is reachable at the configured collector endpoint,
  3. a trace can be exported into Phoenix end-to-end.

Usage:  make smoke   (or:  cd agent && uv run python smoke.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from shopping_demo.mini_webshop import MiniWebshopEnv


def check_webshop() -> None:
    """Drive the in-memory webshop through search -> click -> buy."""
    env = MiniWebshopEnv()
    env.step("search[floral dress]")
    assert "B09P5CRVQ6" in env.observation, "search did not return the catalog"
    env.step("click[B09P5CRVQ6]")
    assert "Floral Summer Dress" in env.observation, "product page did not render"
    _, reward, done, _ = env.step("click[Buy Now]")
    assert done and reward == 1.0, "Buy Now did not complete the order"
    print("  OK  webshop tools: search -> click -> buy")


def check_phoenix() -> bool:
    """Ping Phoenix; if up, export one test span. Returns True when a span was sent."""
    endpoint = (os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006").rstrip("/")
    try:
        urlopen(endpoint, timeout=3)
    except (URLError, OSError) as exc:
        print(f"  SKIP  Phoenix unreachable at {endpoint} ({exc}) — run `make phoenix-up`")
        return False
    print(f"  OK  Phoenix reachable at {endpoint}")

    from opentelemetry import trace

    from instrumentation import setup_tracing

    provider = setup_tracing()
    tracer = trace.get_tracer("alethia.smoke")
    with tracer.start_as_current_span("alethia-smoke-test") as span:
        span.set_attribute("alethia.smoke", True)
    if provider is not None:
        provider.force_flush()
    project = os.environ.get("PHOENIX_PROJECT_NAME", "alethia")
    print(f"  OK  exported a test span — check project '{project}' at {endpoint}")
    return True


def main() -> int:
    print("ALETHIA smoke check")
    try:
        check_webshop()
        check_phoenix()
    except AssertionError as exc:
        print(f"  FAIL  {exc}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
