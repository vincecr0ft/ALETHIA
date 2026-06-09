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

"""Cloud Run entrypoint: serve the ALETHIA ADK agent over HTTP.

`adk` discovers agent packages under ``agents_dir``; ours is ``agent/alethia``.
``get_fast_api_app(web=True)`` mounts both the ADK dev UI (a browser chat — the
hackathon's "runs on web" surface) and the ``/run_sse`` JSON API on one app.

Tracing is registered the moment ``alethia.agent`` is imported (it calls
``setup_tracing()`` at module load), so every turn served here streams
OpenInference spans to the configured Phoenix instance without extra wiring.

Run locally:   uvicorn main:app --port 8080
Container CMD: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Agent packages (and the sibling ``instrumentation`` module they import) live
# under ./agent — put it on the path so ``alethia`` resolves the same way it
# does for the local one-shot runner.
AGENTS_DIR = Path(__file__).resolve().parent / "agent"
sys.path.insert(0, str(AGENTS_DIR))

from google.adk.cli.fast_api import get_fast_api_app

app = get_fast_api_app(
    agents_dir=str(AGENTS_DIR),
    web=True,
    allow_origins=["*"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
