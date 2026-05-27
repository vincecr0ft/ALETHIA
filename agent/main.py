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

"""One ADK turn; tracing via ``instrumentation.setup_tracing``."""

from __future__ import annotations

import asyncio
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google.adk.runners import InMemoryRunner
from google.genai import types

from instrumentation import setup_tracing
from shopping_demo.agent import root_agent


async def run_turn(user_text: str) -> None:
    setup_tracing()
    app_name, user_id, session_id = "hackathon_shopping", "local_user", secrets.token_hex(8)
    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=user_text)]),
    ):
        parts = event.content.parts if event.content and event.content.parts else []
        for part in parts:
            if getattr(part, "text", None):
                print(part.text, end="", flush=True)
    print()


def main() -> None:
    msg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Help me find a floral summer dress and buy size M."
    )
    asyncio.run(run_turn(msg))


if __name__ == "__main__":
    main()
