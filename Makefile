.PHONY: help setup lhapdf phoenix-up phoenix-down phoenix-logs run run-adk smoke test

help:
	@echo "ALETHIA — targets:"
	@echo "  make setup        - uv sync + create .env from .env.example"
	@echo "  make phoenix-up   - start local Phoenix (Docker) at http://localhost:6006"
	@echo "  make phoenix-down - stop local Phoenix (trace data is kept)"
	@echo "  make phoenix-logs - tail the Phoenix container logs"
	@echo "  make run          - one-shot traced agent run (MESSAGE=...)"
	@echo "  make run-adk      - ADK CLI dev loop"
	@echo "  make smoke        - offline setup check (tools + trace pipeline, no Gemini key)"
	@echo "  make test         - run the module unit tests (pytest)"
	@echo "  make lhapdf       - build LHAPDF + CT18NNLO into vendor/ (optional, ~3 min)"

setup:
	uv sync
	@test -f .env || { cp .env.example .env && echo "Created .env — add your GOOGLE_API_KEY."; }

phoenix-up:
	docker compose up -d
	@echo "Phoenix starting at http://localhost:6006 (allow ~20s for healthy state)."

phoenix-down:
	docker compose down

phoenix-logs:
	docker compose logs -f phoenix

run:
	cd agent && uv run python main.py "$(if $(MESSAGE),$(MESSAGE),Help me find a floral summer dress and buy size M.)"

run-adk:
	cd agent && uv run adk run shopping_demo

smoke:
	cd agent && uv run python smoke.py

test:
	uv run pytest

lhapdf:
	bash scripts/install_lhapdf.sh
