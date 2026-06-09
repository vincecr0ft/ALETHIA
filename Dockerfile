# ALETHIA — Cloud Run image for the ADK physics agent.
#
# Python for the agent + Node for the Arize Phoenix MCP server, which the agent
# launches over stdio (`npx @arizeai/phoenix-mcp`) to read its own traces.
FROM python:3.12-slim

# Node 20 (for npx) + curl for healthchecks. Single layer, no apt cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the runtime assets the agent loads at first tool call:
#   - modules/   the physics surrogate stack (numpy-only)
#   - agent/     the ADK agent package, instrumentation, and the packaged
#                Intention FM checkpoint under alethia/assets/
COPY modules ./modules
COPY agent ./agent
COPY main.py .

# Warm the Phoenix MCP package into the image so the first agent turn doesn't
# pay an npx download (best-effort; ignored if the network is unavailable).
RUN npx -y @arizeai/phoenix-mcp@latest --help > /dev/null 2>&1 || true

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
# Enable the agent-level Phoenix MCP toolset in the deployed image.
ENV ALETHIA_PHOENIX_MCP=1

EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
