#!/usr/bin/env bash
# Deploy the ALETHIA ADK agent to Google Cloud Run.
#
# Builds the image remotely with Cloud Build (no local Docker needed), then
# deploys a public web service. Reads secrets/config from .env so nothing
# sensitive is hard-coded here.
#
# Prereqs (one-time):
#   - gcloud CLI installed and authenticated:  gcloud auth login
#   - a project with billing enabled
#   - .env filled in (GOOGLE_API_KEY + Phoenix Cloud endpoint/key)
#
# Usage:
#   ./deploy.sh                 # uses PROJECT_ID/REGION below or from env
#   PROJECT_ID=my-proj ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via environment) -------------------------------------
PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-alethia-agent}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: set PROJECT_ID (e.g. PROJECT_ID=my-gcp-project ./deploy.sh)" >&2
  exit 1
fi

# ---- load .env (export every non-comment KEY=VALUE) ------------------------
if [[ -f .env ]]; then
  set -a; source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a
fi

: "${GOOGLE_API_KEY:?set GOOGLE_API_KEY in .env}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-alethia}"
# For Phoenix Cloud: collector endpoint includes /s/<space>; MCP baseUrl is the
# bare host. Fall back to the collector endpoint if MCP base is unset.
PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-}"
PHOENIX_MCP_BASE_URL="${PHOENIX_MCP_BASE_URL:-${PHOENIX_COLLECTOR_ENDPOINT}}"
PHOENIX_API_KEY="${PHOENIX_API_KEY:-}"

echo "→ project=${PROJECT_ID}  region=${REGION}  service=${SERVICE}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "→ enabling required APIs (run, cloudbuild, artifactregistry)…"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com >/dev/null

# ---- assemble env vars for the service -------------------------------------
ENV_VARS="GOOGLE_API_KEY=${GOOGLE_API_KEY}"
ENV_VARS+=",GEMINI_MODEL=${GEMINI_MODEL}"
ENV_VARS+=",ALETHIA_PHOENIX_MCP=1"
ENV_VARS+=",PHOENIX_PROJECT_NAME=${PHOENIX_PROJECT_NAME}"
[[ -n "${PHOENIX_COLLECTOR_ENDPOINT}" ]] && ENV_VARS+=",PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT}"
[[ -n "${PHOENIX_MCP_BASE_URL}" ]]       && ENV_VARS+=",PHOENIX_MCP_BASE_URL=${PHOENIX_MCP_BASE_URL}"
[[ -n "${PHOENIX_API_KEY}" ]]            && ENV_VARS+=",PHOENIX_API_KEY=${PHOENIX_API_KEY}"

echo "→ deploying (Cloud Build + Cloud Run)…"
gcloud run deploy "${SERVICE}" \
  --source . \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --set-env-vars "^@@^${ENV_VARS//,/@@}" \
  --quiet

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
echo
echo "✓ deployed: ${URL}"
echo "  web chat UI:  ${URL}/dev-ui/?app=alethia"
echo "  health:       ${URL}/list-apps"
