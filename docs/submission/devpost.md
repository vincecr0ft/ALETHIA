# ALETHIA — Devpost submission draft

> Google Cloud Rapid Agent Hackathon · **Arize track**
> Fill the Devpost fields below; replace every `<…>` before submitting.

## Submission checklist (hard requirements)

- [ ] **Hosted project URL** — Cloud Run service URL (`./deploy.sh` prints it).
      Web chat UI: `<cloud-run-url>/dev-ui/?app=alethia`
- [ ] **Public open-source repo** — https://github.com/vincecr0ft/ALETHIA
      (confirm visibility is Public; Apache-2.0 `LICENSE` is present).
- [ ] **Demo video ≤3 min** on YouTube/Vimeo (script: `docs/submission/demo-script.md`).
- [ ] **Text description** — the sections below.
- [ ] **Built with Gemini + Google Cloud** — Gemini 2.5 Flash via ADK, deployed on Cloud Run.
- [ ] **Partner MCP integration** — Arize Phoenix MCP, wired into the agent.
- [ ] **Runs on web** — the ADK dev UI served by the Cloud Run service.

---

## Tagline

An AI agent that runs a physics foundation model, watches itself drift through
Arize Phoenix, and repairs its own knowledge with targeted experiments.

## Inspiration

Foundation models in the physical sciences silently go stale: the world (or the
detector, or the theory cut) shifts, and a model that was trustworthy yesterday
is quietly wrong today. Humans catch this late. We wanted an agent that treats
its own observability stream as a first-class input — one that can *notice* it
has drifted and *act* to fix it, the way a scientist re-measures when a result
stops making sense.

## What it does

ALETHIA is an ADK agent that orchestrates a physics foundation model (the
"Intention" surrogate for SMEFT Drell-Yan cross-sections). In one conversation it:

1. **Predicts** a physical observable μ(m) with a calibrated uncertainty band.
2. **Checks itself for drift** with three physics-aware detectors (a CUSUM on
   standardised residuals, a Benjamini-Hochberg-corrected coverage test, and a
   design-matrix condition-number check).
3. **Recovers** when drift fires: it selects the most informative new
   measurements via closed-form EPIG acquisition, queries an oracle for them,
   folds them into context, and re-calibrates — then re-predicts.
4. **Reads its own traces** through the Arize Phoenix MCP server to report how a
   prediction, drift check, or recovery actually went.

In our demo runs this loop recovers prediction error by ~110× over a 400-cycle
engineered-drift scenario (100 gated retrains) and ~150× when a new physics
operator switches on mid-deployment.

## How we built it

- **Agent:** Google **Agent Development Kit (ADK)**, Gemini 2.5 Flash
  orchestrating six deterministic physics tools over a shared FM/calibrator/
  context state (`agent/alethia/`).
- **Observability:** **Arize Phoenix** via OpenInference — every tool emits a
  span with `aletheia.*` attributes (`agent/instrumentation.py`).
- **Partner MCP (Arize):** the agent carries an ADK `McpToolset` over
  `@arizeai/phoenix-mcp`, so it can query the very traces it just produced
  (`agent/alethia/agent.py`, `ALETHIA_PHOENIX_MCP=1`).
- **Hosting:** **Google Cloud Run** — `main.py` serves the ADK web UI + API via
  `get_fast_api_app`; `Dockerfile` (Python + Node, for the MCP server) builds
  remotely through Cloud Build; `deploy.sh` is one command.
- **The physics:** a closed-form analytic SMEFT oracle and the Intention FM with
  conformal calibration and EPIG acquisition (`modules/`).

## How the partner technology is used (Arize)

Phoenix is not just a dashboard here. The agent's own tools are instrumented to
Phoenix, and the **Phoenix MCP server is a tool the agent calls at runtime** —
the observe→reason→act loop closes on the partner platform. The same Phoenix
data drives the drift evaluations that gate retraining.

## Challenges

- Closing the loop on the agent's *own* telemetry without it hanging on cold
  starts — the MCP toolset is guarded and lazy so the service always boots.
- Keeping the deployed image slim: analytic oracle only, a 24 KB packaged
  checkpoint, no MadGraph/LHAPDF.

## What's next

- Hosted Phoenix experiments comparing acquisition strategies on live traffic.
- The agent proposing A/B promotion of a recovered model version under its
  BH-corrected promotion rule.

## Built with

`google-adk` · Gemini 2.5 Flash · Google Cloud Run · Cloud Build · Arize Phoenix
· OpenInference/OpenTelemetry · `@arizeai/phoenix-mcp` · PyTorch · NumPy/SciPy

## Mapping to judging criteria

- **Technological implementation** — ADK + Gemini + Cloud Run + a genuine
  agent-level partner MCP integration; closed-form, tested physics tools.
- **Design** — a single conversation drives predict → detect → repair, with a
  browser chat UI and a live trace view.
- **Potential impact** — self-maintaining foundation models for the sciences;
  the drift/recover pattern generalises beyond SMEFT.
- **Quality of the idea** — an agent that uses its own observability as a
  control signal, not just a log.
