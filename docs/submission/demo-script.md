# ALETHIA — 3-minute demo video script

Target: **≤ 3:00**, English audio or subtitles, uploaded to YouTube/Vimeo.
Record the Cloud Run web UI (`<url>/dev-ui/?app=alethia`) on screen, with the
Phoenix UI in a second tab. Keep one continuous agent conversation.

---

### 0:00–0:20 — Hook + what it is
*(screen: the deployed Cloud Run chat UI, ALETHIA selected)*

> "This is ALETHIA, running live on Google Cloud Run. It's an ADK agent that
> drives a physics foundation model — and it watches itself for drift using
> Arize Phoenix, then repairs its own predictions. Everything you'll see is one
> agent, one conversation."

Show the URL bar (the public Cloud Run URL) for 2 seconds.

### 0:20–0:55 — Predict
*(type into the chat)*

> Prompt: "Predict μ(m) at m = 0.6, 1.0, 1.5, 2.0 TeV and give the calibrated band."

- Narrate: "Gemini calls the `fm_predict` tool — the foundation model returns
  the observable with a conformal uncertainty interval."
- Cut to the **Phoenix** tab: show the `tool.surrogate.predict` span appearing
  with its `aletheia.fm.*` attributes. "Every tool call is traced to Phoenix."

### 0:55–1:40 — Detect drift
*(continue the conversation)*

> Prompt: "Is the model still reliable in this range? Check for drift."

- Narrate: "It runs three physics-aware detectors — a CUSUM on residuals, a
  coverage test with Benjamini-Hochberg correction, and a condition-number
  check on the design matrix."
- Phoenix tab: show the `chain.drift.evaluate` span tree (three child spans),
  and the aggregate decision flag firing `local_retrain`.

### 1:40–2:25 — Recover
*(continue)*

> Prompt: "Drift fired — recover: pick 5 informative measurements with EPIG,
> query the oracle, fold them in, recalibrate, and re-predict."

- Narrate: "It uses closed-form EPIG acquisition to choose where new data is
  most valuable, queries the oracle, updates context, recalibrates — and the
  band tightens."
- Show the before/after numbers shrink. Phoenix tab: the
  `chain.recover_from_drift` span with its nested EPIG/oracle/update children.

### 2:25–2:55 — The Phoenix MCP loop (the partner integration)
*(continue)*

> Prompt: "Using Phoenix, summarise how that recovery actually went."

- Narrate: "Here's the part that closes the loop: the agent calls the **Arize
  Phoenix MCP server** to read the very traces it just produced, and reports
  back on its own recovery. Its observability is an input, not just a log."
- Show the agent's natural-language summary citing the trace data.

### 2:55–3:00 — Close

> "ADK on Cloud Run, Gemini, and Arize Phoenix — an agent that maintains its
> own foundation model. Code and license are public on GitHub."

Show the GitHub URL on screen.

---

**Pre-record checklist**
- [ ] Phoenix Cloud configured so spans appear in real time on screen.
- [ ] Run the full prompt sequence once as a dry run (cold start warms the FM).
- [ ] Confirm the MCP summary step returns (Node present in the image).
- [ ] Total runtime under 3:00 — trim narration, not the live tool calls.
