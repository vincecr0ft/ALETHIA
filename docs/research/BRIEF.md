# ALETHIA research brief (shared)

This is the briefing every research agent reads. The whole project is a Google Cloud Rapid Agent
Hackathon entry in the Arize track. End-state: physics foundation model + dedicated SMEFT simulator,
monitored by Phoenix, drift detection gating EPIG-driven oracle calls, all wrapped behind an ADK
agent emitting OpenInference traces.

---

## Hard constraint (the whole point of this rewrite)

The foundation model must NEVER:

1. Take a Wilson coefficient (WC) value as input, in any encoding (raw, polynomial,
   one-hot, embedding lookup, conditioning vector — none of it).
2. Emit a WC value as output.
3. Be trained against any analytical ansatz in c. In particular, NOT
   `σ(c) ≈ σ_SM + Σ A_i c_i + Σ B_ij c_i c_j` (the morphing form). That ansatz
   is mathematically exact at the matrix-element level (amplitudes linear in
   c, squared amplitudes quadratic), which is exactly why it's seductive and
   exactly why it is the wrong learning target here: fitting it produces a
   regressor of a known parametric form, not a representation of the manifold.

Wilson coefficients are permitted only in:

- the data-generation process (the oracle samples events / distributions
  conditioned on a chosen c).
- the construction of the loss (e.g. labels are c-tagged samples; the loss
  encourages the representation to distinguish c-conditioned distributions
  without c ever entering the forward pass).

The FM's forward pass sees only "measured" objects: events, unbinned data,
histograms, or whatever observable representation the architecture uses.

"Iterative refinement" means the **oracle** grows richer over time — more
WCs activated, more diagrams, finer detail, higher fidelity. The FM input
interface does NOT change as fidelity grows. What changes is the
distribution from which training samples are drawn. The learned
representation is the model; probing it is the science.

If a proposal puts c into a feature map, conditions a network on c, or fits
a per-bin polynomial in c — it is out of scope. Flag it and replace it.

---

## Tension with the existing code and verification plan

The current `modules/surrogate/features.py` defines

```
phi_c(c) = {1, c_a, c_a c_b}           # the SMEFT morphing basis
phi_joint(c, m) = phi_c(c) ⊗ phi_x(m)   # the FM feature map
```

and `IntentionFM.predict(C, M)` consumes both. This is *exactly* the
forbidden pattern — c is a direct input. The closed-form
`w = A^{-1} Phi^T Y` is also fit against `Y = σ(c, m)` labels, which encodes
the morphing ansatz into the loss. The existing
`aletheia_verification_plan.md` (root of repo) endorses this design. That
plan is therefore in tension with the constraint and needs to be revised, not
extended. Research output should make the relocation of c outside the
forward pass explicit.

---

## Code map (repo root: /home/vince/ALETHIA)

Oracle:
- `modules/analytic_smeft/smeft.py` — closed-form LO Drell-Yan, 14 dim-6
  Warsaw operators, exact `σ(c, m)` decomposed into sm_only / interference /
  bsm_squared. PDF-independent for the `μ = σ_BSM / σ_SM` ratio.
- `modules/analytic_smeft/pdfs.py` — LHAPDF wrapper (CT18NNLO vendored at
  vendor/lhapdf/, auto-discovered).
- `modules/analytic_smeft/constants.py` — SM couplings.
- `modules/surrogate/oracle_smeft.py` — `AnalyticSMEFTOracle` adapter
  exposing pointwise μ(c, m) at fixed-point (c, m).
- `modules/surrogate/oracle_madgraph.py` — MadGraph subprocess wrapper.
- `modules/surrogate/ground_truth.py` — toy synthetic μ(c, m) for tests.

Foundation model (current, to be rebuilt under the constraint):
- `modules/surrogate/features.py` — `phi_c`, `phi_x`, `phi_joint`. Note
  `D_C=15`, `D_X=5`, `D_JOINT=75`. The whole point of this file is the
  forbidden pattern.
- `modules/surrogate/model.py` — `IntentionFM` closed-form ridge:
  fit / predict / leverage / update / state_dict. ~100 lines.
- `modules/surrogate/calibration.py` — `ConformalCalibrator` (leverage-
  stratified split-conformal). Returns coverage-specific sigmas.
- `modules/surrogate/acquisition.py` — `random_acquire`, `leverage_acquire`,
  `epig_acquire`. EPIG closed-form derivation in the module docstring.
- `modules/surrogate/evaluation.py` — metrics.
- `docs/historical/smeft_surrogate-2026-05-25.tar.gz` contains an
  `incontext.py` variant worth comparing — extract to inspect if you need
  it. (Closed-form attention with `c` removed from the forward pass;
  `psi(m)` hand-engineered. See [02-foundation-model/summary.md](02-foundation-model/summary.md) §3.)

Agent runtime / Phoenix:
- `agent/main.py`, `agent/instrumentation.py` — ADK agent + OpenInference
  tracing.
- `agent/shopping_demo/` — placeholder demo from the starter template;
  to be replaced by the physics agent.
- `docker-compose.yml` — Phoenix at http://localhost:6006. `make phoenix-up`.

Docs already present:
- `docs/surrogate/{README,ARCHITECTURE,INTEGRATION,REFERENCE,TOOLS}.md`
- `docs/foundation-model-survey.md` (may be stale; check timestamps).
- `aletheia_verification_plan.md` — existing master plan (in tension with
  the constraint as noted above).

Tests:
- `tests/test_features.py`, `test_model.py`, `test_acquisition.py`,
  `test_calibration.py`, `test_oracle_smeft.py`, `test_oracle_madgraph.py`,
  `test_analytic_smeft.py`.

Scripts:
- `scripts/surrogate_demos/{demo_v3,demo_analytic_oracle,demo_madgraph_oracle,demo_epig,plotting}.py`
- `scripts/install_lhapdf.sh` builds LHAPDF into vendor/lhapdf/.

---

## Hackathon constraints (rubric)

Arize track requires: code-owned agent runtime (ADK qualifies), OpenInference
instrumentation, traces persisted to Phoenix (self-hosted allowed), Phoenix
MCP server configured, evaluations on traces. Drift detection and retraining
gating are explicitly rewarded.

## Dev environment

- Python 3.12 via uv. uv binary is at `/home/vince/snap/code/240/.local/bin/uv`
  (snap-confined VSCode home, NOT `~/.local/bin/uv`). Export PATH first:
  `export PATH="$HOME/snap/code/240/.local/bin:$PATH"`.
- `uv run` prints `VIRTUAL_ENV=/usr does not match …` — ignore it.
- Docker is installed but the `vince` user is not in the docker group and
  sudo requires a password, so Claude's Bash cannot drive
  `docker`/`docker compose`. The user runs `make phoenix-up` themselves.
  Claude can verify Phoenix over HTTP at `http://localhost:6006`.
- LHAPDF: vendored under `vendor/lhapdf/`. `pdfs.py` self-discovers it.

---

## Output expectations for the first pass

Each of the four research areas writes to its own subdirectory under
`docs/research/`:

- 01-oracle/    — agent (a)
- 02-foundation-model/ — agent (b)
- 03-drift/     — agent (c)
- 04-epig-conformal/ — agent (d)

First pass produces, per area:

- `summary.md`  — research summary (3000–6000 words, citations welcome,
  link to specific files in the repo when referring to code).
- `test-plan.md` — what tests would demonstrate the design works. Closed-
  form predictions where the math gives one; small-scale empirical checks
  otherwise. Tests should be runnable against the existing code where
  possible; where the code needs to change first, say so and sketch the
  change.
- `notable.md`  — 1 to 5 numbered points that should be investigated
  empirically for potential improvement, each with a hypothesis and an
  expected outcome.

Second pass (after all four first-pass outputs exist) produces, per area:

- `eval.md`     — how this area's choices interact with the other three;
  where they constrain each other; where the joint design could be tighter.
- `empirical-results.md` — results of running the second-pass empirical
  checks. Include exact commands run, what failed, what was debugged, what
  scaling extrapolation says about the true (10-20 hour) run.

Final synthesis pass writes:

- `synthesis/three-test-cases.md` — three concrete, executable test plans:
  (1) machinery smoke test (everything wires up), (2) theory check on a
  known problem (the algebra predicts X, we measure X), (3) full-chain
  10-20 hour run script.

## Style

Plain prose. Equations in LaTeX where they actually help. Reference papers by
arXiv id or DOI; do not invent citations. When you describe code behaviour,
either point at the file:line or run the file and quote the output.
