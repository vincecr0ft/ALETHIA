# Integration brief

This document is for the agent that will wire `smeft_surrogate` into Phoenix,
Gemini, and a Cloud Run deployment for the hackathon submission. Read this
first, then `TOOLS.md` for the orchestrator-facing tool signatures.

## What this package gives you

A working surrogate. Everything below is implemented, tested, and
demonstrated end-to-end:

- `IntentionFM`: closed-form fit/predict/leverage/update, all sub-millisecond.
  State serialises to a plain dict via `state_dict()`.
- `ConformalCalibrator`: per-leverage-stratum coverage factors, fit on a
  held-out set. Also serialises.
- Three acquisition strategies: `random_acquire`, `leverage_acquire`,
  `epig_acquire`. Same return type (indices into the candidate pool).
- `DummyAnalyticOracle`: placeholder that has the same interface as the
  real oracle you'll plug in. Implements `__call__(c, m, noise=True)` and
  `truth(c, m)` (noiseless).
- Diagnostics: `empirical_coverage`, `decile_calibration`,
  `stratified_coverage`. These are the building blocks the Phoenix
  evaluators will call.
- Two demo scripts that exercise the full loop end-to-end.

## What you have to add, in order

### Step 1: Replace the oracle

`smeft_surrogate.DummyAnalyticOracle` is a deterministic polynomial. Swap
it for the analytic Drell-Yan SMEFT calculator. The required signature is

    class RealOracle:
        def __call__(self, c, m, *, noise=True) -> np.ndarray: ...
        def truth(self, c, m)                  -> np.ndarray: ...

with `c` of shape `(n, N_WC)` and `m` of shape `(n,)`. Return shape
`(n,)`. Nothing else in the package needs to change. Expect 1–2 days.

The real oracle determines the physics-of-record. Until it's in, the rest
of the loop is a regression demo; once it's in, it's a SMEFT emulator.

### Step 2: OpenInference instrumentation

Wrap each public surrogate method with span creation. Use the
`openinference-semantic-conventions` Python package for attribute keys;
where there's no standard key, use the suggested ones below. The
orchestrator's reasoning quality depends on Phoenix being able to surface
these as filterable trace attributes.

Suggested span schemas:

    span: surrogate.predict
      input.c: list[list[float]]               # (n, N_WC)
      input.m: list[float]                     # (n,)
      output.mu_mean: float
      output.mu_max:  float
      output.sigma_mean: float
      output.sigma_max:  float
      leverage.mean: float
      leverage.max:  float
      leverage.over_threshold_fraction: float  # fraction with lev > 10

    span: surrogate.acquire
      strategy: "random" | "leverage" | "epig"
      pool.size: int
      k: int
      target.size: int                         # only for epig
      picked.mean_leverage: float
      picked.mean_c0: float
      picked.mean_m:  float

    span: surrogate.update
      n_new: int
      n_total: int
      noise_frac.before: float
      noise_frac.after:  float

    span: calibration.fit
      n_cal: int
      n_strata: int
      factors.0_683: list[float]
      factors.0_954: list[float]

    span: oracle.query
      n_points: int
      duration_ms: float
      backend: "analytic" | "madgraph" | "dummy"

The agent will filter on `leverage.mean` and `noise_frac.after` heavily;
make sure those are reliable. The picked.mean_c0/m attributes let the
Phoenix dashboard show where the acquisition campaigns are landing.

### Step 3: Phoenix evaluators

Three online evaluators run on the trace stream. Implement as
`phoenix.evals` functions that take a span batch and return a score.

**Drift evaluator.** Sliding window of recent `surrogate.predict` spans.
Compute the mean of `leverage.mean` weighted by `n_points`. Fire when
above threshold (suggested initial value: 5.0). The output should be a
single score per window plus a tagged region (which `c_0` band the
high-leverage queries clustered in, which `m` band). This is what
triggers the orchestrator to acquire.

**Calibration evaluator.** Retroactive trigger: when oracle truth becomes
available for a region that was previously queried via the surrogate
(i.e. the orchestrator has since run an acquisition campaign there),
compute the per-stratum empirical coverage via
`stratified_coverage(model, calibrator, C, M, Y_oracle)`. Fire when any
stratum is more than 0.05 from target. The orchestrator should respond
by refitting the conformal layer.

**Accuracy evaluator.** Same retroactive trigger, but computes
`median |rel err|`. This is the headline regression-quality number the
orchestrator's A/B comparisons will read.

The evaluators are not LLM-graded. They're deterministic numerical
checks, which makes them cheap and reliable. Reserve LLM grading for
the orchestrator's promote/rollback decisions, where the trade-off is
nuanced.

### Step 4: Gemini orchestration

One agent. Five tools (see `TOOLS.md` for full signatures).

The control loop:

    while True:
        drift = read_phoenix_evaluator("drift")
        if drift.score < THRESHOLD:
            sleep(POLL_INTERVAL)
            continue

        # Gemini reasons about which strategy fits the drift pattern.
        # Examples it should learn to make:
        #   broad drift across the probe region        → strategy="leverage"
        #   drift concentrated in a known fit region   → strategy="epig",
        #                                                target=that_region
        #   no clear pattern, exploration phase        → strategy="random"
        strategy, target = decide_strategy(drift)

        idx       = acquire(strategy, k=K_PER_ROUND, target=target)
        c, m      = pool[idx]
        y         = query_oracle(c, m)
        update_model(c, m, y)
        refit_calibration(c_cal, m_cal, y_cal)   # cached cal set

        cmp = ab_compare(model_old, model_new, eval_set=hard_region)
        if cmp.decision == "promote":
            promote_new_version()
        else:
            rollback()

Gemini should never touch the math. The strategy choice is a one-shot
decision per cycle, made on a Phoenix evaluator's structured output;
everything else runs in the deterministic tools.

### Step 5: Phoenix MCP

Standard `arize-phoenix-mcp` setup. Gemini reads trace history and
evaluator outputs through the MCP server; nothing custom. Config goes in
the agent's MCP server list.

### Step 6: Cloud Run deployment

Containerise. The surrogate is stateless except for `IntentionFM`'s
`(w, A_inv, noise_frac, C_train, M_train, Y_train)` and the calibrator's
`(edges, factors)`. Both `state_dict()` methods give you serialisable
dicts. Persist each model version to a bucket; the orchestrator
references models by version ID for A/B comparison.

Order of operations for the deploy:

1. Local Docker with self-hosted Phoenix and the Gemini API as a remote
   call. Run the full loop end-to-end against the analytic oracle. Verify
   trace history accumulates.
2. Switch Phoenix to Phoenix Cloud. Re-verify.
3. Cloud Run for the surrogate service. Phoenix Cloud for tracing.
   Gemini 3 Flash for orchestration during development; promote to Pro
   for the submission video.

## Order of work, calendar

The submission deadline (as of this writing) leaves enough room to do
this in roughly this cadence:

    Days 1-2:   Step 1 (real oracle).
    Days 3-4:   Step 2 (OpenInference) + Step 3 (evaluators).
    Days 5-7:   Step 4 (Gemini orchestration) + Step 5 (MCP).
    Days 8-9:   Step 6 (deployment, end-to-end test).
    Days 10-12: Generate trace history, screen-record demo video.
    Buffer:     2 days.

Each step is independently testable. Don't merge to the next until the
previous step has its own end-to-end test passing.

## Things that are deliberately out of scope here

- Multi-observable joint surrogate (`m_ll` AND `p_T`). One feature-map
  extension; see `ARCHITECTURE.md`.
- B-spline basis for `m_ll` if you need resonances or edges. Same.
- Heteroscedastic-aware EPIG. Five-line change.
- LLM-graded evaluators. Reserve for the promote/rollback decision only.
- Distributional shift detection beyond mean leverage. Sequential change
  detection is one option if you want a more principled trigger than a
  threshold.

The package is small on purpose. Anything not in it is your job to add,
and the package is structured so each addition is local.
