# Orchestrator-facing tool signatures

The Gemini agent sees the surrogate as seven deterministic tools. None of
them call an LLM; all of them call into the `smeft_surrogate` package and
return structured JSON.

These are the suggested tool descriptions. Tune the wording to match the
agent framework's tool-binding convention.

## predict

    name: predict
    description: |
      Query the SMEFT surrogate. Returns calibrated predictive mean and
      standard deviation at each (c, m) input point. Use this for any
      cross-section evaluation the agent needs in production.
    input:
      c: list[list[float]]    # shape (n, N_WC=4); Wilson coefficients
      m: list[float]          # shape (n,);        m_ll in TeV
      coverage: float = 0.683 # 1σ-equivalent by default
    output:
      mu:        list[float]  # predictive mean of mu(c, m)
      sigma:     list[float]  # conformal-calibrated std at target coverage
      leverage:  list[float]  # per-point drift signal (epistemic only)

## get_drift

    name: get_drift
    description: |
      Get the drift signal (leverage) on a set of points without computing
      predictions. Cheaper than predict when the agent only wants to know
      where the model is uncertain.
    input:
      c: list[list[float]]
      m: list[float]
    output:
      leverage:       list[float]
      mean_leverage:  float
      max_leverage:   float
      n_over_threshold: int    # leverage > 10 by default

## acquire

    name: acquire
    description: |
      Decide which points to query the oracle on next, given a pool of
      candidate points. Three strategies:

        "random":   uniform sample, baseline
        "leverage": maximise information about model parameters globally
        "epig":     maximise expected predictive information gain about
                    the supplied target set; use when the agent has a
                    specific region of physics interest in mind

      For "epig", supply target c and m arrays. Without a target, EPIG
      falls back to leverage acquisition.
    input:
      strategy: "random" | "leverage" | "epig"
      pool_c:   list[list[float]]   # candidate pool
      pool_m:   list[float]
      k:        int                  # number of picks
      target_c: list[list[float]] | null   # only for "epig"
      target_m: list[float]         | null
    output:
      indices: list[int]            # picks into the pool
      diagnostics:
        mean_picked_leverage: float
        mean_picked_c0:       float
        mean_picked_m:        float

## query_oracle

    name: query_oracle
    description: |
      Run the analytic SMEFT oracle on the supplied points and return
      observations. This is the expensive step; minimise calls.
    input:
      c: list[list[float]]
      m: list[float]
      noise: bool = true          # set false for evaluation queries
    output:
      y:           list[float]
      duration_ms: float

## update_model

    name: update_model
    description: |
      Fold new observations into the surrogate. Returns the new total
      training set size and the re-estimated noise fraction. Cheap; the
      fit is closed-form.
    input:
      c: list[list[float]]
      m: list[float]
      y: list[float]
    output:
      n_total:          int
      noise_frac_new:   float

## refit_calibration

    name: refit_calibration
    description: |
      Refit the leverage-stratified conformal layer on a calibration set.
      Call this after any update_model or when a calibration evaluator
      flags drift in stratum coverage. Calibration set should cover the
      broader probe region the model is queried on, not just the training
      box.
    input:
      c: list[list[float]]
      m: list[float]
      y: list[float]
      coverages: list[float] = [0.683, 0.954]
    output:
      factors_0_683: list[float]    # per-stratum multipliers, length n_strata
      factors_0_954: list[float]

## ab_compare

    name: ab_compare
    description: |
      Run a Phoenix experiment comparing two model versions on a fixed
      eval set. Returns the per-version metrics and a decision string
      ("promote_b", "keep_a", "ambiguous"). Use after update_model to
      decide whether to promote the new version.
    input:
      model_a_version: str          # version IDs from the model store
      model_b_version: str
      eval_c: list[list[float]]
      eval_m: list[float]
      eval_y: list[float]
    output:
      a_metrics:
        median_rel_err:   float
        stratum_coverage: list[float]
      b_metrics:
        median_rel_err:   float
        stratum_coverage: list[float]
      decision: "promote_b" | "keep_a" | "ambiguous"
