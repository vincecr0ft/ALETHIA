# smeft_surrogate

> **Note (2026-05-26).** This directory documents the current
> `modules/surrogate/` implementation, which puts Wilson coefficients on
> the forward pass and is therefore a regressor, not a foundation model
> under the project's hard constraint. The rebuild plan is at
> [docs/research/02-foundation-model/](../research/02-foundation-model/summary.md);
> the closed-form ridge described below survives as the *probe* on a
> learned embedding, not as the model itself. This README remains
> accurate as a description of the running code until the rebuild PR
> lands.

Intention foundation model for the SMEFT modification factor `mu(c, m)`, with
leverage-stratified conformal calibration and three acquisition strategies
(random, leverage, EPIG).

The model is a closed-form Bayesian linear regression on a tensor-product
feature map: polynomial in the Wilson coefficients (exact SMEFT structure at
dimension 6) times polynomial in `log(m / M_ref)`. With `N_WC = 4` and
`K_X = 5`, the joint dimension is 75 and everything is sub-millisecond on a
laptop.

The intended use is as a fast surrogate sitting behind a Phoenix-monitored
orchestration agent (Gemini). The orchestrator reads a drift signal
(leverage on production traces), decides when to acquire new oracle points,
calls the analytic SMEFT oracle, folds the new observations in, refits the
conformal layer, and Phoenix-A/B-tests the new model version. See
`docs/INTEGRATION.md` for the wiring brief.

## Install

    pip install -e ".[plot,test]"

## Run the demos

    python scripts/demo_v3.py       # main demo: conformal + active vs random
    python scripts/demo_epig.py     # EPIG vs leverage with focused target

Each writes a 4-panel PNG to `/mnt/user-data/outputs/`. Headline numbers from
the included `DummyAnalyticOracle`:

    demo_v3:    active acquisition reduces hard-region mean leverage ×24
                vs random for the same oracle budget. Conformal lifts
                per-stratum 1σ coverage from raw 0.47–0.65 into 0.66–0.78
                (target 0.683) without inflating the central interval.

    demo_epig:  EPIG with a focused target (a tight cloud in Wilson-mass
                space) picks zero overlap with leverage acquisition. Mean
                pick coordinates are pulled toward the target. Target
                predictive variance drops 1.3–4.9× faster than leverage
                acq on the same oracle budget.

## Tests

    pytest -x -q

23 tests, ~0.5 s. Covers feature-map structure, fit/predict/update consistency,
conformal marginal and stratum coverage, acquisition uniqueness, EPIG Cauchy-
Schwarz bound, and the focused-target steering property.

## Documentation

**Read `docs/REFERENCE.md` first** if you've worked with earlier prototypes
(`intention_fm_demo.py` binned v1, `intention_fm_unbinned.py` v2) and need to
align concepts. It's the technical reference: data shapes throughout the
loop, full API contract, end-to-end worked example, mapping table from the
old binned API to this one, common pitfalls, glossary.

`docs/ARCHITECTURE.md` for the math (closed-form posterior, the SMEFT and
log-m polynomial bases, stratified split conformal, the EPIG derivation).

`docs/INTEGRATION.md` for the wiring brief: what's already in the package,
what wiring still has to happen (real oracle, OpenInference instrumentation,
Phoenix evaluators, Gemini orchestration, MCP, Cloud Run), and in what
order.

`docs/TOOLS.md` for the orchestrator-facing tool signatures Gemini will see.
