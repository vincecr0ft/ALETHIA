# ALETHIA full-chain demonstration run

End-to-end exercise of the four design-research areas — oracle, FM, drift,
EPIG / conformal — in one Phoenix-instrumented loop, driving the controlled
drift event from the synthesis test plan
([three-test-cases.md](../../docs/research/synthesis/three-test-cases.md)).

The scenario is fixed: an Intention foundation model with learned
`psi_theta` MLP basis is pretrained on the analytic SMEFT oracle with
`|c_lq^(3)| in [0.6, 1.0]` excluded. At inference we query the FM in that
withheld band (target `c_lq^(3) = 0.8`). The FM has a thin seed context
of 8 observations clustered in `m_ll in [0.5, 1.0]` TeV, so it
extrapolates badly in the tails. The drift detectors fire (DAS-CUSUM on
standardised residuals; BH-corrected per-region binomial coverage;
condition-number kappa(A) plus v_min projection ratio); the aggregator
chooses local_retrain; EPIG picks five m-values inside the high-leverage
band; the analytic oracle returns labels; the context grows; conformal
recalibrates; the loop continues until coverage on the band is restored
or the oracle budget (500 calls) is exhausted.

This is the realistic scaled equivalent of Test 3 from the synthesis
plan. It is NOT 10–20 hours of MadGraph compute: we use only the
analytic-PDF tier T0/T1 from agent (a)'s tier ladder. Everything else
that Test 3 specifies is exercised.

## Layout

```
intention_fm.py         Intention head with learned psi_theta MLP
drift_detectors.py      DAS-CUSUM, BH coverage, kappa-vmin, aggregator
conformal_intention.py  Leverage-stratified split conformal + fingerprint
epig_intention.py       Closed-form EPIG over Intention features
run.py                  Phoenix-instrumented orchestrator
plots.py                Four PNG diagnostics
output/                 trajectory.npz, summary.json, run.log, intention_fm.pt
```

## Configuration

| | |
|---|---|
| Oracle | `AnalyticSMEFTOracle(pdf="analytic")`, fidelity tier T1 |
| Target scenario | `c_lq^(3) = 0.8` (in withheld band [0.6, 1.0]) |
| Pretraining | 1500 steps, batch 24, K=12 context, Q=32 query |
| Seed context | K=8, m clustered in [0.5, 1.0] TeV |
| Calibration set | n_cal=400, m sampled over full [0.3, 2.3] TeV |
| Probes / cycle | 24 (50% tail-biased to m > 1.5 TeV) |
| Oracle budget | 500 calls |
| EPIG batch | k=5 per drift event |
| DAS-CUSUM | h=6.0, w=30, k=0.5 |
| BH | alpha=0.05, 5 m-regions |
| Coverage drift | kappa > 1e3 OR proj_ratio > 2.5 |
| Persistence | cal=1 for 4 consecutive cycles -> escalate to local_retrain |

## Reproducing

```bash
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
uv run python experiments/full-chain-run/run.py        # ~15 min CPU
uv run python experiments/full-chain-run/plots.py      # ~5 seconds
```

`output/run.log` shows progress; `output/summary.json` has the final
metrics; `docs/research/plots/full-chain/*.png` are the four headline
plots. Phoenix traces live in project `alethia` at
http://localhost:6006/projects.

## What this demonstrates

- The four design-research areas compose without modification beyond
  light wiring.
- Drift detection fires before the FM produces silently-wrong answers.
- EPIG picks oracle calls inside the drifted region.
- The closed-form Intention head recovers under context growth without
  any gradient-descent update at inference time — exactly the
  "foundation model in the in-context-learning sense" framing.
- Conformal coverage is restored after each EPIG-driven update,
  validating the refit-after-update invariant.
- Phoenix spans expose every cycle's decision and every contract on a
  filterable schema (`aletheia.*`).
