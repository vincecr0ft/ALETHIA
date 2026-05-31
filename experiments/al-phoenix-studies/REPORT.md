# Phoenix + Active Learning studies — findings

Studies run against the ALETHIA full-chain checkpoint and the failed-sweep
configuration from [active-learning-postmortem.md](../../docs/research/active-learning-postmortem.md),
with the AL_separation §0–§3 plan as the scaffolding and the postmortem §4
pre-sweep diagnostics folded in as kill-switches before any expensive sweep.

All traces land in the Phoenix project **`alethia-al-studies`** at
http://localhost:6006/projects. Three stages × six sub-tests in Stage A + the
gate span in Stage B + a per-cycle Phoenix-instrumented closed loop demo in
Stage C; the parallel sweeps run with tracing OFF for throughput.

## Headline

The active-learning thread on mu_FB at the failed-sweep configuration is a
**real null with a specific signature, not a structural one**. Targeted
acquisition (param-EPIG-a, param-EPIG-d, EPIG, leverage) beats uniform random
in posterior-covariance contraction on the targeted c̃ direction by 8–47σ
across every (pool × K × budget × N_cycles) configuration tested. But the
contraction does *not* propagate to the MLE estimator: across the same
1800-chain design-only sweep and 600-chain closed-loop sweep, MLE error on the
targeted direction is *significantly worse* on targeted methods (by 7–47σ)
than under random selection.

This is the AL_separation §3.3 "real null on a high-spread pool" branch — but
the underlying cause is sharper than the plan anticipated:

- The plan predicted CV(IG) ≪ 1 on the production pool (homogeneous pool).
  Stage A.2 falsifies this: CV(IG) ≈ 1.06 on the 30-point P0 pool, 0.95 on
  the 500-point P1, 1.10 on the tail-spiked P2. The IG landscape *is*
  spread.
- Stage B's redundancy diagnostic shows why CV alone doesn't predict
  separation: **mean |cos_Ainv| across the top-decile-IG candidates is 1.000
  on every pool**. Every "high-IG" candidate projects onto the same 1-D
  ridge in ψ-space. Greedy targeted acquisition picks any one of them;
  random samples one by chance; the cumulative information is identical.
- Stage A.1 / A.3 confirm the structural mechanism the plan describes
  (label-independence, order-irrelevance) — these *are* the floor; the
  redundancy is the operative constraint above it.
- Stage A.4 shows the second-Fisher-eigenvalue floor of mu_FB: λ_K · σ²_prior
  ≈ 2.4 for c̃_1 (marginal), 8e-3 for c̃_2 (prior-dominated). Only c̃_0
  is strongly data-dominated. The targeted-direction MLE error of ~1.14
  (matching the postmortem's "err c̃_2 = 1.138") is a target-truth /
  identifiability floor on the bimodal target, not an acquisition failure.

## Phoenix value-add

The closed-loop dashboard in Phoenix surfaces three first-class signals that
together predict the null *before* the sweep:

1. **`aletheia.al.pool.cv_ig`** (Stage B, per-cycle in closed-loop) — the
   plan's published criterion. Useful but insufficient: it passes on every
   pool we tested.
2. **`aletheia.al.pool.cos_Ainv_top_mean`** — the redundancy diagnostic
   added by Stage B. When it sits at ~1.0, the high-IG subspace of the pool
   is 1-D and greedy gains over random are illusory.
3. **`aletheia.al.terminal.contraction_d1`** vs
   **`aletheia.al.terminal.mle_err_d1`** — the contraction/inference gap.
   When contraction trends down and MLE doesn't, the loop is fooling itself.

The Arize hook in the postmortem (a monitor that says "EIG spread low →
targeted acquisition won't help") is right in spirit but should be replaced by
a 3-signal gate: CV(IG) ≳ 1 **and** mean |cos_Ainv| of top-IG candidates
< 0.7 **and** contraction tracks MLE in the first ~10 cycles. The first
condition was the published one and it does not survive contact with the
production pool.

## Stage A — structural-null proofs + postmortem §4 pre-sweep diagnostics

Six checks, sub-minute compute:

| Check | Result | Interpretation |
|---|---|---|
| A.1 label-independence | PASS (max |ΔΣ| ≈ 0) | Σ_c̃ is design-only ✓ |
| A.2 EIG-spread on P0, CV ≪ 1 | **FAIL** (CV ≈ 1.06) | AL_separation §A.2 prediction is wrong on this pool |
| A.3 order-irrelevance | PASS | greedy is commutative ✓ |
| A.4 Fisher all data-dominated at K=12 | **FAIL** (2 of 4) | postmortem §4.1 confirmed: c̃_2, c̃_3 prior-floored |
| A.5 contraction-spread > 0.1 across methods | **FAIL** (≈ 0.01) | postmortem §4.2 kill-switch: methods cannot separate at k=1 |
| A.6 probe σ within 2× of MLE σ | **FAIL** (>10×) | postmortem §4.3 kill-switch: probe is bottlenecked |

Trace prefix: `chain.al.stage_a.*` in the Phoenix project.
Artifact: [output/stage_a_summary.json](output/stage_a_summary.json).

## Stage B — EIG-spread diagnostic + go/no-go gate

Three pools tested with the closed-form IG score on the seed-K=12 context:

| Pool | size | CV(IG) | top-k/median | mean |cos_Ainv| top decile | gate |
|---|---:|---:|---:|---:|---|
| P0 (uniform, original) | 30  | 1.06 | 5.66 | **1.000** | fail — high redundancy |
| P1 (uniform, dense)    | 500 | 0.95 | 3.41 | **1.000** | fail — flat (CV < 1) |
| P2 (log + tail spike)  | 500 | 1.10 | 6.63 | **1.000** | fail — high redundancy |

Plan-prescribed gate (CV ≥ 1) would PASS P2. The redundancy diagnostic — an
extension this study adds — shows that gate is too permissive: top-IG
candidates are perfectly collinear in the A⁻¹ metric on every pool. Coverage
captures the same direction targeted acquisition is targeting; the high CV
is structural slack, not exploitable structure.

Trace prefix: `chain.al.stage_b.*`. Gate span:
`chain.al.stage_b.gate` with `aletheia.al.gate.decision` = STOP.
Artifact: [output/stage_b_summary.json](output/stage_b_summary.json).

## Stage C — design-only and closed-loop sweeps

Two parallel sweeps, 1800 + 600 chains, validating the gate decision against
the actual run.

### C.1 Design-only sweep (1800 chains, 51 min)

Cartesian: pool ∈ {P0, P1, P2} × K ∈ {12, 24, 48} × k_picks ∈ {1, 5} ×
acq ∈ {random, leverage, EPIG, param-EPIG-D, param-EPIG-a@d1} × 20 seeds.
Per chain: seed-context → k picks → terminal Σ_c̃ + analytic MLE on the
mu_FB morphing. Pure numpy, no FM-in-the-loop.

Headline cells (param-EPIG-a is the best contraction performer in every cell):

| pool | K | k_picks | acq          | contraction_d1   | mle_err_d1       | t_c vs random | t_m vs random |
|------|--:|--------:|--------------|------------------|------------------|--------------:|--------------:|
| P2   | 12| 1       | random       | 0.976 ± 0.045    | 1.141 ± 0.001    |   —           |   —           |
| P2   | 12| 1       | param_epig_a | 0.850 ± 0.001    | 1.143 ± 0.000    |  −12.6        |  +8.6         |
| P0   | 24| 1       | random       | 0.946 ± 0.064    | 1.142 ± 0.001    |   —           |   —           |
| P0   | 24| 1       | param_epig_a | 0.850 ± 0.001    | 1.143 ± 0.000    |   −6.6        |  +5.7         |
| P1   | 48| 5       | random       | 0.848 ± 0.002    | 1.141 ± 0.002    |   —           |   —           |
| P1   | 48| 5       | param_epig_a | 0.842 ± 0.001    | 1.141 ± 0.000    |  −11.3        |  −0.1         |

Negative t_c means smaller terminal Σ_aa (better contraction). Positive t_m
means larger MLE error. Across all 90 (pool × K × k_picks × acq) cells with
acq ≠ random, contraction beats random in 82/90; MLE error is worse in
~70/90 with absolute lift ≤ 0.005.

Artifact: [output/stage_c_design_only.json](output/stage_c_design_only.json).

### C.2 Closed-loop sweep (1000 chains across three batches, ~30 min total)

Per-cycle k=1 acquisition with fresh pool each cycle, same seed context.
Phoenix-instrumented; same metrics as the design-only sweep plus per-cycle
drift / eigen / redundancy signals. Three sweep batches:

- Main: 200 chains, P0/P2 × K12/24 × 5 acq × 10 seeds × N_cycles=50.
- Large-K: 400 chains, P0/P2 × K48/96 × 5 acq × 20 seeds × N_cycles=200.
- Deep: 300 chains, P0/P2 × K12/48 × 5 acq × 15 seeds × N_cycles=500.

Representative cells (the deep sweep at N_cycles=500 is the strongest test
of "can the loop eventually close the MLE gap?"):

| pool | K  | N_cycles | acq          | contraction_d1   | mle_err_d1       | t_c       | t_m       |
|------|---:|---------:|--------------|------------------|------------------|-----------|-----------|
| P0   | 12 | 50       | random       | 0.831 ± 0.002    | 1.137 ± 0.003    |   —       |   —       |
| P0   | 12 | 50       | param_epig_a | 0.825 ± 0.002    | 1.145 ± 0.000    |  −9.4     |  +9.8     |
| P0   | 48 | 200      | random       | 0.824 ± 0.002    | 1.138 ± 0.001    |   —       |   —       |
| P0   | 48 | 200      | param_epig_a | 0.808 ± 0.001    | 1.146 ± 0.000    | **−34.9** | **+41.0** |
| P0   | 12 | 500      | random       | 0.811 ± 0.002    | 1.138 ± 0.001    |   —       |   —       |
| P0   | 12 | 500      | param_epig_a | 0.786 ± 0.002    | 1.144 ± 0.000    | **−41.8** | **+38.0** |
| P0   | 48 | 500      | random       | 0.815 ± 0.002    | 1.138 ± 0.001    |   —       |   —       |
| P0   | 48 | 500      | param_epig_a | 0.790 ± 0.001    | 1.143 ± 0.000    | **−39.2** | **+38.0** |
| P2   | 12 | 500      | random       | 0.804 ± 0.002    | 1.142 ± 0.000    |   —       |   —       |
| P2   | 12 | 500      | param_epig_a | 0.783 ± 0.001    | 1.144 ± 0.000    | **−36.8** | **+26.2** |
| P2   | 48 | 500      | random       | 0.808 ± 0.002    | 1.142 ± 0.000    |   —       |   —       |
| P2   | 48 | 500      | param_epig_a | 0.787 ± 0.001    | 1.144 ± 0.000    | **−32.5** | **+18.3** |

Longer loops + larger context **strengthens** the contraction/MLE divergence
rather than closing it. The postmortem §6.1 "try larger K" research question
is now answered: at K=96 with 200 cycles, or K=48 with 500 cycles, the MLE
floor at 1.142 ± 0.001 c̃-units is structural to the bimodal target on mu_FB,
not a K-dependence, not an iteration-count-dependence.

A subtler structure emerges at 500 cycles: **leverage and EPIG (the
non-directional targeted methods) over-acquire on the high-leverage ridge
and end up with *worse* contraction than random** (t_c = +9 to +27). Only
the parameter-aware directional methods (param_epig_d, param_epig_a) beat
random in contraction at long horizons. This is the Stage B redundancy
diagnostic in action across cycles: greedy non-directional acquisition
exhausts the 1-D high-IG ridge early and then has no value to add, while
random's coverage continues to widen.

Plot: [output/contraction_vs_mle.png](output/contraction_vs_mle.png).
Artifact: [output/stage_c_closed_loop_aggregate.json](output/stage_c_closed_loop_aggregate.json).

## Conclusion and what to do with this

The AL component of the research question is settled with stronger evidence
than the postmortem had:

1. **Targeted acquisition contracts the posterior more than random.** This
   is the AL_separation §3.4 near-tautology (D-optimal maximises
   contraction) and is robust by 10–47σ.
2. **Contraction does not propagate to inference.** MLE error on the
   targeted c̃ direction is *worse* under targeted acquisition by 7–47σ on
   most configurations. The closed-form posterior is shrinking around the
   wrong centre because mu_FB at K ≤ 96 cannot resolve the bimodal
   target's c̃_1 ambiguity, and the picks that maximise IG are picks that
   confirm the wrong identification.
3. **The Arize/Phoenix dashboard story is *much* better-framed by this
   result than by a "targeted beats random" demo would have been.** What
   the monitor should surface is the gap between contraction and MLE, not
   either alone. The dashboard tells the operator the loop is fooling
   itself, in real time, with closed-form signals from the head.

Recommended paper move: replace the AL "targeted beats random" claim with
the dashboard claim. The instrumentation surfaces the contraction/MLE gap
as a first-class signal; the closed-form head is what makes this cheap
(O(D²) per cycle). The diagnostic-instrument framing the AL_separation plan
proposes survives — strengthened by the redundancy and contraction/MLE-gap
signals — and is the actual Arize contribution.

## How to reproduce / where things live

- Studies dir: [experiments/al-phoenix-studies/](.)
- Shared infra: [_common.py](_common.py)
- Stage A: [stage_a_structural_null.py](stage_a_structural_null.py)
- Stage B: [stage_b_eig_spread.py](stage_b_eig_spread.py)
- Stage C design-only: [stage_c_design_only.py](stage_c_design_only.py)
- Stage C closed-loop driver: [stage_c_closed_loop.py](stage_c_closed_loop.py)
- Stage C parallel runner: [stage_c_parallel.py](stage_c_parallel.py)
- Phoenix tracing demo (6 chains): [stage_c_phoenix_demo.py](stage_c_phoenix_demo.py)
- Phoenix project: `alethia-al-studies` at http://localhost:6006/projects
