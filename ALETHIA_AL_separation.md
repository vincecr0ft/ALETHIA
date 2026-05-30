# ALETHIA — active learning: structural null, diagnostic, and the separation experiment

Handoff for Claude Code. The 20-seed sweep returned a null: predictive EPIG, leverage,
parameter-space EPIG (D and A) all tied uniform random in c̃-space (identical 0.305 and 1.14
per-direction errors, identical resolved-subspace entropy). This document does three things:
proves the null is structural rather than a bug, builds the diagnostic that predicts it, and
runs the one separation experiment that can show active learning working if it can work at
all in this physics. None of it changes the head, the acquisition code, or the oracle. It
adds a candidate-pool construction, a closed-form diagnostic, and an MLE-scored run.

The contribution this produces is not "AL beats random." It is a closed-form diagnostic —
the EIG spread across the candidate pool — that predicts a priori whether targeted
acquisition can separate from random, validated in two regimes: the homogeneous pool where it
correctly predicts no separation (the existing null), and a heterogeneous physical pool where
it predicts and delivers separation. That is a general result with a predictive instrument,
and it is the Arize hook: the monitor can tell you in advance whether to spend a targeted
acquisition or fall back to random.

Read the mechanism in §0 before doing anything. It determines what every later stage measures.

---

## 0. Why the null is structural (the mechanism every stage depends on)

The head is Bayesian linear regression on ψ_θ with a Gaussian prior and homoscedastic noise.
The posterior precision is additive and label-independent:

```
A = Ψ_ctxᵀ Ψ_ctx + α I,        Σ_w = σ_y² A⁻¹,        Σ_c̃ = σ_y² P A⁻¹ Pᵀ,   P = Vᵀ W.
```

A grows by Σ_k φ_k φ_kᵀ over the chosen context features φ_k = ψ_θ(M_k). The observed labels
Y enter the posterior *mean* but not the covariance (Rasmussen & Williams 2006, eq. 2.8: the
predictive variance depends only on the inputs). Three consequences, each of which the sweep
exhibited:

1. EIG, D-optimality, A-optimality, and predictive-variance reduction are deterministic
   functions of the *design* (which M are chosen) and σ_y. They do not depend on Y.
2. Acquisition order is irrelevant: adding the same set of points in any order gives the same
   A, because matrix addition commutes. Greedy, batch, and sequential selection of the same
   set yield identical posteriors.
3. Therefore the only way a targeted rule can beat random is by selecting a *different and
   more informative set*. If every candidate in the pool has near-equal EIG, no rule can
   select a more informative set, and all rules tie. This is the null.

The original pool was 30 points drawn *inside the drifted region*. Points clustered in one
kinematic neighbourhood have near-identical leverage, so the pool was EIG-homogeneous and the
tie was forced. This is the single fact that the diagnostic in §2 measures and the experiment
in §3 breaks.

---

## 1. Stage A — Prove the null is structural (cheap, ships regardless)

This converts the reported null into a rigorous statement: the test could not have separated,
and here is the proof. It is required for the paper independent of whether §3 runs.

### A.1 Label-independence check
Take one drift context. Recompute Σ_c̃ under (i) the true labels, (ii) labels permuted across
context points, (iii) labels replaced by noise. Σ_c̃ must be identical across all three to
numerical precision. Emit the three covariances and their max elementwise difference.
**Expected:** difference ~1e-12. This is the numerical proof of label-independence; one
paragraph in the paper.

### A.2 EIG-spread on the original pool
On the original 30-point in-region pool, compute per-candidate leverage and predictive EIG:

```
ℓ_p = ψ_θ(M_p)ᵀ A⁻¹ ψ_θ(M_p),     IG(p) = ½ log(1 + ℓ_p).
```

Report the spread: CV = std/mean and the ratio (max − min)/median across the pool.
**Expected:** CV ≪ 1, ratio small — the quantitative signature of the homogeneous pool.
Emit `eig_spread_original.json` with the full per-candidate distribution, not just summary
stats.

### A.3 Order-irrelevance check
Run the existing greedy acquisition twice on the same pool with the pick order reversed.
Terminal Σ_c̃ and per-direction error must match. Confirms consequence (2) above.

### Keep/cut
Always produced. Output is the structural-null proof: label-independence (A.1), flat EIG
spread (A.2), order-irrelevance (A.3). This is the paper's honest framing of the existing
sweep and it stands whether or not §3 separates.

---

## 2. Stage B — The EIG-spread diagnostic and the go/no-go gate

The diagnostic is the contribution. It predicts, before any sweep, whether a pool can support
separation. Run it on candidate pools of increasing heterogeneity and find whether a
physically meaningful pool with large EIG spread exists.

### B.1 The diagnostic
For a candidate pool `P` and current design `A`, the separation that any targeted rule can
achieve over random is bounded by the EIG spread of `P`. Random draws k points uniformly:
expected information ≈ k · median(IG). A leverage/D-optimal rule draws the top-k: information
≈ k · mean of top-k IG. The achievable separation scales with

```
spread(P) = ( mean_top-k IG  −  median IG ) / median IG,
```

equivalently the coefficient of variation CV(IG) over the pool. The mechanism is exact for
the linear-Gaussian head because IG is design-only and additive. State this in the paper as
the closed-form predictor.

### B.2 Candidate pools to test
Build pools of increasing heterogeneity, all physically meaningful, all on the mass+angular
observable:
- **P0 (control):** the original in-region 30-point pool. Expect CV ≪ 1.
- **P1:** pool spanning the full observable, m_ll ∈ [0.3, 2.5] TeV × cos θ* ∈ [−1, 1], size
  ≥ 500, sampled uniformly in (log m_ll, cos θ*). Leverage varies strongly because the
  high-mass tail and the forward/backward angular extremes carry the SMEFT sensitivity while
  the low-mass central region carries little. Expect CV ~ O(1).
- **P2:** P1 with the budget made scarce — total acquisitions ≤ 5–10% of pool size. Scarcity
  is what makes a wasted random pick expensive; with a generous budget random covers the
  informative region by chance regardless of spread.

### B.3 The gate
Compute spread(P) for P0, P1, P2. **Decision:**
- If P1/P2 reach CV(IG) ≳ 1 (order-unity relative spread, informative candidates rare and
  much more informative than the median) → the diagnostic predicts separation is possible;
  proceed to §3.
- If no physically meaningful pool reaches CV(IG) ≳ 1 → STOP. Active learning genuinely
  cannot separate in this physics: the observable's information is spread too evenly across
  the kinematic axis for targeting to beat coverage. That is a real result. Report the
  diagnostic and the two-regime prediction (P0 flat, all physical pools flat) as the
  contribution, and do not run §3.

Emit `eig_spread_pools.json` with CV and the full IG distribution per pool.

### Keep/cut
The diagnostic is the contribution regardless of the gate outcome. If the gate passes, §3
validates the "separation predicted and delivered" arm. If it fails, the paper reports the
diagnostic plus a physics conclusion: the DY mass+angular observable does not support
active-learning separation, and the diagnostic says so without a sweep.

---

## 3. Stage C — The separation experiment (gated on §2 passing)

Run only if §2 found a pool with CV(IG) ≳ 1. Same head, same acquisition code, same oracle.
The only changes from the failed sweep are the pool (heterogeneous, from §2) and the scoring
estimator (§3.2).

### 3.1 Run matrix
- Pool: the highest-CV physical pool from §2 (P1 or P2).
- Budget: scarce (k total ≤ 5–10% of pool), the regime where a wasted pick is expensive.
- Acquisition modes: uniform random, leverage (D-optimal on A), predictive EPIG,
  parameter-space EPIG-D, parameter-space EPIG-a on the targeted direction.
- Seeds: ≥ 20. Note that seed scatter on the *targeted* arms reflects only noise draws, since
  the design is near-deterministic given the pool; scatter on the random arm reflects its
  selection variance. Report both.

### 3.2 Scoring — do not use the bias-floored probe
The linear probe is bias-dominated on the resolved directions (probe MSE 0.035 on c̃_1, of
which ~0.026 is bias²). Active acquisition reduces the *variance* component; if the metric is
bias-capped, both random and targeted hit the same floor and separation is invisible even
when contraction differs. Score separation in three quantities, in priority order:
1. **Posterior-variance contraction** on the targeted directions, Σ_aa^final / Σ_aa^seed.
   This is design-only and not bias-floored. It is the clean primary metric.
2. **MLE per-direction error.** Use the bounded multi-restart analytic MLE (variance-limited,
   approximately unbiased) as the inference estimator, not the probe. Contraction differences
   propagate to MLE error; they do not propagate to bias-floored probe error.
3. **μ-RMSE** for continuity with the prior sweeps, flagged as scale- and bias-contaminated.

### 3.3 Decision rule
- **Separation established** if, on the heterogeneous pool with demonstrated CV(IG) ≳ 1,
  leverage / D-optimal / parameter-EPIG beat random in posterior contraction (§3.2.1) beyond
  seed scatter, AND the contraction propagates to MLE error (§3.2.2). Report the effect size
  and CI, not the sign.
- **Real null** if the pool has demonstrated EIG spread (the test has power) yet targeted
  acquisition still ties random. This is a genuine and surprising result that needs one
  further diagnosis: check whether the high-EIG candidates are mutually redundant (their
  features are near-collinear), so random's coverage captures the informative subspace anyway.
  Compute the pairwise A⁻¹-inner-product among the top-decile-IG candidates; if they are
  near-collinear, the separation failure is a redundancy effect and a batch-aware selector
  (BatchBALD-style, k>1) is the correct rule, not greedy. Report whichever it is.

### 3.4 The near-tautology to avoid claiming
D-optimal beating random in *contraction* on a heterogeneous pool is close to definitional —
D-optimal maximizes contraction. Do not headline that. The non-trivial claims are (a) the
contraction propagates to inference error under an unbiased estimator (§3.2.2), and (b) the
EIG-spread diagnostic predicted the regime in advance (§2). Frame the contribution as the
predictive diagnostic validated in both regimes, with §3 as the positive-regime validation.

### Keep/cut
If §3.3 establishes separation, the paper carries the diagnostic plus both-regime validation,
and the Arize framing gets its strongest form: monitoring-driven acquisition that pays off,
with a closed-form signal that says in advance when it will. If §3.3 returns a real null on a
high-spread pool, the redundancy diagnosis decides between "needs batch acquisition" and "the
observable's informative directions are captured by coverage" — both publishable.

---

## 4. Pitfalls (read before coding)

1. **Run the §2 gate before the §3 sweep.** The original sweep was null by construction
   because the pool's EIG spread was never measured before committing 100 chains. The gate is
   a sub-minute computation. Do not skip it.
2. **The pool must span the observable.** A pool drawn inside the drifted region reproduces
   the homogeneous null no matter how many seeds. Heterogeneity comes from spanning low-info
   (low-mass central) and high-info (high-mass tail, angular extremes) regions.
3. **Do not score separation with the bias-floored probe.** Use posterior contraction and the
   MLE. The probe's bias floor masks contraction differences (§3.2).
4. **Seed scatter is asymmetric.** Targeted arms are near-deterministic given the pool
   (label-independence); their seed scatter is only the noise draw. The random arm carries
   selection variance. Report the two separately; do not pool them into one error bar.
5. **Redundancy vs. flat spread are different failures.** Flat EIG spread (§2 gate fails)
   means no informative structure to find. High spread but redundant top candidates (§3.3
   real-null branch) means the informative structure exists but coverage captures it; that
   calls for batch acquisition, a different fix. Diagnose which before concluding.
6. **No architecture change.** This is a polishing-stage task. If a step seems to require
   changing the head, the acquisition solver, or the oracle, stop — it does not. The changes
   are the pool, the diagnostic, and the MLE-scored run.

---

## 5. Decision table

| Stage | Action | Computation | Outcome |
|---|---|---|---|
| A structural null | Prove | label-independence, EIG-spread on original pool, order-irrelevance | always; the rigorous framing of the existing sweep |
| B diagnostic + gate | Measure | CV(IG) across P0/P1/P2 | CV ≳ 1 on a physical pool → §3; else STOP, report diagnostic + physics null |
| C separation run | Investigate | heterogeneous pool, scarce budget, contraction + MLE scoring, ≥20 seeds | targeted beats random in contraction and MLE error → separation; high-spread tie → redundancy diagnosis |

## 6. Execution order
1. Stage A on the existing config (cheap; produces the structural-null proof).
2. Stage B: build P1/P2, compute CV(IG), apply the gate.
3. If the gate passes, Stage C on the highest-CV pool, scored by contraction + MLE.
4. If the gate fails, stop at the diagnostic; the paper reports it as the contribution and the
   physics null as the conclusion.

The diagnostic in §2 is the deliverable that makes this worth doing even if §3 never separates:
it predicts the regime, it explains the original null mechanically, and it is computable online
from the design matrix, so the monitor can surface "EIG spread low → targeted acquisition will
not help here" as a first-class signal. That is the Arize contribution, and it does not depend
on §3's outcome.
