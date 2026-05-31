# Audit: drift detection, Phoenix integration, and the closed-loop story

This audit reads Section 6 (`sec:loop`) of `/home/vince/ALETHIA/paper/alethia.tex`
against the implementation in `/home/vince/ALETHIA/modules/surrogate/intention/`,
`/home/vince/ALETHIA/agent/`, and `/home/vince/ALETHIA/experiments/full-chain-run/`.
It is written for the Arize-track submission of the Google Cloud Rapid Agent
Hackathon (deadline 2026-06-11) and judges what is load-bearing versus
decorative in the drift + Phoenix + EPIG chain.

## 1. The real questions

### 1.1 What is Phoenix actually contributing to the closed loop?

Read in sequence:
`/home/vince/ALETHIA/agent/instrumentation.py:62-85`,
`/home/vince/ALETHIA/agent/alethia/tools/check_drift.py:23-94`,
`/home/vince/ALETHIA/agent/alethia/tools/recover_from_drift.py:49-110`,
`/home/vince/ALETHIA/experiments/full-chain-run/run.py:41-43,250-344`.

Phoenix is wired as a one-way OTLP exporter. `setup_tracing()` calls
`phoenix.otel.register(auto_instrument=True)` once at agent boot; every tool
opens a span via `trace.get_tracer("alethia.tools")` and sets `aletheia.*`
attributes; spans flow over OTLP/HTTP to the Docker container at
`localhost:6006` and persist in SQLite (`docker-compose.yml:21-25`). Inside
the closed loop, nothing reads from Phoenix. No `phoenix.client`, no
`px.Client`, no `SpanEvaluations`, no MCP call inside `run.py` or any tool.
The aggregator state lives in `STATE.history_flags`
(`agent/alethia/state.py:57`); the conformal counts live in
`STATE.coverage_counts_68`; the CUSUM buffer lives in `STATE.cusum_state`.
All of these are Python in-memory state. The full-chain run could be
re-pointed at a Jaeger backend, a JSON log file, or `/dev/null` and the
numbers in `experiments/full-chain-run/output/summary.json` would not move.

**This is the gap.** The paper at `paper/alethia.tex:415` claims that "the
rubric MCP queries against the resulting trace tree are answered as
single-span filters", and the design document at
`docs/research/03-drift/summary.md:96-470` carefully lays out a Phoenix
evaluator architecture (`accuracy_drift_das_cusum`,
`calibration_drift_conformal`, `coverage_drift_condition_number`) that
writes back `SpanEvaluations` and is consumed by the orchestrator via the
MCP `get-span-annotations` call. Neither the evaluators nor the read-back
is implemented in the runtime code. The `.gemini/settings.json` MCP wiring
only enables an *external* Gemini CLI session to introspect traces; the
agent itself never queries Phoenix.

This places the current submission at the lower end of the "perfunctory ...
load-bearing" range. The instrumentation is real and conformant; the
control plane is not. Sections 4 and 5 below propose how to fix this
within hackathon timelines.

### 1.2 Are the three drift detectors physics-aware?

DAS-CUSUM on standardised residuals (Ahad-Davenport-Xie 2022) and
BH-corrected binomial coverage tests (Benjamini-Hochberg 1995) are
domain-neutral SPC; they would work on the residual stream of a chatbot
quality regressor with no modification. They are physics-aware only in
the engineering-trivial sense that we feed them a residual computed from
SMEFT cross sections. The threshold *h = 6.0* in
`modules/surrogate/intention/drift.py:33` is calibrated for ARL_0 ~ 1000
under Gaussian residuals (notes at `docs/research/03-drift/summary.md:482`)
which is exactly what an SPC engineer would do for any regressor.

The condition number `kappa(A)` signal in
`modules/surrogate/intention/drift.py:97-124` is the one that touches the
geometry of the learned representation. It uses the Eckart-Young theorem
on the design matrix `A = Psi(M_ctx)^T Psi(M_ctx) + alpha I` in the
Intention head's feature space, and the v_min projection ratio is a
genuine question about the FM's basis. This is closer to *model-aware*
than to physics-aware: it knows the foundation model's internal
representation, not the Standard Model.

A genuinely **physics-aware** drift signal would tie into the SMEFT
operator structure of equation (2) of the paper. Candidates that lie
within the existing infrastructure:

- **Operator-coverage drift** at the linear-probe level. The identifiability
  probe (`paper/alethia.tex:213-238`, `experiments/full-chain-run/output/identifiability_summary.json`)
  fits a linear map from `w_implicit` to `c`. The trace of the residual
  covariance of that map, evaluated on a sliding window of the recent
  context, fires when the FM's representation has lost the ability to
  resolve one of the Warsaw operators. This is physics-aware because the
  failure mode is named in operator space.
- **Energy-growth saturation drift.** Under the dimension-six SMEFT
  morphing decomposition the BSM-squared term scales as `(m_ll/Lambda)^4`.
  A fit of the predicted `log mu(m)` to `4 log m + const` on the tail
  region returns a slope. Slope below 3.5 in the high-mass band signals
  that the FM is undershooting the quadratic, which is the dominant
  failure mode for an under-resolved `c_lq^(3)` direction. This is
  physics-aware because the failure mode is named in the dispersion
  scaling of the underlying QFT.
- **K-factor stability drift.** The MadGraph cross-check at
  `experiments/full-chain-run/output/mg_crosscheck_summary.json`
  produces a per-region K-factor (NLO/LO). Drift in that K-factor under
  the analytic-MG ladder signals that the analytic oracle is being
  exercised in a region where its leading-order assumption fails.

These are concrete and bolt cleanly onto the existing aggregator decision
table.

### 1.3 EPIG vs random was a tie. What multimodal drift event resolves it?

`paper/alethia.tex:311-327` and
`experiments/full-chain-run/output/acquisition_compare.json` document that
on the engineered `c_lq^(3) = 0.8` band, uniform random gets RMSE 1.56,
EPIG 2.65, leverage-greedy 2.66. The paper attributes this to a "flat
information geometry" in the broadband drift. The proposal:

**Bimodal drift experiment design.**

- **Wilson configuration.** `c_lq^(3) = 0.8` *and* `c_Hq^(3) = -0.5`,
  simultaneously withheld at training. This pairs an energy-growing
  four-fermion direction with a rate-shifting vertex direction.
  Under the morphing decomposition the BSM-squared piece is dominated
  by `c_lq^(3)` in the high-mass tail (`m_ll > 1.5 TeV`), and the
  interference piece is dominated by `c_Hq^(3)` in the low-mass region
  near the Z-peak (`m_ll in [0.3, 0.6] TeV`). The information geometry
  has two peaks: one localised in `m_ll` near 0.4 TeV (rate shift), one
  in the tail near 2 TeV (energy growth).
- **Withhold band.** Two disjoint bands in 4D c-space:
  `|c_lq^(3)| in [0.6, 1.0]` AND `|c_Hq^(3)| > 0.4`. Training samples
  the complement of the intersection.
- **Budget.** 500 oracle calls, k = 5 per acquisition, identical to the
  current chain.
- **Acceptance.** EPIG wins if its picks straddle both `m_ll` regions
  (concretely: at least 30% of EPIG picks land in `m_ll < 0.7` and at
  least 30% in `m_ll > 1.5`) and its final-state RMSE on the joint
  band is statistically lower than uniform random by more than the
  bootstrap CI overlap. If EPIG places all picks in one band, the
  hypothesis fails and the result is informative.

The point is that EPIG's closed-form information gain has a
target-distribution-weighted average over a target set `M_target`. On a
multimodal target distribution, the EPIG argmax should sit at the peak
where the residual leverage is highest, *and switch peaks* between
sequential picks as the Sherman-Morrison update of `A_inv` consumes
information at the first peak. Uniform random has no such mechanism.

Effort estimate: 1 day (modify `WITHHOLD_BAND` and the c-sampler in
`experiments/full-chain-run/run.py:48-49,98-107`; the chain itself is
unchanged). Confidence that EPIG separates from random on this design:
**medium-high**.

### 1.4 The eigen-direction proposal: Phoenix-traced eigendecomposition events

This is the load-bearing creative move. Current code computes the full
eigendecomposition of A inside `kappa_drift`
(`modules/surrogate/intention/drift.py:116`) every cycle, uses only
`eigs.max() / eigs.min()` and `vecs[:, 0]`, and discards everything else.
The full `(Lambda, U)` is information the chain throws away.

**Proposal.** On every drift firing emit a Phoenix span
`chain.drift.eigen` carrying `(Lambda, U)` as serialised attributes
(numpy `tolist()`; D = 16 so 16 + 256 floats per event, well within OTLP
limits). Three operational consequences:

1. **Eigen-redirected acquisition.** Replace the EPIG candidate pool
   filter from "uniform in `m_ll`" with "uniform in `m_ll`, weighted by
   how strongly each candidate's `psi(m)` projects onto the top-k
   under-resolved eigenvectors of A". Concretely, define
   `r(m) = sum_{j in S_low} (psi(m) . u_j)^2 / lambda_j`
   where `S_low` is the index set of the k smallest eigenvalues. Replace
   `M_pool = rng.uniform(m_lo, m_hi, 300)` in
   `agent/alethia/tools/recover_from_drift.py:58` with importance sampling
   from `r(m)`. The argmax of `r(m)` is the m-value whose feature vector
   has the largest projection on the worst-resolved direction. This makes
   drift events *re-project* what the chain considers important.
2. **MCP-queryable eigen history.** The MCP rubric query "show me all
   drift firings where the top eigenvalue of A_inv exceeded 10^4" maps
   to `get-spans` filtered on `aletheia.drift.cov.kappa > 1e4` returning
   the serialised `(Lambda, U)` of each, which Gemini reads through MCP
   and uses to *direct* the next acquisition. This is the Phoenix-as-
   control-plane usage that the README claims at lines 379-385 but the
   code does not exercise.
3. **Visualisation.** See Section 6 below.

Literature anchor. The active-subspace line of Constantine
(*Active Subspaces*, SIAM 2015) and the likelihood-informed-subspace line
of Cui-Martin-Marzouk (*Likelihood-informed dimension reduction for
nonlinear inverse problems*, Inverse Problems 30, 2014; arXiv:1403.4680)
both pre-compute a problem-specific subspace by spectral analysis of an
expected gradient/sensitivity matrix and concentrate the active-learning
budget inside that subspace. Constantine's matrix is
`C = E[(grad f)(grad f)^T]`; ours is `A = Psi^T Psi + alpha I` evaluated
on the current context. The relationship is direct: the right singular
vectors of Psi are the active directions of the closed-form ridge head
(Cui-Martin-Marzouk equation 2.3 in the LIS paper; see also Spantini et
al., *Optimal low-rank approximations of Bayesian linear inverse
problems*, SIAM J. Sci. Comput. 37, 2015). The recent "Active learning
for adaptive surrogate model improvement in high-dimensional problems"
(Struct. Multidisc. Optim. 2024, doi:10.1007/s00158-024-03816-9)
implements exactly this loop on a multivariate-Gaussian PDE surrogate
and reports a 30-50% reduction in oracle queries against uniform random
on a multi-peaked target distribution. The Intention head with closed-
form `A^{-1}` is the cleanest setting in which to do this in the
hackathon: no nested Monte Carlo, no gradient samples to estimate
Constantine's `C`, the active subspace *is* the right singular subspace
of Psi.

**Implementability in 1-3 days.** Yes. Concretely:

- Modify `kappa_drift` (`drift.py:97-124`) to return
  `(fired, kappa, proj_ratio, eigs, vecs)` rather than `(fired, kappa,
  proj_ratio)`.
- Add `eigen_redirected_pool(model, M_ctx, n_pool=300, top_k=4)`
  in `acquisition.py` that draws from `r(m)` by rejection sampling on
  a coarse 1000-point grid.
- In `recover_from_drift.py` and `run.py`, when the aggregator action is
  `local_retrain`, call `eigen_redirected_pool` instead of
  `rng.uniform(M_RANGE, 300)` and emit a child span
  `tool.epig.eigen_redirected` carrying `top_k`, `eigs[:top_k].tolist()`,
  and the count of pool points sampled.
- Wire one Phoenix MCP rubric query (see Section 5).

Effort estimate: 1 day for the code, 1 day for the rubric and span
schema, 1 day to re-run the chain and update three figures.

Confidence that this produces a measurable lift on the bimodal drift
experiment: **medium-high** for separation from uniform random, **medium**
for separation from leverage-greedy (because leverage-greedy already
concentrates on high-`lev` regions which correlate with low-`lambda`
directions; the eigen-redirected pool is the *target-aware* version of
that and should win on a multi-peaked target distribution where
leverage-greedy picks one peak only).

### 1.5 The Phoenix MCP angle

Phoenix MCP is configured in `.gemini/settings.json:2-15` but is used
only by an external Gemini CLI session, never by the agent. The hackathon
rubric requires Phoenix MCP to be *configured*, which the project satisfies,
but a competitive submission would have the agent use MCP to read its own
history and direct its next action. The MCP server publicly exposes
`list-projects`, `list-traces`, `get-spans` (with attribute filters),
`get-span-annotations`, `list-datasets`, `list-experiments-for-dataset`,
`get-experiment-by-id` (Phoenix release notes April 2025,
arize.com/docs/phoenix/integrations/phoenix-mcp-server). The rubric MCP
query in Section 1.4 is a single `get-spans` filter; a more ambitious
agent loop would call MCP from inside a tool. See Section 5.

## 2. Literature review (2022-2026)

Active learning, EPIG, and information-gain acquisition:

1. **Bickford Smith, Kirsch, Farquhar, Gal, Foster, Rainforth.** *Prediction-Oriented
   Bayesian Active Learning.* AISTATS 2023 (arXiv:2304.08151). The EPIG
   paper. Key claim: target-distribution-weighted information gain over
   predictions outperforms BALD on regression precisely because BALD
   chases parameter uncertainty in regions of no predictive relevance.

2. **Kirsch, Rainforth, Gal.** *Test Distribution-Aware Active Learning:
   A Principled Approach Against Distribution Shift and Outliers.*
   arXiv:2106.11719. Earlier development of the target-set conditioning
   used in EPIG.

3. **Foster, Ivanova, Malik, Rainforth.** *Deep Adaptive Design.*
   ICML 2021. The amortised Bayesian experimental design construction
   that EPIG inherits.

Active subspaces and likelihood-informed dimension reduction:

4. **Constantine.** *Active Subspaces: Emerging Ideas for Dimension
   Reduction in Parameter Studies.* SIAM, 2015. The foundational text;
   defines the active subspace as the leading eigenspace of
   `C = E[(grad f)(grad f)^T]`.

5. **Cui, Martin, Marzouk, Solonen, Spantini.** *Likelihood-informed
   dimension reduction for nonlinear inverse problems.* Inverse Problems
   30, 2014 (arXiv:1403.4680). The LIS construction; spectral analysis of
   the prior-to-posterior update operator.

6. **Spantini, Solonen, Cui, Martin, Tenorio, Marzouk.** *Optimal low-rank
   approximations of Bayesian linear inverse problems.* SIAM J. Sci.
   Comput. 37, 2015. Direct link from `Psi^T Psi` to the active
   subspace in the linear case.

7. **Cui, Tong, Zahm.** *A unified performance analysis of likelihood-
   informed subspace methods.* arXiv:2101.02417 (2021). Modern treatment.

8. **Constantine, del Rosario, Iaccarino et al.** *Application of Active
   Subspaces for Model Reduction and Identification of Design Space.*
   Springer (Lecture Notes), 2024. Recent applied result that motivates
   the eigen-redirected acquisition.

9. **Active learning for adaptive surrogate model improvement in
   high-dimensional problems.** Struct. Multidisc. Optim. 2024,
   doi:10.1007/s00158-024-03816-9. Implements a Constantine-style active
   subspace acquisition loop on a PDE surrogate; 30-50% oracle reduction
   on multi-peaked targets.

Conformal calibration and drift under shift:

10. **Gibbs, Candès.** *Adaptive Conformal Inference Under Distribution
    Shift.* NeurIPS 2021 (arXiv:2106.00170). Foundational online
    adaptation of conformal split-quantile.

11. **Gibbs, Candès.** *Conformal Inference for Online Prediction with
    Arbitrary Distribution Shifts.* JMLR 25, 2024 (arXiv:2208.08401).
    Strongly adaptive online update; achieves long-run nominal coverage
    irrespective of the shift.

12. **Bhatnagar et al.** *Improved Online Conformal Prediction via
    Strongly Adaptive Online Learning.* ICML 2023 (arXiv:2302.07869).

13. **Araz, Spannowsky.** *Another Fit Bites the Dust: Conformal
    Prediction as a Calibration Standard for ML in HEP.*
    arXiv:2512.17048 (2025). The physics-side case for conformal as the
    HEP calibration standard; cited in the paper bibliography.

Drift detection on streaming residuals:

14. **Ahad, Davenport, Xie.** *Data-Adaptive Symmetric CUSUM for
    Sequential Change Detection.* arXiv:2210.17353 (2022, Sequential
    Analysis 43, 2024). The detector used in our `das_cusum_update`.

15. **DriftLens (Greco et al.).** *Unsupervised Concept Drift Detection
    from Deep Learning Representations in Real-time.* arXiv:2406.17813
    (2024). Embedding-space drift; the right comparator if we ever
    extend kappa(A) to a representation-distance signal.

Observability and Phoenix:

16. **OpenInference semantic conventions.**
    github.com/Arize-ai/openinference. The convention that
    `phoenix.otel.register(auto_instrument=True)` satisfies.

17. **Arize Phoenix release notes, 04.18.2025.** *Tracing for MCP
    Client-Server Applications.* OpenInference instrumentation now
    propagates OTel context across MCP wire protocol; gives the agent
    distributed end-to-end traces when it calls MCP tools.

SMEFT high-mass DY context:

18. **Greljo, Marzocca.** *High-pT dilepton tails and flavor physics.*
    Eur. Phys. J. C 77, 2017 (arXiv:1704.09015). The dimension-six
    cross-section parametrisation we use.

19. **HEPfit Collaboration.** *Constraining new physics effective
    interactions via a global fit of EW, Drell-Yan, Higgs, top, and
    flavour observables.* JHEP 03 (2026) 013. Recent global SMEFT fit;
    establishes the physical relevance of the `c_lq^(3)` direction.

## 3. Findings, with file references

**Phoenix integration is one-way and surface-level.**
`/home/vince/ALETHIA/agent/instrumentation.py:79-84` registers the tracer
via `phoenix.otel.register(auto_instrument=True)`, and tools open child
spans (`agent/alethia/tools/check_drift.py:23,29,38,58`, equivalent in the
other five tools). Span attributes under the `aletheia.*` prefix carry
the drift detector outputs, kappa, EPIG picks, oracle call cost, and
context-size deltas. Nothing reads from Phoenix; no `SpanEvaluations` is
ever uploaded; the `phoenix.client` API is not imported anywhere in
`agent/` or `modules/`.

**The drift aggregator state is local and ephemeral.** The history of
flags lives in `agent/alethia/state.py:57` (`history_flags: list = field`)
and is cleared whenever `set_target_c` is called. There is no persistent
re-creation of the aggregator from Phoenix history. If the agent
restarts, the run resumes from a blank flag history.

**The agent never queries MCP.** `.gemini/settings.json:2-15` configures
`@arizeai/phoenix-mcp` for a *Gemini CLI* session, which is a human-in-
the-loop introspection path, not a runtime control path. There is no MCP
client used inside `agent/alethia/tools/` or `agent/main.py`.

**The closed-loop result is real and the chain *does* recover.** The
acquisition compare at
`/home/vince/ALETHIA/experiments/full-chain-run/output/acquisition_compare.json`
and the trajectory at
`/home/vince/ALETHIA/experiments/full-chain-run/output/summary.json`
confirm the headline RMSE recovery 293 -> 2.65 and the no-better-than-
random EPIG result. The chain is well-engineered and instrumented; the
issue is that the instrumentation is not consumed.

**The drift detectors are statistically conservative.** The BH-corrected
binomial test fires on >99% of post-recovery cycles
(`bh_coverage_pvalues.json: fraction_below_alpha = 0.99`) at a 3.4 pp
over-coverage. The aggregator then escalates via
`CAL_PERSISTENCE_ESCALATE = 4` (`run.py:80`), which is why
`n_local_retrain = 100` in `summary.json`. The chain spends most of its
oracle budget on "fix a 3.4 pp over-coverage" which is real statistical
miscalibration but not a regime where physics is failing. This is the
SPC-vs-physics-aware question of Section 1.2 surfacing in the numbers.

**Eigendecomposition is computed and discarded.**
`/home/vince/ALETHIA/modules/surrogate/intention/drift.py:116`
(`eigs, vecs = np.linalg.eigh(A)`) inside `kappa_drift` returns the full
spectrum but the function returns only `kappa = eigs.max() / eigs.min()`
and `proj_ratio` via `vecs[:, 0]`. The other 15 eigenvectors and
eigenvalues are local variables that go out of scope. This is the
single most actionable finding: the information is in the call site,
just not exported.

## 4. Proposed upgrades, with confidence and effort

### Upgrade A: Phoenix evaluators that write SpanEvaluations back
Implement `accuracy_drift_das_cusum`, `calibration_drift_conformal`,
`coverage_drift_condition_number` as Python functions that read a
sliding window of `chain.cycle` spans via `phoenix.Client().get_spans`,
compute the same three statistics, and write back as
`SpanEvaluations(dataframe=..., eval_name=...)` via
`px.Client().log_evaluations()`. The orchestrator then reads the
annotations via `get-span-annotations` before deciding. This is the
design at `docs/research/03-drift/summary.md:456-523` but never built.

- **Confidence**: high.
- **Effort**: 2-3 days. Phoenix's Python client API and
  `SpanEvaluations` are stable and documented
  (arize-phoenix.readthedocs.io, GitHub Arize-ai/phoenix issue #2197 is
  resolved). The risk is the read-after-write race that
  `docs/research/03-drift/summary.md:519-523` flags; the recommended fix
  is a synchronous client call.

### Upgrade B: Replace random pool with eigen-redirected pool in EPIG acquisition
Section 1.4. Add a `top_k` argument to the aggregator's `local_retrain`
action; on a drift firing, sample the candidate pool from the importance
function `r(m) = sum_{j in S_low} (psi(m) . u_j)^2 / lambda_j` instead
of uniform.

- **Confidence**: medium-high.
- **Effort**: 1 day.

### Upgrade C: Bimodal drift experiment
Section 1.3. Two simultaneously-withheld c-directions producing
information peaks at low and high `m_ll`. Re-run the chain with three
acquisition strategies on the new band.

- **Confidence**: medium-high that EPIG separates from random.
- **Effort**: 1 day code + 1 day to run.

### Upgrade D: Physics-aware drift signal #4 (energy-growth slope)
Section 1.2. Add a fourth detector: on the high-mass tail of the recent
probe stream, fit `log mu(m) ~ alpha log m + const` and fire when
`alpha < 3.5`. Wire it as a fourth row in the aggregator decision table,
firing local_retrain on the high-mass band.

- **Confidence**: medium.
- **Effort**: half a day for the detector, half a day to extend the
  aggregator and add a span.

### Upgrade E: Agent-level MCP query inside `recover_from_drift`
Section 1.5. Before computing the new acquisition pool, the
`recover_from_drift` tool issues an MCP `get-spans` call asking "what
were the last 10 drift firings on this project, and what were their
top eigenvalues?". The tool uses that history to choose whether to do
local_retrain or escalate to global_retrain. Concretely, three drift
events in a row at progressively higher kappa is the signal to
*re-pretrain* the FM, not to keep growing the context. This adds the
MCP-as-control-plane story for the rubric.

- **Confidence**: medium. Engineering risk is small (Phoenix MCP is
  documented); design risk is that the agent over-reacts and triggers
  spurious global retrains.
- **Effort**: 2 days, including end-to-end test against the chain.

### Upgrade F: Replace fixed CUSUM threshold with online-adaptive conformal
Gibbs-Candès 2024 (arXiv:2208.08401). The current chain spends 100 of
500 oracle calls fixing a 3.4 pp steady-state over-coverage, which is
real but doesn't warrant retraining. Replace the "recalibrate then
local_retrain on persistence" pattern with online ACI: on each cal-
drift firing, adjust the per-stratum conformal multiplier by a small
step and reset the BH test window. This frees oracle budget for the
acc/cov detectors.

- **Confidence**: high. ACI is well-understood.
- **Effort**: 1 day in `modules/surrogate/intention/calibration.py`.

### Upgrade G: Eigenvalue trajectory plot
Section 6 below.

- **Confidence**: high.
- **Effort**: half a day.

## 5. The eigen-direction proposal: span schema, MCP rubric, algorithm, visualisation

### 5.1 Span structure

On every drift evaluation (every cycle), emit a child span
`chain.drift.eigen` under `chain.drift.evaluate` with attributes:

```
chain.drift.evaluate
+-- chain.drift.eigen
        aletheia.eigen.lambdas              list[float]   # length D = 16
        aletheia.eigen.U                    list[list[float]]   # 16 x 16 row-major
        aletheia.eigen.kappa                float
        aletheia.eigen.top_k_under_resolved list[int]     # indices of k smallest lambdas
        aletheia.eigen.context_size         int
        aletheia.eigen.target_signal        str           # which detector triggered
```

This sits naturally inside the span tree at
`docs/research/03-drift/summary.md:381-450`.

### 5.2 MCP rubric

A worked MCP rubric query suitable for the hackathon demo:

> **Rubric prompt.** Show me all drift firings in project `alethia` where
> the minimum eigenvalue of A was less than 1e-3, group by cycle range,
> and report the eigenvector of the minimum eigenvalue at each firing.
> Then, for the most recent such firing, what is the m-value `m*` such
> that `psi(m*) . u_min` is maximised?

Internal MCP calls:
1. `list-projects` -> select `alethia`.
2. `get-spans` with filter `aletheia.eigen.lambdas[0] < 1e-3`
   (the first index is the smallest under `np.linalg.eigh`).
3. For each returned span, parse `aletheia.eigen.U[0]` (the first
   eigenvector) and project against a stored `psi` evaluation grid.
4. Return the argmax m-value to the agent, which uses it as a single-
   point acquisition target outside the EPIG pool.

This is the "Phoenix-as-control-plane" usage the hackathon rubric
rewards. The whole MCP call sequence is read-only and synchronous, so it
fits inside one tool call from the agent.

### 5.3 Eigen-redirected acquisition algorithm (pseudocode)

```
def eigen_redirected_acquire(model, M_ctx, Y_ctx, M_target, k=5, top_k=4):
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    Psi_ctx = model.psi_np(M_ctx)
    A = Psi_ctx.T @ Psi_ctx + model.alpha * np.eye(model.d_psi)
    eigs, U = np.linalg.eigh(A)            # ascending
    S_low = np.arange(top_k)               # k smallest indices

    # Build importance function r(m) on a dense grid.
    M_grid = np.linspace(M_RANGE[0], M_RANGE[1], 1000)
    Psi_grid = model.psi_np(M_grid)        # (1000, D)
    projections = Psi_grid @ U[:, S_low]   # (1000, top_k)
    r = (projections ** 2 / np.maximum(eigs[S_low], 1e-12)).sum(axis=1)
    r = r / r.sum()

    # Sample candidate pool from r(m); then run EPIG on this pool.
    M_pool = np.random.choice(M_grid, size=300, p=r, replace=True)
    return epig_acquire_m(model, M_ctx, Y_ctx, M_pool, M_target, k=k)
```

This drop-in replaces lines 358-370 of
`experiments/full-chain-run/run.py` when `ACQUISITION == "eigen"`.

### 5.4 Visualisation: eigenvalue trajectory with drift overlays

Replace one of `chain_trajectory.png` /
`chain_drift_events.png` with a new panel:

- **Top panel.** Top-k smallest eigenvalues of A across the 400-cycle
  run, log y-axis, k = 4. Initially the smallest sits at
  ~`model.alpha = 1e-3` (the ridge floor), the next three sit at
  ~10^-2. As context grows the four climb toward the bulk; the
  smallest jumps in steps coincident with EPIG acquisition events
  (vertical orange marks).
- **Middle panel.** The cosine `|u_min(t) . u_min(0)|` between the
  current smallest-eigenvector and the seed smallest-eigenvector. This
  drops sharply at every drift event and tells the reader *which*
  direction got re-projected. Trajectory is non-monotone: re-projection
  events.
- **Bottom panel.** Aggregator action stripe (same as current
  `chain_drift_events.png`).

The story the figure tells: drift events are not just "fire and
acquire", they are *re-projection events* in feature space. The
smallest-eigenvalue direction *rotates* between local-retrain firings,
which is the signature of the active subspace being filled in.

Effort: half a day. The data already exists in `trajectory.npz` minus
the eigenvector cosine; that requires storing one extra vector per
cycle, ~10 KB total.

## 6. What this audit is asking for

If the goal is to get the submission across the Arize-track line as
"perfunctory but conformant", the current state suffices: Phoenix is
running, OpenInference is wired, traces persist, MCP is configured.

If the goal is to get the submission across the line as
*load-bearing instrumentation*, three things have to land in the next
two weeks before the 2026-06-11 deadline:

1. **Upgrade A** (Phoenix evaluators with read-back), 2-3 days. This
   alone moves Phoenix from logging to control plane.
2. **Upgrade B + G** (eigen-redirected acquisition + eigenvalue
   trajectory plot), 1.5 days. This delivers the creative direction
   that distinguishes the submission and produces one new figure for
   the paper.
3. **Upgrade C** (bimodal drift experiment), 2 days. This resolves the
   "EPIG is no better than random" weakness in Table 4 and makes the
   acquisition comparison informative.

Total effort: ~6 days for one developer. The eigen-direction proposal
in Section 5 is the highest-leverage and the most defensible piece
because it has a clean literature anchor (active subspaces /
likelihood-informed subspace, Constantine 2015, Cui-Martin-Marzouk
2014, Spantini 2015), and it produces a multi-modal lift that
generalises beyond SMEFT.

The audit's central claim is that the Intention head's closed-form
design matrix `A` is an active-subspace object in disguise, and the
chain is currently using only one scalar function of it (`kappa`).
Exporting the rest into Phoenix as queryable structured spans, and
using it to steer acquisition, turns Phoenix from instrumentation into
the substrate that closes the loop.

## Sources

- [Bickford Smith et al., Prediction-Oriented Bayesian Active Learning (EPIG)](https://arxiv.org/abs/2304.08151)
- [Constantine, Active Subspaces (SIAM 2015)](https://www.semanticscholar.org/paper/Active-Subspaces-Emerging-Ideas-for-Dimension-in-Constantine/a4f9650f651a7119a34de4bf0155da85b8cb530c)
- [Cui, Martin, Marzouk et al., Likelihood-informed dimension reduction for nonlinear inverse problems](https://arxiv.org/abs/1403.4680)
- [Spantini et al., Optimal low-rank approximations of Bayesian linear inverse problems](https://epubs.siam.org/doi/abs/10.1137/140977308)
- [Cui, Tong, Zahm, A unified performance analysis of LIS methods](https://arxiv.org/abs/2101.02417)
- [Active learning for adaptive surrogate model improvement in high-dimensional problems (2024)](https://link.springer.com/article/10.1007/s00158-024-03816-9)
- [Gibbs, Candes, Conformal Inference for Online Prediction with Arbitrary Distribution Shifts (JMLR 2024)](https://arxiv.org/abs/2208.08401)
- [Bhatnagar et al., Improved Online Conformal Prediction via Strongly Adaptive Online Learning](https://arxiv.org/pdf/2302.07869)
- [Ahad, Davenport, Xie, Data-Adaptive Symmetric CUSUM (2022)](https://arxiv.org/pdf/2007.16109)
- [DriftLens: Unsupervised Concept Drift Detection from DL Representations](https://arxiv.org/abs/2406.17813)
- [Arize Phoenix MCP server documentation](https://arize.com/docs/phoenix/integrations/phoenix-mcp-server)
- [Phoenix release notes 04.18.2025: tracing for MCP client-server applications](https://arize.com/docs/phoenix/release-notes/04.2025/04.18.2025-tracing-for-mcp-client-server-applications)
- [OpenInference Instrumentation (Arize-ai/openinference)](https://github.com/Arize-ai/openinference)
- [Phoenix Python client: log_evaluations / SpanEvaluations](https://docs.arize.com/phoenix/tracing/how-to-tracing/feedback-and-annotations/llm-evaluations)
- [HEPfit, Constraining new physics effective interactions, JHEP 03 (2026) 013](https://link.springer.com/article/10.1007/JHEP03(2026)013)
- [Greljo, Marzocca, High-pT dilepton tails and flavor physics (arXiv:1704.09015)](https://arxiv.org/abs/1704.09015)
- [Multimodal Information Gain in Bayesian Design of Experiments](https://arxiv.org/pdf/2108.07224)
- [Parallelized Acquisition for Active Learning using Monte Carlo Sampling](https://arxiv.org/pdf/2305.19267)
