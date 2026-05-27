# Drift monitoring for a scientific foundation model (agent (c))

## 0. The shape of the argument

The Arize-track rubric rewards a Phoenix-instrumented agent runtime with online
evaluators driving a closed control loop. Phoenix's catalogue of drift detectors
(PSI, JS, KS, embedding-distance to a baseline, prediction-distribution drift)
was built for chatbots, recommender stacks, and tabular ML - settings where
"drift" means *users started asking different things*. For ALETHIA the right
notion of drift is *not* "the prompt distribution changed". It is three
physics-aware signals derived from the foundation model's own predictive
behaviour against an oracle and against its own design matrix:

1. **Accuracy drift** - the FM's pointwise predictions are deviating from
   oracle ground truth more than the conformal calibration assumed. Detected
   sequentially on standardised residuals via Data-Adaptive Symmetric CUSUM
   (DAS-CUSUM, arXiv:2210.17353).
2. **Calibration drift** - the conformal intervals no longer cover at their
   stated level on at least one stratum/region. Detected per-region with the
   Benjamini-Hochberg correction (BH, JRSS B 1995) so that multi-region
   testing does not produce nominal false positives. The physics-domain
   reference for conformal-as-calibration-standard is arXiv:2512.17048
   (Araz-Spannowsky); the online-adaptive extension we will pre-empt with
   is Gibbs-Candes "Conformal prediction with conditional guarantees"
   (JMLR 2024, arXiv:2305.12616).
3. **Coverage / leverage drift** - the queries the FM is now being asked are
   probing under-supported directions in its feature space. Detected by the
   condition number kappa(A) of the design matrix together with the worst-direction
   right singular vector v_min that identifies *where* the manifold is thin.
   Eckart-Young gives the bound, Sherman-Morrison gives the online update.

The job of this document is to (i) map these three signals into spans,
attributes, and evaluators that Phoenix natively understands; (ii) argue why
the proper-scoring-rule decomposition makes the *order* in which the three
fire informative; (iii) reconcile the leverage signal with the constraint
that the FM may not see Wilson coefficients in its forward pass;
(iv) define an aggregation policy mapping the three flags to an orchestrator
action; (v) spell out the Phoenix wiring (span tree, evaluator API, A/B
experiment policy) concretely enough to implement.

The leverage / accuracy / calibration triple was already gestured at in
`docs/surrogate/INTEGRATION.md:96-122`, but with three differences from what
we land on here. First, INTEGRATION.md proposes a threshold on
`mean(leverage)`; we replace that with kappa(A), which is invariant to data
scale and to which point in the pool you happen to be looking at. Second,
it suggests "fire when any stratum is >0.05 from target" for calibration;
we replace the bare threshold with a finite-sample binomial test corrected
by BH over strata. Third, accuracy is to be monitored as a one-shot
retroactive evaluator; we replace that with DAS-CUSUM on the streaming
residuals, because the question is *when did something change*, not *what
is the current error*.

## 1. What Arize / Phoenix actually provides

The codebase already calls `phoenix.otel.register(auto_instrument=True)` from
`agent/instrumentation.py:79-84`, which means every ADK turn and every
LangChain / OpenAI / Google-GenAI client call inside the agent is instrumented
without further code. The collector at `http://localhost:6006` is the
container in `docker-compose.yml:14-32` (Phoenix 16.0.0). MCP is wired in
`.gemini/settings.json:2-15` as the standard `@arizeai/phoenix-mcp` Node binary.

The primitives Phoenix exposes and the ones we will use:

- **Traces and spans.** A span has `name`, `span_kind` (one of LLM, CHAIN,
  RETRIEVER, TOOL, AGENT, EMBEDDING, RERANKER), arbitrary attributes
  (string / bool / number / array of), and a parent. Traces are graphs of
  spans sharing a root. Attribute keys follow OpenInference semantic
  conventions where possible (`openinference.semconv.trace`). Custom keys
  are allowed; we will namespace ours under `aletheia.*` to keep them
  filterable in the UI.
- **Span evaluations.** A `phoenix.trace.SpanEvaluations` object is a
  DataFrame of `(span_id -> score, label, explanation, metadata)` rows.
  It is uploaded with `px.Client().log_evaluations(SpanEvaluations(...))`.
  Each span can carry multiple named evaluations.
- **Datasets and experiments.** A dataset is a frozen collection of
  `(input, reference)` examples (`px.Client().upload_dataset`). An
  experiment runs a callable against the dataset and persists per-example
  outputs and per-example evaluator scores
  (`phoenix.experiments.run_experiment(dataset, task, evaluators=[...])`).
  Multiple experiments on the same dataset are comparable in the UI and
  via MCP `list-experiments-for-dataset` / `get-experiment-by-id`.
- **MCP server.** The publicly-documented MCP tools that matter for us:
  `list-projects`, `get-project`, `list-traces`, `get-trace`, `get-spans`
  (with attribute and time filters), `get-span-annotations`,
  `list-datasets`, `get-dataset`, `get-dataset-examples`,
  `get-dataset-experiments`, `list-experiments-for-dataset`,
  `get-experiment-by-id`. This is exactly the surface Gemini will use to
  read its own history.

Phoenix's *own* notion of drift is the "embeddings analysis" and "inference
set vs baseline" view, with Euclidean distance and PSI as the default
scores. That comes with a threshold convention (PSI > 0.25 investigate,
> 0.5 act). It is fine for monitoring an LLM's prompt distribution and
useless for monitoring a regression's calibration. The contribution of
this document is to keep Phoenix as the substrate (spans + evaluations +
experiments) while replacing the metric definitions with ones that are
correct for our problem.

## 2. Why generic LLM drift is the wrong frame

The generic "monitoring" pipeline assumes the model is a fixed mapping
*f: X -> Y* trained once. Drift is a distribution shift in *X* that might
or might not translate into a degradation in performance on *Y*. The
standard detector - PSI on a feature, or a KS test on an embedding
distance distribution - answers the question "are my inputs different
this week than last week?". For a chatbot whose retrained-or-not decision
is opaque and where ground truth is mostly unavailable, that is a
reasonable signal.

ALETHIA inverts this. We have an oracle (`modules/analytic_smeft/smeft.py`)
that returns exact ground truth at any query point in finite time. We
have a foundation model with a *closed-form posterior* - its predictive
mean and variance are not bootstrapped or asymptotic, they come out of
a single ridge solve (`modules/surrogate/model.py:38-65`). And we have a
control loop in which the FM is *meant* to be retrained: drift detection
is not a sad signal that the model has stopped working in production; it
is the gating function on the *next acquisition step*.

So the questions worth asking are sharper:

- "Has the FM's pointwise error grown beyond what its conformal
  calibration predicts?" - accuracy drift. Tests an algebraic
  relationship between residual variance and calibrated sigma.
- "Are the conformal intervals still covering at their stated level on
  every region?" - calibration drift. Tests an exchangeability
  assumption that finite-sample conformal prediction makes.
- "Is the FM being asked questions whose feature representation lies far
  from anything in its training set?" - coverage drift. Tests an
  observable geometric property of the design matrix.

None of these are PSI on prompts. All three have closed-form null
distributions or asymptotic guarantees we can lean on.

### 2.1 Accuracy drift via DAS-CUSUM

CUSUM (Page 1954, Lorden 1971) is the optimal sequential change-point
detector under known pre- and post-change distributions in the
Lorden-Pollak sense: minimum expected detection delay subject to a
constraint on the average run length to false alarm (ARL). It does not
need batched data - it operates on a running statistic
*S_t = max(0, S_{t-1} + g(x_t))*, with *g* the log-likelihood ratio
between post-change and pre-change. We do not know the post-change
distribution (we know only that something has shifted), and we do not
trust either the pre-change variance to be stationary across a stream of
physics queries that hits different *m* bins. DAS-CUSUM
(arXiv:2210.17353) solves both: pre- and post-change variance estimated
*from the stream itself* via two sliding windows; symmetric so a shift
of the mean in either direction triggers the same way. The paper proves
that the asymptotic ARL bound from Lorden-Pollak survives the
data-adaptive variance estimation under mild moment conditions.

The streaming input we feed it is the **standardised residual**

    z_t = (y_t^oracle - mu_hat_t^FM) / sigma_t^conformal

where mu_hat_t is the FM's predictive mean and sigma_t is the
conformal-calibrated sigma at the leverage stratum of query *t*
(`modules/surrogate/calibration.py:76-89`). If the FM and the conformal
layer are both correct, *z_t* is approximately *N(0, 1)* under
exchangeability, and the CUSUM is hugging zero. If the FM stops being
able to predict in a region (accuracy degrades while sigma stays put),
the mean of *z_t* drifts away from zero; DAS-CUSUM picks this up with
optimal asymptotic delay. If the *conformal sigma* is wrong (calibration
drift), the variance of *z_t* moves and a symmetric CUSUM still fires,
but the calibration evaluator will fire first (see section 2.4).

### 2.2 Calibration drift via conformal coverage with BH

The conformal calibrator (`modules/surrogate/calibration.py`) gives, per
leverage stratum, a multiplier *f_s* such that for an exchangeable test
draw the interval *[mu_hat - f_s sigma_raw, mu_hat + f_s sigma_raw]*
covers the target probability *p in {0.683, 0.954}*. Coverage drift
means: on at least one stratum the empirical coverage rate is no longer
at *p*.

The natural test per stratum is a two-sided binomial test of
*H_0: Pr[|y - mu_hat| < f_s sigma] = p*. With *S* strata you get
*S* p-values. Reporting "the worst stratum has *p_(1)* < 0.05" is a
multiple-testing failure: if every stratum is well-calibrated the
probability of *min p < 0.05* is roughly *1 - 0.95^S*, which for
*S = 5* is 23%. So we apply Benjamini-Hochberg at *alpha = 0.05* across
strata, which controls the false discovery rate at exactly *alpha* when
the strata are independent, and at *alpha m / sum_{k=1}^m 1/k* when they
are positively dependent (Benjamini-Yekutieli 2001). For our use case
the strata are leverage quantiles on disjoint slices of the data, so
positive dependence is the relevant regime; in practice the BH
conservatism we are giving up is at the percent level.

The Araz-Spannowsky physics paper (arXiv:2512.17048) makes the case for
conformal as the right calibration standard in HEP precisely because
"frequentist test-statistic" workflows have repeatedly produced
miscalibrated intervals on real measurements. The
BH-correction-over-regions extension is ours; it is the natural
multiple-testing layer once you agree to monitor coverage region by
region.

The Gibbs-Candes online-adaptive conformal (arXiv:2305.12616) provides
the right backup: if a stratum *does* drift out of coverage, instead of
immediately retraining we can update the conformal *f_s* online with a
provable long-run coverage guarantee. So calibration drift on its own
buys us a free recalibration step before any retraining is triggered.

### 2.3 Coverage drift via kappa(A) and the worst right singular vector

The current `IntentionFM` stores `A_inv` where *A = Phi^T Phi + lambda I*
on the joint feature space (`modules/surrogate/model.py:43`). Predictive
variance at a query *x* is *sigma^2 (1 + phi(x)^T A^{-1} phi(x))*, i.e.
*sigma^2 (1 + lev(x))* in the notation of `modules/surrogate/model.py:65`.
The standard variance-amplification bound from the SVD is

    lev(x) = phi(x)^T A^{-1} phi(x)  <=  ||phi(x)||^2 / lambda_min(A),

i.e. the worst-case amplification is *kappa(A) = lambda_max(A) / lambda_min(A)*
in the appropriate sense; the Eckart-Young theorem makes this tight in
the direction *v_min*, the right singular vector of Phi associated with
the smallest singular value.

This gives us *two* drift signals at once. First, *kappa(A)* itself is
scale-invariant and population-level; it does not depend on which query
arrived, only on how well-conditioned the FM's design matrix is. It
monitors *whether the FM is theoretically capable of giving low-variance
predictions anywhere*. Second, *v_min* identifies *the worst-supported
direction in feature space*; projecting the recent query stream onto
*v_min* tells us whether the agent has started asking questions
concentrated in that direction. A "coverage drift" event is one in which
the projection of recent queries onto *v_min* is significantly larger
than the projection of training queries onto *v_min*.

The combination of *kappa(A)* and the *v_min* projection is the right
replacement for INTEGRATION.md's "fire when `mean(leverage) > 5`"
suggestion (`docs/surrogate/INTEGRATION.md:99-104`): it is scale-free,
geometrically interpretable, and naturally returns a *location* - the
worst direction - to feed back into EPIG-driven local retraining
(agent (d)'s scope).

### 2.4 Why these three in this order: the Gneiting-Raftery decomposition

The proper-scoring-rule literature (Gneiting-Raftery, JASA 2007;
Broecker, QJRMS 2009) decomposes the squared-error or log-score of a
probabilistic forecaster into three orthogonal terms:

    Score = Uncertainty - Resolution + Reliability.

Reliability is the calibration term: it measures how far the stated
probabilities are from the empirical event rates conditional on the
stated probability. Resolution is the discriminative term - how much the
forecaster's stated probabilities depend on the case. Uncertainty is
the intrinsic difficulty of the task and is independent of the
forecaster.

The relevant consequence here: a forecaster can stop being *reliable*
(calibration drift) without changing its mean predictions (no accuracy
drift), and an accuracy degradation that does not also degrade
calibration is essentially impossible in the standard regime.
Concretely: if the predictive mean is right and the predictive variance
is right and the distribution is exchangeable, calibration holds. If
the mean degrades while the variance stays, the standardised residual
mean moves, calibration fails on at least one stratum, and DAS-CUSUM on
*z_t* moves too. If the variance degrades while the mean stays,
calibration fails directly. So *calibration drift fires first*, and the
design of the aggregation policy in section 4 reflects this.

The Gibbs-Candes online-adaptive conformal mechanism is the formal
guarantee that a calibration-drift event can be absorbed by a single
update to the conformal *f_s* without retraining the FM. This buys us
a cheap stabilising layer between "calibration is off" and "the FM is
wrong", which is what the aggregation policy is designed around.

## 3. Reconceiving the three signals under the FM constraint

The BRIEF.md hard constraint (`docs/research/BRIEF.md:10-43`) forbids
the FM from taking *c* as input, in any encoding. The current feature
map `phi_joint(c, m)` in `modules/surrogate/features.py:60-65` is the
forbidden pattern: it tensor-products `phi_c(c)` with `phi_x(m)` and
feeds both into the design matrix in `modules/surrogate/model.py:40-44`.
The whole current `IntentionFM` is out of scope under the constraint;
agent (b)'s job is to rebuild it.

The relevant question for *this* agent is: *do the three drift signals
survive the constraint?* The answer is yes, and the algebra survives in
form; what changes is the interpretation.

### 3.1 The leverage formula is invariant in form

Whatever agent (b) settles on, the FM will produce some embedding
*z = embed(d)* of a measured object *d* (event sample, histogram,
unbinned dataset). The downstream regression head - whether a
closed-form ridge on *z*, a small MLP, or a Gaussian process - will
have some local design matrix *A* and the predictive variance at a query
embedding *z* will have the form

    sigma_pred(z)^2  ~  z^T A^{-1} z + aleatoric.

The leverage *lev(z) = z^T A^{-1} z* is *exactly the same object* as
the current `leverage` (`modules/surrogate/model.py:67-70`), just over a
*c*-free feature space. Sherman-Morrison still gives O(d^2) online
updates for the EPIG acquisition. Eckart-Young still gives the *kappa(A)*
bound. The condition-number / *v_min* machinery is unchanged.

What changes is the *meaning*. In the current code, high leverage at a
candidate query *(c, m)* means "the user picked a weird Wilson
configuration". Under the constraint, high leverage at an embedding *z*
means "the FM is seeing a measured distribution whose internal
representation is far from anything in training data". The first is a
question about the user; the second is a question about the manifold
the FM has learned. The second is the right framing for a foundation
model: if the FM has not seen anything that embeds to here, its
predictions are unreliable, and we need to acquire more training data
that pushes the embedding manifold to cover this region.

This re-interpretation also fixes a subtle bug in the current
INTEGRATION.md plan. There, "leverage > 10" was suggested as a trigger,
treating leverage as a per-query scalar with a hard threshold
(`docs/surrogate/INTEGRATION.md:99-104`). But leverage is dimensionless
in a *coordinate-dependent* sense: rescaling the feature map by a factor
*alpha* rescales leverage by *alpha^{-2}*. With a learned embedding agent
(b) produces, the absolute scale of leverage is not something to put a
hard threshold on. *kappa(A)* and the *v_min*-projection ratio between
recent queries and training data are scale-invariant and the right
substitutes.

### 3.2 Accuracy and calibration are constraint-neutral

Accuracy drift (DAS-CUSUM on standardised residuals) and calibration
drift (BH-corrected per-region binomial test on coverage) only depend on
*(mu_hat, sigma_conformal, y_oracle)*. They are agnostic about how
mu_hat and sigma are produced internally. So the algebra of sections
2.1 and 2.2 carries over verbatim. The only implementation change is
that the per-region BH stratification needs a new notion of "region".
Under the constraint, "region" is no longer a slice of *(c_0, m)*
space; it is a clustering of the FM's embedding space *z*. K-means or
HDBSCAN on the embeddings of a held-out probe set is sufficient and
doesn't introduce new hyperparameters worth arguing over. Agent (d)
should pick this clustering and we'll consume it.

## 4. Aggregation policy

Three signals fire (or don't). 8 combinations. The policy:

| accuracy | calibration | coverage | action                                            | rationale |
|----------|-------------|----------|---------------------------------------------------|-----------|
| 0        | 0           | 0        | noop                                              | nothing drifted |
| 1        | 0           | 0        | watch                                             | impossible in steady state under Gneiting-Raftery - treat as noise / under-powered conformal test, log and keep watching |
| 0        | 1           | 0        | recalibrate conformal (Gibbs-Candes online step)  | exchangeability assumption violated locally; FM still predicts well in mean |
| 0        | 0           | 1        | local retrain (EPIG on the drifted region)        | the FM has been asked questions outside its training manifold but is not yet making large errors |
| 1        | 1           | 0        | recalibrate; if accuracy persists, partial retrain | order matters: calibration first (cheap), accuracy second (if still off, the FM mean is degrading) |
| 1        | 0           | 1        | local retrain (EPIG); recalibrate after           | rare; FM making errors *and* queries off-manifold but somehow calibration held - typically because the conformal layer over-covered |
| 0        | 1           | 1        | local retrain; recalibrate                        | queries off-manifold; the FM mean still OK, but the conformal layer can't cope |
| 1        | 1           | 1        | global retrain; full conformal refit              | end-to-end failure, EPIG over the union of drifted regions |

Three observations on the table:

- The (1, 0, 0) row is marked "impossible in steady state". This is the
  Gneiting-Raftery prediction: if calibration is fine and the queries
  are in the training manifold, an accuracy drift must come with a
  calibration drift in expectation, so (1, 0, 0) is either a
  finite-sample fluke or a sign that the conformal test is
  under-powered in the region that drifted. The right response is *not*
  to retrain but to log and require persistence - *N* consecutive
  windows with this pattern before any retraining fires. This is what
  cuts down the false-positive retrain rate, which is the single
  biggest source of wasted oracle calls in active learning.
- "Recalibrate" is cheaper than "retrain" by 2-3 orders of magnitude.
  The conformal `fit` is a few quantile computations on the cal set; it
  is sub-millisecond at our sizes. So the policy is biased toward
  trying recalibration first when both calibration and accuracy drift.
- "Local retrain" requires a *location*: the drifted region. Coverage
  drift gives us *v_min*. Calibration drift gives us the failing
  stratum. Accuracy drift gives us - via the CUSUM detection time - the
  embedding cluster of the query that triggered the change. Each signal
  is therefore tagged with the location of its failure, and the union
  of those locations is what EPIG (agent (d)) consumes.

## 5. Phoenix wiring: span tree and attributes

The agent runtime under `agent/main.py:35-51` issues an ADK turn. ADK
auto-instrumentation puts the turn under a top-level CHAIN span. Inside
that, we emit spans for each FM-loop primitive. Names and attributes
below; the `aletheia.*` prefix keeps them filterable in the UI.

Span tree for one orchestrator cycle:

    chain  agent.cycle
    |-- tool  surrogate.predict       (one per FM forward pass)
    |       aletheia.fm.version              str
    |       aletheia.fm.n_query              int
    |       aletheia.fm.embed_norm_mean      float
    |       aletheia.fm.embed_norm_max       float
    |       aletheia.fm.leverage_mean        float    (over query batch)
    |       aletheia.fm.leverage_max         float
    |       aletheia.fm.mu_mean              float
    |       aletheia.fm.sigma_mean           float
    |       aletheia.fm.region_id            str      (clustered region)
    |-- tool  oracle.query              (when ground truth is invoked)
    |       aletheia.oracle.backend         "analytic" | "madgraph"
    |       aletheia.oracle.n_points        int
    |       aletheia.oracle.duration_ms     float
    |       aletheia.oracle.linked_predict_span_id   str   (parent FM call)
    |-- chain drift.evaluate            (runs after each oracle.query)
    |   |-- tool drift.accuracy.das_cusum
    |   |       aletheia.drift.acc.S_t              float
    |   |       aletheia.drift.acc.S_threshold      float
    |   |       aletheia.drift.acc.fired            bool
    |   |       aletheia.drift.acc.delay_estimate   int
    |   |-- tool drift.calibration.bh
    |   |       aletheia.drift.cal.per_stratum_pvalues   list[float]
    |   |       aletheia.drift.cal.bh_threshold_adj      float
    |   |       aletheia.drift.cal.failing_strata        list[int]
    |   |       aletheia.drift.cal.fired                 bool
    |   |-- tool drift.coverage.kappa
    |           aletheia.drift.cov.kappa                 float
    |           aletheia.drift.cov.kappa_threshold       float
    |           aletheia.drift.cov.vmin_norm             float
    |           aletheia.drift.cov.recent_proj_ratio     float
    |           aletheia.drift.cov.fired                 bool
    |           aletheia.drift.cov.worst_region_id       str
    |-- tool drift.aggregate            (the 8-row decision table)
    |       aletheia.drift.combined_flag             str      "000" .. "111"
    |       aletheia.drift.action                    str      "noop"|"watch"|"recal"|"local_retrain"|"global_retrain"
    |       aletheia.drift.target_region_id          str?
    |-- tool epig.select                (only when action in {local_retrain, global_retrain})
    |       aletheia.epig.k                          int
    |       aletheia.epig.pool_size                  int
    |       aletheia.epig.target_region_id           str
    |       aletheia.epig.picked_span_ids            list[str]   (links back to predict spans)
    |-- tool oracle.query              (the EPIG-selected batch)
    |       same attrs as above
    |-- tool fm.update                  (the retrain)
    |       aletheia.fm.version_in                  str
    |       aletheia.fm.version_out                 str
    |       aletheia.fm.n_train                     int
    |       aletheia.fm.regions_retrained           list[str]
    |-- chain experiment.ab            (gated by promote rule, section 7)
        |-- tool ab.run                  (Phoenix experiment under the hood)
        |-- tool ab.decision
                aletheia.ab.promote                bool
                aletheia.ab.improvement_drifted    float
                aletheia.ab.no_regression          bool

The convention `aletheia.epig.picked_span_ids` linking back to the
`surrogate.predict` spans is the key piece that lets the MCP server
answer the rubric question "list spans that triggered retraining and
the EPIG-selected candidates that resulted" via a single `get-spans`
filter on `aletheia.drift.action == "local_retrain"` followed by a
`get-spans` join on the linked span ids.

For the rubric question "show me the last drift event by region",
filter `get-spans` on
`aletheia.drift.action in {recal, local_retrain, global_retrain}`,
group by `aletheia.drift.target_region_id`, take the latest per group.

For "compare RMSE on drifted-region between FM version k and k+1",
filter `get-spans` on `aletheia.fm.version` and join with the
per-region RMSE evaluation (next section).

## 6. Phoenix evaluators

Three Phoenix evaluators run on the trace stream. Each is a Python
function with the same signature:

    def evaluator(spans_df: pandas.DataFrame, *,
                  window: int = 200,
                  **kwargs) -> phoenix.trace.SpanEvaluations:
        ...

It pulls the most recent `window` spans of the relevant kind, computes
a score, and writes a `SpanEvaluations` annotation back via
`px.Client().log_evaluations(...)`. The evaluators do not call an LLM;
they are deterministic numerical checks. The orchestrator queries the
annotations (via MCP `get-span-annotations`) to make its decisions.

### 6.1 accuracy_drift_das_cusum

Pulls the last `window` `oracle.query` spans, joins each to the linked
`surrogate.predict` span (via the `linked_predict_span_id` attribute),
constructs the standardised residual time series

    z_t = (y_t - mu_t) / sigma_t^conformal,

runs DAS-CUSUM with adaptive variance windows of length *w* (suggested
*w = 30*) and detection threshold *h* tuned to give ARL_0 ~ 1000 under
*N(0, 1)* residuals - for the symmetric two-sided form, *h ~ 5.0* on
the standardised score, per Table 1 of arXiv:2210.17353. Returns one
score per evaluated span: the CUSUM statistic at the time of that span.
Annotates with a boolean label `accuracy_drift_fired = (S_t > h)` and
the metadata field `detection_delay_estimate`.

### 6.2 calibration_drift_conformal

Pulls the last `window` `oracle.query` spans, joins to predict spans,
buckets each by `aletheia.fm.region_id` (the clustered region), and for
each region computes empirical coverage at each target level *p in
{0.683, 0.954}*. For each region, does a two-sided binomial test of
*H_0*: coverage = *p*. Collects all *S* p-values, runs BH at *alpha = 0.05*
over them. Annotates each region's representative spans with
`calibration_drift_fired = bool`, with metadata `bh_threshold`,
`region_pvalue`, `failing_regions`.

### 6.3 coverage_drift_condition_number

Pulls the last `window` `surrogate.predict` spans. Reads the FM's
current `A_inv` (via the FM's state-dict in object store, referenced
from the predict span) and computes *kappa(A) = kappa(A^{-1})^{-1}* plus
the right singular vector *v_min* (cached at FM-version boundaries,
refreshed only on retrain). Projects the embeddings of the recent
queries onto *v_min* and compares the variance of that projection to
the same projection on the training set. Fires when either *kappa(A) >
kappa_threshold* (suggested *kappa_threshold = 10^4*) or
*Var_recent(<z, v_min>) / Var_train(<z, v_min>) > 3*. Annotates with
`coverage_drift_fired = bool`, `kappa`, `vmin_projection_ratio`,
`worst_region_id` (the cluster of the queries that contributed most to
the worst projection).

### 6.4 Schedule

All three evaluators run *on every cycle*, on a sliding window
terminated at the most recent `oracle.query`. There is no separate
cron - they fire when the orchestrator emits the `drift.evaluate` span,
the orchestrator waits for the annotations to come back (the synchronous
client return is the natural barrier), and then the `drift.aggregate`
span reads them and emits the action. This keeps the loop simple and
avoids the race between "evaluator hasn't seen this span yet" and
"orchestrator is about to decide".

## 7. Experiments: A/B testing FM versions before promoting

Two Phoenix experiments per promotion cycle, on the same Phoenix dataset
(the *probe set*).

The probe set is a fixed set of measured-distribution embeddings sampled
to (i) cover all regions roughly uniformly, (ii) include the drifted
region of the current cycle at higher density. Stored as a Phoenix
dataset, uploaded once and reused; the dataset id is the cycle-stable
reference object across experiments. Each example is
`{input: measured_distribution_serialised, reference: y_oracle}`.

Two experiments:

- `pre_update`: runs the FM version *before* the proposed retrain
  against the probe set.
- `post_update`: runs the FM version *after* the proposed retrain
  against the probe set.

Each experiment's `task` is a thin wrapper around `model.predict`. The
evaluators attached to each experiment compute:

- per-example squared error and absolute error (for region-stratified
  RMSE and median |rel err|),
- per-example coverage indicator at *p in {0.683, 0.954}* (for
  region-stratified coverage),
- per-example leverage / *v_min* projection (for the geometric
  comparison).

Each experiment writes one row per example. Post-experiment we compare
the two via the Phoenix `get-experiment-by-id` MCP tool: pull both
per-example DataFrames, stratify by region, compute the region-wise
deltas.

The promote rule:

1. RMSE on the *drifted region* must improve by at least 1% (a threshold
   tunable to the noise level of the oracle).
2. No region's RMSE may regress by more than the BH-corrected bootstrap
   CI at FDR 0.05 over the *K* regions. Bootstrap CIs use *B = 1000*
   resamples per region (cheap on the probe-set size).
3. Coverage on the drifted region at both target levels must be within
   +/- 0.02 of target.

If all three hold, the post-update version is promoted and tagged
`current` in the Phoenix dataset metadata; otherwise the pre-update
version is kept and the orchestrator logs a `promote_failed` annotation
on the `experiment.ab.decision` span with the failure reason. The
failure reason is in turn what the next cycle's drift evaluators read
to adjust their `region_id` clustering for the next attempt.

## 8. Loose ends

Two things this design leaves to the partner agents:

- The cluster definition for `region_id`. Agent (d) owns the
  embedding-space clustering. The drift evaluators consume the cluster
  ids; we do not specify the clustering algorithm beyond "k-means or
  HDBSCAN on the FM embedding of a held-out probe set". Whatever agent
  (d) picks must be stable across FM versions (otherwise A/B is
  comparing apples and oranges); the standard trick is to fit the
  clustering once on the *current* embedding and then map new
  embeddings to nearest centroid.
- The shape of the FM embedding *z*. Agent (b) owns this. We need only
  the contract: a vector-valued *z* of fixed dimensionality, plus
  access to *A^{-1}* and *v_min* at the regression head. The DAS-CUSUM
  and BH evaluators do not look at *z* at all; the kappa evaluator does.

## 9. What this delivers against the rubric

The Arize-track rubric (per `docs/research/BRIEF.md:117-124`) rewards:
code-owned agent runtime (ADK via the existing `agent/main.py`),
OpenInference instrumentation (via `agent/instrumentation.py`),
traces persisted to Phoenix (via the local container), Phoenix MCP
configured (via `.gemini/settings.json`), evaluations on traces
(via the three drift evaluators of section 6), drift detection and
retraining gating (via the aggregation policy of section 4 and the
A/B experiment policy of section 7).

The specific argument we make against generic drift detection is that
PSI / KS / embedding-distance to a reference baseline does not answer
the question we care about - *is the FM still predictive?* - and that
the three signals above do, with closed-form null distributions
(binomial for calibration), provable asymptotic guarantees
(Lorden-Pollak ARL for DAS-CUSUM, Gibbs-Candes for online conformal),
and tight theoretical bounds (Eckart-Young for kappa). The Phoenix
substrate is unchanged; we replace only the metric definitions and the
aggregation policy.

## References

- Ahad, Davenport, Xie. *Data-Adaptive Symmetric CUSUM for Sequential
  Change Detection*. arXiv:2210.17353 (2022, published Sequential
  Analysis 43(1), 2024).
- Araz, Spannowsky. *Another Fit Bites the Dust: Conformal Prediction as
  a Calibration Standard for Machine Learning in High-Energy Physics*.
  arXiv:2512.17048 (2025).
- Benjamini, Hochberg. *Controlling the False Discovery Rate*. JRSS B
  57(1), 1995.
- Benjamini, Yekutieli. *The Control of the False Discovery Rate in
  Multiple Testing under Dependency*. Annals of Statistics 29(4), 2001.
- Broecker. *Reliability, sufficiency, and the decomposition of proper
  scores*. QJRMS 135(643), 2009.
- Eckart, Young. *The approximation of one matrix by another of lower
  rank*. Psychometrika 1, 1936.
- Gibbs, Candes. *Conformal Prediction with Conditional Guarantees*.
  arXiv:2305.12616, JMLR 2024.
- Gneiting, Raftery. *Strictly Proper Scoring Rules, Prediction, and
  Estimation*. JASA 102(477), 2007.
- Lorden. *Procedures for reacting to a change in distribution*. Annals
  of Math. Statistics 42(6), 1971.
- Page. *Continuous inspection schemes*. Biometrika 41(1/2), 1954.
- Pollak. *Optimal detection of a change in distribution*. Annals of
  Statistics 13(1), 1985.
- Smith, Bickford Smith, Rainforth. *Prediction-Oriented Bayesian Active
  Learning*. arXiv:2304.08151 (AISTATS 2023). EPIG.
