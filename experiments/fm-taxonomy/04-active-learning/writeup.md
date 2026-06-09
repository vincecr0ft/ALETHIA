# Active learning in relation to foundation models

Branch 04 of the ALETHIA FM taxonomy. Siblings: 01 FM/surrogate (general),
02 FM in HEP, 03 information geometry & morphing. This branch formalises the
**mathematics of the learner** — acquisition as expected information gain — and
the link between information-gain acquisition and the Fisher-information /
posterior-contraction picture that the rest of ALETHIA is built on. It then
distinguishes *parameter*-information acquisition (BALD, Fisher/D-optimal,
A-optimal) from *prediction*-information acquisition (EPIG), and works out what
changes when a pretrained foundation-model backbone supplies the
representation.

Every citation below has been located and verified (title, authors, venue,
arXiv id). The math is tied throughout to the closed-form acquisition code in
`experiments/al-phoenix-studies/` and the documented contraction-vs-MLE gap.

---

## 1. Setup: the learner and the loop

A learner holds a posterior over a quantity of interest given data
$\mathcal D$. In a pool-based active-learning loop it repeatedly:

1. scores each candidate input $x$ in a pool by an **acquisition function**
   $a(x)$;
2. queries the label/observation $y$ at $x^\star=\arg\max_x a(x)$;
3. folds $(x^\star,y)$ into $\mathcal D$ and updates the posterior.

The whole subject is *which* $a(x)$ to use. Two quantities can be the target of
"information": the **parameters** $\theta$ of the model, or the **predictions**
$y_\ast$ at points we will be tested on. The canonical references frame this as
expected reduction in Shannon entropy — Lindley's measure of the information in
an experiment (Lindley 1956) and MacKay's three information-based objectives
for active data selection (MacKay 1992). Settles (2009) is the standard survey
of the query-strategy zoo built on top.

The decision-theoretic skeleton is *Bayesian optimal experimental design*
(BOED): pick the design $\xi$ maximising expected information gain (EIG),
which for a parameter target is the mutual information between $\theta$ and the
prospective outcome $y$,

$$
\mathrm{EIG}(\xi)=\mathbb E_{p(y\mid\xi)}\!\big[\,H[p(\theta)] - H[p(\theta\mid y,\xi)]\,\big]
            = I(\theta;\,y\mid\xi).
$$

Active learning is the sequential, pool-restricted special case of BOED:
$\xi$ ranges over "which pool point to label next" (Chaloner & Verdinelli 1995;
Foster et al. 2019). ALETHIA's loop is exactly this — the "design" is which
invariant-mass value $m$ to query the SMEFT oracle at.

---

## 2. Parameter-information acquisition

### 2.1 BALD — mutual information between parameters and label

For a Bayesian predictor $p(y\mid x,\theta)$ with posterior $p(\theta\mid\mathcal D)$,
the **expected information gain in the parameters** from labelling $x$ is the
mutual information $I(\theta; y\mid x,\mathcal D)$. Houlsby et al. (2011) rewrite
it via Bayes' rule into a form that needs only predictive entropies — the BALD
score:

$$
a_{\mathrm{BALD}}(x)=\underbrace{H\big[\,\mathbb E_{p(\theta\mid\mathcal D)} p(y\mid x,\theta)\,\big]}_{\text{total predictive entropy}}
              -\underbrace{\mathbb E_{p(\theta\mid\mathcal D)}\,H\big[p(y\mid x,\theta)\big]}_{\text{expected entropy under }\theta}.
$$

The first term is large where the *marginal* prediction is uncertain; the
second is large where individual parameter draws are each individually
uncertain. The difference is large precisely where the model is uncertain
*because the parameters disagree* — epistemic uncertainty — which is the
uncertainty a label can remove. Gal, Islam & Ghahramani (2017) made this
practical for deep nets via MC-dropout posteriors.

### 2.2 Fisher information / D-optimal design — the Gaussian-linear closed form

When the predictor is (locally) linear-Gaussian — $y = \psi(x)^\top w + \varepsilon$,
$\varepsilon\sim\mathcal N(0,\sigma_y^2)$, Gaussian prior on $w$ — the posterior
is Gaussian and BALD collapses to a determinant. Writing the design (precision)
matrix as $A=\sigma_y^{-2}\sum_i \psi(x_i)\psi(x_i)^\top + A_0$ (so the
posterior covariance of $w$ is $\sigma_y^2 A^{-1}$), the EIG of adding one
point $x$ is the classical rank-one D-optimal gain

$$
a_{\mathrm{D}}(x)=\tfrac12\log\!\big(1+\psi(x)^\top A^{-1}\psi(x)\big).
$$

This is exactly ALETHIA's `ig_per_candidate` in
`experiments/al-phoenix-studies/_common.py`:
`0.5 * log1p(psi @ A_inv @ psi)`. The quadratic form
$\mathrm{lev}(x)=\psi(x)^\top A^{-1}\psi(x)$ is the **leverage** of the
candidate in the current posterior metric; the design matrix $A$ *is* the
Fisher information of the Gaussian-linear model. So in ALETHIA, "BALD",
"D-optimal", and "Fisher-information acquisition" are literally the same
score, and the loop's `leverage` arm implements it. The link to BOED's
determinant objective is the Bayesian-D-optimal criterion of
Chaloner & Verdinelli (1995).

### 2.3 A-optimal and directional acquisition

D-optimality maximises $\log\det$ of the information (total volume contraction).
**A-optimality** instead minimises the trace of the posterior covariance, or —
when only one direction $a$ matters — the variance along that direction.
ALETHIA's `param_epig_a_acquire` is exactly A-optimal acquisition restricted to
one Wilson-coefficient direction $\hat p_a$: it scores

$$
\Delta H_a(x) = -\tfrac12\log\!\Big(1-\frac{\sigma_y^2\,u_a(x)^2}{(1+\mathrm{lev}(x))\,\Sigma_{aa}}\Big),
\qquad u_a(x)=\hat p_a^\top A^{-1}\psi(x),\;\;\Sigma_{aa}=\sigma_y^2\,\hat p_a^\top A^{-1}\hat p_a,
$$

i.e. the log-reduction in the marginal posterior variance of the targeted
direction from a rank-one update (Sherman–Morrison). `param_epig_d_acquire` is
the same construction with $\det\Sigma$ of a resolved subspace — D-optimality
restricted to the data-dominated rows of the projection $P=V^\top W$. These are
textbook D-/A-optimal criteria specialised to the projected parameter of
interest.

### 2.4 Batch acquisition and the redundancy trap

Greedy top-$k$ on any of the above repeatedly picks near-duplicate points: the
marginal scores ignore that two high-leverage candidates may carry the *same*
information. BatchBALD (Kirsch, van Amersfoort & Gal 2019) fixes this by scoring
the **joint** mutual information $I(\theta;y_{1:k})$ of a batch, which is
submodular and sub-additive, so it down-weights redundant points. The paper's
own finding — naive batch-BALD "acquires similar and redundant points,
sometimes performing worse than random" — is *precisely* the structure ALETHIA
measures: see §5.

---

## 3. Prediction-information acquisition (EPIG)

BALD reduces uncertainty about *all* parameters, including directions no test
input cares about. **EPIG** (Expected Predictive Information Gain; Bickford
Smith et al. 2023) instead targets the predictions on a *target distribution*
$p_\ast(x_\ast)$ we will be evaluated on:

$$
a_{\mathrm{EPIG}}(x)=\mathbb E_{p_\ast(x_\ast)}\,I\big(y_\ast;\,y \mid x_\ast, x,\mathcal D\big),
$$

the expected mutual information between the *prediction* at a random target
point $x_\ast$ and the label of the candidate $x$. BALD is the special case
where the target distribution is "the parameters" rather than future
predictions; EPIG favours only information that lowers downstream predictive
uncertainty, and empirically beats BALD as a drop-in acquisition under
distribution shift. In the Gaussian-linear regime EPIG also has a closed form,
which is what ALETHIA's `epig_acquire_m` implements: for a target set
$\{m_T\}$ it scores each candidate by the mean over targets of

$$
\tfrac12\log\frac{\mathrm{lev}_T}{\mathrm{lev}_T - K_{TP}^2/(1+\mathrm{lev}_P)},
\qquad K_{TP}=\psi(m_T)^\top A^{-1}\psi(m_P),
$$

i.e. the predictive-variance reduction at the targets from a rank-one update at
the candidate, averaged over the target set (Sherman–Morrison again). EPIG and
A-optimal-on-a-direction are the *prediction-space* and *parameter-space*
versions of the same rank-one contraction; ALETHIA carries both, which is why
it can measure the gap between them directly.

---

## 4. Taxonomy axes this branch adds

| Acquisition | Information target | Closed form (Gaussian-linear) | ALETHIA arm |
|---|---|---|---|
| Predictive entropy / variance | marginal label uncertainty | $H[\bar p(y\mid x)]$ | (uncertainty baseline) |
| **BALD** (Houlsby 2011) | parameters $\theta$ | $\tfrac12\log(1+\psi^\top A^{-1}\psi)$ | `leverage` |
| **Fisher / D-optimal** (Lindley 1956; Chaloner–Verdinelli 1995) | $\log\det$ posterior precision | same as BALD here | `leverage`, `param_epig_d` |
| **A-optimal / directional** | one parameter direction | $-\tfrac12\log(1-\sigma_y^2 u_a^2/((1{+}\mathrm{lev})\Sigma_{aa}))$ | `param_epig_a` |
| **EPIG** (Bickford Smith 2023) | predictions on target set | mean target var-reduction | `epig` |
| **BatchBALD** (Kirsch 2019) | joint param info of a batch | submodular det-gain | (diagnosed via `cos_Ainv`) |

Two orthogonal axes organise the whole branch:

- **Axis A — what the information is *about*:** parameters (BALD, Fisher,
  D-/A-optimal) vs predictions (EPIG). Parameter-information is target-agnostic
  and maximises posterior contraction; prediction-information is
  target-weighted and maximises test-time predictive accuracy. These coincide
  only when the parameter directions a query resolves are the ones the test
  distribution loads on — and **come apart exactly when they do not** (§5).
- **Axis B — where the representation $\psi(x)$ comes from:** fit from scratch
  on task data, vs supplied by a **frozen / fine-tuned foundation-model
  backbone** (§6).

A third, dependent axis is **greedy vs joint batch** acquisition (Axis C),
which only bites once the representation makes high-score candidates collinear.

---

## 5. How ALETHIA's studies map onto the math

The Phoenix+AL studies (`experiments/al-phoenix-studies/`, `REPORT.md`) run all
of the above against the ALETHIA full-chain IntentionFM head on the $\mu_{FB}$
observable. They are a direct, instrumented test of the taxonomy:

- **The acquisition functions are the closed-form scores above.** `leverage` =
  BALD/Fisher/D-optimal (§2.2); `param_epig_d`/`param_epig_a` = D-/A-optimal on
  the projected Wilson subspace (§2.3); `epig` = predictive EPIG (§3). The
  posterior covariance of the parameter of interest is
  $\Sigma_{\tilde c}=\sigma_y^2\,P A^{-1}P^\top$ (`sigma_ctilde`), so
  "contraction" is literally the BOED objective shrinking.

- **Contraction confirms the parameter-information theory.** Targeted
  acquisition beats random in posterior-covariance contraction on the targeted
  $\tilde c$ direction by 8–47$\sigma$ across every configuration — the
  near-tautology that D-/A-optimal maximises contraction (Chaloner–Verdinelli
  §2). In the deep closed-loop sweep this reaches $-42\sigma$ at K=12, 500
  cycles.

- **The redundancy diagnostic is the BatchBALD finding, measured.** Stage B
  reports mean $|\cos_{A^{-1}}|$ across top-decile-IG candidates $=1.000$ on
  every pool: every "high-information" candidate projects onto the *same* 1-D
  ridge in the $A^{-1}$ metric. This is exactly the redundancy BatchBALD warns
  about (Kirsch 2019) — greedy marginal acquisition exhausts one direction.
  At 500 cycles the non-directional methods (`leverage`, `epig`)
  *over-acquire* that ridge and end up with **worse contraction than random**
  ($t_c=+9$ to $+27$); only the parameter-aware directional methods keep
  beating random. That is Axis-C (greedy-vs-joint) failure made visible.

- **The contraction-vs-MLE gap is the parameter-vs-prediction distinction,
  inverted.** Targeted acquisition contracts the posterior *and yet makes the
  MLE point estimate on the targeted direction worse* by 7–47$\sigma$
  (REPORT §C.2). Mechanism: at $K\le96$ the $\mu_{FB}$ observable cannot resolve
  the bimodal target's $\tilde c_1$ ambiguity (Stage A.4: that direction is
  prior-dominated, $\lambda_K\sigma^2_{\rm prior}\approx$ marginal), so the
  high-EIG picks are picks that *confirm the wrong mode*. The posterior shrinks
  confidently around the wrong centre. In taxonomy terms: maximising
  parameter-information (Axis A, parameter side) does **not** imply better
  predictions/estimates when the information-rich direction is not the
  identifiable one. EPIG's premise — align acquisition with the downstream
  target, not raw parameter entropy — is the literature's answer; ALETHIA shows
  the failure mode that motivates it, in a closed-form, traceable setting.

- **The operative gate is a representation-geometry quantity, not CV(IG).** The
  AL_separation plan's published criterion CV(IG)$\gtrsim1$ passes on every pool
  (it is structural slack, not exploitable structure). The signal that
  predicts the null is mean $|\cos_{A^{-1}}|$ of the top-IG candidates — a
  property of the *embedding geometry* $\psi(\cdot)$, i.e. of the backbone
  (Axis B). This is the bridge to §6.

---

## 6. How a foundation model changes the picture

A pretrained backbone supplies the feature map $\psi_\theta(x)$ that every
acquisition score above is computed in. Three consequences, each grounded:

1. **Representation reuse / amortised acquisition.** ALETHIA does *not* refit
   the network in the loop. The IntentionFM head exposes
   $A^{-1}=(\sigma_y^{-2}\sum\psi\psi^\top+A_0)^{-1}$ in closed form, so every
   acquisition score is an $O(D^2)$ rank-one update via Sherman–Morrison — no
   retraining, no MC sampling. This is the cheap, amortised end of the BOED
   spectrum whose expensive end is variational EIG (Foster et al. 2019) and
   amortised design policies (Deep Adaptive Design, Foster et al. 2021): a
   pretrained backbone turns a per-step variational optimisation into one
   matrix solve.

2. **What "uncertainty" means changes.** With a frozen backbone the *only*
   epistemic uncertainty BALD/EPIG can see is in the readout head's posterior
   $p(w\mid\mathcal D)$; the features themselves are fixed. The redundancy and
   identifiability pathologies in §5 are therefore **properties of the frozen
   embedding**, not of the labels (Stage A.1 confirms label-independence). If
   the backbone's $\psi$ collapses the directions a target needs onto one
   ridge, no acquisition function can separate them — the ceiling is set by the
   representation. This is why the operative gate is a geometry of $\psi$.

3. **Cold-start and the diversity/uncertainty split.** Uncertainty-based
   acquisition (BALD/EPIG) is unreliable at the start of a loop when the
   posterior is uninformative — the cold-start problem. With a strong
   pretrained representation the productive early-budget strategy is
   *diversity/coverage* in embedding space — TypiClust (Hacohen, Dekel &
   Weinshall 2022, arXiv:2202.02794) and ProbCover (Yehuda et al. 2022,
   arXiv:2205.11320) select typical/covering points from self-supervised
   embeddings and beat uncertainty sampling at low budget. ALETHIA's
   observation that *random's coverage keeps widening while greedy exhausts the
   ridge* is the same phenomenon: in a good frozen representation, coverage is a
   competitive — sometimes dominant — acquisition principle, and pure
   information-greedy acquisition needs the BatchBALD-style joint/diversity
   correction to avoid collapsing onto one direction.

So the FM does three things to active learning: it makes the acquisition score
analytic and cheap (reuse), it relocates the binding constraint from the labels
to the embedding geometry (uncertainty redefinition), and it shifts the
low-budget optimum from uncertainty toward coverage (cold-start).

---

## 7. Taxonomy contribution

This branch contributes the **learner-side axes** of the FM taxonomy and the
formal machinery that connects them to the information-geometry branch (03):

- **Axis A (information target): parameter vs prediction.** Formalised as
  BALD/Fisher/D-/A-optimal (parameter side) vs EPIG (prediction side), unified
  by the rank-one Sherman–Morrison contraction in the Gaussian-linear regime.
  The non-trivial content is that they *separate*, and ALETHIA exhibits the
  separation as the contraction-vs-MLE gap.
- **Axis B (representation source): from-scratch vs frozen/fine-tuned FM
  backbone.** With a frozen backbone the acquisition metric is the head's
  $A^{-1}$ and the binding constraint is the embedding geometry of $\psi_\theta$
  — a quantity the morphing/information-geometry branch (03) characterises via
  the Fisher rotation $V$ and the projection $P=V^\top W$. This is the explicit
  hand-off between branch 03 and branch 04: branch 03 says *what the
  data-dominated directions are*; branch 04 says *how to acquire along them and
  why greedy information-gain can still pick the wrong ones*.
- **Axis C (greedy vs joint batch).** Only binds once the FM representation
  makes top-score candidates collinear; ALETHIA's $|\cos_{A^{-1}}|\to1$ is the
  empirical trigger and BatchBALD is the prescribed fix.

The operative claim for the paper, consistent with `REPORT.md`: the
first-class signal is **not** "targeted beats random" but the *gap* between
posterior contraction and estimator/predictive error — the gap between Axis-A's
two sides — which a frozen FM head makes computable in closed form per cycle.

---

## References (all verified)

- D. V. Lindley (1956). *On a Measure of the Information Provided by an
  Experiment.* Annals of Mathematical Statistics 27(4):986–1005.
- D. J. C. MacKay (1992). *Information-Based Objective Functions for Active Data
  Selection.* Neural Computation 4(4):590–604.
- K. Chaloner, I. Verdinelli (1995). *Bayesian Experimental Design: A Review.*
  Statistical Science 10(3):273–304.
- N. Houlsby, F. Huszár, Z. Ghahramani, M. Lengyel (2011). *Bayesian Active
  Learning for Classification and Preference Learning.* arXiv:1112.5745. (BALD)
- B. Settles (2009). *Active Learning Literature Survey.* Univ. of
  Wisconsin–Madison, Computer Sciences Technical Report 1648.
- Y. Gal, R. Islam, Z. Ghahramani (2017). *Deep Bayesian Active Learning with
  Image Data.* ICML 2017. arXiv:1703.02910.
- A. Kirsch, J. van Amersfoort, Y. Gal (2019). *BatchBALD: Efficient and Diverse
  Batch Acquisition for Deep Bayesian Active Learning.* NeurIPS 2019.
  arXiv:1906.08158.
- A. Foster, M. Jankowiak, E. Bingham, P. Horsfall, Y. W. Teh, T. Rainforth,
  N. Goodman (2019). *Variational Bayesian Optimal Experimental Design.*
  NeurIPS 2019. arXiv:1903.05480.
- A. Foster, D. R. Ivanova, I. Malik, T. Rainforth (2021). *Deep Adaptive
  Design: Amortizing Sequential Bayesian Experimental Design.* ICML 2021.
  arXiv:2103.02438.
- F. Bickford Smith, A. Kirsch, S. Farquhar, Y. Gal, A. Foster, T. Rainforth
  (2023). *Prediction-Oriented Bayesian Active Learning.* AISTATS 2023 (PMLR
  v206). arXiv:2304.08151. (EPIG)
- G. Hacohen, A. Dekel, D. Weinshall (2022). *Active Learning on a Budget:
  Opposite Strategies Suit High and Low Budgets.* ICML 2022. arXiv:2202.02794.
  (TypiClust)
- O. Yehuda, A. Dekel, G. Hacohen, D. Weinshall (2022). *Active Learning Through
  a Covering Lens.* NeurIPS 2022. arXiv:2205.11320. (ProbCover)
</content>
