# Foundation models and surrogate modelling: the general landscape

Branch 01 of the ALETHIA foundation-model taxonomy. Sibling branches: FM in HEP (02),
information geometry and Lagrangian morphing (03), active learning (04). This branch
fixes the vocabulary the other three borrow: it states, formally, what a foundation
model, a surrogate, an emulator, and an amortized posterior each approximate, under
what objective, and what part of the fit is reusable across tasks.

The four terms are routinely used interchangeably. They are not the same object. The
distinctions are not stylistic; they are differences in (i) the function or distribution
being approximated, (ii) the variable the fit is amortized over, and (iii) the
mechanism by which a trained artefact is reused on a new problem. We set up notation
once and then read every mode off it.

## 0. Common notation

Let a simulator (or any data-generating process) be a map from parameters
$\theta \in \Theta$ to a distribution over observables, $x \sim p(x \mid \theta)$.
A *task* is an inference or prediction problem instantiated at some parameter, prior,
or input condition. Write:

- $f_\theta : \mathcal{X}_{\rm in} \to \mathcal{X}_{\rm out}$ for an input-output map the
  simulator (or experiment) realises at fixed $\theta$ — e.g. PDE initial condition
  $\mapsto$ solution field, or detector input $\mapsto$ summary statistic;
- $p(x \mid \theta)$ for the likelihood (forward model);
- $p(\theta \mid x) \propto p(x \mid \theta)\, p(\theta)$ for the posterior;
- $g_\phi$ for the learned artefact with weights $\phi$.

The single axis that separates the four modes is **what $g_\phi$ is conditioned on and
amortized over**, and **what objective ties $g_\phi$ to the truth**.

## 1. Surrogate

A surrogate replaces an expensive forward map by a cheap learned function, *for one
task*. Given draws $\{(u_i, y_i)\}_{i=1}^n$ with $y_i = f_{\theta_0}(u_i) + \varepsilon_i$
at a *fixed* configuration $\theta_0$, the surrogate is

$$
g_\phi \approx f_{\theta_0}, \qquad
\hat\phi = \arg\min_\phi \frac1n \sum_{i=1}^n \big\| y_i - g_\phi(u_i) \big\|^2 ,
$$

i.e. ordinary supervised regression of outputs on inputs at one operating point. The
function class is whatever is convenient — polynomial chaos, a Gaussian process, a
small neural net. What it approximates is a *deterministic (or mean) response surface*
$u \mapsto \mathbb{E}[y \mid u, \theta_0]$. What is reusable: the fit transfers to new
*inputs* $u$ drawn under the same $\theta_0$, and to nothing else. Change $\theta_0$ and
the surrogate is re-fit from scratch. Amortization is over the input $u$ only.

## 2. Emulator

"Emulator" is the surrogate's statistical sibling, with the discipline that it carries
calibrated uncertainty and is meant to be plugged into a *downstream uncertainty or
calibration analysis* of a computer model. The canonical construction is the Gaussian
process emulator with model-discrepancy term of Kennedy and O'Hagan
[Kennedy2001]: the field observation is

$$
z(u) \;=\; \rho\, g_\phi(u, \theta) \;+\; \delta(u) \;+\; e, \qquad
g_\phi \sim \mathcal{GP}(m, k), \quad \delta \sim \mathcal{GP}(m_\delta, k_\delta),
$$

where $g_\phi$ emulates the simulator output, $\delta$ is a discrepancy process for
structural model error, and $e$ is observation noise. The learning objective is the GP
marginal likelihood over the design points; the deliverable is a *posterior predictive
distribution* $p(y \mid u, \text{design})$, not a point. The operational distinction
from a surrogate: an emulator is required to report $\mathrm{Var}[y \mid u]$ honestly so
it can be propagated through a calibration of the physical parameters $\theta$.
Amortization is still over the input $u$; the calibration parameters $\theta$ are
inferred *outside* the emulator, by an MCMC or optimiser that calls it.

Surrogate vs emulator is therefore a distinction of *contract*, not of architecture: a
surrogate promises a value, an emulator promises a calibrated predictive distribution
intended for uncertainty propagation. Both are single-system objects.

## 3. Neural operator (the surrogate generalised over a function space)

A neural operator widens the surrogate one step: instead of fitting $f_{\theta_0}$ for
one $\theta_0$, it learns the *solution operator of a parametric family*. For a PDE with
parameter/coefficient function $a$ and solution $u$, the target is the operator
$\mathcal{G} : a \mapsto u$ between function spaces. The Fourier Neural Operator
[Li2020] parameterises $\mathcal{G}_\phi$ as a stack of kernel-integral layers made cheap
in Fourier space,

$$
v_{t+1}(x) = \sigma\!\Big( W v_t(x) + \mathcal{F}^{-1}\big[ R_\phi \cdot \mathcal{F} v_t \big](x) \Big),
$$

trained on samples $\{(a_i, u_i)\}$ by $\min_\phi \sum_i \| \mathcal{G}_\phi(a_i) - u_i \|^2$.
DeepONet [Lu2021] does the same with a branch/trunk split licensed by the operator
universal-approximation theorem. What it approximates is a map *between functions*; what
is reusable is the whole family — evaluate $\mathcal{G}_\phi$ on a new coefficient field
$a$ with no re-fit, at fixed (or discretisation-invariant) resolution. This is the first
mode that is reusable across *tasks within a family* rather than only across inputs of
one task. It is a surrogate amortized over the family parameter.

## 4. Amortized posterior (simulation-based / likelihood-free inference)

Here the learned object is an inverse: a distribution over parameters given data. When
$p(x \mid \theta)$ can be sampled but not evaluated, simulation-based inference
[Cranmer2020] trains $g_\phi$ on pairs $(\theta_i, x_i)$, $\theta_i \sim p(\theta)$,
$x_i \sim p(x \mid \theta_i)$, in one of three families:

- **Neural posterior estimation (NPE):** fit a conditional density
  $q_\phi(\theta \mid x)$ directly, by maximum likelihood over the joint draws,
  $\max_\phi \mathbb{E}_{p(\theta)p(x\mid\theta)}[\log q_\phi(\theta \mid x)]$, recovering
  $p(\theta \mid x)$ at the optimum. Density estimators are normalizing flows such as
  Masked Autoregressive Flow [Papamakarios2017]; the sequential/proposal-corrected
  variant is Automatic Posterior Transformation [Greenberg2019], building on the
  $\varepsilon$-free conditional-density-estimation idea of [Papamakarios2016].
- **Neural likelihood estimation (NLE):** fit $q_\phi(x \mid \theta)$, then sample the
  posterior with a standard MCMC.
- **Neural ratio estimation (NRE):** train a classifier to approximate the
  likelihood-to-evidence ratio $r(x,\theta) = p(x\mid\theta)/p(x)$ and use it inside MCMC
  [Hermans2020].

The defining property is **amortization over the data / observation**: pay the training
cost once over the prior predictive, then for *any* observed $x_{\rm obs}$ obtain
$q_\phi(\theta \mid x_{\rm obs})$ by a single forward pass — no per-observation
optimisation, no per-observation simulation. What is approximated is a *conditional
distribution over parameters*; what is reusable is the inverse map across all
observations consistent with the trained prior. This is the mode ALETHIA operates in: a
closed-form Bayesian-linear-regression head amortizes the Wilson-coefficient posterior
over event sets.

A neural process [Garnelo2018] is the supervised analogue of the same amortization:
condition on a context set, predict a stochastic-process posterior over targets in one
pass. It amortizes the *posterior over functions given data*, which is why it sits
between the operator (function-valued output) and the amortized posterior (distribution
conditioned on observed data).

## 5. Foundation model

A foundation model is defined by Bommasani et al. [Bommasani2021] as a model trained on
broad data at scale by a *self-supervised* objective and then *adapted* to many
downstream tasks. The two operative components are an objective that needs no
task labels and a reuse mechanism that is not retraining.

**Pretraining objectives** construct a supervisory signal from the structure of the
data itself:

- masked / denoising prediction — BERT masks tokens and predicts them
  [Devlin2018]; the masked autoencoder masks image patches and reconstructs pixels
  [He2021]; the loss is reconstruction of the hidden part conditioned on the visible
  part, $\min_\phi \mathbb{E}\,\ell\big(x_{\rm masked},\, g_\phi(x_{\rm visible})\big)$;
- joint-embedding / predictive objectives predict *representations* of held-out views
  rather than raw pixels (the JEPA line the ALETHIA encoder follows), which removes the
  pressure to model nuisance detail.

The common form is a self-supervised risk $\min_\phi \mathbb{E}_{x \sim \mathcal D}\,
\ell_{\rm ssl}(x; \phi)$ over a broad corpus $\mathcal D$, with no $\theta$-labels and no
single downstream target.

**Adaptation / reuse** is the second axis. The trained representation
$h_\phi : \mathcal{X} \to \mathbb{R}^D$ is reused by (a) a frozen-feature linear probe or
fine-tune, $\min_\psi \sum_i \ell(y_i, w^\top h_\phi(x_i))$ over a small labelled set;
or (b) in-context conditioning, where downstream behaviour is set by a context set
supplied at inference with no weight update. The reuse mechanism — not the architecture —
is what makes the model "foundational": one pretraining run, many tasks.

A foundation model is thus *not* a fifth point estimate. It is orthogonal to the
surrogate/posterior axis: it specifies *how the representation was learned and reused*,
and that representation can then be wired into a surrogate head (regression), an operator
head, or an amortized-posterior head. ALETHIA is exactly this composition: a
self-supervised event-level encoder (foundation-model side) feeding a closed-form
amortized-posterior head (SBI side).

## 6. Taxonomy table

| Mode | Approximates | Objective | Conditioned on | Amortized over | Reuse mechanism |
|---|---|---|---|---|---|
| Surrogate | $\mathbb{E}[y\mid u,\theta_0]$, one system | supervised MSE | input $u$ | inputs $u$ at fixed $\theta_0$ | evaluate on new $u$ |
| Emulator | $p(y\mid u,\theta_0)$ + discrepancy | GP marginal likelihood | input $u$ | inputs $u$ | calibrated predictive, propagated downstream |
| Neural operator | operator $\mathcal G:a\mapsto u$ | supervised MSE over a family | input function $a$ | family parameter $a$ | evaluate on new $a$, no re-fit |
| Amortized posterior (SBI) | $p(\theta\mid x)$ | NPE/NLE/NRE | observation $x$ | observations $x$ | one forward pass per new $x$ |
| Neural process | posterior over functions given context | predictive log-likelihood | context set | datasets / context sets | condition on new context |
| Foundation model | reusable representation $h_\phi$ | self-supervised (mask / JEPA) | raw inputs | the data distribution | probe / fine-tune / in-context |

Reading the table by column makes the confusions precise. Surrogate and emulator differ
only in the *Objective/contract* column. Surrogate and neural operator differ only in the
*Amortized-over* column (one $\theta_0$ vs a whole family). Surrogate/operator and
amortized posterior differ in the *Approximates* column: forward map vs inverse
distribution. Foundation model differs from all of them in the *Objective* column — it is
the only row whose target is a representation rather than a task output — and is
composable with any of the other rows as a front end.

## 7. How this branch frames a physics foundation model

A physics foundation model, in this taxonomy, is the *composition* of a self-supervised
representation learner over a family of physical configurations with a reusable inference
head, where reuse is by conditioning rather than retraining. Concretely for ALETHIA: the
family is dimension-six SMEFT Drell-Yan parameterised by Wilson coefficients $c$; the
self-supervised encoder $\psi_\theta$ learns an event-level representation under a
JEPA-style view-invariance objective (foundation-model row); the closed-form ridge head
turns the pooled representation into $p(c \mid \{x_n\})$ in one forward pass
(amortized-posterior row); and the active-learning loop (sibling branch 04) chooses where
in $c$-space to simulate next. The morphing coefficients of sibling branch 03 are the
analytic operator $\mathcal{G}: c \mapsto$ cross-section structure that the encoder must
linearly recover (properties P3/P4 in the paper). The taxonomy thus places ALETHIA at the
intersection of the foundation-model row (how the representation is learned and reused)
and the amortized-posterior row (what the head outputs), with the operator row
(branch 03) supplying the ground-truth structure the representation is graded against.

## 8. Taxonomy contribution: axes of distinction this branch introduces

This branch contributes three orthogonal axes that the other branches inherit:

1. **Amortization-over-what.** The variable held fixed at train time and varied for free
   at test time: inputs only (surrogate/emulator), a family parameter (operator),
   observations (amortized posterior), or the entire data distribution (foundation
   model). This is the primary discriminator; most confusion in the literature is a
   confusion of this column.
2. **Forward vs inverse target.** Whether $g_\phi$ approximates a forward map
   $\theta \mapsto x$ / $u \mapsto y$ (surrogate, emulator, operator) or an inverse
   distribution $x \mapsto p(\theta\mid x)$ (amortized posterior). Foundation models are
   neither until a head is attached.
3. **Reuse mechanism / objective coupling.** Re-fit per task (surrogate, emulator),
   evaluate within a family (operator), single forward pass per observation (amortized
   posterior), or probe / in-context after a label-free pretraining objective (foundation
   model). The objective and the reuse mechanism are coupled: a self-supervised objective
   is what *licenses* representation reuse, which is the operational definition of
   "foundational".

The demo in this directory makes axis 1 visible numerically: a single-task surrogate fit
to one function (amortized over inputs only) is contrasted against a foundation-style
representation pretrained on a family of functions and then adapted to an unseen function
from a handful of points (amortized over the family). The few-shot gap on the new task is
the quantity the axis predicts.

## References (verified)

- [Bommasani2021] R. Bommasani et al., "On the Opportunities and Risks of Foundation Models," arXiv:2108.07258 (2021).
- [Cranmer2020] K. Cranmer, J. Brehmer, G. Louppe, "The frontier of simulation-based inference," PNAS 117(48):30055–30062 (2020), arXiv:1911.01429.
- [Garnelo2018] M. Garnelo et al., "Conditional Neural Processes," ICML 2018 (PMLR 80), arXiv:1807.01613.
- [Greenberg2019] D. Greenberg, M. Nonnenmacher, J. Macke, "Automatic Posterior Transformation for Likelihood-Free Inference," ICML 2019 (PMLR 97:2404–2414), arXiv:1905.07488.
- [He2021] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, "Masked Autoencoders Are Scalable Vision Learners," arXiv:2111.06377 (2021).
- [Hermans2020] J. Hermans, V. Begy, G. Louppe, "Likelihood-free MCMC with Amortized Approximate Ratio Estimators," ICML 2020 (PMLR 119:4239–4248), arXiv:1903.04057.
- [Kennedy2001] M. C. Kennedy, A. O'Hagan, "Bayesian calibration of computer models," J. R. Stat. Soc. B 63(3):425–464 (2001).
- [Li2020] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," arXiv:2010.08895 (2020); ICLR 2021.
- [Lu2021] L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," Nature Machine Intelligence 3:218–229 (2021); arXiv:1910.03193 (2019).
- [Papamakarios2016] G. Papamakarios, I. Murray, "Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation," NeurIPS 2016, arXiv:1605.06376.
- [Papamakarios2017] G. Papamakarios, T. Pavlakou, I. Murray, "Masked Autoregressive Flow for Density Estimation," NeurIPS 2017, arXiv:1705.07057.
- [Devlin2018] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL 2019, arXiv:1810.04805.
