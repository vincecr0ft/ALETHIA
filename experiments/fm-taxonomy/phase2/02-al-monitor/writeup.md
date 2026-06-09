# An active-learning loop with a monitor at the centre — the mathematics of the learner

Phase-2 deliverable for axis IV (the learner) of the ALETHIA FM taxonomy.
Siblings: branch `04-active-learning/` (the axis-IV formalism and its first
demo). This build is monitor-first: the centrepiece is not the acquisition rule
but the **monitor** that, every round and for every acquisition function and
regime, computes and logs the learner's full mathematical state. The runnable
artifact is `al_monitor.py`; its primary output is the per-round CSV
`monitor_log.csv`.

The claim the run makes visible: posterior contraction and estimator error are
the *same* signal in a well-conditioned embedding and *diverge* on a collinear
ridge, and the monitor detects which regime it is in **live**, from the
embedding geometry alone, as the redundancy diagnostic
`top-decile mean |cos_Ainv|` crosses from spread (~0.5) to collinear (~0.85+).
This reproduces the E1→E2 separation of branch 04 at greater depth: three
regimes spanning the `cos_Ainv` axis, five acquisition functions head-to-head,
and full per-round trajectories rather than terminal z-scores alone.

---

## 1. The learner

The learner is the closed-form Bayesian linear-Gaussian head that ALETHIA's
frozen IntentionFM exposes. A pretrained backbone supplies a fixed feature map
$\psi(x)\in\mathbb R^{D}$, and the readout is

$$
y=\psi(x)^\top w+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma_y^2),
\qquad w\sim\mathcal N(0,\tau^2 I).
$$

After observing a context $\{(x_i,y_i)\}_{i=1}^{K}$, the posterior over $w$ is
Gaussian with **precision** (the design / information matrix)

$$
A=\sigma_y^{-2}\sum_{i=1}^{K}\psi(x_i)\psi(x_i)^\top+\tau^{-2}I,
\qquad \operatorname{Cov}(w)=\sigma_y^2 A^{-1},\quad
w_{\rm MAP}=A^{-1}\Big(\sigma_y^{-2}\textstyle\sum_i\psi(x_i)\,y_i\Big).
$$

This is exactly `LinearGaussianHead.A_inv_and_w`, mirroring
`IntentionFM.A_inv_and_w` (referenced in
`al-phoenix-studies/_common.py`). $A$ is the Fisher information of the
Gaussian-linear model; with a *frozen* backbone the only epistemic uncertainty
the loop can act on lives in this head, so every acquisition score and every
monitored quantity is a function of $A^{-1}$ and $\psi(\cdot)$ — a property of
the embedding geometry, not of the labels.

## 2. The Sherman–Morrison rank-one update

Adding one labelled point $x$ updates the precision by a rank-one term,
$A\mapsto A+\sigma_y^{-2}\psi\psi^\top$ with $\psi=\psi(x)$. The
Sherman–Morrison identity gives the inverse in closed form, with no refactor:

$$
A^{-1}\mapsto A^{-1}-\frac{A^{-1}\psi\,\psi^\top A^{-1}}{\sigma_y^2+\psi^\top A^{-1}\psi}.
$$

Writing the **leverage** of a candidate as $\mathrm{lev}(x)=\psi^\top A^{-1}\psi$
(in the rescaled convention $A\!\to\!A/\sigma_y^{-2}$ used by the acquisition
code, $\mathrm{lev}=\psi^\top A^{-1}\psi$ and the denominator is $1+\mathrm{lev}$),
every acquisition score below is one evaluation of this update. The cost is
$O(D^2)$ per candidate per cycle — the cheap, amortised end of Bayesian optimal
experimental design that a frozen FM head buys you. In `al_monitor.py` the loop
refits $A^{-1}$ each round (numerically identical, marginally simpler to read);
the monitored `dlogdet_A` confirms the update is rank-one.

## 3. The acquisition scores (mirrored from the ALETHIA modules)

All five are derived from the same rank-one update; each is mirrored faithfully
from the cited file.

**`leverage` — BALD / Fisher / D-optimal.** The expected information gain in the
parameters from one label is, in the Gaussian-linear regime, the rank-one
D-optimal gain

$$
a_{\rm lev}(x)=\tfrac12\log\!\big(1+\psi(x)^\top A^{-1}\psi(x)\big).
$$

Mirrors `_common.py:ig_per_candidate` (`0.5*log1p(psi@A_inv@psi)`). In this
regime BALD, Fisher-information acquisition, and Bayesian D-optimality are the
*same* score.

**`param_a` — A-optimal on one direction $\hat p_a$.** A-optimality minimises the
posterior variance along the targeted direction. The log-reduction in
$\operatorname{Var}(\hat p_a^\top w)$ from a rank-one update is

$$
\Delta H_a(x)=-\tfrac12\log\!\Big(1-\frac{\sigma_y^2\,u_a(x)^2}{(1+\mathrm{lev}(x))\,\Sigma_{aa}}\Big),
\quad u_a(x)=\hat p_a^\top A^{-1}\psi(x),\;\;\Sigma_{aa}=\sigma_y^2\,\hat p_a^\top A^{-1}\hat p_a.
$$

Mirrors `acquisition.py:param_epig_a_acquire`.

**`param_d` — D-optimal on a resolved subspace $P$.** With $\Sigma=\sigma_y^2 P
A^{-1}P^\top$ the covariance of the resolved $\tilde c$-subspace and
$u=PA^{-1}\psi$,

$$
\Delta H_D(x)=-\tfrac12\log\!\Big(1-\frac{\sigma_y^2\,u^\top\Sigma^{-1}u}{1+\mathrm{lev}(x)}\Big),
$$

computed via a single triangular solve against the Cholesky factor of $\Sigma$
(no explicit inverse). Mirrors `acquisition.py:param_epig_d_acquire`.

**`epig` — predictive EPIG over a target set $\{x_T\}$.** Targets the
*predictions* on a held-out high-$x$ band rather than the parameters:

$$
a_{\rm EPIG}(x)=\frac{1}{|T|}\sum_{T}\tfrac12\log\frac{\mathrm{lev}_T}{\mathrm{lev}_T-K_{TP}^2/(1+\mathrm{lev}_P)},
\quad K_{TP}=\psi(x_T)^\top A^{-1}\psi(x),
$$

the mean over targets of the predictive-variance reduction from a rank-one update
at the candidate. Mirrors `acquisition.py:epig_acquire_m`. EPIG and
A-optimal-on-a-direction are the prediction-space and parameter-space versions of
the same rank-one contraction.

**`random` — uniform pool draw.** The baseline.

## 4. The monitored quantities

Each round the monitor records, for the current context:

| Column | Definition | Mirrors |
|---|---|---|
| `var_a` | $\operatorname{Var}(c_a)=\sigma_y^2\,\hat p_a^\top A^{-1}\hat p_a$ — **target contraction** | `_common.py:sigma_ctilde` |
| `err_a` | $\lvert\hat p_a^\top w_{\rm MAP}-\hat p_a^\top w_{\rm true}\rvert$ — **estimator (MLE/MAP) error** on the target direction | REPORT.md `mle_err` |
| `pred_rmse` | RMSE of $\psi(x)^\top w_{\rm MAP}$ vs truth on the eval band | — |
| `logdet_A`, `dlogdet_A` | $\log\det A$ and its per-round increment — the total information and the rank-one D-optimal gain actually realised | §2 |
| `cos_ainv_top` | top-decile mean $\lvert\cos_{A^{-1}}\rvert$ — **redundancy diagnostic** | `demo.py:cos_ainv_top_decile`, REPORT.md Stage B |
| `cv_ig` | $\mathrm{CV}$ of the leverage scores over the pool — the AL_separation published gate | REPORT.md Stage B |
| `gap` | `err_a - var_a` — the contraction-vs-MLE gap as a first-class signal | REPORT.md §Headline |
| `acq_score` | the max acquisition score actually selected this round | §3 |

The redundancy diagnostic is the operative one. Take the top-decile-leverage
candidates and form their Gram matrix in the $A^{-1}$ metric,
$G_{ij}=\psi_i^\top A^{-1}\psi_j$; the mean off-diagonal
$\lvert G_{ij}\rvert/\sqrt{G_{ii}G_{jj}}$ is `cos_ainv_top`. When it sits near 1
the high-information candidates are collinear in the metric that defines
information gain: every "high-IG" pick reinforces the *same* posterior direction,
so the marginal gain over a random draw of that one direction is illusory. This
is the BatchBALD redundancy signature (Kirsch et al. 2019), measured live.

## 5. Why contraction and error coincide in E1 and diverge in E2

Project onto the target direction. With $\beta=\hat p_a^\top w$, the rank-one
update both shrinks the posterior variance $\operatorname{Var}(\beta)$
(`var_a`) **and** moves the MAP estimate $\hat\beta=\hat p_a^\top w_{\rm MAP}$.
The estimator error decomposes as variance plus squared bias,
$\mathbb E[(\hat\beta-\beta_\star)^2]=\operatorname{Var}(\beta)+\mathrm{bias}^2$,
where the bias is the prior pull along directions the data have not resolved.

**E1 (well-conditioned embedding, `cos_Ainv`≈0.5).** The feature map (random
Fourier features) spreads leverage across the pool, so successive high-IG picks
load on *different* eigen-directions of $A$. The target direction $\hat p_a$ is
data-identifiable: information acquired along it removes both variance and bias
together. Contraction and error move in lockstep — both z-scores go strongly
negative for the targeted methods.

**E2 (collinear ridge, `cos_Ainv`≈0.85+).** The feature map suppresses all but
one dominant coordinate (`leak=0.02`), so the high-leverage candidates are
collinear in the $A^{-1}$ metric. Greedy parameter-information acquisition
exhausts that one ridge: $A$ still grows along it, so `var_a` keeps shrinking
(the loop reports confident contraction), but the target direction $\hat p_a$ has
a component *off* the ridge that is prior-dominated and never resolved. The bias
on that component does not fall, so `err_a` stalls while `var_a` drops — the
posterior shrinks around a biased centre. `gap = err_a - var_a` opens up; the
contraction-vs-MLE divergence is exactly the bias term that contraction is blind
to. This is the documented ALETHIA finding (`al-phoenix-studies/REPORT.md`):
targeted acquisition beats random in contraction by tens of $\sigma$ yet makes
the MLE on the targeted direction *worse*, because the IG-rich direction is not
the identifiable one — Axis-A's parameter side and prediction side coming apart.

The intermediate regime (`leak=0.12`) sits between the two, confirming the gap
is monotone in the embedding's collinearity rather than a two-point artifact.

## 6. What the monitor is therefore measuring

The monitor is not scoring the acquisition rule; it is reading the **geometry of
the frozen embedding** through the head's $A^{-1}$. `cos_Ainv` is the binding
constraint: when it is low, any sensible acquisition rule works and contraction
predicts error; when it is high, no marginal acquisition rule can separate the
collinear high-IG candidates, and contraction stops predicting error. The
operative gate is therefore `top-decile mean |cos_Ainv|` (a property of
$\psi(\cdot)$), **not** `CV(IG)` — which `cv_ig` shows passing in every regime as
structural slack, matching REPORT.md Stage B. The first-class signal a monitoring
program should surface is the `gap` between contraction and estimator error,
because a greedy information-gain learner on a collinear ridge contracts hardest
precisely while it is fooling itself. The directional, parameter-aware methods
(`param_a`, `param_d`) survive longest on the ridge because they down-weight the
exhausted direction; the non-directional `leverage`/`epig` over-acquire it — the
Stage-B redundancy story, now visible per round.

---

## References

- N. Houlsby, F. Huszár, Z. Ghahramani, M. Lengyel (2011). *Bayesian Active
  Learning for Classification and Preference Learning.* arXiv:1112.5745. (BALD)
- K. Chaloner, I. Verdinelli (1995). *Bayesian Experimental Design: A Review.*
  Statistical Science 10(3):273–304.
- A. Kirsch, J. van Amersfoort, Y. Gal (2019). *BatchBALD: Efficient and Diverse
  Batch Acquisition for Deep Bayesian Active Learning.* NeurIPS 2019.
  arXiv:1906.08158.
- F. Bickford Smith et al. (2023). *Prediction-Oriented Bayesian Active
  Learning.* AISTATS 2023. arXiv:2304.08151. (EPIG)

ALETHIA code mirrored: `experiments/al-phoenix-studies/_common.py`
(`ig_per_candidate`, `sigma_ctilde`); `modules/surrogate/intention/acquisition.py`
(`param_epig_a_acquire`, `param_epig_d_acquire`, `epig_acquire_m`);
`experiments/fm-taxonomy/04-active-learning/demo.py` (`cos_ainv_top_decile`).
Findings reference `experiments/al-phoenix-studies/REPORT.md`.
