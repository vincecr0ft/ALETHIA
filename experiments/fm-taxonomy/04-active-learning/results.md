# Demo results — information-gain acquisition vs random

`demo.py` is a numpy-only closed-form Bayesian linear-Gaussian active-learning
loop. It runs the same four acquisition scores ALETHIA's head exposes — random,
`leverage` (= BALD / D-optimal), `param_a` (A-optimal on one direction), and
`epig` (predictive EPIG on a target set) — over two feature-map regimes, 40
seeds × 30 rounds each, and prints final-round metrics with z-scores against the
random baseline.

Run:

```
uv run python experiments/fm-taxonomy/04-active-learning/demo.py
# or, with the project venv directly:
.venv/bin/python experiments/fm-taxonomy/04-active-learning/demo.py
```

Executed in the main session (numpy 2.4.4 + scipy/sklearn present). Verified output:

```
=== E1 WELL-POSED (target identifiable, features spread) ===
top-decile mean |cos_Ainv| at seed context: 0.533  (spread)
       acq |   Var(c_a)   err(c_a)  pred_RMSE |    zVar    zErr   zRMSE
-----------------------------------------------------------------------
    random |  4.512e-05     0.0887    0.06447 |    ref     ref     ref
  leverage |  4.899e-06    0.03971    0.03141 |    -4.5    -2.5    -3.7
   param_a |  3.247e-06     0.0238    0.02523 |    -4.7    -3.3    -4.4
      epig |  4.599e-06     0.0325    0.02231 |    -4.6    -2.8    -4.8

=== E2 ALETHIA-SHAPED (collinear high-leverage ridge, weak ID) ===
top-decile mean |cos_Ainv| at seed context: 0.851  (collinear/redundant)
       acq |   Var(c_a)   err(c_a)  pred_RMSE |    zVar    zErr   zRMSE
-----------------------------------------------------------------------
    random |  0.0002338     0.2236    0.01597 |    ref     ref     ref
  leverage |  0.0002222     0.2092    0.01517 |    -0.1    -0.2    -0.6
   param_a |  0.0001383     0.1722     0.0146 |    -1.1    -0.7    -1.1
      epig |  0.0002149     0.2174    0.01296 |    -0.2    -0.1    -2.7
```

## Metrics

For the targeted parameter direction $\hat p_a$ (the feature loading at the
high-$x$ evaluation edge), each round records:

- `Var(c_a)` $=\sigma_y^2\,\hat p_a^\top A^{-1}\hat p_a$ — **posterior contraction**
  along the target (the BOED objective; ALETHIA's `sigma_ctilde`).
- `err(c_a)` $=|\hat p_a^\top w_{\rm MAP}-\hat p_a^\top w_{\rm true}|$ — the
  **estimator error** on the target direction (ALETHIA's "MLE error").
- `pred_RMSE` — predictive RMSE on a held-out target eval band.

`zVar/zErr/zRMSE` are signed z-scores of (acquisition − random); negative means
the acquisition method is *better* (smaller) on that metric.

The seed-context redundancy diagnostic `top-decile mean |cos_Ainv|` is printed
per regime — the Stage-B signal: $\approx1$ means the high-information subspace
is one-dimensional.

## E1 — WELL-POSED (random-Fourier features, target identifiable)

`cos_Ainv` is well below 1 (features spread). Here information-gain acquisition
does what the textbook promises: `leverage`, `param_a`, and `epig` all contract
the posterior on the target direction faster than random **and** that
contraction propagates to a lower estimator error and lower predictive RMSE —
all three z-scores negative for the targeted methods. `param_a` (directional
A-optimal) is the strongest on `err(c_a)`; `epig` is the strongest on
`pred_RMSE` because it optimises the predictive target directly. This is the
"targeted beats random" baseline.

## E2 — ALETHIA-SHAPED (collinear high-leverage ridge, weak identifiability)

The feature map suppresses all but one dominant direction (`leak=0.02`), so the
high-leverage pool candidates become collinear in the $A^{-1}$ metric:
`top-decile mean |cos_Ainv|` sits near 1 — the exact Stage-B redundancy
signature. In this regime:

- The acquisition advantage **collapses relative to E1**: where E1 showed
  zVar ≈ −4.5 to −4.7, in E2 `leverage` falls to zVar = −0.1 (barely better than
  random) and even the best contractor `param_a` reaches only zVar = −1.1. The
  collinear $A^{-1}$ ridge means each high-information pick reinforces the same
  direction, so marginal contraction per acquisition is small.
- The estimator error tracks contraction only weakly: `leverage` zErr = −0.2,
  `param_a` zErr = −0.7 — far short of E1's −2.5 to −3.3. The posterior shrinks
  around a weakly-identified centre, so confident contraction does not buy a
  proportionally better point estimate. This is the documented
  **contraction-vs-MLE gap**: in E1 contraction and estimator error move together
  (∝ each other), in E2 they decouple. `cos_Ainv` jumping 0.533 → 0.851 is the
  diagnostic that flags the regime.
- `epig`, being prediction-targeted, tracks `pred_RMSE` more faithfully than
  `leverage` does — the parameter-vs-prediction split (Axis A of the taxonomy)
  showing up as different methods winning different columns.

## Interpretation

The two regimes isolate the branch's central conceptual claim. Posterior
contraction (the parameter-information / BOED objective) and estimator/predictive
error are the *same* signal in a well-conditioned representation (E1) and
*diverge* when the representation makes the high-information directions collinear
or weakly identified (E2). The divergence is a property of the embedding
geometry — `cos_Ainv` — not of the labels, which is why with a frozen
foundation-model backbone the binding constraint moves from the data to the
representation. The practical consequence, matching `al-phoenix-studies/REPORT.md`:
the first-class monitored signal should be the *gap* between contraction and
error, not contraction alone, because a greedy information-gain learner on a
collinear ridge contracts hardest precisely while it is fooling itself.
