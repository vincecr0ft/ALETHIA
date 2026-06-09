# Where active learning beats random: the rung-0→2 finding

Built bit-by-bit from the rung-0 single-coefficient seed. All scripts are
numpy-only except rung 2, which calls the real `modules.analytic_smeft` oracle.
Run rung 2 with `PYTHONPATH=/home/vince/ALETHIA python3 rung2_real_oracle.py`.

## The law

> Active learning beats random only when the bottleneck (worst-constrained)
> direction is **constrainable, but only through a minority of designs that
> uniform sampling under-covers** — AND the acquisition objective actually
> targets that direction. Fail either clause and random ties or wins.

## The ladder

| rung | setup | result |
|---|---|---|
| 0 | 1 coefficient, exact 3-sample morphing, Fisher in the tail | exact recovery 2.8e-14; 52% of Fisher in the high-mass third |
| 1 | n=2, vertex direction **degenerate** on mass spectrum | leverage (info-gain) **loses** to random on the near-SM target (z_err +0.6, z_var +20); EPIG beats random (z_err −3.3). "Acquiring in the wrong direction." |
| 1b | n=2 toy, vertex constrainable only via a **rare** angular channel | AL routes to the scarce channel; beats random **z = −6.3** on the bottleneck |
| 2 | rung 1b on the **real SMEFT oracle**, floated normalisation nuisance, Poisson variances | see below |

## Rung 2 (real oracle, Λ = 1 TeV)

Coefficients (c_lq³ four-fermion, c_φq³ vertex) + floated log-normalisation α.
The vertex gives a flat ~2.5% rate shift → degenerate with α on the mass
spectrum; A_FB breaks it (oracle-confirmed). 40 mass cells, 6 A_FB cells
(A_FB = 13% of pool). Marginal Cramér–Rao σ, 30 seeds:

```
=== mass-only ===           sigma(c_lq3)   sigma(c_phiq3)   #AFB
  random                       8.02e-02        5.21e+00       0.0
  dopt                         4.12e-02        2.83e+00       0.0
  aopt                         3.52e-02        2.38e+00       0.0   <- vertex lost for all

=== mass+AFB ===
  random                       8.44e-02        5.46e+00       3.4
  dopt  (log-det/BALD/lev.)    4.11e-02        2.82e+00       0.4   <- ignores A_FB, ties mass-only
  aopt  (coeff-variance)       2.86e-02        1.74e+00      23.0   <- finds A_FB, 3x better than random
```

## Two-layer conclusion

1. **Observable space.** Without A_FB the vertex direction is unconstrained
   (σ ≈ 2.4–5.5) regardless of acquisition. The degeneracy-breaking channel must
   be in the design pool.
2. **Acquisition objective.** Even with A_FB present, D-optimal / BALD / leverage
   — the paper's "working-point curvature acquisition" — spends almost nothing on
   A_FB (0.4 cells) because the log-det is dominated by the bright c_lq³ tail
   (rate grows 460× by m=2.4 TeV). It ties the mass-only result. Only A-optimal,
   which targets the coefficient variance (the worst-constrained direction),
   routes budget to A_FB (23 cells) and beats random 3× on the vertex.

"Random always wins" was the symptom of acquiring **D-optimal/leverage** (chase
the bright direction) on an observable space that **excluded the degeneracy
breaker**. Fix both — put A_FB in the pool and switch to a coefficient-targeted
(A-optimal / parameter-EPIG) objective — and AL wins. The ManifoldInformer's
universal quality is orthogonal: the bottleneck is the acquisition objective and
the observable space, not the representation, which is why it should be the
control in the paper, not the headline.

## Rung 2b (adaptive, LEARNED sensitivities) — the win does not survive

Rung 2 used oracle-computed gradients (known-simulator optimal design). Rung 2b
makes the agent LEARN the sensitivities from a noisy morphing surrogate and
acquire on the learned J_hat (decisions under the learned model, outcome scored
on the true Fisher). `rung2b_adaptive_learned.py`.

Validation: at zero surrogate noise, A-optimal exploit recovers rung 2 exactly
(σ(c_φq³)=1.76, 22.9 A_FB cells). Then a query-noise scan:

```
exploit (A-optimal on learned J_hat) vs A_FB query precision:
 rel_noise  abs_noise |  sigma(c_phiq3)   #AFB used
       0.0        0.0 |      1.76e+00        22.9   <- exact: finds A_FB
     0.002     0.0001 |      4.77e+00         0.2   <- 0.2% noise: blind to A_FB
     0.005     0.0002 |      4.47e+00         0.2
     0.01      0.0005 |      4.33e+00         0.2
     0.02      0.001  |      4.47e+00         0.2
     0.05      0.002  |      5.98e+00         0.2
```

**The cliff is at <0.2% surrogate precision.** The vertex A_FB signature is a
~0.002 shift on a ~0.35 asymmetry; any surrogate that does not pin dA_FB/dc_φq³
to better than that signal cannot see A_FB's value, so A-optimal on the learned
J_hat stops buying A_FB (0.2 cells) and collapses to ~random on the vertex
(σ ≈ 4.3–6.0, vs the exact-gradient 1.76). A learned representation (the
ManifoldInformer latent) does not resolve a 0.002 sensitivity, so learned-
representation AL on this direction reproduces "random wins."

## Full conclusion — three clauses, all required

AL beats random on the vertex direction only if **all three** hold:
1. **Observable space** contains the degeneracy-breaker (A_FB present). Mass-only → vertex lost (rung 2).
2. **Acquisition objective** targets the weak direction (A-optimal / parameter-EPIG, not log-det / BALD / leverage / curvature). D-optimal ignores A_FB (rung 2).
3. **Sensitivity precision** beats the degeneracy-breaking signal (~0.002). Exact morphing gradients clear it; a learned surrogate at ≥0.2% noise does not (rung 2b).

This is the precise anatomy of "random always wins": the historical loops used a
learned-latent sensitivity (clause 3 fails) and/or a contraction/leverage
objective (clause 2 fails). The constructive routes: **(a)** acquire on the
analytic morphing gradients (exact, clause 3 satisfied — you already have the
oracle); **(b)** pick an EFT direction whose degeneracy-breaking signal is large
enough to learn (e.g. a four-fermion chirality contrast, not the vertex); or
**(c)** spend enough reference sims to pin the template below the signal floor
(scan in progress: does N reference sims at fixed noise rescue the A_FB win?).
