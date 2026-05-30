# ALETHEIA — investigate-or-drop spec (detailed)

Handoff for Claude Code. Each item is promoted to a paper contribution by a computation
that confirms it, or cut. Nothing stays as a reported negative. Every section gives the
claim, the derivation the code must implement, the exact estimands, and the numeric
threshold that decides keep vs cut.

Read the Conventions block first. Most agent errors here come from a coupling-sign
convention, a noise-model mismatch between the Fisher and the data generator, or the
ridge-induced bias in the efficiency check. Those three are called out explicitly in the
Pitfalls section at the end; read it before writing any code.

Repo anchors: `smeft_surrogate/` (oracle, `IntentionFM`, `IntentionFMInContext`, acquisition
strategies, conformal calibrator), `experiments/full-chain-run/`,
`experiments/intention-vs-deepsets/`. CPU as now.

---

## Conventions and notation (read once, used everywhere)

- `m_ll` invariant dilepton mass; `ŝ = m_ll²` partonic CM energy squared at LO with zero
  dilepton p_T. `M_ref = 0.2 TeV`. Feature input scaling `log(m_ll/M_ref)`.
- `θ*` = polar angle of `ℓ⁻` in the Collins-Soper frame. At LO (p_T = 0) the CS frame is the
  dilepton rest frame with z-axis along the boost direction.
- Wilson vector `c = (c_Hq^(1), c_Hq^(3), c_lq^(1), c_lq^(3))` in the order used by the
  oracle. `Λ` the EFT scale. Confirm the index order against the oracle before indexing.
- Rate observable (current): `μ(c, m_ll) = σ(c, m_ll) / σ_SM(m_ll)`, the cross-section ratio.
- Morphing decomposition (current oracle, exact at fixed c):
  `σ(c, m_ll) = σ_SM(m_ll) + Σ_i A_i(m_ll) c_i + Σ_{i≤j} B_ij(m_ll) c_i c_j`.
- Feature map `ψ_θ: R^d → R^D`, `D = 16`. Current `d = 1` (input `log(m_ll/M_ref)`).
- Design matrix `A = Ψᵀ Ψ + α I`, `Ψ` stacks `ψ_θ(M_ctx,k)` over the K context points,
  `α = 10⁻³`. Ridge weight `w = A⁻¹ Ψᵀ Y_ctx`. Prediction `Ŷ_q = ψ_θ(M_q)ᵀ w`.
- Leverage `lev(q) = ψ_θ(M_q)ᵀ A⁻¹ ψ_θ(M_q)`. Predictive variance `σ_y²(1 + lev(q))`.
- Disclosure probe: linear map `ĉ = W w + b`, `W ∈ R^{4×16}`, fit by least squares on
  training-box scenarios (`w` evaluated per scenario from a fresh K=12 context).
- Fisher eigenbasis: `F = V diag(λ) Vᵀ`, rotated coefficients `c̃ = Vᵀ c`. Current eigenvalues
  `{19.6, 8.7, 3.0e-4, 1.3e-4}`. Directions 1,2 are four-fermion combinations; 3,4 are vertex
  combinations.
- `σ_y²` is the noise variance on `Y` (the ratio), estimated from context residuals. The
  noise is additive on `Y`, which fixes the Fisher form below. Do not silently switch to
  a relative-noise model.

---

## Three project claims the results must make legible (output structuring, not paper prose)

The four core investigations correct local results. The paper additionally rests on three
framing claims. The agent does not write the framing, but it must produce a named result
artifact behind each claim so the writeup has evidence to cite rather than assertion. The
paper prose is out of scope here; what follows are the result objects.

- Claim A — the system compares different foundation-model readings of the same `(M_ctx,
  Y_ctx)` context, and the four analytic properties (predictive variance, information gain,
  sequential update, Eckart-Young sensitivity) are not generic across them. Evidence vehicle:
  INV-6 (architecture-vs-properties table) plus the surviving scaled-JEPA point from INV-5
  and the existing Section 6 / Appendix C numbers.
- Claim B — the drift detectors are a scientific instrument: they diagnose where the model
  has failed to carry the physics and select the matching intervention. Not loop quality
  control. Evidence vehicle: the INV-3 drift decision trace (and the same trace applied
  retroactively to the Section 4 1D loop).
- Claim C — the c-agnostic head performs amortized neural simulation-based inference: context
  in, calibrated Wilson posterior out. Evidence vehicle: the INV-2 SBI posterior object plus
  INV-4 coverage.

Each evidence vehicle below carries an "Output artifact" subsection marked with the claim it
serves. Produce those artifacts as first-class outputs, not as logging.

---

## INV-1 — Angular observable extension to the analytic oracle

### Purpose
The current observable is the angle-integrated rate. Integrating over `cos θ*` annihilates
the antisymmetric (forward-backward) part of the distribution, which carries the chiral
structure. On the rate alone the two vertex operators `c_Hq^(1,3)` produce modifications
degenerate with the SM normalisation, and the two four-fermion operators produce tails
differing only by up/down luminosity weights. The Fisher eigenvalues 3e-4 and 1.3e-4 on the
vertex directions are the quantitative statement of this. The forward-backward asymmetry
breaks the degeneracy (Boughezal, Huang, Petriello arXiv:2303.08257). INV-2 and INV-3 both
require this observable.

### Derivation the code must implement
At tree level, massless fermions, `q q̄ → ℓ⁺ℓ⁻` via s-channel γ/Z plus dim-6 contact. Define
chiral amplitudes `M_ij(ŝ)` for quark chirality `i ∈ {L,R}` and lepton chirality
`j ∈ {L,R}`. The polar-angle dependence for massless 2→2 is fixed by helicity conservation:

```
dσ̂/dcosθ*  ∝  |M_LL|²(1+cosθ*)² + |M_LR|²(1−cosθ*)² + |M_RL|²(1−cosθ*)² + |M_RR|²(1+cosθ*)²
```

Expanding `(1 ± cosθ*)² = (1 + cos²θ*) ± 2cosθ*` gives the symmetric/antisymmetric split:

```
dσ̂/dcosθ*  ∝  (1 + cos²θ*) · S(c, ŝ)  +  2cosθ* · D(c, ŝ)

S(c, ŝ) = |M_LL|² + |M_LR|² + |M_RL|² + |M_RR|²        (symmetric; survives ∫dcosθ*)
D(c, ŝ) = |M_LL|² − |M_LR|² − |M_RL|² + |M_RR|²        (antisymmetric; ∫dcosθ* = 0)
```

Angular integrals: `∫_{-1}^{1}(1+cos²θ*)dcosθ* = 8/3`, `∫_{-1}^{1} 2cosθ* dcosθ* = 0`. So the
angle-integrated rate is `(8/3) S` and the antisymmetric part is invisible to it. The
forward-backward asymmetry and the A4 angular coefficient are

```
A_FB = (3/4)(D / S),     A4 = 2 D / S.
```

Any `cos θ*` binning is a fixed linear combination of `S` and `D`. For a bin spanning
`[u_lo, u_hi]` in `cos θ*`:

```
observable(bin) = w_S(bin) · S(c, ŝ) + w_D(bin) · D(c, ŝ),
w_S(bin) = ∫_{u_lo}^{u_hi}(1+u²)du,   w_D(bin) = ∫_{u_lo}^{u_hi} 2u du.
```

Compute `S` and `D` once per `(c, m_ll)`; all bins follow. Start with 4 symmetric bins.

### Chiral amplitudes (exact SM part, structural SMEFT part)
SM NC chiral couplings, normalisation `(g/cosθ_W) Z_μ f̄ γ^μ (g^f_L P_L + g^f_R P_R) f`:

```
g^f_L = T3^f − Q_f s_W²,   g^f_R = − Q_f s_W²,   s_W² = sin²θ_W.
up:    g^u_L = 1/2 − (2/3)s_W²,   g^u_R = −(2/3)s_W²
down:  g^d_L = −1/2 + (1/3)s_W²,  g^d_R = (1/3)s_W²
lepton:g^ℓ_L = −1/2 + s_W²,       g^ℓ_R = s_W²
```

Per-channel amplitude (schematic normalisation; match the oracle's existing constant):

```
M_ij(ŝ) = e² Q_q Q_ℓ / ŝ
        + (g_Z²/(ŝ − m_Z² + i m_Z Γ_Z)) · gq_i(c) · gℓ_j
        + contact_ij(c) / Λ²
```

SMEFT enters two ways, both of which the rate oracle already encodes for `S`; reuse its
coefficients:

1. Vertex operators `c_Hq^(1,3)` shift only the LEFT-handed quark couplings,
   proportional to `v²/Λ²`, in the SU(2) combinations
   `gq_L(up)  = g^u_L + δ_up,   δ_up  ∝ −(c_Hq^(1) − c_Hq^(3)) v²/Λ²`
   `gq_L(down)= g^d_L + δ_down, δ_down∝ −(c_Hq^(1) + c_Hq^(3)) v²/Λ²`.
   The right-handed quark couplings and the lepton couplings are untouched by `c_Hq^(1,3)`.
   The exact proportionality constants and overall sign are convention-dependent — take them
   from the same source the rate oracle uses (Greljo, Marzocca arXiv:1704.09015 give the
   modified high-p_T couplings explicitly). Do not hand-pick the constant; lift it and verify
   by the regression test below.

2. Four-fermion operators `c_lq^(1,3)` add a contact term to the LL channel only
   (left quark doublet × left lepton doublet), zero in LR/RL/RR. The SU(2) structure of
   `O_lq^(3)` gives opposite signs for up and down:
   `contact_LL(up)   ∝ (c_lq^(1) − c_lq^(3)) · ŝ/Λ²   (relative to SM, grows as ŝ)`
   `contact_LL(down) ∝ (c_lq^(1) + c_lq^(3)) · ŝ/Λ²`.
   Again lift the exact normalisation from the oracle source and verify.

### Hadronic level and the forward-backward dilution
At a pp collider the quark direction is unknown per event. Use the Collins-Soper frame:
`cos θ*_CS` defined with z-axis along the sign of the dilepton longitudinal momentum. The
antisymmetric observable `D` survives only because valence-quark PDFs exceed sea-antiquark
PDFs, so it is diluted relative to the parton-level value. Implement by computing both
quark-direction assignments per `m_ll` and combining with the PDF luminosity difference:

```
D_had(m_ll) ∝ ∫ dy [ f_q(x1)f_q̄(x2) − f_q̄(x1)f_q(x2) ] · D_parton(ŝ)   (summed over flavours)
S_had(m_ll) ∝ ∫ dy [ f_q(x1)f_q̄(x2) + f_q̄(x1)f_q(x2) ] · S_parton(ŝ)
```

with `x1,2 = (m_ll/√s) e^{±y}`. CT18NNLO via LHAPDF, identical to the rate oracle. Do not
re-derive the dilution analytically; the MadGraph cross-check below is the safety net.

### Recommended implementation (reuse, do not rebuild)
The morphing fit in `c` is unchanged. The oracle already obtains `A_i(m_ll), B_ij(m_ll)` for
the rate by evaluating the cross section at a set of `c` points and solving a linear system.
Apply the identical procedure to the antisymmetric phase-space integral to obtain
`A_i^D(m_ll), B_ij^D(m_ll)`. Concretely: where the rate integrand carries angular weight that
reduces to `S`, the antisymmetric integrand carries the projector `sign(cos θ*_CS)` and
reduces to `D`. The only change is the angular weight inside the existing phase-space
integral. Then:

- `S(c, m_ll) = σ_SM + Σ A_i^S c_i + Σ B_ij^S c_i c_j` (the existing rate, up to the 8/3).
- `D(c, m_ll) = Σ A_i^D c_i + Σ B_ij^D c_i c_j` (D vanishes at c=0 only if the SM A_FB is
  separately included; keep the SM A_FB as the `D_SM(m_ll)` constant term — it is nonzero
  from Z/γ interference. Verify `D_SM` reproduces the SM A_FB(m_ll)).

Feature map: set `d = 2`, input `(log(m_ll/M_ref), cos θ*_bin_center)` or feed the bin index
through a small embedding. The closed-form layer (ridge, Sherman-Morrison, leverage, EPIG,
conformal) is abscissa-agnostic and needs no change beyond `d`. Confirm by diffing: nothing
in `IntentionFMInContext`'s solve path should change.

### Validation gates (must pass before INV-2/INV-3)
1. `∫ over cos θ*` of the new oracle reproduces the existing rate oracle to ~1e-10 relative.
   This catches any error in `S` and the angular weights.
2. `A_FB(m_ll)` and `A4(m_ll)` for `c=0` reproduce the known SM forward-backward asymmetry
   (rises through the Z pole, approaches the asymptotic high-mass value). Sanity-plot it.
3. For 3–5 non-zero `c` points, `A_FB(m_ll)` matches MadGraph at the per-cent level for
   |c| ≤ 1 across `m_ll ∈ [0.3, 2.5] TeV`. The MC tier is already wired. This is the gate
   that certifies the chiral construction and the dilution.
4. EFT validity: do not reuse |c|=0.8 / Λ=1 TeV / μ→870 for any physics claim. Run two
   configs: (a) linear-only (drop all `B_ij`), a controlled in-validity target; (b)
   linear+quadratic with |c| reduced and/or Λ raised so `ŝ/Λ² < 1` holds across the m_ll
   range used. Report both. Boughezal et al. discuss when quadratics matter; mirror that.

### Keep/cut
Infrastructure. Ships if gates 1–3 pass. If A_FB cannot be matched to MadGraph at per-cent,
stop and debug the chiral amplitudes (most likely a coupling sign or the up/down contact
combination) before doing anything downstream.

---

## INV-2 — Bayesian-efficiency of the disclosure probe

### Claim to establish
The disclosure probe attains the best per-direction precision the observable permits: its
realised error matches the Bayesian Cramér-Rao bound. The vertex directions at R² ≈ 0 are
then evidence of optimality — the observable carries no information about them, so any
estimator returns the prior — rather than a recovery failure. This replaces the Section 5
"physically expected" assertion with a proof.

### Why the frequentist CR bound is the wrong bound, and which to use
The head is Bayesian ridge: `w | data ~ N(w*, σ_y² A⁻¹)`, inducing `c̃ | data ~ N(Vᵀ W w*,
Σ_c̃)` with `Σ_c̃ = σ_y² (Vᵀ W) A⁻¹ (Vᵀ W)ᵀ` in the Fisher eigenbasis. Ridge shrinks toward
the prior mean, so the probe is a biased estimator. The frequentist bound `F_K⁻¹` is for
unbiased estimators and will read as "inefficient" purely from shrinkage. Use the Bayesian
(van Trees) bound, which includes the prior and is the correct floor for a Bayes estimator:

```
BCRB(c̃)  =  ( F_K(c)  +  Σ_prior⁻¹ )⁻¹       evaluated and diagonalised in the eigenbasis.
```

`Σ_prior` is the induced prior covariance of `c̃` from the training prior the head was meta-
trained against (the box |c_i| ≤ 0.7; estimate `Σ_prior` empirically from that prior, then
rotate by `Vᵀ`). In the eigenbasis both `F_K` and `Σ_prior⁻¹` are approximately diagonal, so
BCRB is per-direction `1 / (λ_a^{(K)} + 1/σ_{prior,a}²)`.

This bound makes the vertex behaviour exact: where `λ_a^{(K)} ≪ 1/σ_{prior,a}²`, the
direction is prior-dominated, BCRB ≈ prior variance, the optimal estimator returns the prior
mean, and R²(direction a) → 0 by construction. Where `λ_a^{(K)} ≫ 1/σ_{prior,a}²`, the
direction is data-dominated and BCRB ≈ `1/λ_a^{(K)} = F_K⁻¹`.

### The K-point Fisher (match the noise model exactly)
Contexts are generated with additive Gaussian noise on `Y` (the ratio), variance `σ_y²`. The
Fisher for estimating `c` from K observations `Y_k = μ(c, x_k) + ε_k`, `ε_k ~ N(0, σ_k²)` is

```
F_K(c)_ij = Σ_{k=1}^{K} (∂_i μ(c, x_k)) (∂_j μ(c, x_k)) / σ_k²,
∂_i μ(c, x_k) = A_i(x_k) + Σ_j (B_ij + B_ji)(x_k) c_j      (analytic, from the morphing).
```

This is additive-noise Fisher on `μ`. The paper's Eq. (9) uses `∂ log μ`, which is the
relative-noise (`σ_k ∝ μ_k`) Fisher. Pick the one matching how contexts are actually
generated and use the same one in the BCRB, the probe, and the oracle MLE. A mismatch here
makes the efficiency number meaningless. State the chosen noise model in the run config.

### Estimands and procedure (revised)
The previous spec used `η_a = BCRB_a / MSE_a ≥ 0.8` on resolved directions as the keep
threshold. This was wrong: at `K = 12` and `σ_y = 0.05` the resolved directions are in the
pre-asymptotic regime where the Cramér-Rao bound is a valid lower bound but is not tight,
and the realised MLE MSE itself sits one to two orders of magnitude above `F_K⁻¹`. `F_K⁻¹`
is therefore not the operative benchmark; the finite-sample MLE MSE is. The criterion is
rewritten to MSE-to-MSE against the MLE.

For each held-out `c` (in-box pool and withheld band, both):
1. Draw `N_rep ≥ 200` fresh K=12 contexts from the oracle at this `c`.
2. Per context, evaluate two probes on the same `(M_ctx, Y_ctx)`:
   - the linear-OLS probe `ĉ = W w* + b` from the implicit ridge weight alone,
   - the augmented linear-OLS probe `ĉ_aug = W_aug [w*; log M_ctx] + b_aug` from the
     `(w, log M_ctx)` sufficient pair; `w` alone is the ridge-summary statistic for predicting
     μ at new M, but the map back to c̃ throws away the conditioning on where in M the
     context sat, and the augmented input restores it; the disclosure stays inside the
     standard Bayesian-linear-regression family.
3. Also fit a parameter-matched MLP probe `ĉ_mlp` on the same augmented features. This is
   the disclosure-integrity gate (below); it is not a competing disclosure object.
4. Per direction `a` and per probe `p ∈ {linear, augmented, MLP}`:
   - bias `β_{p,a} = mean_r(ĉ_{p,a}) − c̃_a^true`,
   - variance `v_{p,a} = var_r(ĉ_{p,a})`,
   - **MSE** `MSE_{p,a} = β_{p,a}² + v_{p,a}` (always bias² + variance; never
     variance-to-variance against an unbiased MLE — the probe is a shrinkage estimator and
     comparing variances alone would reward the probe for trading variance for bias).
5. MLE cross-check (same `N_rep` contexts, `scipy.optimize.least_squares` on the morphing
   residuals with `c0 = 0`, additive-Gaussian noise model on Y matching the context generator):
   - bias `β_{MLE,a}`, variance `v_{MLE,a}`, **MSE** `MSE_{MLE,a} = β_{MLE,a}² + v_{MLE,a}`.
   - The MLE is approximately unbiased in this setup; report `β_{MLE,a}` to verify.
6. **The two-benchmark report** per direction:
   - pre-asymptotic gap `MSE_{MLE,a} / F_K⁻¹_a`: tells the reader how far the MLE is from the
     CR asymptote at this `K, σ_y`. Large values are the regime where `F_K⁻¹` is not the
     operative benchmark (this is true on `c̃_1` at K=12, σ_y=0.05 in the SMEFT problem),
   - achievable-efficiency `MSE_p / MSE_{MLE,a}`: tells the reader how close the probe is to
     the finite-sample-achievable benchmark.
7. Calibration check in c̃-space: compare the probe's claimed `Σ_c̃` (diagonal) to `v_a`. A
   ratio near 1 means the posterior variance tracks the realised spread.

Run twice:
- **Mass-only** (current oracle): expect on the resolved directions `MSE_{MLE,a} ≫ F_K⁻¹_a`
  (pre-asymptotic, `c̃_1` at the leading energy-growing direction is the canonical case);
  the augmented linear probe should be within a small factor of `MSE_{MLE,a}` on `c̃_1, c̃_2`,
  and on the vertex `c̃_3, c̃_4` the MLE itself runs at the prior-mean variance and the probe
  correctly reports the prior floor.
- **Mass+angular** (INV-1): expect `λ_3, λ_4` to rise above the prior precision once A_FB
  is in the observable; vertex directions move from prior-dominated to data-dominated;
  recompute MSE-to-MSE ratios and show the vertex MSE drops onto the new MLE benchmark.

### Disclosure-integrity gate (new)
Fit a parameter-matched MLP probe on the same `(w, log M_ctx)` features and compare
`MSE_{linear,a} / MSE_{MLP,a}` per direction on the withheld band. The disclosure claim that
the c̃ structure lives in the representation rather than in the probe head requires this
ratio to be near 1 on the resolved directions OOD: if the linear and MLP augmented probes
agree, ψ_θ carries the structure. If the MLP pulls ahead on `c̃_1` OOD, work has moved from
the representation into the probe family, and the "Bayesian linear regression in a learned
basis" framing weakens correspondingly. The integrity ratio is reported per direction.

### Cross-check that the floor is the observable, not the probe
Implement a direct oracle MLE: per held-out context, fit `c` by maximising the analytic
Gaussian likelihood of the K observed ratios (`scipy.optimize.least_squares` on the morphing
residuals, weights `1/σ_k²`). Report its bias and variance per direction; combine into MSE.
Do **not** assume the MLE matches `F_K⁻¹` — that is the asymptotic claim and at K=12,
σ_y=0.05 it is two orders of magnitude off on `c̃_1`. The MLE MSE at finite K is the
operative benchmark; the gap to `F_K⁻¹` is reported as the pre-asymptotic caveat.

### Keep/cut (revised)
Let `R_p = MSE_p / MSE_{MLE}` be the probe-to-MLE MSE ratio per direction.

- KEEP as an efficiency result if all three hold:
  1. on the resolved directions (mass-only: `c̃_1, c̃_2`; mass+angular: also `c̃_3, c̃_4`),
     the augmented linear probe satisfies `R_augmented ≤ 3` with `|β_augmented|` smaller
     than the standard error of the mean across the held-out pool;
  2. the vertex directions sit at the prior floor: `MSE_{augmented} ≈ σ²_{prior}` and
     `MSE_{MLE} ≫ σ²_{prior}` (the MLE without prior runs free; the prior is what gives the
     probe its sensible floor on the unresolved directions);
  3. the disclosure-integrity gate passes: `MSE_{linear} / MSE_{MLP} ≈ 1` per direction on
     the withheld band.
  Report all three benchmarks per direction: `F_K⁻¹` (CR asymptote — sub-physical at K=12
  is the diagnostic, not a target), `MSE_{MLE}` (finite-sample-achievable benchmark), and
  the probe MSE. Also report bias and variance separately, not just MSE, so the
  shrinkage-vs-bias profile is visible.
- FIX, do not report, if any of:
  1. `R_augmented ≫ 3` on a resolved direction while `MSE_{MLE}` is comparable to `F_K⁻¹`
     (this is the asymptotic regime where the probe is unambiguously the bottleneck —
     debug `W` shape and `ψ_θ` per the original spec);
  2. the disclosure-integrity gate fails (MLP pulls ahead OOD on a resolved direction by
     more than a stated factor) — the framing claim weakens and ψ_θ-only structure
     cannot be claimed in the abstract;
  3. the probe's claimed `Σ_c̃` (posterior covariance) and the realised `v_a` disagree
     beyond their standard errors — the credible-interval upgrade is not calibrated and
     the SBI artifact's coverage flag is meaningless.

### Output artifact (Claim C — amortized SBI posterior)
The probe output is the SBI deliverable, not a diagnostic by-product. Emit
`sbi_posterior_summary.json`: per held-out scenario, save the context `(M_ctx, Y_ctx)`, the
posterior mean over the four Warsaw directions and the four rotated `c̃` directions, the
posterior covariance `Σ_c̃`, the 68% per-direction interval, the true `c`, and the INV-4
coverage flag (true `c̃_a` inside its interval, per direction). This is a forward-pass
posterior estimator conditioned on a context, which is amortized neural SBI; label it as such
in the artifact and the run log. The methodological sibling is TabPFN-as-SBI (paper ref 37).
The deliverable is the calibrated posterior itself, evaluated identically on simulator and
(future) measured contexts because the head is c-agnostic. Do not describe it as feedstock
for a downstream SBI pipeline; the head is the inference engine.

---

## INV-3 — Parameter-space EPIG vs random on the angular observable

### What the §4.7 result is, mechanically
On a 1D `m_ll` axis, 500-call budget, 300-point pool, scored in μ-RMSE, random beats EPIG
(1.38 vs 2.08). This is the regime the active-learning literature flags as adverse: dense
random coverage of one axis leaves nothing for a targeted rule to exploit, and greedy
information-gain over-exploits the dominant information mode (the BatchBALD redundancy Kirsch
et al. flag; sequential-greedy removes intra-batch redundancy but not cross-mode starvation).
μ-RMSE on one axis is near-isotropic, so design cannot beat coverage. The anisotropy that
rewards design lives in Wilson space, where the Fisher eigenvalues span five orders of
magnitude; μ-RMSE averages it away.

Two paper consequences:
1. Remove the random-vs-EPIG race from the 1D loop. The 1D demonstration's contribution is
   loop closure (drift fires → acquire → refit → coverage holds → entropy monotone). EPIG is
   the rule there because it is the principled closed-form one, not because it wins a race it
   cannot win.
2. Run the acquisition comparison where directions compete for a scarce budget: the
   mass+angular observable, scored in Wilson c̃-space.

### Corrected, falsifiable hypothesis
Targeted acquisition beats random when resolvable directions compete for budget, and the gain
is visible only when recovery is scored in the parameter metric whose information geometry is
anisotropic. If random still ties parameter-EPIG on the angular observable scored in c̃-space
at a stressed budget, active learning adds nothing here and is cut.

### Derivation of the closed-form parameter-space acquisition score
The c̃ posterior covariance is `Σ = σ_y² P A⁻¹ Pᵀ`, with `P = Vᵀ W ∈ R^{r×D}` projecting the
ridge weight onto the `r` resolved directions. Acquiring a point with feature `ψ_p` updates
`A → A + ψ_p ψ_pᵀ`. Sherman-Morrison:

```
A_new⁻¹ = A⁻¹ − (A⁻¹ ψ_p ψ_pᵀ A⁻¹)/(1 + lev_p),   lev_p = ψ_pᵀ A⁻¹ ψ_p.
```

Push through `P (·) Pᵀ`. Let `u = P A⁻¹ ψ_p ∈ R^r` (the acquisition's influence projected
onto the resolved directions). Then

```
Σ_new = Σ − σ_y² u uᵀ / (1 + lev_p).
```

D-optimal score (joint differential entropy of the resolved subspace; the principled
"information" target, accounts for cross-direction correlations) via the matrix determinant
lemma `det(Σ − β u uᵀ) = det(Σ)(1 − β uᵀ Σ⁻¹ u)`:

```
ΔH_D(p) = ½ [ log det Σ − log det Σ_new ]
        = −½ log( 1 − σ_y² (uᵀ Σ⁻¹ u) / (1 + lev_p) ).
```

Single-direction A-optimal score (target one direction `a`, e.g. the weakly resolved c̃_2):

```
ΔH_a(p) = −½ log( 1 − σ_y² u_a² / ((1 + lev_p) Σ_aa) ).
```

Positivity: `uᵀ Σ⁻¹ u ≥ 0` and Cauchy-Schwarz keeps the log argument in `(0, 1]`, so
`ΔH ≥ 0`, exactly as for the predictive EPIG of paper Eq. (7). Implement `ΔH_D` and `ΔH_a`
as new acquisition targets reusing the existing Sherman-Morrison cache; this is a new
quadratic form, not a new solver. `P` comes from `W` (INV-2) and `V` (the Fisher rotation);
the loop must hold both. Choose `r` = the data-dominated directions identified in INV-2.

### Distinction from predictive EPIG (state precisely in the paper)
Predictive EPIG (paper Eq. 7) reduces predictive variance at query points in `μ`-space:
target functional is `lev_T` at a predictive target T. Parameter-EPIG reduces posterior
variance in `c̃`-space: target functional is `Σ` above. Same closed-form machinery, different
target. On a 1D observable the predictive target is near-isotropic; the parameter target is
not.

### Run matrix
- Observable: mass+angular (INV-1).
- Target activation: see the open decision below.
- Acquisition modes: uniform random; leverage (D-optimal on `A`, the existing one);
  predictive-EPIG (existing); parameter-EPIG-D (`ΔH_D`); parameter-EPIG-a on c̃_2 (`ΔH_a`).
- Score per cycle: resolved-subspace posterior entropy `H = ½ log det Σ`; per-direction
  recovery error on c̃_1 and c̃_2 against the oracle MLE (INV-2); and μ-RMSE for continuity
  with §4.
- Stress the budget so a pick matters: `k = 1` per fire and pool shrunk to ~30, alongside the
  current `k=5` / 300-pool. The random-coverage advantage only lifts when a wasted pick costs
  something.
- Seeds: ≥ 20. Report paired contrasts (parameter-EPIG vs random) with effect size and CI,
  not just the sign. The current 5-seed σ ≈ 0.4–0.5 cannot resolve a 0.2 effect.

### Open decision (resolve before building the harness)
Which directions to activate as the competing target. Two different questions:
- c̃_1 vs c̃_2 (both four-fermion, both resolved on mass alone): tests whether parameter-EPIG
  allocates a scarce budget between two comparably-resolved directions better than coverage.
- four-fermion (tail) vs vertex (made resolvable only by A_FB): tests whether parameter-EPIG
  discovers the newly-resolvable direction and the loop's value is "acquire a new observable."
The second is the stronger digital-collider narrative; the first is the cleaner test of the
competition hypothesis. Pick one as primary; the spec supports both.

### Keep/cut
- KEEP if parameter-EPIG reduces `H` faster than random and the c̃_2 recovery error separates
  from random beyond seed scatter (paired test, ≥20 seeds, report effect size). Contribution:
  closed-form parameter-space EPIG drives the Wilson posterior down faster than coverage when
  directions compete.
- CUT active learning from the paper if random still ties on the angular observable in
  c̃-space at the stressed budget. Then report loop closure only. The test has told you the
  hypothesis was wrong for this observable; that is the adjustment, not a null to publish.

### Output artifact (Claim B — drift-to-action decision trace)
This is the evidence that the drift detectors are a scientific instrument. Without it Claim B
is assertion. Log the loop as a sequence of scientific decisions, not detector firings. Emit
`decision_trace.jsonl`, one record per acquisition event:

```
{ cycle,
  detectors_fired:      subset of {accuracy, calibration, coverage},
  failure_mode:         the mapped diagnosis (point predictions wrong / intervals
                        miscalibrated / context extrapolating into new directions),
  localization:         which c̃_a direction is unresolved (read from Σ_c̃ / leverage in
                        c̃-space) and/or which (m_ll, cosθ*) region under-covers (from the
                        per-region conformal test and the condition-number monitor),
  action:               recalibrate / acquire-in-region / acquire-new-observable,
  outcome:              Δ(c̃ posterior entropy) and Δ(per-direction error) attributable to
                        this action }
```

The summary the paper cites is the narrative this trace makes explicit: a coverage signal
localizes an unresolved direction, the agent acquires the observable that breaks the
degeneracy (on the angular observable this is "acquire A_FB information," a recognizable
physics decision), and the posterior entropy on that direction falls. Apply the same trace
retroactively to the Section 4 1D loop — the loop already runs, so this is logging plus the
c̃-space localization, no new run. The decision trace and the SBI posterior object together
are what make the loop legible as drift-instrumented inference rather than model maintenance.

---

## INV-4 — Replace the c̃_2 R² with calibrated coverage

### The artifact
Table 3 has c̃_2 recovering better out-of-distribution (0.39) than in-box (0.06).
`R² = 1 − SS_res/SS_tot`; inside |c| ≤ 0.7 the c̃_2 excursions are small so `SS_tot` is tiny
and R² collapses regardless of slope; the annulus has larger c̃_2 spread, larger `SS_tot`,
larger R². The inversion measures the pool's dynamic range, not the model. R² is the wrong
summary for a regression whose regressor variance differs across pools.

### Estimands (replace R² as the headline)
- Probe slope and standard error per direction: regress c̃_a on the relevant `w` components,
  report coefficient ± SE. A consistent slope in-box vs withheld with divergent R² confirms
  the artifact.
- Per-direction credible-interval coverage: from the c̃ posterior (INV-2), form the 68%
  interval per held-out `c`, report the fraction of held-out scenarios whose true c̃_a lies
  inside, per direction, in-box and withheld. This is the SBI-consumable output and is not
  distorted by pool spread. Target coverage ≈ 0.68; report the band.

### Keep/cut
KEEP c̃_2 as the second resolvable direction (eigenvalue 8.7: resolvable, but sampling-
sensitive — the natural acquisition target in INV-3), reported by interval and coverage.
Drop the R² table as primary and drop the in-box/withheld inversion discussion entirely.
No standalone run; fold these numbers into INV-2/INV-3 output.

---

## INV-5 — Cut matched-budget JEPA

### Decision
Matched-budget JEPA is a strawman: JEPA's premise is capacity, so pinning it to ~6k
parameters tests nothing about JEPA, and the y-shuffle/cross-decoding diagnostics built on
its collapse add nothing the reader needs. Delete from the paper: the matched-budget JEPA-FM
row and discussion; the y-shuffle invariance table (Appendix C.2); the JEPA-loss ablation
(the 0.003); the +0.007 latent-distance/R² correlation; the cross-decoding numbers.

### What survives (one paragraph, no new compute)
Scaled-budget JEPA-FM reaches median R² ≈ 0.95–0.97 at 186k parameters and 10k scenarios; the
closed-form head reaches 0.9999 at 5.3k parameters and 200 scenarios. Mechanism, one clause:
the ridge solve is a function of the joint M–Y assignment by construction, so a pooling or
attention aggregator must spend capacity learning the alignment the closed form has built in,
which is the parameter and data gap. Keep only Intention-vs-scaled-JEPA c-recoverability
(0.75 vs 0.62) and effective rank (11 vs 21) as support.

---

## INV-6 — Foundation-model comparison artifact (Claim A)

### Purpose
Make legible that the system compares foundation-model readings of the same `(M_ctx, Y_ctx)`
context and that the four analytic properties are not generic across architectures. Section 6
and Appendix C already carry the numbers; this consolidates them into the single artifact
that states the comparison as a comparison across architectures on the M–Y-alignment axis.

### Artifact
Emit `architecture_comparison.json`, one row per architecture {closed-form Intention
(learned ψ), fixed-ψ Intention, Deep Sets, constraint-violating regressor, scaled JEPA} with
columns:
- closed-form predictive variance: yes/no + reason (e.g. Deep Sets has no design matrix, so
  no closed-form variance; the regressor consumes c and is not c-agnostic);
- closed-form information gain: yes/no;
- closed-form sequential update (Sherman-Morrison): yes/no;
- Eckart-Young condition-number sensitivity: yes/no;
- M–Y-alignment retained = `1 − y-shuffle cosine similarity`, the single scalar per
  architecture (this is the only piece of the matched-budget y-shuffle material that survives
  INV-5; the table and the rest of that diagnostic suite stay cut);
- held-out median and p5 R² (Table 4);
- c-recoverability R² (linear probe), in-box and withheld;
- parameter count.
Order rows by alignment retained. The table content is the claim: the four closed-form
properties and the c-recoverability track alignment retention, and only the closed-form head
carries all four properties by construction.

### Keep/cut
Always produced; it is a consolidation of existing numbers plus the one alignment scalar, no
new training beyond INV-2/INV-5. If the alignment ordering is not monotone in the properties
or the c-recoverability, report the exception — that is itself a finding about the alignment
axis, not something to suppress.

---

## Numerical and implementation pitfalls (read before coding)

1. **Noise-model consistency (INV-2, INV-3).** The Fisher, the BCRB, the probe, and the
   oracle MLE must all use the same noise model as the context generator. Additive-on-Y uses
   `∂μ`; relative/multiplicative uses `∂log μ` (paper Eq. 9). Pick one, record it, use it
   everywhere. This is the single most likely source of a meaningless efficiency number.

2. **Ridge bias in the efficiency check (INV-2).** The probe is biased by shrinkage. Do not
   compare its variance to the frequentist `F_K⁻¹`; use the Bayesian bound
   `(F_K + Σ_prior⁻¹)⁻¹`. Always report bias `β_a` alongside variance; define `η` on MSE,
   not variance.

3. **Near-singular Fisher (INV-2, INV-3).** Eigenvalues spanning 1e-4 to 1e1 make `F_K`
   ill-conditioned. Never invert `F_K` directly. Work in the eigenbasis and either use the
   Bayesian form `(F_K + Σ_prior⁻¹)⁻¹` (well-conditioned) or threshold tiny eigenvalues
   explicitly and report them as prior-dominated. The same applies to `Σ⁻¹` in `ΔH_D`: if the
   resolved subspace `r` includes a near-degenerate direction, `det Σ` underflows — restrict
   `r` to data-dominated directions from INV-2.

4. **cos θ* dilution sign (INV-1).** The hadronic A_FB sign depends on the CS-frame
   convention and the quark-direction assignment. Getting the sign wrong still passes
   validation gate 1 (angular integral) because `D` integrates to zero. Gate 3 (MadGraph
   A_FB) is the one that catches it. Do not skip gate 3.

5. **Morphing reuse, not rebuild (INV-1).** `A_i^D, B_ij^D` come from the existing morphing
   procedure with the angular weight swapped to `sign(cos θ*_CS)`. Re-deriving the full
   hadronic angular cross section from scratch is unnecessary and error-prone. Reuse the rate
   oracle's c-morphing solver verbatim.

6. **`W` and `V` coupling across INV-2/INV-3.** Parameter-EPIG needs `P = Vᵀ W`. `W` is fit
   in INV-2 on training-box scenarios and must be frozen (not refit per held-out `c`); `V` is
   the Fisher rotation. The acquisition loop must carry both. Fit `W` once, cache it.

7. **EFT validity (INV-1).** The |c|=0.8 / Λ=1 TeV setup is EFT-invalid above ~1 TeV. Use it
   for nothing quantitative. Run linear-only and an in-validity linear+quadratic config.

8. **Eigenbasis is per-Fisher (INV-2).** The mass-only and mass+angular observables have
   different Fisher matrices and therefore different `V`. Recompute `V` for each observable;
   do not carry the mass-only rotation into the angular run.

9. **Conformal in c̃-space (INV-4).** The existing conformal calibrator is in predictive
   `μ`-space. The c̃ credible interval is the probe posterior, a different object. Do not
   reuse the μ-space conformal quantiles for the c̃ coverage table.

---

## Decision table

| Item | Action | Core computation | Keep threshold |
|---|---|---|---|
| INV-1 angular oracle | Build | S/D split via reused c-morphing + PDF dilution | gate1 angular-integral ≈ rate to 1e-10; gate3 A_FB matches MadGraph per-cent |
| INV-2 probe efficiency | Investigate | K-point Fisher, Bayesian CRB, η_a, MLE cross-check | η_a ≥ 0.8 + small bias on resolved dirs; MLE confirms vertex floor → efficiency result; else fix probe |
| INV-3 parameter-EPIG | Investigate | ΔH_D / ΔH_a closed form, angular observable, c̃-space score, ≥20 seeds | parameter-EPIG beats random on H and c̃_2 beyond seed scatter → keep; else cut active learning |
| INV-4 c̃_2 calibration | Recompute | slope±SE, c̃-space interval coverage | always; replaces R²; c̃_2 becomes the acquisition target |
| INV-5 JEPA | Cut + 1 paragraph | none | scaled-JEPA efficiency point only |
| INV-6 architecture table | Consolidate | architecture × {4 properties, alignment scalar, R², c-recoverability} | always; supports Claim A; report any non-monotone exception |

Claim artifacts (produced by the INVs above, named so the paper can cite them):
- Claim A → `architecture_comparison.json` (INV-6).
- Claim B → `decision_trace.jsonl` (INV-3, and retroactively the Section 4 loop).
- Claim C → `sbi_posterior_summary.json` (INV-2 + INV-4 coverage).

## Execution order
1. INV-1; pass gates 1–3 (blocks 2 and 3).
2. INV-2 mass-only (proves the floor) → fit and freeze `W` → INV-2 mass+angular. Emit the SBI
   posterior object (Claim C).
3. INV-3 on the angular observable (shares the harness and `W`, `V` with INV-2 mass+angular).
   Emit the decision trace (Claim B); apply it retroactively to the Section 4 loop.
4. INV-4 recompute (cheap; run alongside INV-2); feeds coverage into the Claim C artifact.
5. INV-6 consolidate the architecture table (Claim A); no new training.
6. INV-5 deletion (no compute).

INV-2 and INV-3 are the two that convert reported negatives into contributions. If INV-2's
MLE cross-check shows the probe sub-optimal, fix the probe before claiming efficiency. If
INV-3 shows random still ties in c̃-space at a stressed budget, cut active learning rather
than reporting it — the test has then established the hypothesis was wrong for this
observable, which is the adjustment to make.

The three claim artifacts are produced as a by-product of INV-2, INV-3, and INV-6; they cost
no extra runs beyond logging and one consolidation. They are what make Claims A, B, and C
legible in the results. The framing prose that states the claims is written separately and is
not the agent's job.
