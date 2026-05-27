# Physics reasoning patches for the ALETHIA manuscript

Working document for the coding agent. The current draft lives at
`paper/alethia.tex`, compiled PDF at `paper/alethia.pdf`. The draft was
written before the MadGraph T1 vs T2 cross-check and before the linear
identifiability probe were produced; the new figures
`paper/figures/mg_crosscheck.png` and
`paper/figures/identifiability_probe.png` are already copied into the
paper directory but are not yet referenced. The companion literature
review and notes that informed the draft are in
`docs/research/synthesis/full-chain-run.md` (which now includes the
MadGraph and identifiability sections) and in this conversation's
literature survey of HEP foundation models.

Two principles for the patches below. First, fix the science before
fixing the prose: every paper edit should land after the supporting
experiment has been run and the numbers pinned. Second, do not add
material the data does not yet support; partial-disclosure should be
named as such, not euphemised.

The patches are ordered by physics weight: a referee for JHEP, SciPost
or PRD will hit patches 1 to 4 first.

---

## Patch 1: add the identifiability probe as the second headline result

### Hole

The paper asserts that the model "learns a representation of measured
Drell-Yan distributions" without ever evaluating whether the
representation has actually disclosed the SMEFT structure. Prediction
R^2 on mu(m) is necessary but not sufficient; it does not separate
"learned the EFT manifold" from "memorised the (M_ctx, Y_ctx) -> Y_q
regression on this corpus". This is exactly the failure mode the BRIEF
warns about (`docs/research/BRIEF.md`) and that the literature
identifies as the operational definition of a foundation model.

The linear probe at `experiments/full-chain-run/identifiability_probe.py`
produces the test the paper is missing. Per-operator held-out R^2
(reproduced from `docs/research/synthesis/full-chain-run.md`):

| Operator | inside training box | withheld band |
|---|---:|---:|
| cHq3 (vertex) | -0.005 | -0.227 |
| cHq1 (vertex) | -0.013 | -0.090 |
| clq3 (four-fermion, energy-growing) | +0.699 | +0.778 |
| clq1 (four-fermion, energy-growing) | +0.146 | -0.473 |

This is a partial-disclosure result: clq3 is recovered at R^2 = 0.78 on
the extrapolation band that the model never saw in training, but the
other three operators are not disclosed at this configuration. The
paper currently advertises the model as "a foundation model for SMEFT
inference" with no caveat for this.

### Fix

Insert a new Section 6 (or new Section 5.3) titled
"Identifiability: which operators does the representation disclose?",
between the existing Section 5 (Results) and Section 6 (The full
inference loop). The section should:

1. Define the implicit per-scenario latent
   `w_implicit(c) = (Psi_ctx(c)^T Psi_ctx(c) + alpha I)^{-1} Psi_ctx(c)^T Y_ctx(c)`,
   stated as the closed-form ridge weight vector that the Intention head
   computes at inference time.

2. Define the linear-probe test (Ridge regression `c ~ W @ w_implicit + b`
   fit on 400 training-box scenarios, evaluated on 200 in-box and 200
   withheld-band scenarios). Cite the Component-3 test 9 specification
   in `aletheia_verification_plan.md`.

3. Include the table above as a numbered Table 2.

4. Include `paper/figures/identifiability_probe.png` as a numbered Figure.

5. State the partial-disclosure result plainly: clq3 is disclosed
   including out-of-distribution; the other three are not at this
   configuration. State the physical reason for each non-disclosure:
   vertex operators (cHq3, cHq1) carry `(v/Lambda)^2` rate shifts only,
   no m_ll-shape signature, and a mean-pool encoder over a 1-d
   kinematic observable cannot resolve a constant rate shift; clq1
   shares its dominant m_ll signature with clq3 modulo a u/d PDF
   weighting that the toy analytic PDF mis-represents at the 13% level
   (documented in `docs/research/01-oracle/empirical-results.md` §2).

6. State the threshold this result clears: R^2 > 0.5 is the qualitative
   disclosure threshold from the verification plan; R^2 > 0.8 is the
   strong-form claim. clq3 sits at +0.78 OOD, so it clears the
   qualitative bar everywhere and is at the edge of the strong-form bar
   on the extrapolation set.

### Acceptance

The paper has a Section titled Identifiability, a numbered Table 2 with
the four R^2 numbers, a numbered Figure with the probe plot, and an
explicit statement of partial disclosure with operator-by-operator
physical reasoning. The abstract is updated to reflect partial
disclosure (see Patch 7).

### Why it matters

This is the single largest physics gap in the current draft. The
contemporary HEP-FM literature is conspicuously silent on whether
foundation-model representations actually disclose physics; the
literature review identified this as a niche ALETHIA can credibly own.
Without the identifiability section the paper does not earn the
"foundation model" framing. With it, the paper makes a precise claim
the field is not currently making.

---

## Patch 2: add the MadGraph fidelity-ladder section

### Hole

The current draft Section 6.2 says, of the analytic SMEFT oracle,
"the chain shape is unchanged by oracle fidelity and only the cost
ladder differs". That assertion is reasonable but unsupported in the
manuscript. The cross-check has been run: 50 m-values across
m in [0.4, 2.2] TeV at target c_lq^(3) = 0.8, T1 versus T2 with
MadGraph LO, nevents=500 per call, 22 min wall time. The results are
in `experiments/full-chain-run/output/mg_crosscheck_summary.json` and
plotted at `paper/figures/mg_crosscheck.png`. The median relative
disagreement is 3.0%, the 95th percentile is 4.3%, the maximum is
5.0%, zero of 50 fidelity-drift firings against a 5% MC sigma; T2
sits ~3% systematically above T1, consistent with the NLO QCD K-factor
at this m-range (Mangano et al. arXiv:1610.07922).

The paper currently disclaims this work as future, when it has been
done.

### Fix

Insert a new subsection inside Section 6 (after the existing
subsection 6.3 "Recovery from the engineered drift"), titled "Fidelity
ladder: MadGraph cross-check on the band", or fold the equivalent
material into the existing Section 7 (Discussion) as a numbered
paragraph. The text should:

1. State the test setup (50 points across m in [0.4, 2.2] TeV at the
   target c, LO MadGraph at nevents=500 for each).

2. Report the four numbers: median 3.0%, p95 4.3%, max 5.0%, zero
   fidelity-drift firings against a 5% MC sigma.

3. Identify the ~3% systematic as the NLO QCD K-factor and cite
   Mangano et al. arXiv:1610.07922.

4. Include `paper/figures/mg_crosscheck.png` as a numbered Figure.

5. Propagate the physics implication: at mu = 800 (m = 2.25 TeV,
   c_lq^(3) = 0.8), the oracle's intrinsic 3% NLO bias corresponds to
   roughly 24 RMSE units, so the chain's reported 2.65 RMSE recovery
   is finer than the oracle's accuracy on the same observable. This
   reframes the recovery as "limited by the oracle ceiling, not by the
   chain", which is the correct reading.

### Acceptance

The paper has a subsection or numbered discussion paragraph reporting
the MG cross-check, the figure is included, and the relationship
between the 2.65 RMSE recovery and the 3% oracle ceiling is explicit.
The hand-waving disclaimer in the current draft Section 6.2 is
removed.

### Why it matters

The literature review identified "downstream physics impact" as a
consistent silence in the HEP-FM cluster. Showing that the analytic
oracle agrees with MadGraph LO inside MC noise across the engineered
drift band is the physics-side validation that turns the chain
demonstration from a software stunt into a publishable result.

---

## Patch 3: add the empirical motivation for the constraint

### Hole

The constraint that the Wilson coefficient never enters the forward
pass is the central architectural decision of the paper. The current
draft Section 3.1 ("The architectural constraint") gives a structural
argument for it but no empirical evidence. The structural argument
alone will not convince a sceptical referee.

The empirical evidence exists. The constraint-violating regressor
fitted against the morphing ansatz produces per-operator R^2 ≈ -144
when one Wilson operator is held at zero during training (cited in
`README.md` and in
`docs/research/02-foundation-model/empirical-results.md`). This is
the strongest evidence for the constraint.

### Fix

Extend Section 3.1 with one paragraph reporting this empirical result.
The paragraph should:

1. Describe the setup: a regressor with feature map phi(c, m) =
   phi_c(c) tensor phi_x(m), phi_c the SMEFT polynomial basis, trained
   by closed-form ridge on labels Y = sigma(c, m), with one operator
   held at zero during training and evaluated at nonzero values of that
   operator at test time.

2. Report the result: per-operator R^2 ≈ -144 on the held-out
   operator. State that this is the "regression-fits-the-ansatz"
   failure mode; the model has the parameters of the morphing ansatz
   but no representation of the physics manifold outside the training
   c-box.

3. Cite `docs/research/02-foundation-model/empirical-results.md` for
   the full study.

If `docs/research/02-foundation-model/empirical-results.md` does not
have a self-contained reproduction script for this exact number, write
one at `experiments/constraint-motivation/holdout_operator.py` that
loads the legacy regressor, retrains it with one operator held at zero,
and reports the four per-operator held-out R^2 values. Cap the experiment
at 60 s on CPU.

### Acceptance

Section 3.1 contains the empirical paragraph with the negative R^2,
the cited reference, and (if a fresh script was needed) the script
runs end-to-end and produces the reported number to within rounding.

### Why it matters

A referee with a physics background will object to the constraint
without an empirical hook. The negative R^2 is one of the most
striking numbers in the project; not having it in the paper is leaving
the strongest argument off the table.

---

## Patch 4: add bootstrap confidence intervals on Table 1

### Hole

The headline R^2 numbers in Table 1 are point estimates over 50
scenarios. There is no uncertainty quantification on the comparison.
A referee will ask, fairly, whether the gap between Intention and
DeepSets is significant at N=50, and whether the gap between
Intention_Learned and Intention_Fixed is robust to scenario
resampling.

### Fix

In `experiments/intention-vs-deepsets/`, add a script
`bootstrap_metrics.py` that takes the existing per-scenario R^2 and
MSE arrays from `/tmp/fm_compare/results.npz`, draws 1000 bootstrap
resamples of the 50 in-box and 50 out-of-box scenarios, and reports
the median and 5th-percentile R^2 and MSE for each architecture with
95% bootstrap confidence intervals. Wall time should be a few seconds.

Update Table 1 in the manuscript to include the 95% confidence
intervals on each cell, formatted as `0.9997 [0.9994, 0.9999]`. If
this makes the table too wide, drop the MSE columns (which add little
beyond R^2) and keep the medians and fifth percentiles with CIs.

State in the table caption that the confidence intervals are 95%
bootstrap intervals over scenario resampling.

### Acceptance

The script runs end-to-end against the existing results.npz, the table
in the paper has CIs on every reported number, and the caption
explains them. If the bootstrap CIs reveal that any comparison is not
significant at the 95% level (for example, Intention_Learned vs
Intention_Fixed on median R^2), the result text in Section 5 is
updated to reflect that.

### Why it matters

The "Intention beats DeepSets by a factor 700 in MSE" claim is the
quantitative heart of Section 5. Without CIs, the claim is suggestive.
With CIs, it is either definitive (which strengthens the paper) or
qualified (which is more honest); either way the paper improves.

---

## Patch 5: re-run the architecture comparison on the analytic SMEFT oracle

### Hole

The headline benchmark in Sections 4 and 5 uses the polynomial-toy
oracle. The current draft acknowledges this in Section 4 and in
Section 7, but the consequence is that the fixed-basis Intention
variant has the exact analytic structure of the target written into
its feature map and reaches R^2 = 0.99994 by construction. The
learned-basis variant does not exploit this, but on the polynomial
toy the paper cannot demonstrate that the learned basis is necessary,
only that it is sufficient.

A second benchmark on the analytic SMEFT oracle (which has parton
distribution function thresholds, real m_ll-shape structure, and
realistic relative rates between operators) is what closes this gap.
The infrastructure for this is in place: the full-chain run uses
exactly that oracle and the Intention encoder pretrains against it
successfully.

### Fix

Add a second run of the four-architecture comparison
(`Intention_Fixed`, `Intention_Learned`, `DeepSets-FM`,
`Cheat regressor`), now against `AnalyticSMEFTOracle(pdf="analytic")`
or, time permitting, `AnalyticSMEFTOracle(pdf="CT18NNLO")`. Reuse the
infrastructure at `experiments/intention-vs-deepsets/`. Settings:

- 200 training scenarios, c ~ U([-0.7, 0.7]^4)
- K = 12 context per scenario, Q = 32 query, m in [0.3, 2.3] TeV
- 1500 Adam steps for the trainable variants
- 50 in-box held-out + 50 out-of-box (L-infinity annulus 0.7 < max|c_i| <= 1.0)
- Wall time budget: 1-2 h CPU if analytic PDF, 4-6 h if CT18NNLO

Report median and p5 R^2 with bootstrap CIs (Patch 4) per architecture,
per region. Save the results to
`experiments/intention-vs-deepsets/results_analytic_smeft.npz` and
produce a companion `intention_vs_deepsets_inside_outside_smeft.png`
figure alongside the existing toy-oracle plot.

In the paper, either replace Table 1 with the analytic-SMEFT results
(and demote the polynomial toy to an appendix), or add a second
results table alongside it. The text in Section 5 should be updated to
make clear that the analytic-SMEFT results are the headline and the
polynomial toy is a controlled ablation. Section 7's caveat about the
fixed-basis tautology is then expected, not flagged.

### Acceptance

The script reproduces, the new table is in the paper, and the
narrative makes the analytic-SMEFT comparison the primary one. The
fixed-basis Intention is expected to lose ground here; if it does not,
the paper notes that surprise and discusses it.

### Why it matters

Without this patch, the most pointed referee question writes itself:
"Your headline benchmark uses an oracle that has the answer written
into the fixed basis. What happens on the real physics?" Producing
the answer in advance pre-empts the question.

---

## Patch 6: test the rate-aware encoder fix and report the result

### Hole

Patch 1 documents that the current model discloses one of four
operators. The rate-aware-encoder hypothesis is recorded in
`docs/research/02-foundation-model/empirical-results.md` §4 and tested
at small scale: adding a single scalar `log(sigma_total / sigma_SM)`
at the set summary lifts clq3 from 0.59 to 0.77 in 80 batches; cHq3
and cHq1 remain near zero, and the diagnosis there is that the
vertex-shift signal is small enough at the (m, c) scales sampled that
80 batches do not resolve it. This is an architecture conjecture
that has not been pushed to the full 1500-batch budget.

### Fix

Run the rate-aware encoder at full budget (1500 steps), with the same
configuration as the headline benchmark plus an auxiliary log-rate
scalar at the set summary computed at two reference points
(m = 1.5 TeV and m = 2.0 TeV at the scenario's Wilson configuration).
Implementation:

1. In `experiments/intention-vs-deepsets/intention_learned.py` (or a
   new sibling script `intention_rateaware.py`), extend the per-event
   feature map so that each context entry carries an additional scalar
   `log Y_aux(m_aux | c_ctx)` computed by the oracle at a fixed
   reference m_aux. The rate scalar enters as an extra column of
   `psi_theta(M_ctx)` and the ridge solve proceeds as before. The
   scalar is the same for every context entry in a given scenario, so
   the augmented design matrix has an extra rank-one block.

2. Train end-to-end as for the headline benchmark.

3. Re-run the identifiability probe of Patch 1 against the new model.
   Report the per-operator R^2 inside and outside the training c-box,
   alongside the existing four numbers.

4. If the result lifts at least one of cHq3, cHq1 to R^2 > 0.5
   in-distribution, add the table to the paper as Table 3 or as a new
   row in Table 2, and update the identifiability section to discuss
   the architectural fix.

5. If the result does not lift the vertex operators, report the
   negative result in the discussion section as a documented limit of
   the current encoder, and frame the multi-observable encoder (m_ll
   plus p_T^lepton or y_lepton-lepton) as the next iteration.

### Acceptance

The rate-aware experiment runs, the identifiability probe is re-run
against the new model, and the per-operator R^2 is reported in the
paper alongside the original four numbers. Whether or not the fix
works at full budget, the paper has a concrete experimental answer
rather than a hypothesis.

### Why it matters

This converts the partial-disclosure result of Patch 1 from "we
disclose only one operator" to "we disclose one operator at the
current encoder; here is the architectural fix and what it does".
The latter framing is what physics referees reward; it shows that the
authors understand why their representation is incomplete and have
tested a remedy.

---

## Patch 7: tighten the "foundation model" framing given partial disclosure

### Hole

The current draft uses "foundation model" throughout, including in the
title, the abstract, and the introduction. The literature review of
the HEP-FM cluster identifies "foundation model" as a load-bearing
term that carries implicit claims about generality, transfer, and
representation quality. The identifiability result from Patch 1
qualifies the claim: the model is a foundation model for one operator
and a regression surrogate for the others.

### Fix

Three small edits, all in `paper/alethia.tex`:

1. The abstract: keep "foundation model" in the lead but add a
   sentence at the end of the abstract that reads, approximately,
   "A linear probe from the model's implicit scenario representation
   to the underlying Wilson coefficients recovers c_lq^(3) at
   R^2 = 0.78 in the withheld extrapolation band, while the
   vertex operators (c_Hq^(3), c_Hq^(1)) and the second four-fermion
   operator c_lq^(1) are not disclosed at this configuration; we
   discuss the architectural cause and a tested remedy in
   Section 6." Replace 6 with the actual identifiability-section
   number.

2. The introduction: in the contribution paragraph, add a sentence
   between the existing "Second, the learned basis is visibly not a
   rediscovery..." and "Third, the foundation model composes...":
   "Second-and-a-half, a linear probe from the implicit scenario
   representation to the Wilson coefficients discloses the
   energy-growing four-fermion operator c_lq^(3) at R^2 = 0.78 even
   in the withheld extrapolation band; the rate-only vertex operators
   are not disclosed at this configuration and require an
   architectural extension that we describe and test."

3. Section 7 (Discussion): add a paragraph that names the partial-
   disclosure result, repeats the operator-by-operator physical
   reasoning, and is explicit that calling the current model a
   "foundation model for SMEFT inference" is shorthand for "a
   foundation model for the operators whose signature lies in the
   m_ll shape, and a representation surrogate for the others".

Do not change the title; "foundation model" in the title is supported
by the in-context, amortised, simulator-prior structure of the
architecture, independent of the partial-disclosure result.

### Acceptance

The abstract and introduction both flag the partial-disclosure result,
the discussion section names it as a limitation, and the title is
unchanged. The paper is honest about what its representation has
learned without giving up the framing that the paper has earned.

### Why it matters

Overstating disclosure is the kind of mistake that gets caught at
review and forces a difficult rewrite under deadline pressure. Pre-
emptively qualifying the claim costs no scientific ground (the result
is still novel) and reads as scholarship rather than salesmanship.

---

## Patch 8: separate marginal from conditional conformal coverage in the text

### Hole

Section 6 ("The full inference loop") reports per-region empirical
68% coverage stabilising into [0.65, 0.72]. The text refers to this
as conformal coverage. Strictly, split-conformal calibration
guarantees marginal coverage under exchangeability; per-region
coverage is conditional, which is a stronger property that requires
either stratified conformal (which the BH-binomial test layer
implements) or a Mondrian-conformal extension. The Araz-Spannowsky
paper (arXiv:2512.17048) is explicit about this distinction and warns
that conflating them is a common error. The current draft conflates
them.

### Fix

Edit Section 3 (or wherever the conformal calibrator is introduced)
to state the marginal-coverage guarantee
`P(Y_{n+1} in Gamma_alpha(X_{n+1})) >= 1 - alpha`, cite
`Lei:2018conformal` and `Vovk:2005alg`, and then explicitly state
that the chain's per-region calibration is implemented by leverage-
stratified split-conformal so that the marginal guarantee carries
over to each leverage stratum.

Edit Section 6 to use "stratified marginal coverage" or "per-stratum
empirical coverage" rather than "conformal coverage" when reporting
the [0.65, 0.72] band, and to cite `Araz:2025conformal` explicitly
when motivating the stratified approach.

Add one sentence to Section 7 noting that conditional coverage (in
the sense of `P(Y in Gamma_alpha(X) | X = x)` exactly equal to 1 -
alpha for every x) is not guaranteed by split conformal and is not
claimed here; the BH-binomial per-region test of Section 6 is the
mechanism by which the loop detects coverage degradation in
sub-regions.

### Acceptance

The text in Sections 3, 6 and 7 distinguishes marginal from
conditional coverage cleanly, with citations. No claim in the paper
overreaches the marginal guarantee that split conformal supplies.

### Why it matters

The Araz-Spannowsky paper is the calibration paper for HEP and is
explicit about this distinction. A referee who has read it (or who
is the author) will catch the conflation. Cleaning it up is twenty
lines of text and protects the calibration claim that is one of
ALETHIA's differentiating contributions.

---

## Patch 9: add a Wilson-coefficient confidence-interval plot

### Hole

The literature review identified "downstream physics impact" as the
consistent silence in the HEP-FM cluster: papers stop at
machine-learning metrics and do not propagate to a published physics
measurement. The ALETHIA infrastructure can close that gap. The
implicit scenario representation `w_implicit(c)`, plus the linear
probe from Patch 1, plus the conformal coverage from the chain,
together supply a propagated confidence interval on c_lq^(3) from a
synthetic measurement.

### Fix

Write a new script
`experiments/full-chain-run/wilson_coefficient_ci.py`. The flow:

1. Choose a target c (the same `c = [0, 0, 0.8, 0]` as the chain).
2. Generate a synthetic "measurement" of K_meas observations at chosen
   kinematic points, with realistic statistical fluctuations
   (Poisson per bin given an integrated luminosity assumption; the
   chain already supplies the cross-section ratios, multiplying by
   sigma_SM(m) and the luminosity gives expected event counts).
3. Pass the measurement as a fresh context into the Intention head;
   compute w_implicit; project through the linear probe to recover
   c_lq^(3)_hat.
4. Use the conformal calibrator's predictive interval on the
   measurement-bin predictions to propagate uncertainty through the
   probe; report a 68% CI on c_lq^(3). The simplest approach is a
   parametric bootstrap: sample 1000 noisy measurements at the same
   true c, run the pipeline on each, and report the 16th-84th
   percentile of the recovered c_lq^(3)_hat.
5. Repeat at three or four luminosity points (e.g.
   1 fb^{-1}, 30 fb^{-1}, 300 fb^{-1}, 3 ab^{-1}) and plot the CI
   width versus luminosity on a log-log plot. Save to
   `paper/figures/wilson_ci_vs_luminosity.png`.

In the paper, add a final results subsection or a new short Section
inside the discussion titled "From representation to a Wilson-
coefficient interval". Include the figure and one paragraph of text
walking the reader from the implicit representation to the CI. State
that the figure is for c_lq^(3) only, on the strength of the
identifiability result, and that the same construction is what would
extend to other operators once they are disclosed (Patch 6).

### Acceptance

The script runs end-to-end on the existing pretrained model and
analytic oracle, the figure is generated, the paper has a short
section that propagates the chain output to a physical confidence
interval. The CI scaling with luminosity should look qualitatively
like Lambda_eff / sqrt(L); flag deviation from that scaling in the
text.

### Why it matters

This is the deliverable that turns "we built a foundation model" into
"we built a foundation model that delivers a calibrated bound on a
Standard Model effective operator". The literature review identified
this as the strongest available differentiator from the HEP-FM
cluster. It is also the form a JHEP or PRD physics referee will most
clearly recognise as a physics result.

---

## Patch 10: add comparison to physics-side baselines

### Hole

The current draft compares to DeepSets (a generic ML baseline) and a
cheat regressor (a deliberately broken architecture). The physics-side
baselines that the paper does not compare to are: a parametrised
classifier of the form Chen et al. arXiv:2211.02058 use, and any of
the SALLY / CARL / ROLR estimators from the MadMiner line (Brehmer
et al. arXiv:1907.10621). These are the methods a physics referee
will ask about by name.

### Fix

Two parts. The first part is a literature paragraph; the second part
is a small empirical comparison.

1. In Section 7 (Discussion), expand the existing paragraph that
   compares to MadMiner and Chen et al. to be more concrete about
   the methodological difference. State that the parametrised
   classifier of Chen et al. fits a network per Wilson configuration
   and interpolates across a trained Wilson grid at deployment; that
   the joint-likelihood-ratio estimators of MadMiner exploit the
   simulator's joint score and likelihood ratio per event; and that
   ALETHIA amortises the inference itself, so a single learned
   feature map handles arbitrary Wilson configurations through the
   context. State explicitly what is gained (a single deployment-time
   matrix solve, no retraining per Wilson region) and what is given
   up (the model is restricted to the observable representation in
   which psi_theta is defined; here, binned cross-section ratios).
   Cite all three lines.

2. Add a small empirical comparison if feasible inside a one-day
   budget: train a parametrised-classifier baseline (Chen et al.
   Eq. 3.9 style) on the same training scenarios and the same
   analytic SMEFT oracle, evaluate on the same held-out scenarios,
   report median R^2 on mu(m) reconstruction alongside ALETHIA. Save
   to `experiments/parametrised-classifier-baseline/`. If the
   budget for a clean implementation is too large for the hackathon
   timeline, drop this part and rely on the literature paragraph.

### Acceptance

Section 7 has a paragraph that compares ALETHIA to the parametrised
classifier and MadMiner explicitly, with all three lines cited; if
the empirical baseline was run, its number sits in the headline
table.

### Why it matters

The first review question from a JHEP referee on this paper will be
"why is this better than MadMiner?". The literature paragraph alone
answers the question conceptually; the empirical comparison answers
it quantitatively. Either is much stronger than the current text.

---

## Cross-cutting hygiene

A few smaller items that do not deserve their own patch but should be
swept up while the larger patches are being made.

The current Section 2 ("SMEFT in Drell-Yan and the oracle") describes
the analytic oracle but does not state that it uses LO QCD only, no
QED FSR, no detector smearing, and the high-mass tail region
m in [0.3, 2.3] TeV. The MadGraph cross-check of Patch 2 supplies
exactly this scope statement; lift it into Section 2 so it appears
before any results are reported.

The current Section 6 refers to "physics-aware drift detectors" as
the framing for the three monitors. The literature review noted that
this is a strong framing that the HEP-FM cluster does not currently
match, and that it reads well. Keep it.

The acknowledgements omit the foundation-of-the-ALETHIA-stack credits
that the README has, namely the upstream `gemini-hackathon` starter
template. Add a one-sentence acknowledgement.

The current draft uses "Greljo and Marzocca" as the citation for the
Drell-Yan SMEFT amplitude decomposition. Verify that the dim-6
operator labelling matches their notation rather than the Warsaw
convention; if not, add a footnote stating which.

The current draft never cites the MadGraph paper itself. With Patch
2 the citation becomes load-bearing; include `Alwall:2014mg5`
explicitly in the new MadGraph paragraph.

---

## Suggested execution order

The patches are independent enough to be done out of order, but a
reasonable sequence is:

1. Patches 4 and 8 first; both are small text-and-script edits that
   tighten the existing claims without depending on new physics work.
2. Patches 1 and 2 next; both ingest already-run experiments and
   close the largest physics gaps.
3. Patch 3 after Patch 1 (the partial-disclosure framing is easier to
   write once the constraint motivation is in place).
4. Patch 7 after Patches 1 and 3 (the framing tightens around the
   science that is now in the paper).
5. Patch 9 after Patch 1 (the Wilson-coefficient CI uses the same
   probe machinery).
6. Patches 5 and 6 if the compute budget allows before the deadline;
   both are 1 to 6 h CPU experiments that strengthen the paper but
   are not strictly required to ship a defensible draft.
7. Patch 10 last; the literature paragraph is one hour, the empirical
   baseline is a day if attempted.

Pin a fresh PDF after each patch and re-read the abstract and the
discussion section end-to-end; the partial-disclosure framing tends
to drift unless re-read every cycle.
