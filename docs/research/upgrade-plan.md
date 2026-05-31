# ALETHIA upgrade plan

A single consolidated plan to finetune the existing results and close the
reasoning holes identified in four parallel audits:

- [audit-physics.md](audit-physics.md)
- [audit-ml.md](audit-ml.md)
- [audit-drift-phoenix.md](audit-drift-phoenix.md)
- [audit-al-sbi.md](audit-al-sbi.md)

Deadline: **2026-06-11, 2pm PT** (hackathon submission). Today is 2026-05-28.
Working budget: ~14 calendar days.

## 1. Executive synthesis

Three issues recur across every audit, and one unifying upgrade resolves
most of them.

### 1.1 The three convergent issues

**I1. Number-source / parameter-count inconsistency** (physics audit). Table 2
quotes a c_lq^(3) probe R² = 0.94 captioned under the 5,328-parameter model
but the number comes from `output_supercharged/summary.json` — a 20,896-param
run. The 5,328-param run actually achieves R² ≈ 0.78. This is referee-bait;
fix is one paragraph and one re-run.

**I2. Three-of-four operators silently undisclosed** (physics + ML audits).
Only c_lq^(3) is reported. c_Hq^(3), c_Hq^(1) sit at R² ≈ 0; c_lq^(1) is
negative in extrapolation. There is a clean physics reason — vertex operators
produce rate-only modifications degenerate with the SM on m_ll alone, and
c_lq^(1) is u/d-isospin-degenerate with c_lq^(3) on the same observable —
but the paper hides the full table. This converts a "one of four worked"
result into a quantitative identifiability statement.

**I3. The closed-loop machinery is built on a single scalar from a much
richer object** (drift + AL/SBI audits). The design matrix A has an entire
eigendecomposition (Λ, U) computed every cycle at
[modules/surrogate/intention/drift.py:116](../../modules/surrogate/intention/drift.py#L116);
the chain uses only κ = λ_max / λ_min and v_min. The likelihood-informed
subspace of the model is the right-singular subspace of Ψ_θ on the current
context, which is computed and discarded. This is also the gap that makes
Phoenix one-way instrumentation (drift audit): there is no rich-enough
state object for Phoenix to broker.

### 1.2 The unifying upgrade: Fisher-eigen-basis as a thread

The four audits independently arrive at the same construction from four
directions:

| Audit | Independent motivation |
|---|---|
| Physics | SMEFT global fits routinely use the Fisher eigenbasis of Wilson space (SMEFiT, fitmaker — Costa-Marzocca-Mimasu-Salko; Ellis-Madigan-Mimasu-Sanz-You). Rotating the disclosure probe target into Fisher eigendirections converts I2 into "the leading Fisher direction is disclosed at R² = X". |
| ML | Rotated MFVI (Stuart et al. 2025) and the Spectral Identifiability Principle (arXiv:2511.16288, 2025) supply mathematical foundations for Fisher-aligned posterior rotation. |
| Drift / Phoenix | The full eigendecomposition (Λ, U) is the natural payload for a Phoenix drift span. MCP rubrics over (Λ, U) spans become the mechanism that turns Phoenix from a logger into a control plane. |
| AL / SBI | Active subspaces (Constantine 2015), likelihood-informed subspace (Cui-Martin-Marzouk 2014), optimal low-rank inverse problems (Spantini 2015) — all give the same prescription: restrict acquisition to the subspace where the design matrix is rank-deficient. |

**The single load-bearing change**: stop discarding (Λ, U). Export it, use it
in acquisition, use it in the disclosure probe, use it as the drift signal,
and expose it through Phoenix spans + MCP. Every other proposed upgrade
hangs off this.

## 2. Tier-1 upgrades (must-do, 14-day window)

In priority order. Confidence is the probability the upgrade lands in time
*and* answers the criticism that motivates it.

### U1. Fix the parameter-count inconsistency in Table 2

**Effort**: 0.5 dev-day. **Confidence**: high.
**Files**: `paper/alethia.tex` Table 2 + Section 5.4; re-source the number
from `experiments/full-chain-run/output/identifiability_summary.json`
(5328-param run) rather than `experiments/intention-vs-deepsets/output_supercharged/summary.json`
(20896-param run).
**Success**: the parameter count in the table caption matches the model
that produced the cited R². No referee can flag the discrepancy.

### U2. Report the full operator disclosure table with physics explanations

**Effort**: 1 dev-day. **Confidence**: high.
Add rows for c_lq^(1), c_Hq^(3), c_Hq^(1) with their actual R² (probably
near zero on the m_ll-only observable). Frame the c_Hq nulls as
**predicted by physics**: dimension-six vertex operators produce rate-only
modifications proportional to (v/Λ)² that are degenerate with the SM
normalisation on m_ll alone; the model correctly fails to disclose them
because the kinematic observable does not resolve them. Frame the c_lq^(1)
null as the u/d-isospin partner degeneracy with c_lq^(3) under the flavour
assumption used in the calculator.
**Success**: a four-row table replaces the one-row Table 2; the paper reads
as a quantified identifiability map of dim-6 DY rather than a one-out-of-
four positive result.

### U3. Add a Prior-Fitted Network (PFN) baseline

**Effort**: 2 dev-days. **Confidence**: high.
The ML audit identifies this as the load-bearing missing experiment.
Train a small transformer ICL head (~6k params, matched to Intention) on
the same SMEFT scenario prior using the PFN posterior-predictive recipe
(Müller et al. 2022, Hollmann et al. TabPFN, Schmitt-Radev 2025
arXiv:2504.17660). Add the row to Table 1.
**Success**: the comparison table includes Intention vs. fixed-Intention vs.
DeepSets vs. PFN vs. cheat regressor — and either Intention beats PFN
(strengthening the closed-form-ridge claim) or PFN matches Intention (in
which case Section 7 honestly positions ALETHIA as a closed-form PFN
specialisation, which is a defensible contribution).

### U4. Cite the concurrent SMEFT foundation-model paper (arXiv:2512.15862, Dec 2025)

**Effort**: 0.5 dev-day. **Confidence**: high.
The ML audit flags this as the natural sister work. Add a discussion
paragraph in Section 7 positioning ALETHIA relative to it: that paper
likely uses standard transformer ICL; ALETHIA's contribution is the
closed-form ridge head + the closed-loop runtime + the disclosure probe.
**Success**: the paper is not vulnerable to "did the authors know about
2512.15862?" at review.

### U5. Eigen-decomposition payload in drift spans + MCP rubric

**Effort**: 2 dev-days. **Confidence**: high (Phoenix-side wiring),
medium (whether the MCP rubric actually closes the runtime loop in time).
Three pieces:

1. **Export** (Λ, U) on every `chain.drift.eigen` Phoenix span from
   `modules/surrogate/intention/drift.py` (one new field on the span).
2. **MCP rubric**: a query that asks Phoenix "since cycle T0, return all
   drift firings where the smallest-k eigenvalues of A fall below θ, and
   the union of their eigenvectors". The drift audit has a draft span
   schema.
3. **Eigen-redirected acquisition**: replace the uniform candidate pool in
   `experiments/full-chain-run/run.py:358` with importance sampling from
   `r(m) = Σ_{j ∈ S_low} (ψ_θ(m) · u_j)² / λ_j` over the k smallest
   eigen-directions. Sherman-Morrison update unchanged.

**Success**: when the engineered drift event fires, Phoenix can be queried
to return the under-resolved eigen-directions of A, and the next
acquisition demonstrably places oracle calls in the subspace those
directions span. This is the Phoenix-as-control-plane story.

### U6. Bimodal drift experiment to discriminate EPIG from random

**Effort**: 1.5 dev-days. **Confidence**: medium-high.
Section 6.4 admits EPIG = random on the engineered unimodal drift. Design
a bimodal drift event: simultaneously withhold c_lq^(3) = 0.8 AND
c_Hq^(3) = -0.5, producing information peaks at high m_ll (4-fermion tail)
and low m_ll (vertex-operator rate region). Re-run the acquisition
comparison at identical budget.
**Success**: EPIG-with-eigen-redirection visibly outperforms random on the
bimodal event in the recovery curve. If it does not, that is itself a
publishable null result and Section 6.4 gets framed as "we have a
candidate failure-mode probe for AL on a 1D regression manifold".

### U7. Rotate the disclosure probe into the Fisher-information eigenbasis

**Effort**: 1.5 dev-days. **Confidence**: medium-high.
The physics audit's eigen-component-C variant. Compute the empirical Fisher
information of the analytic oracle on the training c-distribution. Its
eigenvectors define a rotated Wilson basis {c̃_1, c̃_2, c̃_3, c̃_4} ordered
by leading sensitivity. Re-run the disclosure probe in this basis. The
leading direction will be a linear combination dominated by c_lq^(3) and
the probe R² will be high; the trailing directions will be undisclosed but
*by construction* are the directions m_ll cannot resolve.
**Success**: the disclosure section reports "the leading Fisher eigen-
direction (a c_lq^(3)-dominated combination) is disclosed at R² > 0.9 on
the magnitude-extrapolation band". This is a strictly stronger statement
than the current one-of-four result.

**Tier-1 total**: 9 dev-days. With one developer, leaves margin within the
14-day window for paper edits and chain re-runs.

## 3. Tier-2 upgrades (high-value, if budget allows)

### U8. Expose morphing-decomposition channels in the oracle

**Effort**: 1 dev-day. **Confidence**: high.
The AL/SBI audit identifies this as the single largest underexploited
asset. [modules/analytic_smeft/smeft.py:516-520](../../modules/analytic_smeft/smeft.py#L516)
already computes `sm_only`, `interference`, `bsm_squared` as named channels;
[modules/surrogate/oracle_smeft.py:104-112](../../modules/surrogate/oracle_smeft.py#L104)
collapses them into a scalar μ. Surface them. This unblocks U9 and U10
and costs nothing on its own.

### U9. Joint-score auxiliary loss (Mining Gold)

**Effort**: 2 dev-days. **Confidence**: medium-high.
The Brehmer-Cranmer-Louppe-Pavez "Mining Gold" trick: the analytic
oracle gives the joint score ∂ log σ / ∂c_i in closed form (it is just the
derivative of the morphing polynomial). Add a small MSE auxiliary loss
between the linear-probe prediction's gradient w.r.t. c and the analytic
joint score. This is a free training signal that the current loss throws
away.
**Success**: the disclosure probe R² for the under-disclosed operators
moves up by something measurable (target: from R² ≈ 0 to R² > 0.3 on
c_Hq directions).

### U10. Posterior-covariance disclosure (probe upgrade)

**Effort**: 2 dev-days. **Confidence**: high.
The current probe is a point estimate (`RidgeCV.predict`,
[experiments/full-chain-run/identifiability_probe.py:114](../../experiments/full-chain-run/identifiability_probe.py#L114)).
Upgrade to a Bayesian linear regression that returns a posterior
covariance Σ_hat over c. This makes the disclosure result an SBI result:
posterior point + covariance + coverage. Reuses the closed-form ridge
machinery already in the chain.
**Success**: Section 5.4 reports both ĉ and a calibrated 68% interval per
Wilson coefficient; the disclosure becomes a proper amortised SBI claim.

### U11. Scaling sweep on ψ_θ width / depth

**Effort**: 1 dev-day. **Confidence**: high.
The ML audit notes that the paper invokes "scaling-behaviour analyses now
standard in the jet-level foundation model literature" but performs none.
A one-day sweep of ψ_θ output dimension D ∈ {4, 8, 16, 32, 64, 128} on
held-out R² and on the disclosure probe gives a scaling figure.
**Success**: a one-panel scaling figure in Section 7.

### U12. Cite TabPFN-as-SBI (Schmitt-Radev 2025, arXiv:2504.17660) and reposition

**Effort**: 0.5 dev-day. **Confidence**: high.
The AL/SBI audit identifies this as the direct methodological competitor.
ALETHIA is structurally the closed-form-ridge specialisation of the PFN
approach to SBI on a physics target. Acknowledging this strengthens the
paper rather than weakens it: closed-form attention specialises a
fully-generic ICL head to a regression problem where the inductive bias
of "Bayesian ridge in a learned basis" is exact.

**Tier-2 total**: 6.5 dev-days. With Tier-1 this is 15.5 dev-days — tight
but feasible with one developer if Tier-2 begins on day 4.

## 4. Tier-3 stretch (post-deadline or paper v2)

- **U13**. Multi-fidelity dispatcher LO → MG with active selection of fidelity
  (4-5 dev-days, medium confidence). Multi-fidelity AL is its own literature
  (Kandasamy-Krishnamurthy-Schneider-Poczos 2017; MFBOpt).
- **U14**. JEPA-style pretext task for ψ_θ (3-4 dev-days, medium confidence).
  Predict masked m_ll context entries from unmasked ones to pretrain ψ_θ
  before the meta-training step.
- **U15**. Set-Transformer / Attentive Neural Process baselines (2 dev-days,
  medium-high confidence). Strengthens the architectural comparison
  beyond DeepSets.
- **U16**. Multi-observable extension (m_ll + p_T + |y_ll|) (5-7 dev-days,
  medium confidence). Lets the c_Hq operators become disclosed and
  changes the disclosure table qualitatively.

## 5. The Fisher-eigen module: concrete design

This is the load-bearing technical artefact. Sketch in one place.

### 5.1 Span schema

A new Phoenix span `chain.drift.eigen`, emitted once per drift evaluation
from `modules/surrogate/intention/drift.py`. Attributes:

```
aletheia.eigen.lambda          float[D]   # full eigenvalue spectrum of A
aletheia.eigen.U_top            float[D,k] # top-k eigenvectors
aletheia.eigen.U_low            float[D,k] # bottom-k eigenvectors
aletheia.eigen.kappa            float      # condition number (existing)
aletheia.eigen.lis_rank          int       # # eigenvalues > tau*lambda_max
aletheia.drift.action            string    # aggregator action (existing)
```

### 5.2 MCP rubric

```
since cycle T0, return drift firings where
  any(lambda_j < theta * lambda_max)
group by the eigenvector u_j responsible
return the union of {u_j} as the "current LIS gap subspace"
```

### 5.3 Eigen-redirected acquisition

Replace the uniform candidate pool (currently `np.linspace` in
`experiments/full-chain-run/run.py:358`) with importance sampling from

    r(m) = Σ_{j ∈ S_low} (ψ_θ(m) · u_j)² / λ_j

where S_low is the bottom-k eigen-direction set returned by the MCP
rubric. EPIG scoring runs on the resampled pool; everything downstream
(Sherman-Morrison update, conformal refit, drift detector) is unchanged.

### 5.4 Disclosure-side rotation

Compute the empirical Fisher information matrix of the analytic oracle on
the training c-prior: F_ij = E[∂_i log σ · ∂_j log σ]. Eigendecompose
F = V D V^T. The rotated Wilson basis is c̃ = V^T c. Run the disclosure
probe on c̃ rather than c. Report per-eigendirection R² in the new
Table 2.

### 5.5 Visualisation

A four-panel figure replacing `chain_drift_events.png`:

- top: top-3 and bottom-3 eigenvalues of A through the chain, log scale
- top-middle: |cos(u_low,1(t), u_low,1(t-1))| trajectory (eigenvector
  stability — drift events visible as cosine drops to near-zero)
- bottom-middle: oracle calls placed in m_ll, coloured by their projection
  onto u_low,1 at the time of acquisition
- bottom: drift-detector firings + aggregator actions (existing)

This figure makes "drift = eigen-collapse = re-projection event" visible
in a single glance.

## 6. 14-day critical path

| Day | Work |
|----|------|
| 1 | U1 (fix Table 2 number-source) + U4 (cite arXiv:2512.15862) + U12 (cite TabPFN-as-SBI). One commit; paper-only changes. |
| 2 | U2 (full operator table with physics framing). Paper-only. |
| 3-4 | U3 (PFN baseline implementation + run + Table 1 row). |
| 5-6 | U5 (eigen-payload in drift spans + MCP rubric + eigen-redirected acquisition). Code-heavy. |
| 7 | U7 (Fisher-basis disclosure rotation). Reuses U5 infrastructure. |
| 8-9 | U6 (bimodal drift experiment design + run + Section 6.4 rewrite). |
| 10 | U8 (expose morphing channels) + U11 (scaling sweep, runs overnight). |
| 11-12 | U9 (joint-score auxiliary loss + retrain + disclosure probe re-run) + U10 (posterior covariance probe). |
| 13 | Figure regeneration (especially the new eigen-trajectory figure from §5.5); paper edits to Sections 5, 6, 7. |
| 14 | Final read; submission. |

The first 4 days are paper-only + one ML baseline; that is the safe core.
Days 5-9 are the eigen-direction module — this is the high-leverage block.
Days 10-12 are Tier-2 if on schedule, else cut.

## 7. Confidence in the overall plan

Tier-1 alone (U1-U7) addresses every concrete criticism the four audits
raise, with the eigen-direction module as a coherent through-line that
connects the physics, ML, drift and AL/SBI stories. Even if Tier-2 slips,
the paper at the end of Tier-1 reads as quantitatively stronger and
better-positioned than the current submission.

The single highest-risk item is U5 (eigen-direction MCP rubric closing the
runtime loop). If the MCP wiring slips, the fallback is to emit the spans,
run the eigen-redirected acquisition from a local query against Phoenix's
HTTP API, and present the MCP rubric as a design rather than a live
demonstration — which still satisfies the Arize track requirements and
still tells the Phoenix-as-control-plane story.

## 8. Consolidated literature additions

Sixteen references that should appear in the bibliography after the
upgrades, organised by where they cite:

**Active learning / BOED**:
- MacKay 1992 (information-theoretic AL foundations)
- Houlsby et al. 2011 (BALD)
- Smith-Foster-Rainforth 2023 (EPIG — already cited)
- Kirsch-Amersfoort-Gal 2019 (BatchBALD)
- Bickford Smith et al. 2023 (target-aware AL, EPIG failure modes)

**Simulation-based inference**:
- Cranmer-Brehmer-Louppe 2020 (SBI frontiers review)
- Tejero-Cantero et al. 2020 (sbi package)
- Lueckmann et al. 2021 (sbibm benchmarks)
- Schmitt-Radev 2025, arXiv:2504.17660 (TabPFN-as-SBI — direct competitor)
- Hermans-Begy-Louppe 2020 (NRE)

**In-context learning / PFNs**:
- Müller et al. 2022, arXiv:2112.10510 (PFN foundations)
- Nagler 2023 (PFN statistical foundations)
- Akyürek et al. 2022, arXiv:2211.15661 (what algorithm is ICL)
- Mahankali et al. 2023, arXiv:2307.03576 (one step of GD is optimal ICL)

**Fisher-eigen / active subspace**:
- Constantine 2015 (active subspaces book)
- Cui-Martin-Marzouk 2014 (likelihood-informed subspace)
- Spantini et al. 2015 (optimal low-rank Bayesian inverse)
- Stuart et al. 2025 (rotated MFVI)

**Physics-side (additional)**:
- arXiv:2512.15862 (concurrent SMEFT FM paper — must cite)
- Costa-Marzocca-Mimasu-Salko (fitmaker, Fisher eigen-basis in SMEFT)

## 9. The summary

Three years of SMEFT, four months of foundation-model architecture, two
weeks of closed-loop engineering, and one week of audit work all
independently say the same thing: the chain is computing the right object
and using one scalar from it. Expose the rest of the eigendecomposition,
let Phoenix broker queries on it, let acquisition restrict to its rank-
deficient subspace, and let the disclosure probe be evaluated in its
eigenbasis. The paper's existing claims become stronger, the Phoenix
integration becomes load-bearing rather than instrumental, the EPIG-vs-
random result becomes a designed experiment rather than an accident, and
the disclosure result becomes an amortised-SBI posterior on the Fisher
eigenbasis of the dimension-six SMEFT operator space.
