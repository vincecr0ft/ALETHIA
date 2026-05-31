# Audit: ALETHIA's overlap with active learning / BOED and simulation-based inference

This audit complements the existing three (`audit-physics.md`, `audit-ml.md`,
`audit-drift-phoenix.md`) by reading the paper at
`/home/vince/ALETHIA/paper/alethia.tex` against two adjacent literatures that
the submission engages only glancingly: (i) Bayesian active learning / optimal
experimental design (BOED) and (ii) simulation-based inference (SBI). The
question this audit settles is: where in those two taxonomies does ALETHIA
actually sit, what does it claim to be doing relative to them, and what cheap
upgrades use machinery the repo already has.

## The real positioning

ALETHIA's closed loop is, viewed from the AL literature, a
**drift-gated, model-based, batch-mode, sequential acquisition** scheme with a
**closed-form (Bayesian-linear-regression) information criterion** (EPIG; Smith
et al. 2023) on a **learned feature map**. The acquisition decision factors
into two layers: a *gating layer* (the three drift detectors of Sec. 6.1)
chooses whether to acquire at all this cycle; an *inner scoring layer* (EPIG
or random or leverage-greedy) chooses which kinematic points to query inside
the drifted region. The gating layer is the more unusual contribution; the
inner scoring layer is textbook D-/A-optimal-style BOED in a learned basis.
The Sherman-Morrison update of `A^{-1}` is the streaming-Bayesian-ridge
realisation of an online BOED scheme.

From the SBI literature the framing is sharper: the problem statement
("amortised inference of Wilson coefficients `c` from a thin sample of
observations `(M_ctx, Y_ctx)`") is the standard amortised-SBI problem. The
disclosure linear probe `c ~ W w_implicit + b` is a degenerate posterior
point-estimate of an NPE flow head; the implicit ridge weight vector
`w_implicit` is a sufficient summary statistic in the kinematic basis
`psi_theta(m)`. The paper cites MadMiner (Brehmer et al.) and the
parametrised-classifier line (Chen et al.) but does not engage with the modern
SBI methodology — NPE, NLE, NRE, sbi-package, sbibm. The closest published
direct competitor is **TabPFN-as-SBI / NPE-PFN (Schmitt-Radev et al. 2025;
arXiv:2504.17660)**, which is not cited; ALETHIA is structurally the same
"prior-fitted in-context posterior" recipe with the transformer block replaced
by a closed-form ridge head and the prior specialised to dimension-six SMEFT.

The "foundation model" label sits awkwardly with both literatures. From AL it
is "Bayesian linear regression with a meta-learned basis"; from SBI it is "a
PFN-style amortised SBI model that returns only the predictive mean instead of
the full posterior". Both are defensible contributions in their own
terminology; neither is "foundation model" in the contemporary jet-FM sense
(no self-supervised pretext, no fine-tune transfer).

## AL / BOED literature review (2018-2026)

1. **Houlsby, Huszar, Ghahramani, Lengyel 2011 — BALD** (arXiv:1112.5745). The
   parametric Bayesian active-learning criterion that EPIG sharpens by
   target-distribution-weighting. ALETHIA's setup is exactly the regime
   where BALD chases parameter uncertainty in regions of no predictive
   relevance — the high-leverage corners of the design matrix are not
   necessarily where the FM's downstream prediction error is largest.
2. **Kirsch, van Amersfoort, Gal 2019 — BatchBALD** (arXiv:1906.08158).
   Jointly-scored batch acquisition; the natural comparator for ALETHIA's
   `k=5`-at-a-time EPIG selection. The current chain uses sequential-greedy
   Sherman-Morrison updates within a batch, which is a known suboptimal
   approximation to BatchBALD on correlated candidates.
3. **Kirsch, Farquhar, Gal 2021 — PowerBALD / stochastic acquisition**
   (arXiv:2106.12059). Stochastic acquisition outperforms argmax on smooth
   information surfaces — directly relevant to the EPIG-vs-random tie in
   Sec. 6.4 of the paper.
4. **Bickford Smith, Foster, Rainforth 2023 — EPIG** (arXiv:2304.08151). The
   acquisition function the paper uses. Section 4 of that paper documents
   that EPIG collapses to uniform-on-support when the target distribution is
   broad and the information geometry is approximately flat; the paper's
   Sec. 6.4 result (uniform random ties EPIG on the single-direction band) is
   the canonical case Smith et al. flag, not a novel finding.
5. **Kirsch, Rainforth, Gal 2021 — Test Distribution-Aware Active Learning**
   (arXiv:2106.11719). The target-set-conditioning machinery EPIG inherits;
   relevant to the eigen-direction proposal because the target distribution
   can be rotated into the active subspace.
6. **Pukelsheim 1993 / Chaloner-Verdinelli 1995** — D-optimal and Bayesian
   optimal experimental design classical foundations. The leverage-greedy
   acquisition in `experiments/full-chain-run/acquisition_compare.py` is
   D-optimal in the Intention-feature basis. The two acquisitions (EPIG and
   D-optimal) collapsing to the same RMSE on a single-direction drift is a
   known result: under Bayesian-linear-regression with a uniform target,
   EPIG and D-optimal coincide (Chaloner-Verdinelli 1995 §2.3).
7. **Forrester, Sobester, Keane 2008; Kandasamy et al. 2017 —
   Multi-fidelity AL / multi-fidelity Bayesian optimisation**
   (arXiv:1603.06288). The analytic LO oracle plus MadGraph oracle are a
   two-tier fidelity ladder; ALETHIA wires both but never exercises them as
   a multi-fidelity AL problem. The "tier" annotation in
   `modules/surrogate/oracle_madgraph.py` is currently a cost label, not an
   acquisition input.
8. **Mukhopadhyay, Pal, Tewari 2022 — Conformal triggers for safe
   exploration** (arXiv:2202.11203). Concrete literature anchor for the
   "drift-gated AL" pattern. ALETHIA's BH-corrected binomial coverage test
   is exactly the conformal-trigger primitive these authors formalise.
9. **Riis et al. 2022 — Bayesian Active Learning with Fully Bayesian
   Gaussian Processes** (NeurIPS 2022). The right comparator for the
   per-scenario GP-Matern baseline the paper already reports. The ALETHIA
   ridge head with a learned basis is a structured-kernel BAL with a
   meta-learned kernel.
10. **Constantine 2015 — Active Subspaces** (SIAM). Plus Cui-Martin-Marzouk
    2014 (arXiv:1403.4680), Spantini et al. 2015. The
    likelihood-informed-subspace (LIS) line. These give *independent
    motivation* for Fisher-eigenbasis-aligned acquisition: the active
    subspace is the eigenspace of `E[(∇f)(∇f)^T]`, and in the closed-form
    Bayesian-linear-regression special case that ALETHIA implements, the
    active subspace **is** the right-singular subspace of
    `Psi_theta(M_ctx)`. The full eigendecomposition of `A = Psi^T Psi +
    alpha I` is computed every cycle in `drift.py:116` and discarded
    (audit-drift-phoenix.md §3).
11. **Müller, Ulmer et al. 2025 — Active in-context learning with PFNs**
    (arXiv:2502.00135). Closest published work on AL-driven *context
    curation* for in-context learners. They show that selecting context
    points by acquisition score on the PFN's predictive variance halves
    the context size for a fixed target accuracy. This is exactly the
    inner scoring layer ALETHIA already implements; the paper does not cite
    it.
12. **Renggli, Bachem, Bachem et al. 2022 — Active context selection
    for tabular ICL** (arXiv:2206.05131). Earlier work in the same vein.

The literature pattern: ALETHIA's machinery overlaps cleanly with the
recently-converging "AL on PFNs" subfield. The differentiation from that
subfield is the *physics-aware* drift gating and the *closed-form*
acquisition (EPIG-on-Bayesian-linear-regression is closed-form; EPIG on a
neural PFN is Monte-Carlo).

## SBI literature review (2018-2026)

1. **Papamakarios, Murray 2016; Greenberg, Nonnenmacher, Macke 2019 —
   Sequential Neural Posterior Estimation (SNPE)** (arXiv:1605.06376,
   arXiv:1905.07488). The original NPE recipe: train a normalising flow
   `q_phi(c | x)` on simulator samples, refine sequentially on
   observation-targeted batches. ALETHIA's disclosure probe is a degenerate
   (point-estimate-only) form of NPE; the natural upgrade is to replace
   the `RidgeCV` probe in `experiments/full-chain-run/identifiability_probe.py:114`
   with a Bayesian linear regression that returns `c_hat ± Sigma_hat`, or
   with a small flow.
2. **Hermans, Begy, Louppe 2020 — Neural Ratio Estimation (NRE)**
   (arXiv:1903.04057). The likelihood-ratio formulation that the
   parametrised-classifier line of Chen et al. (cited in the paper)
   instantiates. ALETHIA's `mu(c, m)` regression is the regressor form of
   exactly this construction.
3. **Lueckmann, Bassetto, Karaletsos, Macke 2019 — Sequential Neural
   Likelihood Estimation (SNLE)** (arXiv:1805.07226). The
   complementary-to-NPE line: estimate the likelihood instead of the
   posterior. Mentioned for completeness; ALETHIA does not currently target
   this regime.
4. **Cranmer, Brehmer, Louppe 2020 — Frontier of Simulation-Based
   Inference** (PNAS 117:30055). The reference review; positions
   MadMiner-style "Mining Gold" as one of three SBI families. The paper
   cites Brehmer et al. (the SMEFT instantiation) but does not cite
   Cranmer-Brehmer-Louppe.
5. **Brehmer, Cranmer, Louppe, Pavez 2018 — Mining Gold from Implicit
   Models** (arXiv:1805.12244). The joint-score/joint-likelihood-ratio
   training signal. **Crucial point for ALETHIA**: the analytic oracle in
   `modules/analytic_smeft/smeft.py` returns the morphing decomposition
   `(sm_only, interference, bsm_squared)` as three named channels of its
   output dict (see `differential_xs:516-520`). The Wilson-coefficient
   derivative `∂σ/∂c_i = A_i(m) + 2 Σ_j B_ij(m) c_j` is **closed-form
   extractable** from those channels. The current `AnalyticSMEFTOracle`
   wrapper at `oracle_smeft.py:103-112` collapses to a scalar `mu` and
   discards the per-channel breakdown. This is the single underexploited
   asset in the repo.
6. **Tejero-Cantero, Boelts, Cranmer et al. 2020 — sbi package**
   (arXiv:2007.09114). The reference open-source toolkit; would let the
   paper run NPE / NLE / NRE on the same prior with a few hundred lines
   and would settle the "would a flow beat ridge?" question.
7. **Lueckmann, Boelts, Greenberg et al. 2021 — sbibm benchmark**
   (arXiv:2101.04653). Standardised SBI benchmarks; ALETHIA's pretrained
   model could be evaluated on the closest sbibm task (linear-Gaussian
   with structured priors) as a sanity check on the disclosure result.
8. **Sharrock, Simons, Liu, Beaumont 2022 — Score-based SBI**
   (arXiv:2210.04872), and **Geffner, Papamakarios, Mnih 2022**. The
   score-matching SBI line. Relevant because Schmitt-Radev 2025 builds
   on this for their TabPFN-as-SBI construction.
9. **Schmitt, Radev et al. 2025 — Effortless Simulation-Efficient
   Bayesian Inference using Tabular Foundation Models**
   (arXiv:2504.17660). **The direct methodological competitor.** TabPFN
   is repurposed as a training-free SBI engine; a single forward pass
   returns an approximate posterior on simulator parameters. ALETHIA is
   the closed-form-attention specialisation of the same idea on SMEFT.
   The differentiation is: (a) ALETHIA's closed-form ridge admits exact
   sequential Sherman-Morrison updates, TabPFN-as-SBI does not; (b)
   ALETHIA's prior is HEP-specific and the ridge weights are
   physics-interpretable via the morphing decomposition; (c) TabPFN-as-SBI
   returns a full posterior, ALETHIA returns a predictive mean. The paper
   does not cite this work.
10. **Müller, Hollmann, Arango, Grabocka, Hutter 2022 — Prior-Fitted
    Networks** (arXiv:2112.10510). Cited in the paper but not compared
    against; this is the parent line of which ALETHIA + TabPFN-as-SBI are
    branches.
11. **Nagler 2023 — Statistical foundations of PFNs** (ICML 2023).
    Convergence theorem: PFNs converge to true Bayes-optimal posterior
    predictive at the appropriate rate. The closed-form-attention variant
    inherits this guarantee under conjugate-Gaussian assumptions; the
    paper could state this as a theorem.
12. **Heinrich, Stevens, Cranmer 2024 — neos / pyhf-integrated NPE
    workflows** (arXiv:2403.14773). Position the SBI-in-HEP line forward
    from MadMiner; relevant because the ALETHIA closed-loop architecture
    is closer to neos's pyhf-in-the-loop than to MadMiner's offline
    classifier training.

The pattern: ALETHIA reads cleanly as "TabPFN-as-SBI for SMEFT with a
closed-form ridge head, plus a drift-gated active-acquisition loop". That is
a sharper positioning than the paper's current "closed-form attention
foundation model"; it has named parent lines (PFN + NPE + EPIG + LIS), all of
which the paper could cite.

## What is underexploited in the current repo

### The morphing-decomposition channel is computed and discarded

`modules/analytic_smeft/smeft.py:516-520` returns `sm_only`,
`interference`, and `bsm_squared` as named keys of the oracle output dict.
These are exactly the `A_i(m)` and `B_ii(m) + Σ_j B_ij(m) c_j` morphing
coefficients (paper Eq. 2). The wrapper at
`modules/surrogate/oracle_smeft.py:104-112` discards the decomposition by
computing `mu = differential_xs / sm_only` and returning the scalar. This
means the closed-form joint score and joint likelihood ratio that the
"Mining Gold" line of Brehmer et al. 2018 trains against are available for
free from the oracle and not exposed to any downstream code.

### The implicit ridge solve is a sufficient statistic but only its mean is read

The disclosure probe at
`experiments/full-chain-run/identifiability_probe.py:114-117` calls `RidgeCV`
and returns `c_hat = probe.predict(w_implicit)` — a point estimate. The
Bayesian linear regression in the ridge basis admits a full posterior
covariance in closed form: `Sigma_hat = (X^T X + lambda I)^{-1} sigma^2`.
Adding three lines to return `(c_hat, Sigma_hat)` upgrades the disclosure
from a regression score to a posterior with uncertainty.

### The design-matrix eigendecomposition is computed and discarded

This is the load-bearing observation from `audit-drift-phoenix.md` §3 and
recurs here. `drift.py:116` calls `np.linalg.eigh(A)` and returns only
`kappa = eigs.max() / eigs.min()`. The full `(Lambda, U)` is the active
subspace of the closed-form ridge in the LIS / Constantine sense
(Spantini 2015 connects them directly).

### The two-tier oracle ladder is wired but not used as multi-fidelity AL

`modules/surrogate/oracle_madgraph.py` implements the MadGraph oracle with a
disk cache and noise-floor calibration. The full chain runs against the
analytic LO oracle exclusively (paper §6.2). A genuine multi-fidelity AL
loop — start cheap, escalate to MG on uncertain points — is supported by
the existing infrastructure and is not implemented.

### The forward-backward and pT observables are wired but not in the loop

`oracle_smeft.py:127-238` exposes `truth_pt`, `truth_mu_fb`, `truth_afb`.
The disclosure failure of the audits — `c_lq^(1)` not separable from
`c_lq^(3)` on `m_ll` alone — is fixable by adding `A_FB(m_ll)` as a
second observable; the oracle method exists, the surrogate context is
single-channel.

## Proposed upgrades

Each is sized in developer-days with a one-line success criterion.

### S1. Morphing-decomposition channels surfaced as oracle outputs (1 day, confidence HIGH)

Modify `AnalyticSMEFTOracle.truth` in `oracle_smeft.py:86-113` to return a
named tuple `(mu, sm_only, interference, bsm_squared)` or a structured
array. Add a `morphing_decomposition` flag that, when set, exposes the
three channels alongside `mu`. The training loop in
`experiments/intention-vs-deepsets/experiment_smeft.py` can then optionally
add an auxiliary loss on the interference and BSM-squared channels.
*Success criterion*: the channels appear in `summary.json` of the next
training run and one auxiliary-loss ablation lands within 1% of the
no-aux baseline on held-out R^2.

### S2. Posterior-covariance disclosure probe (2 days, confidence HIGH)

Replace `RidgeCV` in `identifiability_probe.py:114` with a Bayesian linear
regression returning `(c_hat, Sigma_hat)`. Add per-operator credible
intervals to the output JSON. Compare nominal vs empirical coverage of the
68% interval on a held-out scenario pool. This is the SBI-flavoured
upgrade: the disclosure becomes a *posterior*, not a regression score.
*Success criterion*: per-operator 68% empirical coverage in
`[0.60, 0.76]` on the in-distribution pool, demonstrably so for at least
two of the four operators.

### S3. Joint-score signal from the morphing oracle (2-3 days, confidence MEDIUM-HIGH)

The analytic oracle's morphing decomposition gives `∂σ/∂c_i = A_i(m) + 2
Σ_j B_ij(m) c_j` in closed form. Add a method
`AnalyticSMEFTOracle.joint_score(c, m) -> ∂ log p / ∂c` returning the
N-Wilson-dimensional score at the scenario. Add an auxiliary regression
head to the Intention training that predicts the score from `w_implicit`,
weighted by a hyperparameter `lambda_score`. This is the "Mining Gold"
move applied to the closed-form ridge.
*Success criterion*: with `lambda_score > 0`, the disclosure probe R^2 on
the *currently-undisclosed* `c_lq^(1)` direction rises above 0.5 in the
withheld band.

### S4. NPE baseline against the same prior (3-4 days, confidence MEDIUM)

Implement a vanilla NPE flow head (e.g. masked autoregressive flow,
sbi-package default) on the same SMEFT prior and the same K=12 context.
The flow's posterior is `q_phi(c | M_ctx, Y_ctx)`. Report posterior mean
and credible interval on the same in-distribution and withheld-band
pools, side-by-side with the ridge probe. This is the apples-to-apples
SBI baseline the paper currently lacks.
*Success criterion*: the comparison lands as one of three rows in a
`disclosure_npe_vs_ridge.json`; if the NPE flow's posterior log-density
on the true `c` exceeds the ridge probe's by more than the bootstrap CI
on either pool, the paper has a new headline result; if not, the paper
has a defensible claim that closed-form ridge matches NPE at a fraction
of the parameter count.

### S5. Fisher-eigen-basis-aligned acquisition (3-4 days, confidence MEDIUM)

The active-subspace / LIS literature (Constantine 2015; Cui-Martin-Marzouk
2014; Spantini 2015) gives a closed-form rotation of `psi_theta` into the
likelihood-informed subspace. Implement two variants:
(a) compute the empirical Wilson-Fisher information `F_ij = Σ_k (∂μ/∂c_i)
(∂μ/∂c_j) / σ_y^2` from the morphing decomposition on the current
context (closed-form, ~10 µs at d_psi=16);
(b) modify the EPIG candidate pool to importance-sample from the
projection of `psi_theta(m)` onto the leading Fisher eigenvectors.
This is the eigen-direction proposal in §1.4 of `audit-drift-phoenix.md`
made physics-aware (Wilson-space Fisher rather than feature-space Gram).
*Success criterion*: on a bi-modal drift event (the experiment design in
`audit-drift-phoenix.md` §1.3), Fisher-eigen-redirected EPIG separates
from uniform random by a factor >2 on final-state RMSE on the joint
target band.

### S6. Multi-fidelity acquisition loop (4-5 days, confidence MEDIUM)

Add a cost-weighted acquisition that scores `EPIG / cost(tier)` and picks
the analytic-LO oracle by default, escalating to MadGraph only when the
top analytic-LO candidate falls below an information-per-cost threshold.
The two-tier infrastructure exists; only the dispatcher is missing.
Anchored on Kandasamy et al. 2017 multi-fidelity BO.
*Success criterion*: on the engineered drift event, the multi-fidelity
loop reaches RMSE 2.65 at half the wall-clock cost of the analytic-only
loop, by deferring MG calls to the high-leverage cycles only.

## Connection to the eigen-direction theme from the prior audits

The three existing audits converge on the same proposal from three angles:
audit-physics §"eigen-direction" wants a Fisher-aligned disclosure-probe
rebasis; audit-ml §5 wants an online-PCA / Fisher rotation of `psi_theta`;
audit-drift-phoenix §5 wants an eigen-redirected EPIG pool. The AL/SBI
literature gives the unifying motivation:

The **likelihood-informed subspace** (Cui-Martin-Marzouk 2014;
Spantini et al. 2015) is *defined* as the leading eigenspace of the
prior-to-posterior update operator, which in a linear-Gaussian model with
prior covariance `Sigma_0` and observation operator `G` and noise covariance
`Gamma` is the leading eigenspace of `Sigma_0^{1/2} G^T Gamma^{-1} G
Sigma_0^{1/2}`. In ALETHIA's closed-form ridge with homoscedastic noise, the
observation operator is `Psi_theta(M_ctx)` and the LIS reduces to the
right-singular subspace of `Psi_theta`, which is exactly the eigenspace of
the design matrix `A`.

This gives the paper a literature-anchored claim that the
chain-throws-away-then-reconstructs eigendecomposition is not an engineering
afterthought but the *natural variable* in which the closed-form ridge head
operates. Three orthogonal pieces follow from this:

1. *Drift signal.* `audit-drift-phoenix` §1.2 — coverage drift in the
   smallest-eigenvalue direction. This is LIS-rank-deficiency detection
   (Spantini 2015 §3.2).
2. *Acquisition.* `audit-drift-phoenix` §1.4 and S5 above — sample the EPIG
   pool from the projection onto the worst-resolved LIS directions. This is
   active-subspace AL (Constantine et al. 2014; the 2024 Struct. Multidisc.
   Optim. paper cited in audit-drift-phoenix §2).
3. *Disclosure.* `audit-physics` Component C — rotate the linear probe
   target into the Wilson-space Fisher eigenbasis. This converts
   "one-of-four operators recovered" into "the leading Fisher direction
   recovered at R^2 = 0.94, second at R^2 = 0.3, third and fourth below
   threshold", which is how SMEFiT and fitmaker report their global-fit
   results (Ellis et al. 2022; Giani-Marzocca-Marzo et al. 2023).

The AL/SBI literature thus gives **independent motivation, from two
different communities (BOED-LIS and SBI-active-subspace), for the same
Fisher-eigenbasis story** that the prior audits derived from the
engineering of the chain. This is the strongest single piece of literature
positioning the paper can adopt that does not require a new experiment;
the citations are 8 lines.

## Sources

- Houlsby et al. 2011, BALD: https://arxiv.org/abs/1112.5745
- Kirsch et al. 2019, BatchBALD: https://arxiv.org/abs/1906.08158
- Kirsch et al. 2021, PowerBALD: https://arxiv.org/abs/2106.12059
- Bickford Smith et al. 2023, EPIG: https://arxiv.org/abs/2304.08151
- Kirsch et al. 2021, Test-distribution-aware AL: https://arxiv.org/abs/2106.11719
- Kandasamy et al. 2017, Multi-fidelity BO: https://arxiv.org/abs/1603.06288
- Mukhopadhyay et al. 2022, Conformal triggers: https://arxiv.org/abs/2202.11203
- Constantine 2015, Active Subspaces (SIAM)
- Cui-Martin-Marzouk 2014, LIS: https://arxiv.org/abs/1403.4680
- Spantini et al. 2015, Optimal low-rank Bayesian inverse: https://epubs.siam.org/doi/abs/10.1137/140977308
- Müller et al. 2025, Active in-context learning with PFNs: https://arxiv.org/abs/2502.00135
- Papamakarios-Murray 2016, SNPE: https://arxiv.org/abs/1605.06376
- Greenberg et al. 2019, APT: https://arxiv.org/abs/1905.07488
- Hermans et al. 2020, NRE: https://arxiv.org/abs/1903.04057
- Lueckmann et al. 2019, SNLE: https://arxiv.org/abs/1805.07226
- Cranmer-Brehmer-Louppe 2020, Frontier of SBI: PNAS 117:30055
- Brehmer et al. 2018, Mining Gold: https://arxiv.org/abs/1805.12244
- Tejero-Cantero et al. 2020, sbi package: https://arxiv.org/abs/2007.09114
- Lueckmann et al. 2021, sbibm: https://arxiv.org/abs/2101.04653
- Sharrock et al. 2022, Score-based SBI: https://arxiv.org/abs/2210.04872
- Schmitt-Radev et al. 2025, TabPFN-as-SBI: https://arxiv.org/abs/2504.17660
- Müller et al. 2022, PFN: https://arxiv.org/abs/2112.10510
- Nagler 2023, Statistical foundations of PFNs (ICML 2023)
- Heinrich-Stevens-Cranmer 2024, neos/pyhf-NPE: https://arxiv.org/abs/2403.14773
