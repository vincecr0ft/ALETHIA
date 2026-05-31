# Physics audit of ALETHIA paper (alethia.tex)

Audit of the physics content of the hackathon submission at `/home/vince/ALETHIA/paper/alethia.tex`. The focus is on whether the SMEFT phenomenology and Drell-Yan claims are defensible at the level a referee would push back, not on the ML machinery (already exercised elsewhere). Code citations are to absolute paths in the repo at the audit date.

## The real questions

The paper makes four falsifiable physics claims:

1. **Architectural claim.** A closed-form linear-attention head that never ingests `c` on its forward pass is a strictly better primitive for amortised SMEFT inference than (a) a Wilson-conditioned regressor, (b) a parameter-matched DeepSets pool, (c) a fixed-basis ridge in `log(m/M_ref)`. Tested: yes, on the Drell-Yan oracle, with bootstrap CIs.
2. **Disclosure claim.** A linear probe from `w_implicit` recovers `c_lq^(3)` at R^2 = 0.94 in a magnitude-extrapolation band. Tested: yes, but **only `c_lq^(3)` is reported in Table 2 of the paper**; the same scripts under `/home/vince/ALETHIA/experiments/full-chain-run/output/identifiability_summary.json` and `/home/vince/ALETHIA/experiments/intention-vs-deepsets/output_*/summary.json` show every other operator at R^2 ≈ 0 or negative. The disclosure is **one-out-of-four**, framed as a result on the one that worked.
3. **Closed-loop recovery claim.** EPIG-driven acquisition + conformal calibration + drift gating recovers a 110x RMSE improvement on the engineered drift band. Tested: yes; but the paper's own Table 3 (acquisition_compare) shows uniform random sampling does *better* (187x). This is reported but the framing soft-pedals it.
4. **Oracle fidelity claim.** Analytic LO Drell-Yan agrees with MadGraph at ~3%, "consistent with the missing NLO K-factor". Tested: yes, at one operator point (`c_lq^(3) = 0.8`) on 50 m-bins, MG @ nevents=500 (5% MC noise floor), see `mg_crosscheck_summary.json`. Median |rel err| = 3.0%, p95 = 4.3%, max = 5.0%.

What is *not* tested and the paper does not flag:

- **Coverage of operator space.** Only 4 of the 14 operators the calculator supports (`smeft.py:69-77`) are exercised in the benchmark. The unexercised 10 include `c_eu, c_ed, c_lu, c_ld, c_qe` (RR-chirality four-fermion) which are precisely the directions that show up in lepton-charge asymmetry and `A_FB`, and they are not pure relabellings of the four operators that *are* in scope.
- **Realism of the oracle.** Pure LO + CT18NNLO convolved with parton luminosities. No EW Sudakov logs (a 10-30% effect at 2-3 TeV invariant mass), no NLO K-factor, no PDF uncertainty band, no photon-induced channels, no detector smearing, no efficiency. None of these is wired even as a placeholder.
- **Whether the disclosure result is a Fisher-information artefact.** `c_lq^(3)` is the highest-leverage direction in the high-mass tail (energy-growing four-fermion, M^4 / Lambda^4); the probe could be reading off the dominant Fisher eigenvalue and the "result" is a tautology of which Wilson direction is most kinematically distinguishable from the SM. The other three operators are exactly the ones that are nearly flat in the Fisher metric on `m_ll` alone.

## Literature review (2022-2026)

1. **Brehmer, Cranmer, Kling, Espejo 2020 — MadMiner** ([arXiv:1907.10621](https://arxiv.org/abs/1907.10621)). The reference simulation-based-inference tool for SMEFT; fits a neural likelihood-ratio estimator per Wilson configuration. The ALETHIA framing of "we amortise the inference, not the conditional density" is genuine differentiation — MadMiner trains a separate estimator per analysis.
2. **Chen, Glioti, Marzocca, Nardini, Wulzer 2021 — Parametrised classifiers for optimal EFT sensitivity** ([JHEP05(2021)247](https://link.springer.com/article/10.1007/JHEP05(2021)247)). Trains a classifier parametrised in `c`; takes the cross-section ratio as the discriminator's target. This is the closest direct competitor; ALETHIA's contribution over it is that the inference at deployment is one matrix solve in the learned basis, with no per-`c` retraining.
3. **Maltoni, Mantani, Mimasu et al. 2023 — Unbinned multivariate observables (ML4EFT)** ([JHEP03(2023)033](https://link.springer.com/article/10.1007/JHEP03(2023)033)). Builds optimal unbinned observables for global SMEFT fits using ML; demonstrates on `tt-bar` and `H+Z`. ALETHIA's binned-`m_ll` observable is strictly less informative than this; the multi-observable extension promised in the paper's outlook would have to compete with ML4EFT, not strawmen.
4. **Greljo, Marzocca 2017** ([arXiv:1704.09015](https://arxiv.org/abs/1704.09015)). The high-`pT` Drell-Yan tail constraint paper the analytic oracle's amplitude structure is built from. Cited correctly; the implementation in `modules/analytic_smeft/smeft.py` faithfully follows the chirality decomposition.
5. **Iranipour, Ubiali 2022 — SIMUnet / joint PDF+EFT fits in DY tails** ([arXiv:2104.02723](https://arxiv.org/abs/2104.02723) and follow-ups). Shows that high-mass DY constraints on four-fermion operators are correlated with PDF uncertainties at high-x. Absent from ALETHIA; the oracle uses CT18NNLO with no PDF replica spread.
6. **Mantani, ATLAS/CMS group analyses 2024** — SMEFiT 3.0 ([arXiv:2412.08311](https://arxiv.org/abs/2412.08311)). The current state-of-the-art global SMEFT fit at NLO QCD, 50 Warsaw operators across Higgs/top/diboson/EWPO. Confirms that the global-fit world routinely runs at NLO with PDFs, EW corrections, K-factors and full systematic-error treatment. ALETHIA's LO-only oracle is two fidelity tiers below this.
7. **Ellis, Madigan, Mimasu, Sanz, You 2020-22 — fitmaker** ([JHEP08(2022)308](https://link.springer.com/article/10.1007/JHEP08(2022)308) and predecessor). The reference linear (dim-6) global fit code; routinely reports principal Wilson directions as the eigenvectors of the inverse covariance / Fisher information. This is the eigen-basis literature the user pointed to.
8. **HighPT — Allwicher, Faroughy, Jaffredo, Sumensari, Wilsch 2022** ([arXiv:2207.10756](https://arxiv.org/abs/2207.10756); ScienceDirect 2023). A dedicated tool for high-`pT` DY in SMEFT with operators up to dim-8, including charged-current and EW Sudakov tails. A useful upgrade benchmark — ALETHIA's analytic oracle could be cross-validated against HighPT at no MG cost.
9. **Heinrich et al. 2024 — Masked Particle Modelling** ([arXiv:2401.13537](https://arxiv.org/abs/2401.13537)) and **Birk et al. — OmniJet-α** ([arXiv:2403.05618](https://arxiv.org/abs/2403.05618)). Self-supervised jet foundation models; cited by ALETHIA. The classification-focused claim about the jet FM cluster is accurate.
10. **Mikuni, Nachman 2024-25 — OmniLearn / OmniLearned** ([arXiv:2510.24066](https://arxiv.org/abs/2510.24066) and predecessor). Label-supervised jet FM that does *both* classification and generation; outperforms OmniJet-α on fine-tuning at small sample. ALETHIA's claim of foundation-model status would benefit from a fine-tuning ablation in the OmniLearn style.
11. **Katel et al. 2025 — HEP-JEPA** ([arXiv:2502.03933](https://arxiv.org/abs/2502.03933)). Joint-embedding predictive jet FM; 4-6 % accuracy advantage from few-shot fine-tuning. Cited; appropriate.
12. **Araz, Spannowsky 2025 — Conformal prediction as a calibration standard** ([arXiv:2512.17048](https://arxiv.org/abs/2512.17048)). The conformal-prediction-for-HEP reference the paper explicitly anchors on. Date is December 2025, which is consistent with the paper's "Araz:2025conformal" tag.
13. **Müller, Hollmann et al. 2022 — Prior-Fitted Networks (PFN)** ([arXiv:2207.05202](https://arxiv.org/abs/2207.05202)). The in-context-Bayesian-inference reference; PFN trains a transformer to do in-context prediction, ALETHIA does the same with a closed-form attention head in place of the transformer. Cited correctly.
14. **Garnelo, Czarnecki 2023 — Exploring Key-Value-Query Models with Intention** ([arXiv:2305.10203](https://arxiv.org/abs/2305.10203)). The closed-form linear attention "Intention" head. Cited correctly. Note: published as a preprint, not at a major ML venue at the time of paper writing; a referee might ask why the design choice is anchored on this rather than on standard linear-attention or Transformer formulations.
15. **NNPDF / Iranipour et al. 2023 — SIMUnet joint PDF+SMEFT** ([arXiv:2302.06660](https://arxiv.org/pdf/2302.06660) and SMEFiT 3.0 follow-on). Cleanest statement of why high-mass DY SMEFT constraints get the PDF uncertainty *folded in*; relevant for the realism-of-oracle critique.

## Findings

### What holds up

- **Architectural constraint and the cheat-regressor failure mode.** `/home/vince/ALETHIA/paper/alethia.tex:91-93` and `:131` argue, and Table 1 demonstrates, that a regressor consuming `c` directly has its parametric answer written into its feature map and exhibits a long catastrophic-failure tail (`p5 R^2 = -23` in-distribution, bootstrap lower bound `-889`). The argument is structurally sound: equation (2) of the paper is a closed-form polynomial in `c`, and a tensor-product basis recovers the morphing coefficients to numerical precision on training data and fails outside. This is a real architectural insight, even if it is not novel relative to the parametrised-classifier line.
- **Implementation of the SMEFT amplitude decomposition.** `/home/vince/ALETHIA/modules/analytic_smeft/smeft.py:146-202` faithfully implements the chirality decomposition of Greljo & Marzocca (cited correctly). The `c_lq^(3)` operator carries the isotriplet `tau^3` sign-flip on up-vs-down quarks (`smeft.py:152-153`), which is the textbook structure. The interference vs. BSM-squared decomposition (`smeft.py:199-201`) is correct.
- **Closed-form linear attention head.** `/home/vince/ALETHIA/modules/surrogate/intention/model.py:57-67` is a correct implementation of equation (3) of the paper. The Sherman-Morrison sequential update in the appendix is standard.
- **MG cross-check, on its face.** `/home/vince/ALETHIA/experiments/full-chain-run/output/mg_crosscheck_summary.json` shows median |rel err| = 3.00%, max 5.01%, 0 fidelity-drift firings. The analytic oracle is internally consistent with MG at the 3% level *at this one Wilson point and at the 5%-MC-noise floor at nevents=500*.

### What does not hold up

**The disclosure result is one-out-of-four, not "the strong-form disclosure result of the present work".** The probe summary at `/home/vince/ALETHIA/experiments/full-chain-run/output/identifiability_summary.json` reports:

```
cHq3 inside  R^2 = -0.005    withheld  R^2 = -0.227
cHq1 inside  R^2 = -0.013    withheld  R^2 = -0.090
clq3 inside  R^2 = +0.699    withheld  R^2 = +0.778    <- the one the paper reports
clq1 inside  R^2 = +0.146    withheld  R^2 = -0.473
```

The paper Table 2 (line 233) reports `c_lq^(3)` at `+0.86` inside and `+0.94` withheld; those numbers come from `/home/vince/ALETHIA/experiments/intention-vs-deepsets/output_supercharged/summary.json`, which is a *different model* — `d_psi=32, hidden=128, 20896 parameters, 3000 steps, 500 train scenarios*, i.e. **roughly 4x the parameter count and 2x the training compute of the headline 5328-parameter Intention model** the paper claims this number for. The paper's Table 1 says 5328 parameters, the disclosure number is from a 20896-parameter run, and the paper does not flag this. In the matching 5328-parameter probe in `identifiability_summary.json`, even `c_lq^(3)` only reaches R^2 ≈ 0.78 in the withheld band — still positive, but well below the "R^2 > 0.8 strong evidence" threshold the probe script itself defines at `identifiability_probe.py:23`.

**The c_Hq^(3) and c_Hq^(1) failures are not "expected by physics" — they are diagnostic of model capacity.** The paper's section 3.1 (lines 93) argues "the two vertex operators c_Hq^(3), c_Hq^(1) produce rate shifts of order (v/Lambda)^2, small enough that the held-out R^2 remains above 0.99". That is a prediction about *predictive* R^2 on `mu(m)`, which is fine — but the *identifiability* R^2 is at chance level (`-0.005, -0.013` inside, `-0.227, -0.090` withheld). The model predicts `mu(m)` accurately even when it has not identified which combination of vertex operators is responsible, because rate-only modifications are degenerate in shape on `m_ll` alone. This is a real physics statement, and it is the actual finding the paper has — but it gets buried. The user's pre-audit instinct is correct: the c_Hq operators are kinematically degenerate with the SM rate on `m_ll`, and *that* is why the model cannot disclose them, not because the model fails.

**The c_lq^(1) negative R^2 is the more interesting failure.** `c_lq^(1)` is also energy-growing four-fermion, same M^4/Lambda^4 scaling as `c_lq^(3)`; the only difference is the isotriplet structure. That the probe gets R^2 = +0.15 in-box and R^2 = -0.47 in extrapolation says the model has learned *the combination* `(c_lq^(1) - c_lq^(3))` on up-quarks plus `(c_lq^(1) + c_lq^(3))` on down-quarks (the `_contact` function in `smeft.py:152-153`) and cannot disentangle the two. This is a genuine kinematic identifiability problem in `m_ll`-only Drell-Yan; the natural disambiguator is `A_FB` (the LL+RR vs LR+RL signed combination, encoded in `_AFB_SIGNS` in `smeft.py:210-212`). The paper's own AFB experiment (`output_multiobs_afb/summary.json`) shows that even with `A_FB` added, only `c_lq^(3)` is disclosed; `c_lq^(1)` worsens. This is not noise; it is a u/d isospin degeneracy at the parton-luminosity level (`d_bar/u_bar < 1` at high-x but not zero), and resolving it requires explicit flavour separation (charge-tagged W+/W- DY or `A_FB` per quark flavour).

**The oracle is unrealistic by current global-fit standards.** SMEFiT 3.0 ([arXiv:2412.08311](https://arxiv.org/abs/2412.08311)) runs at NLO QCD with full EWPO and PDF uncertainties on 50 operators. ATLAS and CMS high-mass DY analyses (e.g. CMS-PAS-EXO-19-019) include EW Sudakov logs (10-30% at 2-3 TeV), photon-induced channels (~5% of high-mass cross section), detector smearing/efficiency and lepton-charge asymmetry. None enter ALETHIA's oracle. The 3% MG cross-check is also blind to this: MadGraph at LO without EW corrections agrees with the analytic LO calculation at LO, which is unsurprising — both miss the same physics. The paper's claim "consistent with the missing NLO-QCD K-factor (Mangano)" is loose: NLO K-factors in DY at 1-2 TeV are 1.2-1.3 (i.e. +20-30%, well outside the 3-5% MG-MC noise floor), and the agreement is *despite* both calculations missing the same NLO effects, not consistent with their absence. A genuine fidelity statement would be "LO analytic agrees with LO MG at 3%; NLO not validated".

**The "3% agreement consistent with K-factor" claim hides a sign-error blind spot.** A 3% relative error at 5% MG MC noise is below the noise floor on every point (no z > 3 firings, n_fidelity_fired = 0). At this noise level, a 5% interference sign error would be invisible: flipping the sign of the `A_i` term shifts `mu` by 2*|A_i*c|/sigma_SM, which at c = 0.8 and m = 1 TeV is ~5-10% on the interference term — comparable to the MG MC noise. The cross-check as run cannot exclude such a sign error on the interference structure. (It would be excluded by a c-sign-flip test: compute mu(+c) and mu(-c) and check the interference flips sign relative to SM; that test is not in the script.)

**The EPIG result inverts the intended story.** Table 3 in the paper (line 320) shows uniform random sampling gives 187x recovery vs EPIG's 110x. The text at line 327 says "we do not interpret this as evidence against EPIG". A more direct reading is that on a smooth, one-dimensional, energy-growing band, EPIG's information geometry is flat and any in-band sampler suffices — which is precisely a falsification of the *practical* utility of EPIG on this problem. The closed-loop machinery's actual value is the *drift gating*, which decides when to query, not which point to query. The paper now states this but it dilutes the headline.

### What is novel and defensible

- The closed-form linear-attention head as an in-context primitive for SMEFT inference is a genuine architectural contribution. The competition (MadMiner, parametrised classifiers, ML4EFT) all retrain per Wilson configuration; ALETHIA does not.
- The disclosure-from-w_implicit framing — falsifying memorisation by reading the latent — is a clean experimental design. The probe protocol is correct; only the framing of the result is misleading.
- The leverage-stratified conformal calibration is a real piece of infrastructure that the jet FM literature does not currently provide and that any deployed inference loop needs.

## Proposed upgrades

Each upgrade is sized to land in the 1-7 day window, with confidence rated against whether it closes the specific criticism.

### U1. Report the full disclosure table, not just c_lq^(3) (1 day, confidence HIGH)

Edit Table 2 of the paper to include all four operators with the actual probe outputs from `identifiability_summary.json` (the **5328-parameter model**, not the supercharged one). Add a 1-paragraph physics interpretation:

- c_lq^(3) discloses (R^2 = 0.78 withheld) because it carries the dominant M^4/Lambda^4 tail and the model has resolved its m_ll shape.
- c_lq^(1) does not disclose because it is kinematically degenerate with c_lq^(3) under the isotriplet-vs-isosinglet structure; resolving it requires charge-separated DY or explicit u/d quark tagging.
- c_Hq^(3), c_Hq^(1) do not disclose because they produce rate-only modifications at order (v/Lambda)^2 that are degenerate with the SM rate on m_ll; resolving them requires multi-observable inputs (A_FB, pT, lepton angular).

This converts a "1/4 disclosure dressed as the result" into "1/4 explained by the kinematic structure", which is a stronger physics result.

**Why this lands:** the numbers are already on disk; it's a paper edit. **Why HIGH confidence:** the explanation is textbook DY phenomenology and will read as competent rather than evasive.

### U2. Add the AFB and pT_l observables to the disclosure probe (2-3 days, confidence MEDIUM-HIGH)

The `differential_afb` and `differential_xs_pt` functions in `modules/analytic_smeft/smeft.py:362-463` already exist. The multi-observable experiment in `experiments/intention-vs-deepsets/experiment_multiobs_afb.py` already runs but with disappointing results (`output_multiobs_afb/summary.json`: c_lq^(3) still the only operator disclosed). The actionable extension is to combine all three observables (m_ll, A_FB, pT_l) in a single Intention head with shared `psi_theta` and a 3-channel context, and re-run the probe. The combination is what discriminates the chirality structure (LL vs RR vs LR) and isospin (u vs d), which is exactly what is needed to break the c_lq^(1) / c_lq^(3) degeneracy.

**Risk:** A_FB at LO with PDFs that are u/d-asymmetric at high-x is *small* (a few percent), and may not carry enough signal at the 5% MC noise floor of MG. Estimate: if A_FB(c_lq^(3) = 0.8) - A_FB(SM) > 0.02 at m_ll = 1 TeV, the disclosure probe should see it; otherwise it gets buried in noise.

**Why MEDIUM-HIGH:** the infrastructure is built; the experimental result might be that A_FB is too weak to disclose c_lq^(1), in which case the paper's c_lq^(1) honestly-unidentifiable conclusion is strengthened (avoiding the banned word: the paper would *explain* the un-identifiability rather than hiding it).

### U3. Add a sign-flip and K-factor test to the MG cross-check (1 day, confidence HIGH)

In `experiments/full-chain-run/mg_crosscheck.py`, add two passes:

1. **Sign-flip pass:** at each m, also compute mu(-c_lq^(3) = -0.8) and check that mu(+c) + mu(-c) - 2 ≈ 0 (interference cancels) and that the absolute value of mu(+c) - mu(-c) tracks 2*interference. This is a free closed-form test of the interference sign.
2. **K-factor band pass:** apply a multiplicative K-factor K(m_ll) = 1.2 + 0.05 * log(m_ll/1 TeV) (the empirical NLO QCD shape, Mangano fits) to the analytic mu before comparison. The expectation is that residual disagreement shrinks below 1%, which would validate the "consistent with missing K-factor" claim with quantitative force rather than handwaving.

**Why HIGH:** existing infrastructure, one extra MG call per m-point is the only cost (50 calls, ~25 min), and the result either confirms or falsifies a concrete claim in the paper.

### U4. Expose the design-matrix spectrum and Fisher-ranking on every disclosure event (3-5 days, confidence MEDIUM)

See "The eigen-direction proposal" below. Confidence MEDIUM because the implementation is straightforward but the *physics* of the Fisher-eigenbasis rotation for in-context inference is not obviously what the user thinks it is (the design-matrix A is over `psi_theta(m)`, a basis in *kinematic* space, not in Wilson space; the Fisher information over Wilsons is a different matrix that has to be computed by the probe). The proposal below makes this precise.

### U5. Add a "dual physics application" — top-quark or Higgs (5-7 days, confidence LOW-MEDIUM)

The paper's claim of being a "foundation model" stands or falls on whether the same `psi_theta` (or `psi_theta + light fine-tune) transfers from neutral-current DY to a structurally similar process. Two candidates:

- **Top-quark pair production tail.** SMEFT four-fermion operators in the (cQ_q, cQ_t) family produce M_tt^4/Lambda^4 tail growth; the same closed-form attention head should work with `m_tt` as the kinematic variable and the same `psi_theta` *before* fine-tuning. If the foundation-model claim is real, a held-out predictive R^2 > 0.9 on m_tt should be achievable with ~50 training scenarios.
- **EWPO at LEP / Z-pole observables.** The same vertex operators c_Hq^(3), c_Hq^(1) that ALETHIA cannot disclose in DY are *the* dominant directions in EWPO global fits (see fitmaker, SMEFiT). Building a Z-pole oracle would let the paper claim that the model discloses different operators *in different contexts*, which is a much stronger foundation-model story than the current single-process result.

**Why LOW-MEDIUM:** 5-7 days is tight for a new oracle even with the analytic infrastructure ready; the top-quark case might be 5 days, the EWPO case is closer to 7. **Falsification risk:** the same `psi_theta` may transfer poorly because it is overfit to the m_ll grid; this would itself be a useful negative result, but it does not help the paper.

### U6. Add a Wilson-Fisher information matrix readout to the closed-loop run (2 days, confidence MEDIUM-HIGH)

For each context size, compute the Fisher information matrix `F_ij = E[d log p(Y|c) / dc_i  d log p(Y|c) / dc_j]` on the current context. Under the homoscedastic Gaussian regression assumption, this reduces to a sum of `(dmu/dc_i)(dmu/dc_j) / sigma_y^2` over the context, and `dmu/dc_i` is closed-form-extractable from the morphing decomposition `A_i + 2 B_ii c_i + ...`. Then report the principal Fisher directions and their eigenvalues. This is the eigen-direction proposal in concrete form.

**Why MEDIUM-HIGH:** the morphing decomposition is closed-form in the oracle; the Fisher matrix is a 4x4 sum of outer products; the eigenproblem is trivial. The integration with the loop reporting is the only labour. **Risk:** the Fisher eigenbasis may not be a stable feature across the run (it depends on the current context, which is growing); the paper would have to report the time evolution of the principal directions, which is a richer story but also a more committed one.

### U7. Add a PDF-uncertainty band to the analytic oracle (3 days, confidence LOW)

CT18NNLO has 58 Hessian eigenvectors. Convolving the SMEFT cross section with the full set, then propagating to a per-`m_ll` PDF uncertainty band on `mu`, is a few hundred lines of LHAPDF wrapping. The Intention head's predictive variance would then be compared against the PDF-induced spread.

**Why LOW confidence:** this is the closest to a "rabbit hole" item. The PDF-induced spread on `mu = sigma/sigma_SM` largely cancels (numerator and denominator share PDFs), so the band may be uninterestingly tight. Worth one prototype hour to check; not worth the full 3 days unless the cancellation argument fails.

## The eigen-direction proposal

The user's suggestion has two components that deserve separating:

### Component A: Design-matrix spectrum exposure (HIGH confidence, 1 day)

`modules/surrogate/intention/model.py:102-107` already exposes `kappa_A`; the full eigenspectrum is one extra line (`np.linalg.eigvalsh`). The closed-loop run already records `kappa_A` over time. Adding the full 16 eigenvalues to the trajectory.npz and to the Phoenix span attributes is trivial. This is the **A**-side eigenbasis: it lives in the `psi_theta(m)` feature space (R^16) and its blow-up signals an under-resolved *kinematic* direction.

This is **not** a Fisher information matrix in Wilson space. It's a kinematic-basis Gram matrix. Reporting it as "the Intention head's design-matrix spectrum" is precise and useful but should not be conflated with the Wilson-direction Fisher matrix.

### Component B: Wilson-Fisher information from the morphing decomposition (MEDIUM-HIGH confidence, 2 days)

The genuine "Fisher direction over Wilson space" is

```
F_ij(c, context) = sum_k [(dmu/dc_i)(m_k, c) (dmu/dc_j)(m_k, c)] / sigma_y^2(m_k)
```

where the sum runs over the context. The derivatives are closed-form from the morphing decomposition: `dmu/dc_i = A_i(m) / sigma_SM(m) + 2 B_ii(m) c_i / sigma_SM(m) + sum_{j!=i} 2 B_ij(m) c_j / sigma_SM(m)`. The morphing coefficients `A_i, B_ij` are in fact what `smeft.py` computes per-piece (`sm_only, interference, bsm_squared`).

Implementation: add a function `wilson_fisher(c, M_context, oracle, sigma_y)` to `modules/analytic_smeft/` that returns the 4x4 Fisher matrix and its eigendecomposition. Wire it into `experiments/full-chain-run/run.py` to log Fisher eigenvalues per cycle. Add a panel to the trajectory plot.

### Component C: Phoenix-triggered rebasis of psi_theta (LOW confidence, 5+ days, probably won't land)

The user's proposal that drift fires a *rotation* of `psi_theta` to the Fisher eigenbasis is conceptually appealing but operationally fraught:

- `psi_theta` is in m-space, not c-space. A "rotation to the Fisher eigenbasis" would have to be a rotation of the *latent* basis (the 16-dim output of the MLP), not of the feature map itself.
- The closed-form ridge solve is invariant under orthogonal rotations of `psi_theta` — the prediction `Psi_q^T A^{-1} Psi_ctx^T Y` doesn't change. So a rotation alone does not change the model's predictions.
- What *would* change the predictions is a *fine-tune* of `psi_theta` on a Wilson-Fisher-weighted loss, but that is no longer the closed-form-no-gradient-at-inference setup the paper sells.

A more tractable variant: on drift, **the disclosure probe** rotates its target basis to the Fisher eigenbasis. That is, instead of reporting per-operator R^2 in the Warsaw basis, report R^2 in the principal Fisher directions of the current context. This is what fitmaker and SMEFiT do globally; ALETHIA could do it locally per context. **Confidence MEDIUM:** straightforward to implement (~2 days), and it converts a one-out-of-four disclosure into "we disclose the leading Fisher direction at R^2 = 0.94, the second at R^2 = 0.3, the third and fourth are below the threshold", which is a substantially more legitimate physics framing because the leading Fisher direction is *defined* to be the most kinematically distinguishable. The risk: the leading Fisher direction may coincide so strongly with `c_lq^(3)` (in this kinematic regime) that the rotated-basis result is operationally identical to the current Warsaw-basis result. The experiment to settle this is one Fisher diagonalisation on a representative context and reading off the alignment of the leading eigenvector with the Warsaw operators.

### Concrete design for the disclosure-probe rebasis

1. At each disclosure event, compute `F = Fisher(c=c_probe_mean, M_context, sigma_y)` (4x4 in Wilson space).
2. Diagonalise: `F = V D V^T`, eigenvalues `d_1 >= ... >= d_4`.
3. Rotate the probe target: instead of `c -> W w_implicit`, fit `V^T c -> W w_implicit`. This puts the regression in the principal directions.
4. Report `R^2` per principal direction *and* report `V` itself (the alignment of each principal direction with the Warsaw operators).
5. Phoenix span: `aletheia.disclosure.fisher_eigenvalues` (the spectrum) and `aletheia.disclosure.fisher_alignment` (the V matrix flattened).
6. On drift firing, the agent re-evaluates whether the leading Fisher direction has changed; if it has, the loop reports the new principal direction along with the disclosure profile.

**Why this is a real upgrade:** the current paper makes a Warsaw-basis disclosure claim, which is naturally one-out-of-four because the Warsaw basis is not aligned with the kinematic sensitivity of m_ll DY. A Fisher-basis disclosure claim is naturally "the leading direction is disclosed", which is the correct phenomenological statement and matches how global SMEFT fits report their results.

## Summary

The paper has a real architectural contribution (closed-form attention as an SMEFT inference primitive), a real but oversold disclosure result (one operator out of four, where the other three are explainable by kinematic degeneracies), an unsurprising EPIG-vs-random comparison that the paper's own framing softens, and an oracle that is two fidelity tiers below the current global-fit state-of-the-art. None of the upgrades requires new theory; all six recommended ones (U1-U4, U6, and Eigen-Component B) close concrete criticisms within the 1-7 day window. The Fisher-eigenbasis disclosure rotation is the single most valuable single change to make: it converts the disclosure framing from "we recovered one operator" to "we recovered the leading Fisher direction", which is the correct phenomenological claim and the one a SMEFT-fit referee would accept.

Sources:
- [SMEFiT 3.0](https://arxiv.org/pdf/2412.08311)
- [SMEFiT toolbox](https://arxiv.org/pdf/2302.06660)
- [MadMiner](https://arxiv.org/abs/1907.10621)
- [Parametrised classifiers](https://link.springer.com/article/10.1007/JHEP05(2021)247)
- [ML4EFT](https://link.springer.com/article/10.1007/JHEP03(2023)033)
- [Greljo-Marzocca DY tails](https://arxiv.org/abs/1704.09015)
- [Iranipour-Ubiali PDF+EFT](https://arxiv.org/abs/2104.02723)
- [fitmaker mW SMEFT](https://link.springer.com/article/10.1007/JHEP08(2022)308)
- [HighPT](https://arxiv.org/abs/2207.10756)
- [HEP-JEPA](https://arxiv.org/abs/2502.03933)
- [OmniLearned](https://arxiv.org/abs/2510.24066)
- [OmniJet-α](https://arxiv.org/abs/2403.05618)
- [Masked Particle Modelling](https://arxiv.org/abs/2401.13537)
- [Araz-Spannowsky conformal](https://arxiv.org/abs/2512.17048)
- [Prior-Fitted Networks](https://arxiv.org/abs/2207.05202)
- [Garnelo-Czarnecki Intention](https://arxiv.org/abs/2305.10203)
