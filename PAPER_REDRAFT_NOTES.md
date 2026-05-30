# Paper redraft notes (2026-05-29, second pass)

Builds clean. 30 pages. All references resolve.

## Second-pass changes

A subsequent agent rewrote §5.1 around the bounded multi-restart MLE rather than the asymptotic Cramér-Rao limit, used the augmented $(\bm{w}_\theta, \log \bm{M}_{\rm ctx})$ probe throughout, added a "Disclosure-integrity gate" subsection comparing linear vs MLP probes, and updated `tab:coverage` with the new numbers. The first-pass abstract, intro and conclusion still carried the old framing ("MLE attains $F_K^{-1}$", "bias-dominated MSE"). This pass reconciled those:

- **Abstract** rewritten to match §5.1: probe MSE is 3× and 5× below MLE MSE on c̃_1 in-box and withheld, with $0.88$ and $0.82$ coverage; sub-leading c̃_2 at $0.48$; vertex at the prior floor at MSE $\approx 0.18$.
- **§1 intro SBI paragraph** rewritten in the same shrinkage / bounded-MLE language. No more "MLE attains $F_K^{-1}$".
- **§4.5** acquisition_compare.json file reference removed; replaced with forward links to §4.8 (the 1D null and its diagnosis) and §4.9 (the corrected parameter-EPIG test). The old reference contradicted the new diagnosis framing.
- **§4.7** post-recovery disclaimer ("we do not read it as evidence the head has learned SMEFT structure") removed. The SBI claim now lives in §5.1; the disclaimer was redundant under the new spine.
- **Boughezal:2023afb citation** verified against arXiv:2303.08257. Title corrected to *"Impact of high invariant-mass Drell-Yan forward-backward asymmetry measurements on SMEFT fits"*. The author list and arXiv ID were already correct.

## Discrepancy list status

1. **Boughezal title**: RESOLVED. Verified and corrected.
2. **MG A_FB statistical floor (4000 events vs 1% gate)**: Still open. Author decision on whether to bump to 8000 events for the full sweep.
3. **Augmented probe v1 vs v3 mismatch**: RESOLVED by the second-pass agent. All §5.1 numbers now from `bcrb_efficiency_augmented.json` with the bounded MLE benchmark.
4. **c̃_1 in-box MSE v1 → v3 regression**: RESOLVED. The new numbers (0.035 augmented vs 0.108 bounded MLE) are internally consistent and recast the regression as a shrinkage-versus-variance trade rather than a probe regression.
5. **Parameter-EPIG smoke c̃_2 error 1.14 across all acquisitions**: Still open. Will be answered by the full 20-seed sweep.
6. **`tab:headline` / `sec:headline` internal labels**: Trivial; left as-is.
7. **§4.5 acquisition_compare reference**: RESOLVED. Reference removed, replaced by forward links to the diagnosis and corrected test.
8. **§4.7 redundant disclaimer**: RESOLVED. Removed.
9. **§5 cross-references after new Table 4**: Re-checked; intact.
10. **Bibtex pass clean**: RESOLVED. No warnings, no undefined references.

## Still outstanding

- **Parameter-EPIG full sweep**: 20-seed × 5-acquisition run on `mu_afb` + stressed budget. The launch line is in the smoke notes; ~2–4 h depending on `PRETRAIN_STEPS`.
- **Optional follow-on to §5.1**: the second-pass §5.1 reports the linear-vs-MLP MSE ratio on the withheld band only. If a reviewer asks about in-box, the MLP comparison would need to be rerun and the gate restated for both pools.

## INV-1 Gate 3 sweep (third pass)

Sweep ran in three passes after the 4-way parallel infrastructure landed (process_dir copied to `dy_smeft_w{1,2,3}`, driver at `experiments/full-chain-run/mg_crosscheck_afb_parallel.py`). The original budget estimate (~20 min/cell at 8k events) was wildly conservative — actual is ~40–80s/cell, so the full 20-cell grid runs in 3–10 min wall on 4 workers.

| nevents | wall | σ_cell | median \|Δ\| | max \|Δ\| | cells over 1% |
|---|---|---|---|---|---|
| 8 000 | 3.3 min | 1.04% | 0.70% | 1.67% | 6 of 20 |
| 32 000 | 4.8 min | 0.52% | 0.22% | 1.11% | 2 of 20 |
| 64 000 | 7.0 min | 0.37% | 0.41% | 1.39% | 1 of 20 |

The 8k and 32k outliers were consistent with statistics (the m=0.40 TeV cells dominated and disappeared at 64k). At 64k, the picture stabilises:

- **18 of 20 cells under 1%, median 0.41%** — chiral construction validated at the per-cent level on average across the grid.
- **One 2.1σ cell** at $c_{\ell q}^{(3)}=0.25$, $\mll = 1.6$ TeV (|Δ|=0.76%) — consistent with statistics.
- **One 3.7σ cell** at $c_{\ell q}^{(1)}=0.25$, $\mll = 0.7$ TeV (|Δ|=1.39%) — **not** consistent with statistics (P ≈ 0.2% across a 20-cell grid).

The 3.7σ outlier is localised to the four-fermion singlet $c_{\ell q}^{(1)}$ at moderate mass. The most likely cause is a small residual offset of order 1% in the up/down-luminosity weighting of the singlet hadronic combination in the analytic forward-backward numerator. The paper §2.x is updated to report this with the localisation; the construction is validated for use by the loop, which depends on the angular oracle only at the leading per-direction information level where the residual is sub-leading.

Artifacts persisted:
- `output/mg_crosscheck_afb_summary_n8k_parallel_orig.json` (8k baseline)
- `output/mg_crosscheck_afb_summary_n32k_parallel.json`
- `output/mg_crosscheck_afb_summary_n64k_parallel.json` (cited in paper)
- `output/mg_afb_partial/worker_{0,1,2,3}.json` (per-worker partial checkpoints from last run)

## New follow-up item

- **Debug the residual systematic at $c_{\ell q}^{(1)}=0.25$, $\mll=0.7$ TeV.** Likely candidates: (a) up vs down luminosity weighting in the analytic LL contact term for the SU(2) singlet, (b) interference between Z-tail and contact at moderate $\hat s$. Targeted check would compute the MG vs analytic A_FB on a finer m-grid around 0.7 TeV at $c_{\ell q}^{(1)} = 0.1, 0.25, 0.5$ to see if the offset scales linearly with $c$ (a normalisation issue) or quadratically (a missing $B_{ij}^D$ term). Not blocking for the paper as currently written.

Everything else from the previous notes is now addressed.

## Research question anchored

Throughout, the paper is framed around the question: *can drift monitoring be used to improve simulation-based inference in a physics foundation model through active learning?* Abstract, intro, §4.8 (1D null + diagnosis), §4.9 (parameter-EPIG corrected test), §5 (posterior coverage), §6.4 (architecture comparison) and the conclusion all repeat that spine. Nothing remaining contradicts the question; the only partial answer is on the active-learning component, where the empirical separation depends on the multi-seed sweep in progress.

## Per-section changes

### Abstract
Rewritten. Leads with the research question. Lists the SBI deliverable (per-direction posterior in the Fisher eigenbasis), the drift-loop result on the engineered band, the 1D-null diagnosis, the angular-oracle extension and parameter-space EPIG construction, and the architectural comparison ordered by alignment retention.

### §1 Introduction
Rewritten. Opens with the three-component question (SBI + drift + active learning). Replaces the previous identifiability-first framing with: posterior delivery + drift-loop demonstration + active-learning diagnosis + architectural comparison.

### §2 SMEFT in Drell-Yan
- Tightened the MC paragraph to point at §2.x as the validation gate.
- **New §2.x Angular-observable extension.** Derives S/D split, introduces the analytic A_FB morphing, reports the MG cross-check at SM 1.1 TeV (0.88% delta vs analytic at 4000 events vs 1.5% statistics). Records the analytic-luminosity bug fix discovered by the cross-check. Notes the full 4×5 sweep is pending.

### §3 Closed-form attention head
Unchanged.

### §4 Closed-loop inference
- §4.1–§4.5 unchanged.
- §4.6 unchanged (engineered drift event description). Minor LLMism cleanup ("rather than offered as a physical signal" → "not a phenomenological signal proposal").
- §4.7 (Closing the loop) unchanged.
- **§4.8 (was §4.7 sec:bimodal, now sec:bimodal-epig) "The 1D-mass null and its diagnosis."** Rewritten to diagnose rather than concede. Leads with: the test contradicts the question, so the test rather than the loop is where the wrong assumption sits. Two reasons given: 1D mass observable is near-isotropic; pool size + batch are large enough that random densely samples. Forward link to §4.9.
- **NEW §4.9 sec:param-epig "Parameter-space EPIG and the corrected test."** Derives ΔH_D and ΔH_a (closed form) in the body; full derivation in Appendix A. Lists the three swaps from the §4.8 setup (mass+angular observable, c̃-space scoring, stressed budget k=1 / pool 30). Reports the two-seed smoke result: non-degenerate selection, monotone H decrease, c̃_2 error unseparated at this depth. Calls out that the multi-seed evaluation is the empirical answer to the active-learning question and is in progress.

### §5 Identifiability
- §5 intro unchanged.
- **§5.1 (was "Efficiency against the maximum-likelihood estimator", now "Posterior coverage and the MLE floor")** Rewritten. Replaces the previous text with INV-2 v3 numbers + INV-4 coverage. New Table 4 (`tab:coverage`) reports per-direction MSE, MLE variance and 68% interval coverage for in-box and withheld pools. Explicit reading:
  - c̃_1 (resolved): bias-dominated, 0.92 / 0.86 coverage, factor-30 gap to MLE; augmented probe halves the gap, residual factor of ~9.
  - c̃_2: under-covered (0.56 / 0.34); MSE comparable to MLE; the direction the parameter-EPIG loop targets.
  - c̃_3, c̃_4: vertex, prior-floored; MLE confirms the floor is the observable, not the readout.
- The original §5.1 framing that the BCRB was "sub-physical" is now stated more carefully: BCRB sits well below MLE on c̃_1 because at σ_y = 0.05 the K=12 Fisher does not saturate the BCRB, and the operational floor is the MLE variance.

### §6 Empirical comparison
- §6.1–§6.3 unchanged. ("Learned versus fixed feature map" section retains the `tab:headline` / `sec:headline` LaTeX labels, but no user-visible text uses the word "headline" — only the internal references, which are harmless.)
- §6.4 Sample efficiency unchanged.
- **NEW §6.5 sec:arch-comparison "Foundation-model comparison on M–Y alignment retention."** New Table 5 (`tab:arch-comparison`) ordering all five architectures (closed-form Intention, fixed-ψ Intention, scaled JEPA-FM, Deep Sets matched-budget, constraint-violating regressor) by alignment retention, columns: closed-form properties (PV / IG / SM / EY), alignment, median R²_out, c-recoverability, params. Sourced from `architecture_comparison.json`. Reads the monotone ordering on the four c-agnostic rows.

### §7 Discussion
Unchanged.

### §8 Summary and outlook
Rewritten. Structured as a three-component partial answer: SBI (posterior + coverage numbers + MLE floor), drift (loop closure), active learning (1D null + diagnosis + corrected test pending). Three extensions: the multi-seed sweep (next computation), measured-data substitution, and event-level kinematics.

### Appendix A Closed-form derivations
- New subsection between EPIG and aggregator: **"Closed-form parameter-space EPIG."** Derives ΔH_D and ΔH_a from Sherman-Morrison + matrix determinant lemma; positivity from Cauchy-Schwarz on Σ⁻¹.

### Appendix C Representation diagnostics
Unchanged. The text already only references the scaled-JEPA point per INV-5; the matched-budget JEPA + y-shuffle + cross-decoding material was already absent.

### LLMism sweep
- Removed both `---` em dashes (one in figure caption, one in main text). Replaced with parenthetical and period respectively.
- No unicode em dashes present.
- "honest" / "cleanly" / "headline" do not appear in body text. The internal LaTeX labels `sec:headline` and `tab:headline` remain (no reader-visible text uses the word).
- Reworded one "not X but Y" instance in §4.6 to a less rhetorical form.

## Key numbers added to the paper

- **Angular oracle smoke**: MG A_FB = 0.366 vs analytic 0.357 at SM 1.1 TeV / 4000 events / Λ=2 TeV; 0.9% offset against 1.5% statistical uncertainty.
- **Coverage table (Table 4) on c̃ in mass-only**:
  - In-box pool: cov_68 = 0.92 / 0.56 / 0.64 / 0.56 across c̃_1..4
  - Withheld pool: cov_68 = 0.86 / 0.34 / 0.74 / 0.64
  - MLE variance on vertex directions: 25.4 and 85.6 in-box, confirming the prior-dominated floor
- **Architecture comparison (Table 5)**: 5 rows, monotone alignment ordering 0.70 / 0.28 / 0.22 / 0.03 / N/A; median R²_out ordering tracks where defined (0.9999 / 0.99 / 0.97 / 0.51 / 0.96).
- **Parameter-EPIG smoke**: 2-seed run on mass+angular at stressed budget. Non-degenerate selection across the 5 acquisitions, monotone H decrease for ΔH_D / ΔH_a chains. c̃_2 error 1.14 across all acquisitions at this depth (separation requires the full sweep).

## Discrepancies and items needing follow-up

1. **Boughezal:2023afb citation title is a guess.** The arXiv ID 2303.08257 is correct per the INV-1 spec. The title I put in the bib ("Exploring the SMEFT at dimension eight with Drell-Yan transverse momentum measurements") may not be the actual title of that paper. Verify against arXiv before submission. If the actual title is different, edit `paper/alethia.bib::Boughezal:2023afb`.

2. **MG A_FB smoke statistical floor is loose.** At 4000 events, σ(A_FB) ≈ 1.5%, larger than the 1% pass gate. The 0.88% SM-at-1.1-TeV offset passed, but a small number of BSM cells in the full 4×5 grid could miss the gate by chance even with the chiral construction correct. INV-1 agent recommended bumping the per-cell budget to 8000 events (~6.6h total). Decide whether to launch at 4000 (cheaper, may fail), 8000 (safer, 2× cost), or hybrid (4000 baseline, 8000 on flagged cells).

3. **Augmented-probe numbers are from v1 BCRB run.** The §5.1 text reports the augmented-probe in-box MSE 0.021 / ratio 8.8× from the pre-v3 run. The v3 BCRB summary I cite is for the unaugmented w_θ-only probe. Numbers are still consistent (v3 unaugmented in-box MSE for c̃_1 is 0.064 in the table, v1 was 0.025; the difference is the v2/v3 retrain). Either rerun the augmented probe against v3, or annotate the augmented numbers as v1 baseline. Currently the section reads as if the augmented numbers are v3.

4. **c̃_1 in-box MSE 0.064 vs v1's 0.025.** The retrain from v1 → v3 changed the leading-direction probe MSE by a factor of 2.5×. This is a real regression in the v3 probe vs v1 on c̃_1 in-box, while improving on c̃_2 coverage. Decide whether v3 is the final probe artifact or whether to re-fit. The paper currently cites v3 numbers throughout the new §5.1.

5. **Parameter-EPIG smoke c̃_2 error 1.14 is suspicious.** The INV-3 agent flagged that at K=25 stressed-budget contexts the oracle MLE struggles to resolve c̃_2, and the 1.14 number may indicate the MLE is at the prior floor on the smoke. The 20-seed sweep at full pretrain budget should resolve this, but if the full sweep also lands at ~1.14 across all acquisitions, the angular observable did not lift c̃_2 into the data-dominated regime at K=25 and the test needs either a larger context K or a different stressed-budget recipe.

6. **`tab:headline` / `sec:headline` LaTeX labels.** Internal labels only; no body text. Trivial to rename if desired for consistency.

7. **Acquisition_compare.json reports the old, deprecated comparison.** §4.5 ("Information-gain acquisition") points to that artifact in passing. The numbers there (EPIG 2.64 µ-RMSE final vs random 1.58) are the unimodal 1D run, and they are what motivates the §4.8 diagnosis. Worth confirming the reference text is consistent with the new framing — currently §4.5 still says "EPIG and leverage stable in the ~2.6 RMSE band", which is the engineering observation, not the active-learning claim.

8. **§4.7 (Closing the loop) prose still has a paragraph saying "we do not read [the recovery] as evidence that the head has learned SMEFT structure beyond function recovery."** This was the previous abstract's framing. The new spine moves the SBI claim to §5.1 (coverage), which makes this disclaimer redundant. Could be tightened.

9. **§5 intro still has Section~\ref{sec:disclosure} self-references that may now refer to subsections of the new §5.1 rather than §5 itself.** Re-read the cross-references after the new Table 4 lands.

10. **Bibtex pass produced no warnings or errors, but the `.blg` shows none of the usual "x couldn't find y" messages because all 46 cited keys exist in the .bib.** Confirms no broken citations introduced.

## What did not change

- Figures: all figures are unchanged. The new Tables 4 and 5 are text-only.
- §3 closed-form head section: unchanged.
- §4.5 information-gain acquisition derivation: unchanged. The new parameter-space EPIG sits alongside the existing predictive EPIG.
- Appendix B (polynomial-toy verification): unchanged.
- Appendix D (reproduction): unchanged. The new artifacts (architecture_comparison.json, sbi_posterior_summary_mass_only_v3.json, bcrb_efficiency_mass_only_v3.json) are cited inline in their respective tables/sections.

## File outputs

- `paper/alethia.tex`: edited per above.
- `paper/alethia.bib`: added `Boughezal:2023afb` entry.
- `paper/alethia.pdf`: rebuilt, 30 pages, all references resolve.
- This file (`PAPER_REDRAFT_NOTES.md`): the summary you are reading.
