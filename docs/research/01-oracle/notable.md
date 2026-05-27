# Oracle — empirical investigations (notable.md)

Five numbered investigations worth running. Each has a one-sentence
hypothesis and a one-sentence expected outcome. Order is roughly by
information-yield-per-effort.

---

1. **PDF dependence of μ for low- and high-mass tails.**
   *Hypothesis*: at LO, the SMEFT modification factor μ(c, m) is
   PDF-independent to within a few percent across the realistic LHC range
   `m_ll ∈ [0.2, 3] TeV`, because the same parton luminosity factorises out
   of the SM, interference, and BSM² pieces.
   *Expected outcome*: median |μ_toy − μ_CT18| / |μ_CT18| < 0.03 in the
   bulk; tails approach 0.05 only when the parton-flavour mix becomes
   non-trivial near `m_ll ≈ √s/2`.

2. **Vectorised LHAPDF wrapper.**
   *Hypothesis*: the ~30× gap between toy-PDF (0.6 ms) and CT18NNLO (19
   ms) per call is dominated by per-point Python overhead in
   `LHAPDFSet.xf` (`modules/analytic_smeft/pdfs.py:162-170`), not by
   LHAPDF's own grid lookup.
   *Expected outcome*: replacing the loop with a single
   `np.frompyfunc(pdf.xfxQ, 3, 1)` or a Cython batch wrapper closes the
   gap to ~2–3× of the toy speed, dropping T1 to ~2 ms/call.

3. **Cache hit rate under realistic EPIG querying.**
   *Hypothesis*: in a 100-round EPIG loop with a fixed 200-point target
   set and a 1000-candidate pool resampled each round, cache hit rate on
   target-set re-evaluations is 1.0, on EPIG-acquired candidates is < 5%,
   and global hit rate is dominated by the target-set component.
   *Expected outcome*: observed global hit rate within ±5pp of
   `n_target / (n_target + n_acquisitions_so_far)`; the EPIG candidates
   provide no detectable cache reuse, confirming that the cache pays off
   exactly for the target set.

4. **MG–analytic disagreement as a function of |c|.**
   *Hypothesis*: in the high-mass tail (`m_ll > 1.5 TeV`), MG and the
   analytic oracle disagree at constant ~3% from MC noise plus a |c|³
   contribution from dim-6 squared interference with NLO QCD that the
   analytic backend does not model.
   *Expected outcome*: fitting `log|μ_MG − μ_analytic|` vs `log|c|` for
   `|c| ∈ [0.05, 0.5]` yields a slope ~ 3 ± 0.5 with intercept consistent
   with 3% noise floor at `|c| → 0`, and disagreement remains under 10%
   for `|c| ≤ 0.3` — the analytic oracle is trustworthy in that band.

5. **Tier-T1 wall-clock budget under realistic 15-hour run.**
   *Hypothesis*: a single-machine run with one core dedicated to the
   agent, six to MG workers, and one to LHAPDF-T1 calls can complete the
   tiered schedule (≈50 000 T1 + ≈5 000 T2 + ≈100 T5) within 15 hours and
   under 1 GB cache footprint.
   *Expected outcome*: total wall time 8–10 hours dominated by T2 MG
   compute; T1 cost negligible (~ 16 minutes); cache reaches ~ 500 MB,
   index.sqlite < 10 MB; no parquet-shard contention observed at this
   throughput.
