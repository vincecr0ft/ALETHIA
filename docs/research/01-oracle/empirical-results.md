# Oracle — empirical results (second pass, agent a)

## Investigation 1: Time the analytic oracle at toy PDF and CT18NNLO

**Command.**
```bash
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
uv run python -c "
import time, numpy as np
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
o_toy = AnalyticSMEFTOracle(pdf='analytic', noise_frac=0.0)
o_ct  = AnalyticSMEFTOracle(pdf='CT18NNLO',  noise_frac=0.0)
o_ct.truth(np.zeros((1,4)), np.array([1.0]))  # warmup
N = 200; c = np.random.default_rng(0).normal(scale=0.1, size=(N,4))
m = np.random.default_rng(1).uniform(0.3, 2.5, size=N)
t0=time.perf_counter(); o_toy.truth(c,m); print((time.perf_counter()-t0)/N*1000)
t0=time.perf_counter(); o_ct.truth(c,m);  print((time.perf_counter()-t0)/N*1000)
"
```

**Observed.** Toy PDF 0.627 ms/call; CT18NNLO 18.69 ms/call. Ratio 29.8×. Matches first-pass headline numbers in `docs/research/01-oracle/summary.md:158`.

**Debugged via cProfile.** The bulk of CT18NNLO time is **NOT** the LHAPDF inner loop. It is `LHAPDFSet.__init__` (`modules/analytic_smeft/pdfs.py:157-160`) being called 50× during 50 oracle calls (one `mkPDF` per call). Profile output: `pdfs.py:157(__init__)` accumulated 0.814 s out of 1.016 s total for 50 calls — i.e. 16.3 ms per call dedicated to PDF re-instantiation. The actual lookup loop `pdfs.py:162(xf)` cost 0.158 s = 3.2 ms/call.

The root cause: `oracle_smeft.py:108` passes the string `self.pdf="CT18NNLO"` to `differential_xs`, which calls `get_pdf(spec)` on `smeft.py:345`, which calls `LHAPDFSet(spec)` on `pdfs.py:188`, which calls `lhapdf.mkPDF(name, member)` on `pdfs.py:159`. The `mkPDF` call is not amortised across rows. Fix: instantiate `PDFSet` once in `AnalyticSMEFTOracle.__init__` and pass the object.

**Scaling extrapolation.** With the fix, T1 cost is ~3.2 ms/call. The 50k-T1 budget for 15h (`docs/research/01-oracle/summary.md:520-548`) costs `50000 × 0.0032 = 160 seconds = 2.7 minutes`. Without the fix it costs `50000 × 0.019 = 15.8 minutes`. Both fit. The 6× improvement materially loosens cache hit-rate sensitivity: misses are 5× cheaper than the summary assumed.

## Investigation 2: PDF independence of μ on (c, m) grid

**Observed.** Overall median |rel disagreement| = 0.33%; max = **13.49%**. Distribution:
- `cHq3`, `cHq1` (vertex operators): all disagreements ≤ 0.4%. PDF-independent to spec.
- `clq3`: all disagreements ≤ 2.9%. Within "few percent" claim.
- `clq1`: 12 of 18 points exceed 5%; max 13.49% at `(c=0.5, m_ll=0.5 TeV)`. **Outside the claim.**

**Debugged.** The `clq1` failure is a flavour-weighting effect. `c_lq^(1)` contributes per chirality channel `(L,L), (L,R)` with explicit u- and d-quark coefficients that the up/down PDF ratio governs. The toy `AnalyticPDF` (`modules/analytic_smeft/pdfs.py:39-94`) has `_n_uv = 2/Beta(0.5, 4)` and `_n_dv = 1/Beta(0.5, 5)` which gives a u/d ratio at fixed x that differs from CT18NNLO's. For operators that couple equally to u and d (`cHq*` vertex), the flavour-dependence cancels in the ratio. For `clq1` it does not. `clq3` partially escapes because the `tau^3` projection (`smeft.py:141, 153`) makes it asymmetric in u vs d but the sum-over-flavour structure largely cancels.

**Implication for design.** First-pass test `A5` (`docs/research/01-oracle/test-plan.md:30-42`) at `tol=2%` will **fail** as currently written for `clq1` at `m_ll=1.5 TeV`. The test as written needs to either (a) restrict to `cHq3/cHq1/clq3` or (b) raise tolerance to 15% (which makes the test toothless). Recommend: rewrite as a per-operator test with operator-specific tolerances calibrated against this measurement. Also recommend agent (b)'s pre-training should not use the toy PDF for `clq1`-active examples — use CT18NNLO or the toy will systematically misrepresent the `clq1` direction in embedding space.

**Scaling extrapolation.** This finding affects the cache strategy: it kills the T0/T1 cross-tier interoperability for `clq1`. Cache rows from `pdf="analytic"` and `pdf="CT18NNLO"` are **not** interchangeable for `clq1`-active commissions. The cache key already discriminates on `pdf` (`docs/research/01-oracle/summary.md:343`); the practical consequence is that the analytic-PDF cache is useless for `clq1` directions, halving its hit rate in that direction.

## Investigation 3: Recomposition exactness

**Observed.** For every test point, `differential_xs == sm_only + interference + bsm_squared` to **exactly zero** (not just `rtol=1e-10`). The recomposition is a literal in-place sum in `smeft.py:355`, not a separate recomputation, so machine-exact agreement is correct.

**Implication.** The `test_analytic_smeft.py:239-244` recomposition test as written (`rtol=1e-12`) is more strict than necessary; it could be tightened to `np.array_equal`. More importantly, this confirms the oracle can advertise an exact `pieces` decomposition without floating-point caveats — `OracleResult.pieces["sm_only"] + pieces["interference"] + pieces["bsm_squared"] == total` bit-for-bit.

## Investigation 4: LHAPDF wrapper profile (notable #2)

**Observed.**
- Current loop in `pdfs.py:162-170`: 2.10 ms / 1000 pts = 2.10 µs/pt.
- `np.frompyfunc`: 1.94 ms / 1000 pts = 1.94 µs/pt (8% faster).
- Raw `_pdf.xfxQ` in a Python loop: 1.96 ms / 1000 pts = 1.96 µs/pt.

**Debugged.** The notable #2 hypothesis (that Python-loop overhead dominates and vectorisation closes the gap to ~2 ms/call) is **false**. LHAPDF's own per-call cost (raw `_pdf.xfxQ`) is 1.96 µs/pt; the wrapper adds 0.14 µs/pt of overhead. Vectorisation can save at most ~10% of the total. The 30× gap between toy and CT18NNLO is dominated by `LHAPDFSet.__init__` (investigation 1), not the lookup loop. **Update notable #2** to reflect this.

The new arithmetic, with PDF instance reused: per oracle call we do ~20 calls to `pdf.xf` on 64-point arrays (5 quark flavours × 4 lookups × 64 nodes ≈ 1280 lookups). At 1.96 µs/lookup that's 2.5 ms, plus ~0.7 ms of numpy overhead → ~3.2 ms/call. Matches the measured 3.17 ms/call with reused PDF.

**Scaling extrapolation.** Even with a hypothetical zero-cost LHAPDF wrapper, T1 cannot fall below ~0.7 ms/call (numpy overhead in `_partonic_xs` + `_luminosity`) which is already 0.6 ms (the toy PDF speed). So the floor for T1 is set by the analytic kernel cost, not by LHAPDF. The realistic post-fix budget is 3.2 ms/call, not 2 ms.

## Investigation 5: Scaling projection for 10-20 h run

**Assumptions** (each made explicit):
1. The PDF-caching fix (investigation 1) is applied. T1 cost = 3.2 ms/call.
2. T2 cost from agent (a)'s summary: 30 s/call with `nevents=1000` on 7 cores (4.3 s/call wall when parallel).
3. T5 cost: 600 s/call (NLO MG); on 7 cores, 86 s/call wall.
4. T0 cost: 0.6 ms/call, never the bottleneck.
5. Cache write costs are absorbed into the call (~10 µs/row on parquet append).
6. Drift evaluator overhead: ~100 ms/cycle (DAS-CUSUM + BH + kappa update); 1000 cycles ⇒ 100 s; negligible.
7. EPIG acquisition cost: O(n_pool × d^2) per pick. For d=75, n_pool=2000, that's 11.25 Mops ≈ 5 ms/pick. 50000 picks ⇒ 250 s ≈ 4 min; negligible.

**Projected wall-time budgets** (15 h total = 54000 s):
- T1: 50000 × 0.0032 s = 160 s = **2.7 min**.
- T2: 5000 × 30 s / 7 cores = **5.95 h**.
- T5: 100 × 600 s / 7 cores = **2.4 h**.
- Total measurable load: 8.4 h. Slack: 6.6 h for the FM forward/backward, the drift loop, the demo, and false starts.

**Cache size projection.**
- One `OracleResult` row: ~512 bytes raw (kinematics grid + 3 pieces + total + mu + metadata strings). With parquet compression: ~200 B/row.
- 50k + 5k + 100 = 55100 rows. Raw bytes: ~28 MB. Compressed: ~11 MB. Plus partial event arrays for T2+ that include `pieces["events"]`: at `nevents=1000` and F=4 floats per event, ~16 KB/row uncompressed. 5000 T2 rows ⇒ 80 MB. Plus 100 T5 with `nevents=10000` ⇒ 16 MB.
- Total cache: ~110 MB. Index sqlite (one row per cache entry, ~200 B/row): 11 MB. **Total < 200 MB**, well under the 1 GB ceiling agent (a) projected (`docs/research/01-oracle/summary.md:401-408`).

**Risk concentrated in T2 wall-time.** Each T2 call has a fixed ~15 s overhead in `generate_events` per `mmll` window (compile + grid + I/O). The first-pass design (`docs/research/01-oracle/summary.md:294-297`) recommends widening the `mmll` window and binning post-hoc to capture multiple `m` points per MG call. Verified: with one MG call yielding 10 binned `m` points, T2 amortised cost drops to ~3 s/m-point. The 5000-T2-call budget then becomes 500 MG runs × 30 s / 7 cores = 36 minutes. This is a 10× win and brings T5 to the dominant cost. **Recommend** the orchestrator default T2 calls to wide-window batched mode; T5 stays single-point.

## Key files (absolute paths)

- `/home/vince/ALETHIA/modules/surrogate/oracle_smeft.py:65-111` — primary target for Diffs 1, 2, 3, 5, 6, 8.
- `/home/vince/ALETHIA/modules/surrogate/oracle_madgraph.py:147-333` — target for Diffs 2, 4, 5, 6.
- `/home/vince/ALETHIA/modules/analytic_smeft/pdfs.py:157-170` — diagnosed source of the 30× slowdown.
- `/home/vince/ALETHIA/modules/analytic_smeft/smeft.py:309-366` — `differential_xs` public API.
- `/home/vince/ALETHIA/tests/test_analytic_smeft.py:239-244` — recomposition test verified.
- `/home/vince/ALETHIA/docs/research/01-oracle/notable.md:18-25` — notable #2 hypothesis falsified; update needed.
- `/home/vince/ALETHIA/docs/research/01-oracle/summary.md:104-111` — PDF-independence claim needs operator-specific qualifier for `clq1`.
