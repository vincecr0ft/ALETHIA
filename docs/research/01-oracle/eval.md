# Oracle — eval (second pass, agent a)

## 1. Cross-area interactions

**Oracle × Foundation Model (agent b).** The most load-bearing intersection. Agent (b)'s recommendation (`docs/research/02-foundation-model/summary.md:99`) explicitly asks the oracle to expose `sample_events(c, n_events)` returning unbinned events for an event-set encoder. The current `AnalyticSMEFTOracle.truth(c, m) -> mu` (`modules/surrogate/oracle_smeft.py:84-111`) only computes a scalar ratio; it does not sample events. The analytic backend cannot sample events at all without an additional Monte Carlo step, because `smeft.py` integrates the partonic cross-section analytically and convolves with PDFs via Gauss-Legendre. Architecture A is therefore only natively reachable through the MadGraph adapter (`modules/surrogate/oracle_madgraph.py:147`) at T2+.

For agent (b)'s recommendation to be realised, the oracle interface needs one of:
- a `sample_events(c, n_events, observable_window) -> np.ndarray[n_events, F]` method on the MadGraph adapter, with the analytic adapter raising `NotImplementedError` (or sampling from the analytic distribution by an inverse-CDF method on `dsigma/dm`); or
- a `pieces` extraction in `OracleResult` rich enough that agent (b)'s architecture B (histogram autoencoder) is the natural consumer — the scalar `mu` is throwing away exactly the differential structure B needs.

The first-pass `OracleResult` structured dict already includes `pieces["sm_only"]`, `pieces["interference"]`, `pieces["bsm_squared"]` on a kinematic grid (`docs/research/01-oracle/summary.md:252-280`). This already covers Architecture B. For Architecture A the missing primitive is `sample_events`.

**Oracle × Drift (agent c).** The drift area assumes a `linked_predict_span_id` attribute on `oracle.query` spans (`docs/research/03-drift/summary.md:399`) and consumes `aletheia.oracle.backend`, `aletheia.oracle.n_points`, `aletheia.oracle.duration_ms`. The first-pass oracle proposes `oracle.fidelity_tier`, `oracle.backend`, `oracle.cache.hit`, `oracle.cost_seconds`, `oracle.wilson_norm` (`docs/research/01-oracle/summary.md:430-447`). These overlap but use different namespaces (`oracle.*` vs `aletheia.oracle.*`). Agent (c)'s `aletheia.*` namespacing is the right choice — every project-specific attribute Phoenix sees should be filterable by a single prefix. The oracle attributes should be renamed.

Agent (c) also needs `oracle.mc_noise` and `oracle.lambda_gev` and `oracle.observable` to drive the DAS-CUSUM evaluator's standardised residual: the calibration `sigma` term `(y_oracle - mu_FM) / sigma_conformal` (`docs/research/03-drift/summary.md:151-155`) requires the variance contribution from `mc_noise` to be folded in (the conformal layer cannot ingest sigma it doesn't know about). This is currently implicit in agent (c)'s summary but the oracle has to advertise it.

**Oracle × EPIG-conformal (agent d).** Agent (d) proposes `epig_acquire_commissions(model, commissions, embed_preview, z_target, k, mc_samples)` (`docs/research/04-epig-conformal/summary.md:107-134`). A `Commission` is a tuple `(wilson_region, kinematic_window, fidelity_tier)` — and the oracle is then called on each chosen commission. This requires the oracle to accept a *bounded sampling region* for `c`, not a single point — the commission spec says "sample over wilson_region". The current `AnalyticSMEFTOracle.truth(c, m)` accepts only a single `c` row. The interface must extend to accept either a scalar `c` (current behaviour) or a sampler description.

Agent (d) also leans on a `cost(kappa)` accessor for cost-normalised EPIG (`docs/research/04-epig-conformal/summary.md:139-141`). The `OracleResult.metadata.wall_time_s` is a measured number after the call; cost-normalised EPIG needs an *a priori* estimate before the call. The oracle should expose `estimate_cost(commission) -> float` (cheap; based on tier and `n_events` only). This was not in the first-pass oracle summary and must be added.

**Drift × EPIG.** Both agents (c) and (d) consume a "region" identifier. Agent (c)'s `region_id` is a clustering of embedding space (`docs/research/03-drift/summary.md:333`); agent (d)'s `target_region` is a bounded subset of commission space (`docs/research/04-epig-conformal/summary.md:111-114`). These are different objects — one in latent space `z`, one in commission space `(c, m, fidelity)`. The oracle is the bridge: a commission produces an embedding via `embed_preview`. The interface must make the join obvious — `OracleResult.metadata.commission_id` and downstream span `aletheia.embedding.region_id` need to be cross-referenceable. Agent (c)'s `aletheia.fm.region_id` is the consumer; agent (d) generates it; the oracle propagates it.

## 2. Conflicts and reconciliations

**Conflict 1: PDF-independence claim.** The oracle first-pass (`docs/research/01-oracle/summary.md:104-111`) claims μ is PDF-independent to ~1% in the bulk. Empirical results (this pass) show this is **false for `c_lq^(1)`**: relative disagreement reaches 13.5% at `(c=0.5, m_ll=0.5 TeV)` and 12.2% at `(c=0.2, m_ll=1 TeV)`. The other three surrogate-front-end operators (`c_phi_q^(3)`, `c_phi_q^(1)`, `c_lq^(3)`) confirm PDF independence to <3%. `c_lq^(1)` is special because the up- vs down-quark contributions enter with opposite-sign coefficients and the toy PDF's u/d ratio differs from CT18NNLO's. This is a flavour-dependent weighting effect — exactly what the summary said was a "few-percent effect", except it is not few-percent for `clq1`.

Reconciliation: the surrogate cannot use the analytic toy PDF safely if `clq1` is in scope. Either the analytic PDF must be replaced by a CT18NNLO-matched toy (re-fit u/d ratios), or the T0 tier should be marked as "approximate, not for clq1" in `OracleResult.metadata`, or T0 should *only* be used for non-`clq1` directions.

**Conflict 2: LHAPDF bottleneck is misdiagnosed.** Notable #2 (`docs/research/01-oracle/notable.md:18-25`) hypothesises the 30× gap is the per-point Python loop in `pdfs.py:162-170`. The empirical profile (this pass) shows the loop costs 2 µs/point, total ~2.5 ms per call — but the *measured* per-call cost is 19 ms. The remaining 16 ms is **PDF re-instantiation**: `oracle_smeft.py:108` passes `self.pdf="CT18NNLO"` as a string to `differential_xs`, which calls `get_pdf(spec)` (`pdfs.py:173`) per call, which calls `LHAPDFSet(spec)` (`pdfs.py:157`), which calls `lhapdf.mkPDF` (`pdfs.py:159`) — 16 ms each time.

Reconciliation: `AnalyticSMEFTOracle.__init__` (or a separate `warmup` method) should instantiate the `PDFSet` once and pass the *object* to `differential_xs`. Empirically this drops per-call cost from 19 ms to 3.2 ms — a 6× speedup. The vectorisation notable #2 proposed is irrelevant; vectorising the loop saves at most 10% of the 2.5 ms LHAPDF time. The PR-shape fix is in section E below.

**Conflict 3: Output shape and `sample_events`.** The first-pass `OracleResult` shape (`docs/research/01-oracle/summary.md:252-280`) returns `pieces["sm_only"]`, `pieces["interference"]`, `pieces["bsm_squared"]` on a kinematic grid, plus `mu`. Agent (b)'s Architecture A wants unbinned events. These are not the same — events are a 2D array `(n_events, n_features)`; pieces is a 1D array `(n_grid,)`. The interface must accommodate both. Recommendation: add a `pieces["events"]` slot, optional, populated only when `return_events=True` is passed and the backend supports it. The analytic backend never populates it; MadGraph backend does.

**Conflict 4: cache key and noise stream.** The first-pass cache key includes `seed` for T2+ (`docs/research/01-oracle/summary.md:344`) but the analytic oracle has its own `_noise_rng` (`oracle_smeft.py:82`) that is not in the key. For T0/T1 the `noise=True` path is deterministic only across a single oracle instance. Two `AnalyticSMEFTOracle` instances with the same seed produce the same noise stream; two with `seed=0` and `seed=1` differ. The cache should not record `noise=True` results without putting the *RNG state* in the key, or only cache the noiseless `truth()`. Recommend cache only `truth()` results and synthesise `noise=True` from them on demand. This also simplifies agent (c)'s drift work: the standardised residual ratio is well-defined only against `truth`, not against a single noisy realisation.

**Conflict 5: target-set re-use across tiers.** Agent (a) recommends caching at the same `(c, m)` across all tiers (`docs/research/01-oracle/summary.md:391-397`): "do not evict the cheap when the expensive arrives". Agent (c)'s drift test compares predictions against oracle calls (`docs/research/03-drift/summary.md:475-480`). If the cache contains a T1 result and a T2 result at the same `(c, m)`, drift detection must always pick the *highest* tier available; the policy is implicit but unstated. Recommendation: `Oracle.call(...)` accepts `fidelity=` as the *minimum* required; the cache hit logic returns the highest-tier cached value that meets the floor. This is one line of code but it has to be in the contract.

## 3. Joint design strengths and weaknesses

**Strongest.** The four-area joint design composes cleanly around three load-bearing invariants:
- The FM never sees `c`; the oracle is the only `c`-aware component; EPIG ranks `phi(z)` not `c`; drift consumes `z` and conformal residuals only. This is consistent across all four summaries (`01-oracle:228-235`, `02-foundation-model:163-174`, `03-drift:288-298`, `04-epig-conformal:88-103`).
- The closed-form ridge + Sherman-Morrison + leverage + EPIG machinery is feature-map-agnostic. Agent (b) keeps it as the *probe*; agent (d) keeps it as the *acquisition function*. Agent (c) uses `kappa(A)` and `v_min` of the same matrix. The mathematics is the same object played in four different roles. Strongest single design choice.
- Phoenix span tree is the runtime backbone. Every cross-area object has a span attribute name and a clear consumer. The `aletheia.*` namespace is consistent.

**Weakest.** Three areas need tightening:
- **The "commission" abstraction is half-built.** Agent (d) introduces `Commission = (wilson_region, kinematic_window, fidelity_tier)` but no other area uses that name. The oracle takes `(c, m, fidelity)`; agent (b) takes events; agent (c) takes spans linked by `linked_predict_span_id`. The shared vocabulary should be: an *oracle call* takes a `Commission`, returns an `OracleResult`. The FM ingests `OracleResult.events` or `OracleResult.histogram`. EPIG ranks commissions. Drift compares `OracleResult.mu` against `FM.predict(events)`. One name throughout.
- **The `phi_preview` problem.** Agent (d)'s EPIG over commissions requires `embed_preview(c)`, a cheap proxy for what the FM would compute on the actual events. The summaries don't say where `embed_preview` comes from. Options: (i) run a cheap T0 oracle call and embed the result; (ii) train a learned proxy; (iii) use the deterministic mean of the embedding distribution. Agent (d) mentions (iii) parenthetically (`04-epig-conformal:101-103`); none of the other agents have committed. Without this, EPIG cannot actually rank commissions. Agent (d) should own this; agent (a) should expose `cheap_preview_embedding(commission)` as part of the oracle interface.
- **Calibration refit after every update.** Agent (d)'s Guarantee B (`04-epig-conformal:243-260`) says calibration must be refit after every Sherman-Morrison update. The drift area presumes this implicitly when it triggers `recalibrate` (`03-drift:340 row 5`). Agent (b)'s probe layer does not say where this happens. The orchestrator must do it on *every* `fm.update` span; no area says this in normative terms. Recommend stating in synthesis.

## E. Interface improvement proposals (PR-implementable)

All in `modules/surrogate/oracle_smeft.py` and `modules/surrogate/oracle_madgraph.py`.

**Diff 1 — instantiate PDF once.** Critical correctness/performance fix.
- `modules/surrogate/oracle_smeft.py:65-82` (`__init__`): after setting `self.pdf = pdf`, add `self._pdf_resolved = get_pdf(pdf)` (import `get_pdf` from `modules.analytic_smeft.pdfs`).
- `modules/surrogate/oracle_smeft.py:108`: change `pdf=self.pdf` to `pdf=self._pdf_resolved`.
- Expected impact: 6× faster T1 calls (19 ms → 3.2 ms). No behaviour change.

**Diff 2 — add `warmup()` method.** Required by `docs/research/01-oracle/summary.md:459-463` and agent (c)'s `oracle.query` span ordering.
- New method `AnalyticSMEFTOracle.warmup(self) -> None` that calls `self._pdf_resolved.xf(2, 0.1, 1000.0)` (forces LHAPDF data load), with a `self._warm = True` guard so a second call is a no-op.
- Same on `MadGraphSMEFTOracle.warmup` — call `precompute_sm` on a small grid `np.array([0.5, 1.0, 1.5])`.

**Diff 3 — add `OracleResult` return shape.** Section 3.2 of the first-pass.
- New module `modules/surrogate/oracle_result.py` defining `OracleResult = TypedDict(...)` matching `docs/research/01-oracle/summary.md:252-280`.
- `AnalyticSMEFTOracle.call(self, c, m, *, observable="m_ll", fidelity="T1", order=None, return_pieces=True) -> OracleResult` — new method that internally invokes `differential_xs` and packages the result. Existing `truth()` retained, becomes a thin wrapper that returns `result["mu"]`.
- `MadGraphSMEFTOracle.call(...)` — same signature; internally invokes the existing `_xs_for` / `_sigma_sm_at` flow and packages.
- Backward-compatible: existing `tests/test_oracle_smeft.py` continues to pass because `truth()` keeps its signature.

**Diff 4 — add `sample_events()` to MG adapter.** Required by agent (b) Architecture A.
- `MadGraphSMEFTOracle.sample_events(self, c, m_window, n_events) -> np.ndarray[n_events, 4]` returns unbinned `(m_ll, pT_l, eta_l, phi_l)` events. Internally parses `Events/run_*/unweighted_events.lhe.gz` rather than the banner. About 50 lines of LHE parsing.
- `AnalyticSMEFTOracle.sample_events` raises `NotImplementedError("analytic oracle is a calculator, not a generator")`. A future inverse-CDF sampler on `dsigma/dm` could replace this raise.

**Diff 5 — add `estimate_cost()` and `supported_fidelity`.** Required by agent (d) cost-normalised EPIG.
- `Oracle.estimate_cost(self, c, m, *, fidelity, n_events=None) -> float` returns expected wall-time in seconds. Analytic returns 0.003 (or 0.0006 for T0); MG returns `0.030 + nevents * 3e-5` (calibrated from this run).
- `Oracle.supported_fidelity` property returns `("T0", "T1")` for analytic, `("T2",)` for MG.

**Diff 6 — add `call_async()`.** Required by agent (c) Phoenix-span "T2 pending" pattern.
- New method `Oracle.call_async(self, c, m, **kw) -> concurrent.futures.Future[OracleResult]`. Implemented via a module-level `ProcessPoolExecutor` (analytic) and `ThreadPoolExecutor` (MG, which forks `generate_events`). Pool size = `os.cpu_count() - 1`.

**Diff 7 — Phoenix span attributes with `aletheia.*` namespace.** Already in agent (c)'s spec; the oracle's first-pass list (`docs/research/01-oracle/summary.md:430-447`) needs renaming.
- Change `oracle.fidelity_tier` to `aletheia.oracle.fidelity_tier`, `oracle.backend` to `aletheia.oracle.backend`, etc.
- Add `aletheia.oracle.commission_id` (the UUID that lets agent (d) join EPIG selections back to oracle calls and agent (c) link drift events to their gating commission).

**Diff 8 — cache only `truth()`, synthesise `noise=True` from cached truth.**
- The cache key (`docs/research/01-oracle/summary.md:343`) drops `seed` and `noise_frac` for T0/T1. The cached value is the noiseless `mu`. The `__call__` method draws fresh noise from `self._noise_rng` *after* the cache lookup. This sidesteps the RNG-state-in-key problem.

**Diff 9 — extend `call(fidelity=...)` semantics to "minimum required tier".**
- Cache lookup returns the highest cached tier ≥ requested. If no cached value meets the floor, oracle runs at the requested tier.

All nine diffs implementable in a single PR. Total LOC delta: ~250 lines added, ~10 modified.
