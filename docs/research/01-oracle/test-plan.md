# Oracle — test plan (agent a)

Tests that demonstrate the oracle does what section 1–6 of the summary
claim. Closed-form predictions where the math gives one; small-scale
empirical checks otherwise. Where the code needs to change first, the
required change is sketched.

---

## A. Structure tests (extend the existing `tests/test_analytic_smeft.py`)

A1. **SM-limit at toy PDF and CT18NNLO** — already passes
(`tests/test_analytic_smeft.py:45-76`). Closed-form prediction: μ(c=0,m)=1
exactly. Empirical: `test_sm_limit` and `test_sm_limit_ct18nnlo` pin
SM cross-section to 5% of tabulated reference.

A2. **Linear interference scaling** — already passes
(`tests/test_analytic_smeft.py:79-96`). Prediction: `σ_int / c` is
independent of `c` to floating-point precision. Pin assertion
`rtol=1e-8`.

A3. **Sign flip** — already passes (`tests/test_analytic_smeft.py:99-107`).
Prediction: `σ_int(c) = −σ_int(−c)` exactly.

A4. **Quadratic energy growth** — already passes
(`tests/test_analytic_smeft.py:110-145`). Prediction:
`(σ_BSM2 / σ_SM)(m) = K (m/Λ)^4` with constant `K` to ≤15% in
`(500, 2000) GeV`. Empirical: pins `K(m)/K(m_ref) − 1 < 0.15`.

A5. **PDF independence of μ** — NEW. Prediction (section 1.5):
μ(c, m_ll=1.5 TeV, clq1=0.3) computed with `pdf="analytic"` and
`pdf="CT18NNLO"` agree to ~1% (the residual is the quark-flavour weighting
inside the chirality channel sum). Sketch:

```python
def test_pdf_independence_of_mu():
    mu_toy  = AnalyticSMEFTOracle(pdf="analytic").truth(
        np.array([[0., 0., 0., 0.3]]), np.array([1.5]))
    mu_ct18 = AnalyticSMEFTOracle(pdf="CT18NNLO").truth(
        np.array([[0., 0., 0., 0.3]]), np.array([1.5]))
    assert abs(mu_toy[0] - mu_ct18[0]) / abs(mu_ct18[0]) < 0.02
```

A6. **Vertex piece does not energy-grow** — NEW. Prediction (section 1.6):
turning on a vertex-only operator (`c_phi_l^(1) = 0.3`) gives μ approaching
a constant `1 + O((v/Λ)^2)` at large `m_ll`, not growing as `(m_ll/Λ)^4`.
Sketch:

```python
def test_vertex_no_energy_growth():
    wc = {"c_phi_l^(1)": 0.3}
    res = simulate_analytic_smeft(wc, observable="m_ll",
        bins=np.array([200., 500., 1000., 2000., 3000.]),
        order="linear", pdf="analytic")
    mu = res["differential_xs"] / res["sm_only"]
    # vertex interference is mass-independent above the Z pole
    high_mass = mu[-1]
    mid_mass  = mu[-3]
    assert abs(high_mass / mid_mass - 1.0) < 0.15
```

A7. **Recomposition exactness** — already passes
(`tests/test_analytic_smeft.py:239-244`). Prediction:
`differential_xs = sm_only + interference + bsm_squared` at `rtol=1e-12`.

---

## B. Capability tests

B1. **AnalyticSMEFTOracle SM-limit and energy growth** — already passes
(`tests/test_oracle_smeft.py:28-46`). Pinned.

B2. **Noise stream determinism** — already covered
(`tests/test_oracle_smeft.py:48-53`). At `noise_frac=0`, `__call__` and
`truth` agree.

B3. **End-to-end MG vs analytic agreement (planned)** — NEW. Prediction:
at low `|c|` and moderate `m`, MG and analytic μ agree at the 5% level
(MC noise on MG ~3% combined with NLO/missing-effects ~3%). Sketch:

```python
@pytest.mark.slow
def test_mg_vs_analytic_low_c():
    analytic = AnalyticSMEFTOracle(pdf="CT18NNLO")
    mg       = MadGraphSMEFTOracle(nevents=5000)
    c = np.zeros((1, N_WC)); c[0, WC_NAMES.index("clq1")] = 0.05
    m = np.array([1.0])
    mu_a = analytic.truth(c, m)
    mu_m = mg.truth(c, m)
    assert abs(mu_a[0] - mu_m[0]) / abs(mu_m[0]) < 0.05
```

B4. **MG-vs-analytic disagreement growth with |c|** — NEW. Prediction:
disagreement grows as the magnitude of the missing dim-6 squared term
divided by SM (i.e. `(c/Λ)^4` relative). Sketch: regress
`log|mu_mg − mu_analytic|` vs `log|c|`; expected slope > 1 in the
high-mass tail.

---

## C. Interface tests (require the `OracleResult` wrapper, section 3.4)

C1. **Output shape invariant across tiers** — Prediction: identical
top-level keys for T0, T1, T2 calls. Sketch:

```python
def test_result_shape_invariant():
    for fidelity in ("T0", "T1"):   # T2 in @slow variant
        r = oracle.call(c, m, fidelity=fidelity)
        assert set(r) == {"kinematics", "pieces", "total", "mu",
                          "wilson_coefficients", "metadata"}
        assert set(r["metadata"]) >= {"fidelity_tier", "backend",
            "pdf", "lambda_gev", "sqrt_s_gev", "order",
            "wall_time_s", "mc_noise", "nevents",
            "cache_hit", "oracle_call_id"}
```

C2. **`metadata.fidelity_tier` is the only carrier of tier semantics** —
Prediction: stripping metadata makes T0 and T1 results indistinguishable
in shape. Sketch:

```python
def test_metadata_carries_tier_only():
    r0 = oracle.call(c, m, fidelity="T0")
    r1 = oracle.call(c, m, fidelity="T1")
    for k in ("pieces", "total"):
        assert r0[k].keys() == r1[k].keys() if isinstance(r0[k], dict) else r0[k].shape == r1[k].shape
    assert r0["metadata"]["fidelity_tier"] != r1["metadata"]["fidelity_tier"]
```

---

## D. Cache tests

(Require `modules/oracle/cache.py` to be built; sketch the tests
alongside.)

D1. **Exact key for exact query** — Prediction: same `(c, m, fidelity,
order, λ, √s, pdf, code_version)` produces the same key bit-for-bit.

D2. **Key rounding at 1e-9** — Prediction: `c = 0.1234567891` and
`c = 0.12345678905` produce the same key.

D3. **Tier separation** — Prediction: T0 and T1 calls with identical
`(c, m)` produce different keys.

D4. **Code-version partition** — Prediction: after editing
`smeft.py`, the same `(c, m, fidelity, order, …)` produces a different
key; old rows still queryable explicitly by version.

D5. **Hit/miss reporting** — Prediction: second identical call after the
first hits the cache; `metadata.cache_hit = True`. Wall time second call
< 10% first call.

D6. **EPIG target-set cache benefit** — empirical. Sketch: 200-point
target set, evaluate once; then 100 EPIG rounds. Predict: target-set
re-evaluation hit rate = 1.0; EPIG-picked candidates' hit rate ≈ 0.

D7. **SM-piece cache hit at same m** — Prediction (extending the MG
adapter's pattern): explicit SM cache hit rate ~100% across distinct `c`
at fixed `m`.

D8. **Parquet round-trip integrity** — Prediction: writing then reading
back an `OracleResult` reproduces every numeric field to floating-point
precision.

---

## E. Interactive tests

E1. **Warmup idempotence** — Prediction: `oracle.warmup()` called twice
returns immediately on the second call (< 10 ms). Sketch:

```python
def test_warmup_idempotent():
    o = AnalyticSMEFTOracle(pdf="CT18NNLO")
    t0 = time.perf_counter(); o.warmup(); dt1 = time.perf_counter()-t0
    t0 = time.perf_counter(); o.warmup(); dt2 = time.perf_counter()-t0
    assert dt2 < 0.01 * dt1
```

E2. **T1 latency ≤ 100 ms for n_points = 1** — Prediction (section 5.3):
after warmup, single-point T1 call returns under 100 ms. Sketch:

```python
def test_t1_latency():
    o = AnalyticSMEFTOracle(pdf="CT18NNLO"); o.warmup()
    t = time.perf_counter(); o.truth(c1, m1); dt = time.perf_counter() - t
    assert dt < 0.1
```

E3. **T2 timeout** — Prediction: an MG call with `timeout_s=1` raises
`OracleTimeoutError`. Sketch (slow):

```python
@pytest.mark.slow
def test_mg_timeout():
    o = MadGraphSMEFTOracle(nevents=10000)
    with pytest.raises(OracleTimeoutError):
        o.call(c, m, timeout_s=1.0)
```

E4. **Async escalation** — Prediction: `call_async` returns a Future;
result is a valid `OracleResult` after completion; concurrent calls to T1
during a pending T2 are not blocked.

---

## F. Phoenix span tests

(See `aletheia_verification_plan.md` §1 for the Phoenix infrastructure;
this section adds oracle-specific attributes.)

F1. **Span emitted per oracle call** — Prediction: one span with
`name=oracle.call` per non-cached call. Cached calls still emit a span
with `oracle.cache.hit=True` but `oracle.cost_seconds≈0`.

F2. **`oracle.wilson_norm` reported, individual `c_i` NOT reported** —
Prediction: span attributes contain `oracle.wilson_norm` but no key of
the form `oracle.c_i`.

F3. **`oracle.fidelity_tier` reported and unique per call** — Prediction:
every span has exactly one `oracle.fidelity_tier` attribute.

---

## G. Scaling extrapolation tests

(Empirical-only; in a `@pytest.mark.scaling` group.)

G1. **T0 throughput** — Prediction: > 1000 calls/s.
G2. **T1 throughput** — Prediction: > 50 calls/s sustained.
G3. **Worker pool linear scaling for T2** — Prediction: with N workers,
T2 throughput is `N × single-worker` ± 10%.

---

## H. Test order and infrastructure

- `pytest` with `slow` marker for MG and any timing-sensitive test.
- All tests <1 s wall time except `slow`.
- `pytest-benchmark` for E2/G1/G2.
- Fix `numpy` seed in every random test; pin reference values from first
  passing run.
- For D-series (cache), `tmp_path` fixture so disk artefacts don't leak.

Order:
1. A1–A7 first (no code change; A5/A6 are new but trivial additions).
2. B1–B3 (already pass) plus B4 in a `slow` variant.
3. C/D/E require new code (`OracleResult`, `cache.py`, `warmup`,
   `call_async`); implement in that order.
4. F/G after Phoenix wiring is up.
