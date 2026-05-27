# Technical reference

This document is the single source of truth on what the package contains,
how its pieces fit together, and how its API differs from the earlier
prototypes. Read this end-to-end if the package looks unfamiliar; the rest
of the docs assume you already have this mental model.

For the math derivations see `ARCHITECTURE.md`. For wiring to Phoenix and
Gemini see `INTEGRATION.md`. For the orchestrator tool surface see
`TOOLS.md`.

---

## 1. Mental model in one paragraph

The surrogate predicts a single scalar `mu(c, m)` — the SMEFT modification
factor `sigma_BSM(c, m) / sigma_SM(m)` — at any point in
`(c, m_ll)` space. It does this via closed-form Bayesian linear regression
on a 75-dim feature space that captures the SMEFT polynomial structure in
the Wilson coefficients times a smooth polynomial in `log(m_ll)`. There is
no binning anywhere in the model. To get a binned cross-section, the
orchestrator integrates the surrogate over an `m_ll` bin numerically;
binning is a presentation layer, not a model concept.

---

## 2. What changed from the earlier prototypes

The package went through three iterations before becoming this package.
The agent has probably seen artifacts from one or more of them in the
trace history. The conceptual shifts are:

### v1 (binned, `intention_fm_demo.py`)

Predict surface: `mu(c) -> vector of shape (n_bins,)`. The model was a
matrix `W` of shape `(D_C, n_bins)` and prediction was `phi_c(c) @ W`.
Drift signal was averaged across bins. Conformal calibration was per-bin.

### v2 (unbinned, `intention_fm_unbinned.py`)

Predict surface: `mu(c, m) -> scalar`. The feature map became the tensor
product `phi_c(c) ⊗ phi_x(m)`, so the model is one weight vector
`w` of shape `(D_JOINT,)` and prediction is `phi_joint(c, m) @ w`. Drift
signal became per-point leverage. Conformal calibration was still global.

### v3 (`intention_fm_v3.py`) and the package

Same predict surface as v2. Conformal calibration was redone to be
stratified by leverage quantile, fixing the high-extrapolation
under-coverage that global conformal couldn't see. The package adds
serialisation, EPIG acquisition, a clean module split, and tests.

### Concept mapping for an agent coming from v1

| v1 concept                          | Package equivalent                                  |
| ----------------------------------- | --------------------------------------------------- |
| `predict(c)` returns shape `(B,)`   | `predict(c, m)` returns shape `(n,)`                |
| `n_bins` and bin edges              | `K_X` polynomial modes and a smooth basis in `log m`|
| Per-bin `W` matrix                  | One weight vector `w` of shape `(D_JOINT,)`         |
| Per-bin conformal factors           | Per leverage-stratum conformal factors              |
| "Drift in bin k"                    | "Leverage on a query point at mass m_k"             |
| "Plot the spectrum at fixed c"      | Evaluate the model on an `m` grid and plot         |
| "Histogram with bin integrals"      | Numerical integration over `m` between bin edges    |

### What is NOT in the package, despite being in earlier scripts

- No `n_bins` parameter anywhere.
- No bin-edge arrays in any object's state.
- No per-bin coverage in the calibrator. Coverage is per leverage stratum.
- No "drift score per bin". Drift is per-query-point leverage.
- No discretisation of the observable axis. The smooth log-poly basis is
  evaluated at exact `m` values.

If the agent is reading old artifacts and expects to find any of the
above in the package, it won't. The equivalent operations are all
expressible in terms of `(c, m)`-point queries and aggregations chosen at
display time.

---

## 3. Data shapes throughout the loop

Concrete shapes for every array the agent will pass or receive. The
constants `N_WC`, `K_X`, `D_C`, `D_X`, `D_JOINT` are module-level in
`modules.surrogate.features` and import like

    from modules.surrogate import N_WC, K_X, D_C, D_X, D_JOINT

With defaults: `N_WC = 4`, `K_X = 5`, `D_C = 15`, `D_X = 5`,
`D_JOINT = 75`.

### Input arrays

| Variable | Shape         | Meaning                                       |
| -------- | ------------- | --------------------------------------------- |
| `C`      | `(n, N_WC)`   | Wilson coefficients, columns ordered as       |
|          |               | `("cHq3", "cHq1", "clq3", "clq1")`.            |
| `M`      | `(n,)`        | `m_ll` in TeV (continuous, NOT bin indices).   |
| `Y`      | `(n,)`        | Observations of `mu(c, m)`, scalar per point.  |

Every function in the package follows this convention. There is never a
`(n, n_bins)` array.

### Feature matrices

| Function     | Input shapes                  | Output shape       |
| ------------ | ----------------------------- | ------------------ |
| `phi_c(c)`   | `(n, N_WC)`                   | `(n, D_C)`         |
| `phi_x(m)`   | `(n,)`                        | `(n, D_X)`         |
| `phi_joint(c, m)` | `(n, N_WC), (n,)`        | `(n, D_JOINT)`     |

The joint feature is the elementwise tensor product:
`phi_joint[i, c*D_X + x] = phi_c[i, c] * phi_x[i, x]`.

### Model state (after `IntentionFM.fit`)

| Attribute    | Shape                  | Meaning                              |
| ------------ | ---------------------- | ------------------------------------ |
| `w`          | `(D_JOINT,)`           | Fitted ridge weights.                |
| `A_inv`      | `(D_JOINT, D_JOINT)`   | Inverse Gram matrix; used for        |
|              |                        | leverage, EPIG, and Sherman-Morrison.|
| `noise_frac` | scalar                 | MAD-estimated relative noise.        |
| `C_train`    | `(n_train, N_WC)`      | Retained for `update()` refits.      |
| `M_train`    | `(n_train,)`           |                                      |
| `Y_train`    | `(n_train,)`           |                                      |

### Calibrator state (after `ConformalCalibrator.fit`)

| Attribute  | Shape                 | Meaning                              |
| ---------- | --------------------- | ------------------------------------ |
| `edges`    | `(n_strata + 1,)`     | Leverage quantile cut points.        |
|            |                       | `edges[0] = -inf, edges[-1] = inf`.  |
| `factors`  | `{cov: (n_strata,)}`  | Per-stratum sigma multipliers,       |
|            |                       | keyed by target coverage (0.683,     |
|            |                       | 0.954 by default).                   |

### Acquisition output

All three acquisition functions return integer indices into the
candidate pool:

    indices: np.ndarray of shape (k,), dtype int

They do not return `(c, m, y)` tuples. The orchestrator does
`C_pool[indices], M_pool[indices]` to get the points, then calls
`query_oracle(...)` to get `y`, then `update_model(...)` with all three.

---

## 4. Public API reference

This section is the API contract. Every public symbol exported by
`modules.surrogate` is listed with full signature, return type, and
behavioural notes. The agent should treat this as authoritative; if the
docstring in the source disagrees, the source is right.

### Constants

```python
from modules.surrogate import (
    N_WC, WC_NAMES, K_X, M_REF, PAIRS,
    D_C, D_X, D_JOINT,
)
```

| Name      | Value (default)                              | Role                          |
| --------- | -------------------------------------------- | ----------------------------- |
| `N_WC`    | `4`                                          | Number of Wilson coefficients.|
| `WC_NAMES`| `("cHq3", "cHq1", "clq3", "clq1")`           | Display labels, ordered.      |
| `K_X`     | `5`                                          | Polynomial order in `log m`.  |
| `M_REF`   | `1.0` (TeV)                                  | Reference mass for `log m`.   |
| `PAIRS`   | `[(0,0), (0,1), ..., (3,3)]`                 | `combinations_with_replacement`|
| `D_C`     | `1 + N_WC + len(PAIRS) = 15`                 | Feature dim of `phi_c`.       |
| `D_X`     | `K_X = 5`                                    | Feature dim of `phi_x`.       |
| `D_JOINT` | `D_C * D_X = 75`                             | Feature dim of `phi_joint`.   |

### Feature functions

```python
phi_c(c: np.ndarray) -> np.ndarray
```

Returns the SMEFT-structure features `{1, c_a, c_a c_b}` evaluated at
each row of `c`. The column order is

    [const, c_0, c_1, ..., c_{N_WC-1},
     c_0*c_0, c_0*c_1, ..., c_0*c_{N_WC-1},
     c_1*c_1, c_1*c_2, ..., c_{N_WC-1}*c_{N_WC-1}]

i.e. constant, then linear in canonical order, then quadratic in the
order given by `PAIRS`. Input may be `(N_WC,)` or `(n, N_WC)`; output is
always `(n, D_C)`.

```python
phi_x(m: np.ndarray) -> np.ndarray
```

Returns `[1, log(m/M_REF), log(m/M_REF)^2, ..., log(m/M_REF)^{K_X-1}]`
at each `m`. Input is `(n,)` or scalar (broadcast to `(1,)`); output is
`(n, D_X)`.

```python
phi_joint(c: np.ndarray, m: np.ndarray) -> np.ndarray
```

Tensor product of `phi_c(c)` and `phi_x(m)`. Both inputs must have the
same `n`. Output shape `(n, D_JOINT)`. The column index `c*D_X + x`
selects the `(c-th SMEFT mode) * (x-th log-m mode)` product.

### Oracle protocol

```python
class Oracle(Protocol):
    def __call__(self, c, m, *, noise: bool = True) -> np.ndarray: ...
    def truth(self, c, m) -> np.ndarray: ...
```

Real implementations must take `c` of shape `(n, N_WC)` and `m` of shape
`(n,)`, return shape `(n,)`. `noise=True` adds multiplicative MC-like
noise; `noise=False` returns the noiseless physical value. `truth(c, m)`
is the noiseless oracle and is what evaluators should compare against.

### `DummyAnalyticOracle`

```python
DummyAnalyticOracle(seed: int = 0, noise_frac: float = 0.05)
```

Placeholder. Replace with the analytic SMEFT calculator. Returns
`phi_joint(c, m) @ W_random.flatten()` where `W_random` is constructed
once at init from `seed`. Useful for tests and demos; should not appear
in production.

### `IntentionFM`

```python
class IntentionFM:
    def __init__(self, lam: float = 1e-3): ...
    def fit(self, C, M, Y) -> IntentionFM: ...
    def predict(self, C, M, return_std: bool = True): ...
    def leverage(self, C, M) -> np.ndarray: ...
    def update(self, C_new, M_new, Y_new) -> IntentionFM: ...
    def state_dict(self) -> dict: ...
    @classmethod
    def from_state_dict(cls, state: dict) -> IntentionFM: ...
```

`fit(C, M, Y)` computes `A_inv`, `w`, and estimates `noise_frac` via MAD
on relative residuals. Sets `C_train, M_train, Y_train` from the inputs
(by reference; do not mutate them externally). Returns `self`.

`predict(C, M, return_std=True)` returns either `mu` (shape `(n,)`) if
`return_std=False`, or `(mu, sigma)` (both shape `(n,)`) otherwise.
`sigma` is `noise_frac * |mu| * sqrt(1 + leverage)`. This is the RAW
sigma; for calibrated coverage, use `ConformalCalibrator.coverage_sigma`
instead.

`leverage(C, M)` returns `phi(C, M) @ A_inv @ phi(C, M).T` per row,
shape `(n,)`. Non-negative by construction. Higher leverage = more
extrapolation = more epistemic uncertainty. This is the drift signal.

`update(C_new, M_new, Y_new)` does a full refit on the concatenated
data. NOT an incremental rank-1 update. The refit is cheap (sub-ms);
choosing full refit keeps `noise_frac` and `A_inv` exact.

`state_dict()` returns a plain dict with `(lam, w, A_inv, noise_frac,
C_train, M_train, Y_train)`. Pickle-clean. `from_state_dict(state)`
reconstructs without rerunning the fit.

### `ConformalCalibrator`

```python
class ConformalCalibrator:
    def __init__(self, n_strata: int = 5): ...
    def fit(self, model, C, M, Y, coverages=(0.683, 0.954)) -> ConformalCalibrator: ...
    def stratum_index(self, lev: np.ndarray) -> np.ndarray: ...
    def coverage_sigma(self, model, C, M, coverage=0.683) -> np.ndarray: ...
    def state_dict(self) -> dict: ...
    @classmethod
    def from_state_dict(cls, state: dict) -> ConformalCalibrator: ...
```

`fit(model, C, M, Y, coverages)` is called AFTER `model.fit(...)`. The
calibration set `(C, M, Y)` should cover the broader probe region the
model will be queried on, not just the training box. See "Common
pitfalls" below.

`stratum_index(lev)` returns the stratum each leverage value falls in,
shape `(n,)`, dtype `int`, values in `[0, n_strata - 1]`.

`coverage_sigma(model, C, M, coverage)` returns the calibrated sigma
such that `|y - mu| < sigma` covers `coverage` fraction of points
within each stratum. Use this rather than the raw `predict(..)[1]` for
any production interval. Raises `KeyError` if the calibrator was not fit
for the requested coverage.

`state_dict()` returns `{n_strata, edges, factors}`. Pickle-clean.

### Acquisition functions

```python
random_acquire(rng: np.random.Generator, pool_size: int, k: int) -> np.ndarray
leverage_acquire(model, C_pool, M_pool, k: int) -> np.ndarray
epig_acquire(model, C_pool, M_pool, C_target, M_target, k: int) -> np.ndarray
```

All return `np.ndarray` of `k` integer indices into the pool. No
duplicates.

`random_acquire` is `np.random.Generator.choice` without replacement.

`leverage_acquire` is sequential greedy: at each step, pick the pool
candidate with maximum current leverage, Sherman-Morrison-update `A_inv`
as if that pick had been observed, then continue. This prevents
clustering.

`epig_acquire` is sequential greedy on Expected Predictive Information
Gain. At each step it scores every available pool candidate `p` by

    score(p) = mean over targets i of 0.5 * log( lev_T_i / (lev_T_i - k_Tp_i^2 / (lev_p + 1)) )

where `lev_T = phi_T A_inv phi_T^T`, `lev_p = phi_p A_inv phi_p^T`, and
`k_Tp = phi_T A_inv phi_p^T`. Picks the argmax, Sherman-Morrison
updates `A_inv`, continues. With a uniform target set, this approximates
leverage acquisition; with a focused target set, it picks differently.

### Evaluation helpers

```python
empirical_coverage(y, mu, sigma) -> float
decile_calibration(y, mu, sigma, n_bins=10) -> tuple[np.ndarray, np.ndarray]
stratified_coverage(model, calibrator, C, M, Y, coverage=0.683) -> list[dict]
```

`empirical_coverage` is `mean(|y - mu| < sigma)`. Simple but used
everywhere.

`decile_calibration(y, mu, sigma, n_bins=10)` sorts by predicted sigma,
splits into deciles, and returns `(mean_sigma_per_decile,
rms_err_per_decile)`. Used to validate that the predicted sigma is
quantitatively right per-decile (not just on average).

`stratified_coverage(...)` returns one dict per leverage stratum, each
with keys `stratum`, `n`, `mean_leverage`, `raw_coverage`,
`conformal_coverage`, `target_coverage`. This is what the Phoenix
calibration evaluator calls.

---

## 5. State and persistence

Two stateful objects: `IntentionFM` and `ConformalCalibrator`. Both
expose `state_dict()` / `from_state_dict()`. The state dicts contain
only numpy arrays and python scalars, so they're pickleable, JSON-able
(after numpy serialisation), and bucket-storable as-is.

A model version, end to end, is

    version = {
        "model":      fm.state_dict(),
        "calibrator": cc.state_dict(),
        "metadata":   {"created_at": ..., "n_train": ..., "noise_frac": ...},
    }

The orchestrator should store these by version ID so it can A/B-compare
two versions on a held-out eval set before promoting. Eval set should
itself be versioned and frozen for the comparison to be honest.

---

## 6. The end-to-end loop, with code

This is a complete worked example. The agent should be able to paste
this and run it; if it's confused about how some piece of the API fits
together, this is the reference.

```python
import numpy as np
from modules.surrogate import (
    IntentionFM, ConformalCalibrator, DummyAnalyticOracle,
    leverage_acquire, epig_acquire,
    stratified_coverage,
    N_WC,
)

oracle = DummyAnalyticOracle(seed=42)
rng    = np.random.default_rng(0)

# ---- 1. initial fit ----
C_train = rng.uniform(-1, 1, (200, N_WC))
M_train = rng.uniform(0.2, 2.5, 200)
Y_train = oracle(C_train, M_train, noise=True)

fm = IntentionFM(lam=1e-3).fit(C_train, M_train, Y_train)

# ---- 2. initial calibration ----
# Calibration set on the BROADER probe region, not the training box.
C_cal = rng.uniform(-2, 2, (200, N_WC))
M_cal = rng.uniform(0.2, 2.5, 200)
Y_cal = oracle(C_cal, M_cal, noise=True)

cc = ConformalCalibrator(n_strata=5).fit(fm, C_cal, M_cal, Y_cal,
                                         coverages=(0.683, 0.954))

# ---- 3. predict at production query points ----
C_q = rng.uniform(-2, 2, (10, N_WC))
M_q = rng.uniform(0.2, 2.5, 10)

mu, sigma_raw = fm.predict(C_q, M_q)                      # raw Gaussian sigma
sigma_cc      = cc.coverage_sigma(fm, C_q, M_q, 0.683)    # calibrated 1σ
lev           = fm.leverage(C_q, M_q)                     # drift signal

# ---- 4. drift triggers acquisition ----
# Suppose the drift evaluator says mean leverage > threshold in a region
# centred on (c_0 ~ 1.8, m ~ 2.0). The orchestrator decides EPIG with a
# focused target there.
n_T = 200
target_c0 = 1.8
target_m  = 2.0
C_target = np.tile([target_c0, 0.0, 0.0, 0.0], (n_T, 1)) \
           + rng.normal(0, 0.06, (n_T, N_WC))
M_target = np.full(n_T, target_m) + rng.normal(0, 0.03, n_T)

C_pool = rng.uniform(-2, 2, (500, N_WC))
M_pool = rng.uniform(0.2, 2.5, 500)

picks = epig_acquire(fm, C_pool, M_pool, C_target, M_target, k=20)
C_new, M_new = C_pool[picks], M_pool[picks]

# ---- 5. query oracle ----
Y_new = oracle(C_new, M_new, noise=True)

# ---- 6. update model ----
fm = fm.update(C_new, M_new, Y_new)         # returns self after refit

# ---- 7. refit calibration after update ----
# (Refit on the SAME cal set; the cal set is a fixed reference, not
# something that grows with the training data.)
cc = ConformalCalibrator(n_strata=5).fit(fm, C_cal, M_cal, Y_cal)

# ---- 8. evaluate per-stratum coverage on a held-out eval set ----
C_eval = rng.uniform(-2, 2, (1000, N_WC))
M_eval = rng.uniform(0.2, 2.5, 1000)
Y_eval = oracle(C_eval, M_eval, noise=True)
rows   = stratified_coverage(fm, cc, C_eval, M_eval, Y_eval, coverage=0.683)
for r in rows:
    print(r)

# ---- 9. persist this version ----
version = {"model": fm.state_dict(), "calibrator": cc.state_dict()}
# pickle/json/etc. to bucket, keyed by version ID
```

That's the entire lifecycle. Every Gemini tool in `TOOLS.md` is a thin
wrapper around one of these calls plus span instrumentation.

### How to plot a binned spectrum from this surrogate

Common request, but the model has no concept of bins. The answer is
"evaluate on a grid, then aggregate at display time":

```python
# Spectrum at fixed c (e.g. c_0 = 1.5, others = 0)
m_grid = np.linspace(0.2, 2.5, 200)
c_grid = np.tile([1.5, 0.0, 0.0, 0.0], (len(m_grid), 1))
mu, sigma = fm.predict(c_grid, m_grid)
# plot(m_grid, mu) with sigma error bars

# Bin-integrated prediction across edges m_edges
def bin_integral(c, m_edges, n_quad=20):
    out = np.zeros(len(m_edges) - 1)
    for k in range(len(m_edges) - 1):
        mg = np.linspace(m_edges[k], m_edges[k+1], n_quad)
        cg = np.tile(c, (n_quad, 1))
        mu_local = fm.predict(cg, mg, return_std=False)
        out[k] = np.trapezoid(mu_local, mg)
    return out
```

Integration is on the orchestrator side. Don't try to put binning into
the model.

---

## 7. Common pitfalls

The agent will hit these if it's not careful. They're all easy to avoid
once you know they exist.

### 1. Calling `predict(c)` without `m`

Old binned API was `predict(c)`. New API requires both `c` and `m`. If
you find code calling `model.predict(c)`, it's referencing the v1 API
and will fail with a `TypeError`. Insert an `m` grid.

### 2. Looking for `n_bins` anywhere

There is no `n_bins`. There is `K_X` (the polynomial order of the
smooth basis in `log m`). These are not the same thing. `K_X = 5` does
NOT mean five bins; it means a degree-4 polynomial in `log m`.

### 3. Calibrating on the training box

The calibration set MUST cover the broader probe region the model will
be queried on. If you draw the cal set from `|c| <= 1` but query at
`|c| <= 2`, the high-leverage stratum's conformal factor is
extrapolated and produces the same under-coverage you started with.
Both demos use `|c| <= 2` for cal; the comment in `demo_v3.py` flags
this explicitly.

### 4. Confusing `noise_frac` with conformal factors

`fm.noise_frac` is the model's estimate of the data noise rate. It's a
single scalar set during fit, used to scale the raw predictive sigma.
The conformal factors `cc.factors[0.683]` are per-stratum multipliers
ON TOP OF the raw sigma, learned to hit nominal coverage. They are
different objects with different roles. Don't try to use one in place
of the other.

### 5. Confusing leverage with predictive variance

`leverage(C, M)` is the epistemic part of the predictive variance,
divided out by `noise_frac^2 * |mu|^2`. It's a dimensionless drift
signal: high leverage means "model is extrapolating here". It is NOT
the actual predictive variance. To get the predictive variance, call
`predict(C, M)` and look at `sigma**2`.

### 6. Expecting `update` to be incremental

`fm.update(C_new, M_new, Y_new)` does a full refit on the concatenated
data. It's cheap enough that this is the right choice (sub-ms at the
relevant scales), and it keeps `noise_frac` and `A_inv` exact rather
than letting them drift. Don't try to implement an incremental version.

### 7. Calling EPIG without a target

`epig_acquire` requires `C_target` and `M_target`. There is no fallback
to leverage acquisition. If the orchestrator wants leverage behaviour,
call `leverage_acquire`. If it wants EPIG behaviour but has no
preferred target, supply a target set drawn uniformly from the probe
region; that gives EPIG behaviour that approximates leverage but is
mathematically EPIG.

### 8. Calling `coverage_sigma` for a coverage that wasn't fit

`coverage_sigma(model, C, M, coverage=0.5)` raises `KeyError` because
`0.5` isn't in `cc.factors`. The calibrator only knows the coverages
it was fit for. Default is `(0.683, 0.954)`. If you need 0.5 or 0.99,
pass them to `cc.fit(coverages=...)`.

### 9. Mutating C_train, M_train, Y_train externally

The model retains references (not copies) to the training arrays for
later refits via `update()`. If the caller mutates those arrays after
calling `fit`, the model's internal state is silently invalidated.
The demos use `.copy()` defensively where this matters; the agent
should too.

### 10. Calling `ConformalCalibrator.fit` before `IntentionFM.fit`

`cc.fit(fm, C, M, Y)` reads `fm.A_inv` and `fm.noise_frac`, which are
`None` until `fm.fit(...)` has been called. The error message is
"NoneType has no attribute" and will be confusing. Always fit the
model first.

### 11. Pool size too small for sequential greedy

`leverage_acquire(fm, C_pool, M_pool, k)` requires `len(C_pool) >= k`.
The sequential greedy picks without replacement, so `k > len(C_pool)`
returns malformed output (it won't raise, but you'll get fewer than
`k` valid indices because `argmax` keeps returning `-inf`). Validate
pool size in the tool wrapper.

### 12. Heteroscedastic noise in EPIG

The closed-form EPIG derivation in `acquisition.py` assumes
homoscedastic noise in the FIT (which is what's actually used in the
ridge regression). The `noise_frac * |mu|` heteroscedasticity is only
applied at predict time. For a strict heteroscedastic EPIG, multiply
`lev_p + 1` in the denominator by `(sigma_p / sigma_avg)^2`. It's a
five-line change; not currently in the package.

---

## 8. Glossary

| Term                  | Meaning                                                |
| --------------------- | ------------------------------------------------------ |
| `mu(c, m)`            | SMEFT modification factor: `sigma_BSM / sigma_SM` at   |
|                       | a point in Wilson-coefficient and mass space.          |
| `c`                   | Wilson coefficient vector, shape `(N_WC,)`.            |
| `m`                   | `m_ll` invariant mass, scalar TeV.                     |
| Wilson coefficients   | Coupling parameters in the SMEFT Lagrangian. Each      |
|                       | parameterises one effective operator's strength.       |
| SMEFT polynomial      | The fact that `mu` is exactly polynomial in `c` (linear|
| structure             | and quadratic at dim-6). Captured by `phi_c`.          |
| Feature map           | Basis function expansion: takes raw inputs to a high-  |
|                       | dimensional space where regression is linear.          |
| Joint feature         | `phi_c(c) ⊗ phi_x(m)`, the tensor product giving the  |
|                       | 75-dim space the closed-form is solved in.             |
| Leverage              | `phi A_inv phi.T` per point. Drift signal,             |
|                       | proportional to epistemic predictive variance.         |
| Drift                 | Increase in mean leverage over time on production      |
|                       | queries. Trigger for orchestrator acquisition.         |
| Acquisition           | Choosing which oracle points to query next.            |
| Oracle                | Function returning ground-truth `mu(c, m)`. Real one   |
|                       | is the analytic SMEFT calculator; placeholder is       |
|                       | `DummyAnalyticOracle`.                                 |
| Conformal calibration | Rescaling predicted sigma so empirical coverage hits   |
|                       | nominal. Stratified here means per-leverage-bin.       |
| Stratum               | A leverage quantile bin in the calibrator. Default 5.  |
| EPIG                  | Expected Predictive Information Gain. Acquisition      |
|                       | strategy that maximises expected variance reduction    |
|                       | on a user-supplied target set.                         |
| Target set            | For EPIG: the points the orchestrator wants the model  |
|                       | to be accurate at. Drives what EPIG picks.             |
| Sherman-Morrison      | Rank-1 update formula for matrix inverse. Used in      |
|                       | sequential greedy to update `A_inv` without a full     |
|                       | matrix inverse per pick.                               |
| `D_JOINT`             | Joint feature dimension, 75 by default.                |
| `noise_frac`          | Estimated relative noise rate. One scalar per model.   |

---

If the agent is still confused after reading this, the issue is
probably outside what's documented here: it might be looking at a stale
artifact, a different repository, or something downstream of the
package proper. The package itself is small and complete; everything in
it is in this doc.
