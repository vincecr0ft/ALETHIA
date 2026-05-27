# EPIG, conformal, and the loop — eval (second pass, agent d)

This document reconciles agent (d)'s EPIG / conformal / loop mathematics with agent (a)'s oracle interface, agent (b)'s c-free foundation model, and agent (c)'s drift detectors. EPIG sits at the centre of the loop: drift flags generate a region R, EPIG selects k commissions in R, the oracle returns labelled embeddings, the FM updates, the conformal layer recalibrates, and the loop closes.

## 1. The candidate space is the oracle-commission type

Agent (a) §3.2 defines `OracleResult` as the structured return of one commission. Agent (b) §4 / §8 builds an encoder `Enc({events_1, ..., events_N}) -> z in R^d`. EPIG must rank over commissions whose execution produces an embedding `z`, plus a label (the observable of interest, e.g. `mu` extracted from `OracleResult.mu`).

Joint type definition (load-bearing — the constraint stands or falls on whether `c` appears anywhere on the FM forward path):

```python
@dataclass(frozen=True)
class Commission:
    wilson_region: tuple[float, float] | dict[str, tuple]   # box / sampler config
    kinematic_window: dict                                  # (mmll_lo, mmll_hi, bins)
    fidelity_tier: Literal["T0", "T1", "T2", "T3", "T5"]    # agent (a) tiers
    observable:    Literal["m_ll", "pT_l"] = "m_ll"
    order:         Literal["linear", "quadratic"] = "quadratic"

@dataclass(frozen=True)
class Commitment:
    commission: Commission
    result:     OracleResult       # agent (a) §3.2 shape
    z:          np.ndarray         # (D_Z,) FM encoder output
    y:          np.ndarray         # (n_T,) probe label vector (e.g. mu at m-grid)
```

The fields `wilson_region`, `fidelity_tier`, `order`, and `c` (inside `OracleResult.wilson_coefficients`) are **commission metadata**. They live on the oracle side and on the loss-construction side. They do not enter the FM encoder. The encoder consumes only `result.pieces` (or whatever measurable representation agent (b) picks) and produces `z`.

The exact type signature of EPIG acquisition under this regime:

```python
def epig_acquire_commissions(
    model:          ProbeModel,                # closed-form ridge on phi(z)
    commissions:    list[Commission],          # candidate jobs
    embed_preview:  Callable[[Commission], np.ndarray],
                                              # cheap preview that returns z
                                              # without paying the full
                                              # oracle cost (T0/T1, mean over
                                              # a small MC if applicable)
    z_target:       np.ndarray,                # (n_T, D_Z) target embeddings
    k:              int,
    mc_samples:     int = 1,                   # >1 -> outer expectation over
                                              # z(kappa) | kappa for noisy
                                              # preview oracles
    cost:           Callable[[Commission], float] | None = None,
                                              # optional cost-normalisation
                                              # (MacKay 1992 / agent (a) §3)
) -> list[Commission]:
    ...
```

Agent (a)'s `oracle.call(c, m, fidelity="T0", return_pieces=True)` is the natural `embed_preview` body: it returns an `OracleResult` cheaply at T0; the FM encoder turns the result into `z`. With agent (a)'s in-memory LRU cache, repeat previews during scoring are free.

What is **not** in the signature: no `C_pool`, no `M_pool`, no `phi_joint`. The current `epig_acquire(model, C_pool, M_pool, C_target, M_target, k)` at `modules/surrogate/acquisition.py:105-146` violates the brief because `phi_joint` consumes `c`. The rewrite changes the pool type from `(C_pool, M_pool)` to `list[Commission]` and replaces `phi_joint(c, m)` with `model.feature_map(z) = phi(z)`. Algebra inside the loop is the same (Sherman-Morrison on `A_inv`, mean over targets, Cauchy-Schwarz floor).

This also tightens the contract with agent (c). The "drifted region" agent (c) produces is now a region in the **embedding** space (a cluster id, a ball in `z`, the v_min direction of `A = Phi^T Phi`). Agent (c) §3.2 already made this explicit: "Under the constraint, 'region' is no longer a slice of `(c_0, m)` space; it is a clustering of the FM's embedding space `z`." EPIG's `z_target` argument is exactly the embeddings sampled from the drifted cluster.

## 2. Conformal recalibration is the bug-class checkpoint

The bug-class finding in §4.3 of summary.md: split-conformal exchangeability between cal scores and test scores **breaks** at every Sherman-Morrison update of the model. Agent (c) §2.2 / §6.2 detects this via BH-corrected binomial coverage drift on the per-region empirical coverage, and agent (c) §2.2 advertises Gibbs-Candès (arXiv:2305.12616) as the no-retrain recalibration option. Coordination point: the orchestration must either

  (i)  refit `ConformalCalibrator.fit(model_t, C_cal, M_cal, Y_cal)` after *every* `model.update(...)`, with `C_cal, M_cal` drawn from the *current probe region* — or

  (ii) replace `ConformalCalibrator` by a Gibbs-Candès online-adaptive conformal layer that updates `f_s` after each test point via a Robbins-Monro step `f_s <- f_s + gamma_t (1{|y - mu| > f_s sigma} - alpha)`.

(i) is mandatory in the current code (calibration.py:35-70). The recommended pattern is the loop variant: step 8 of summary.md §5.1 says "Recalibrate on a cal set drawn from the current probe region", and agent (c)'s `drift.calibration.bh` evaluator should fire whenever a region drifts out of coverage. If the evaluator does not fire, refit is still required after a model update — because the **calibration-preserved-iff-refit** guarantee (§5.3) is statistical: empirical coverage may stay within nominal slack by luck for a few cycles but is no longer the guarantee of split-conformal. The orchestration code must call refit *unconditionally* on every `model.update` event, not "when the BH evaluator fires".

Spelling this out as an invariant the test suite enforces:

```python
def test_recalibrate_after_update():
    fm0, cc0 = ...
    fm1 = fm0.update(C_new, M_new, Y_new)
    # The state_dict timestamp on cc0 is now older than fm1's. The runtime
    # must reject coverage_sigma calls with an OutdatedCalibratorError if
    # the model has been updated since the calibrator was fit.
    with pytest.raises(OutdatedCalibratorError):
        cc0.coverage_sigma(fm1, C_te, M_te, 0.683)
```

This is a runtime check, not a heuristic. The `ConformalCalibrator.fit` should stamp a `model_fingerprint = hash(model.A_inv.tobytes() + model.w.tobytes())`, and `coverage_sigma` should compare against the model passed in. Agent (c)'s evaluator catches the *symptom* (coverage drift); the runtime check catches the *cause* (using a stale calibrator).

Empirical justification: section 2 of empirical-results.md below shows that when 100 picks are concentrated in the highest-leverage cal stratum, mean no-refit 68% coverage = 0.7769 (+9.4pp from nominal), versus refit = 0.6984 (within 1.5pp). The bug is real when the picks have high cross-leverage with cal; in benign cases the effect is at the noise floor; the runtime check is the safe default because the orchestrator cannot in general predict the cross-leverage of the next acquisition batch.

## 3. Target-set total predictive entropy H_T is a Phoenix span attribute

H_T = 0.5 sum_{i in T} log(2 pi e sigma^2 lev_{T_i}) is a single scalar per loop cycle (summary.md §6, quantity 1). It is the headline EPIG monitor. Agent (c) §5 sketches the span tree; the natural attachment is the `fm.update` span:

```
aletheia.fm.target_entropy_H_T_pre     float   (before EPIG-driven update)
aletheia.fm.target_entropy_H_T_post    float   (after the update + refit)
aletheia.fm.target_var_sum             float   (sum_i sigma^2 lev_{T_i})
aletheia.fm.condition_number_kappa     float   (agent (c) §2.3)
aletheia.fm.kappa_delta                float   (kappa_pre - kappa_post)
aletheia.fm.target_set_size_n_T        int
```

Plus per-region (clustered z-space) entropy as a list-valued attribute:

```
aletheia.fm.target_entropy_by_region   list[float]
```

Phoenix supports list-valued attributes; agent (c) already plans to attach `aletheia.drift.cal.per_stratum_pvalues` (list[float]) under the same convention.

The Phoenix MCP server can then answer: "Plot H_T over the last 50 cycles" — `get-spans` filtered on `name == "fm.update"`, project `aletheia.fm.target_entropy_H_T_post` over time. "Which retrain reduced H_T the most" — sort by `H_T_pre - H_T_post` descending. The dashboard becomes single-attribute queries; no derived computation on the Phoenix side.

## 4. Calibration set drawn from the broader probe region — the data-flow

Stated as an invariant: **the calibration set for `ConformalCalibrator.fit` must be drawn from the same distribution the model will be queried on, not from the training distribution.** summary.md §4.3 (this agent's §4.3) and the existing test at `tests/test_calibration.py:14-15` already use this pattern. Under iterative refinement (summary.md §5.1 step 8), this invariant must be preserved as the probe region drifts.

Concrete data-flow per loop iteration:

1. Agent (c)'s drift detector fires on cluster `R` in embedding space.
2. EPIG selects `k` commissions whose execution produces embeddings concentrated in `R`. Picks now training data are pulled toward `R`.
3. `model.update(C_new, M_new, Y_new)` (or its Sherman-Morrison equivalent).
4. **Replenish the calibration set from `R`**: sample `n_cal_R / k` new commissions from `R` *at the oracle*, label them, append to the cal store while evicting the oldest cal points from `R` to keep `n_cal_R` constant.
5. `ConformalCalibrator.fit(model_t, C_cal_new, M_cal_new, Y_cal_new)`.

The key sub-invariant: the cal-set composition tracks the probe-region composition. If 30% of probe queries hit cluster `R` after the update, then 30% of cal-set points must live in `R`. Implementation: maintain a per-cluster cal pool of size proportional to recent probe traffic in that cluster. Agent (a)'s parquet cache layer (oracle §4.2) is the right substrate — cal points are just oracle results with a `metadata.cal_for_region` tag. Agent (c)'s `aletheia.fm.region_id` attribute on the `surrogate.predict` span identifies which cluster a probe query hit, so the proportionality update is a span-aggregation step.

Failure mode without this: cal points concentrated in the original training box; after several EPIG updates pull the support outwards, the cal set no longer represents the probe distribution. Coverage on the new high-leverage queries silently degrades. The Barber-Candès-Ramdas-Tibshirani bound (arXiv:2202.13415 Theorem 1) makes this precise: the coverage deficit is upper-bounded by the Lipschitz constant of the score CDF times the W_1(P_cal, P_probe). Empirical-results section 3 quantifies this: BAD cal (cal on |c|<=1) at |c| in [1.5, 2] gives cov_68 = 0.523, while GOOD cal (cal on |c|<=2) at the same band gives cov_68 = 0.729; the difference is the 21pp coverage deficit predicted by the bound.

## 5. The two guarantees, formalised

**Guarantee A (variance reduction in expectation).** Let `T = {x_T^{(1)}, ..., x_T^{(n_T)}}` be the target set, and let `p* in P` be the EPIG pick. Under the conjugate Gaussian linear model with homoscedastic noise, for each `i`,

    sigma_post^2(x_T^{(i)} | p*) = sigma_pre^2(x_T^{(i)}) - sigma^2 * k_{T_i, p*}^2 / (1 + lev_{p*})

with `k_{T_i, p*} = phi(x_T^{(i)}) A^{-1} phi(p*)^T`. Cauchy-Schwarz on the A^{-1} inner product gives `k_{T_i, p*}^2 <= lev_{T_i} * lev_{p*}`, hence the second term is non-negative, hence

    sigma_post^2(x_T^{(i)} | p*) <= sigma_pre^2(x_T^{(i)})

with strict inequality whenever `k_{T_i, p*} != 0`. Summing over `i`,

    sum_i sigma_post^2 <= sum_i sigma_pre^2,

and the EPIG-optimal `p*` maximises the mean of `0.5 log(sigma_pre^2 / sigma_post^2)`, which is monotone increasing in the sum-of-variance reduction. Therefore the EPIG pick achieves the largest possible expected target-set variance reduction in the conjugate Gaussian model.

Conditions:
- Conjugate Gaussian likelihood (homoscedastic, see §1.4 of summary.md).
- Correctly fit ridge (the A^{-1} cache is current).
- Cauchy-Schwarz holds exactly; no numerical breakdown.

**Guarantee B (calibration preserved iff refit).** Let `(mu_hat_0, sigma_hat_0, f_s^{(0)})` be the model state and conformal multipliers before a Sherman-Morrison update with pick `p*`. Let `(mu_hat_1, sigma_hat_1, f_s^{(1)})` be the state after. Let `q^{(0)}` denote the empirical quantile of nonconformity scores `{s_i^{(0)}} = {|y_i - mu_hat_0(x_i)| / sigma_hat_0(x_i)}` on cal set `D_cal`. Split-conformal coverage at level `1 - alpha` holds for the interval `[mu_hat_t - q_t sigma_hat_t, mu_hat_t + q_t sigma_hat_t]` if and only if cal scores `{s_i^{(t)}}` are exchangeable with test scores `s_test^{(t)}`.

After the update:
- `mu_hat_1 = mu_hat_0 + Delta mu(p*)`, with `Delta mu(p*)` a function of `A^{-1} phi(p*)^T` and `y_{p*}`;
- `sigma_hat_1` changes through both the predictive variance (because `lev(x)` shrinks at every `x`) and through the per-stratum factor `f_s^{(t)}` (because stratum edges, which depend on `lev`, shift);
- the cal score `s_i^{(1)}` is therefore not equal to `s_i^{(0)}`.

Quantitative statement: by Barber-Candès-Ramdas-Tibshirani 2023 (arXiv:2202.13415 Theorem 1), the coverage deficit is bounded as

    |Pr(s_test^{(1)} <= q_t^{(0)}) - (1 - alpha)| <= Lip(F_s^{(0)}) * W_1(s^{(0)}, s^{(1)})

with `W_1(s^{(0)}, s^{(1)}) <= (1/n_cal) sum_i |s_i^{(0)} - s_i^{(1)}|`. Both `Delta mu_i` and `Delta sigma_i` are linear-in-direction `A^{-1} phi(p*)^T` at leading order; their magnitudes are proportional to the cross-leverage between `p*` and `x_i`. Hence coverage is preserved iff refit; the deviation scales with the cross-leverage between picks and cal set. Empirical-results section 2 confirms this empirically.

## 6. Interface improvements (concrete)

1. **`epig_acquire_commissions`** as in section 1 above. New file `modules/surrogate/acquisition_commissions.py` (or rename of `acquisition.py`) once agent (b)'s `phi(z)` encoder lands. Backwards-compatible wrapper that calls into the existing closed-form machinery with `Phi_pool := model.feature_map(z_pool)` and `Phi_T := model.feature_map(z_target)`.

2. **`ConformalCalibrator` invariant enforcement.** Stamp a model fingerprint at `fit` time; raise on `coverage_sigma` if the model has changed. Patch:

   ```python
   class ConformalCalibrator:
       def fit(self, model, C, M, Y, ...):
           ...
           self._model_fingerprint = hashlib.sha256(
               np.ascontiguousarray(model.A_inv).tobytes()
               + np.ascontiguousarray(model.w).tobytes()
           ).hexdigest()
           # also stamp the probe-region cover sketch so the data-flow
           # invariant of section 4 can be enforced
           self._cal_region_sketch = _region_sketch(C, M)
           return self

       def coverage_sigma(self, model, C, M, coverage=0.683):
           fp = hashlib.sha256(...).hexdigest()
           if fp != self._model_fingerprint:
               raise OutdatedCalibratorError(
                   "Model has been updated since this calibrator was fit. "
                   "Call ConformalCalibrator.fit on the current model and "
                   "a cal set drawn from the current probe region."
               )
           ...
   ```

3. **`loop_monitor`** in `modules/surrogate/monitoring.py`:

   ```python
   def loop_monitor(
       model, calibrator,
       z_target: np.ndarray,
       z_probe:  np.ndarray, y_probe: np.ndarray,
       region_ids: np.ndarray,
       A_inv_fresh: np.ndarray | None = None,
       coverages: tuple[float, ...] = (0.683, 0.954),
   ) -> dict:
       """Return the four panels of agent (d) §6: H_T, per-region coverage,
       kappa(A), Sherman-Morrison residual."""
       Phi_T = model.feature_map(z_target)
       lev_T = np.einsum("id,de,ie->i", Phi_T, model.A_inv, Phi_T)
       H_T = 0.5 * np.sum(np.log(2 * np.pi * np.e * np.maximum(lev_T, 1e-15)))
       eigs = np.linalg.eigvalsh(np.linalg.inv(model.A_inv))
       kappa = float(eigs.max() / max(eigs.min(), 1e-30))
       cov_by_region = {}
       for r in np.unique(region_ids):
           m = region_ids == r
           mu_r = model.predict_z(z_probe[m], return_std=False)
           cov_by_region[int(r)] = {
               cov: float(np.mean(
                   np.abs(y_probe[m] - mu_r)
                   < calibrator.coverage_sigma_z(model, z_probe[m], cov)
               )) for cov in coverages
           }
       sm_residual = None
       if A_inv_fresh is not None:
           sm_residual = float(
               np.linalg.norm(model.A_inv - A_inv_fresh, ord="fro")
               / np.linalg.norm(A_inv_fresh, ord="fro")
           )
       return dict(H_T=float(H_T), target_var_sum=float(np.sum(lev_T)),
                   kappa=kappa, cov_by_region=cov_by_region,
                   sm_residual=sm_residual)
   ```

   Called once per loop cycle inside the `fm.update` span; the return dict is dumped into `aletheia.fm.*` span attributes (section 3).

## 7. Scaling — predictions and stall conditions

Predictions consistent with empirical work:

- **Target-set variance decay.** `sum_i sigma^2 lev_{T_i}^{(N)} ~ C/N` in the EPIG-effective regime. Empirically (section 4 of empirical-results), the log-log slope of `sum_i lev_{T_i}` vs `N` for `N >= 50` is `-1.66`, *faster* than the predicted `-1`. The super-1/N behaviour comes from two compounding effects: (i) the initial regime `N < D_JOINT = 75` is rank-limited and each new pick fills a rank deficiency; (ii) EPIG actively rejects redundancy.

- **Target-set entropy decay.** `H_T(N) = a + b/N + c/log N`. Empirical fit (section 4): `H_T = -121.26 + 11779.12/N + 149.73/log N`, SSE = 2139. Pure 1/N fit has SSE = 2174 (within 2%). Pure 1/log N has SSE = 13764 (6.3× worse). The 1/N component dominates; we are firmly in the EPIG-effective regime.

- **Coverage stabilisation.** O(1/sqrt(n_cal)) by Dvoretzky-Kiefer-Wolfowitz 1956. Empirically (section 2), after refit the 68% coverage is within 1.5pp of nominal at n_cal = 400, consistent with `1.96/(2*sqrt(400)) ~ 5pp` 95% CI; observed std across 20 seeds is 2.04pp.

**Stall conditions.** The loop stalls when:

1. **Target-set is rank-saturated.** `rank(Phi_T A^{-1} Phi_T^T) < |T|` relative to A's rank. Symptom: `H_T` plateaus while `kappa(A)` continues to drop. Remedy: enlarge T, or bump encoder dimension D_Z.
2. **Pool is exhausted.** Every candidate's `lev_p` is small. Symptom: EPIG scores converge to near-zero. Remedy: broader Wilson regions; promote tier.
3. **Heteroscedasticity dominates.** Realised noise scale much larger than assumed homoscedastic noise (e.g. MadGraph shot noise on T2+). Symptom: empirical `H_T` decay shallower than closed-form prediction. Remedy: cost-normalised EPIG.
4. **Calibrator staleness.** Variance falls while frequentist guarantees decay silently. The runtime check of §6 catches this.
5. **Cal-set support detaches from probe-region support.** Agent (c)'s drift detector catches via BH-corrected coverage; §4 of empirical-results quantifies the consequence (16pp deficit at |c| in [1.5, 2]).

In the 10-20 hour run, the most likely stall is (2) once `N >> D_JOINT * n_distinct_directions_in_pool`. With D_JOINT = 75 and the analytic oracle producing essentially arbitrary commissions, this is not a practical bound at `N = O(10^4)`.
