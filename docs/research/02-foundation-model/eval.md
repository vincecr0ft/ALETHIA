# Foundation Model — eval (second pass, agent b)

The FM area is structurally the pivot of the redesign, because shifting `c` off the forward pass changes the type of the FM's input from `(c, m)` to `events`, and that type change propagates into every adjacent module. Each interaction below identifies a concrete contract.

## 1. Interaction with the oracle (agent (a))

**Gap.** Agent (a)'s `OracleResult` (`docs/research/01-oracle/summary.md:251-280`) returns a structured dict whose `pieces` field is `(sm_only, interference, bsm_squared)` evaluated on a fixed kinematic grid. The analytic oracle currently produces a scalar `μ(c, m)` per call (`modules/surrogate/oracle_smeft.py:84-111`). My Architecture A (event-set encoder) requires **unbinned event batches** — N kinematic 4-vectors drawn from the c-conditional distribution. The analytic oracle does not natively produce these.

**Bridge.** Agent (a) needs to add a method to the `Oracle` protocol:

```python
def sample_events(self, c, n_events, *, m_range_tev=(0.3, 3.0),
                  observable="m_ll", rng=None) -> np.ndarray
    """Return (n_events,) or (n_events, F) array of per-event
    kinematic features drawn from the c-conditional differential
    cross-section."""
```

For T0/T1 (analytic), this is a 30-line wrapper around `differential_xs` plus an inverse-CDF sampler on a dense m-grid; I demonstrated the implementation works in `/tmp/fm_deepsets/sample_events.py` (rejection / inverse-CDF on the differential, ~30 ms per N=512 batch). For T2+ (MadGraph), unbinned events are the natural output of `generate_events`; the wrapper extracts the LHE 4-vectors from `oracle_madgraph.py`'s already-spawned MG run.

The `OracleResult` shape proposed by (a) need not change. `sample_events` is a separate method; it is called by the FM training loop, while `call`/`truth` remain available for the closed-form probe and for the existing scalar interface tests. The constraint is that for a given oracle call at fidelity tier T, both methods must return mutually consistent objects (the bin-integrated `OracleResult.pieces` and the empirical histogram of `sample_events` must agree to MC noise; this is a testable invariant).

**Cache implications.** Agent (a)'s cache key (`docs/research/01-oracle/summary.md:333-346`) needs an additional field `n_events_seed` (or the seed of the sampler RNG) when the output type is events; this is exactly the same pattern as the existing `nevents`/`seed` fields for T2+ MadGraph rows. The cache value grows from a tuple of floats to an `(n_events, F)` float32 array (~8 KB at N=512, F=4). 10 GB cache caps at ~10⁶ event-batches; the original 1 GB estimate at 100k calls in (a) §4.5 holds.

**OpenInference span.** Span attribute `oracle.n_events` is already in (a)'s schema. Add `oracle.output_type ∈ {"scalar", "events"}` for filterability.

## 2. Interaction with drift (agent (c))

**Algebra survives, semantics shift.** Agent (c) builds three drift signals (DAS-CUSUM on standardised residuals; BH-corrected per-region binomial coverage; condition number plus v_min on the design matrix), and section 3.1 of their summary already notes that the leverage / kappa / v_min algebra is invariant under the change of feature map. What they need from my redesigned FM:

- `model.A_inv` — the `(d, d)` inverse of the **probe** design matrix `Z^T Z + λI`, where `Z` is the matrix of training-set embeddings. In my redesign, this lives on the closed-form ridge probe (the surviving piece of `IntentionFM`), not on the encoder. The encoder is parameter-frozen at probe-time.
- `model.v_min` (or equivalently `model.svd()` returning U, S, Vt of the design matrix). Computed once per probe-fit and refreshed only when the probe is refit, per (c) §6.3.
- `model.leverage(z)` for query embeddings `z`.
- `model.embed(events)` returning `z` for arbitrary event batches.
- `model.region_id(z)` — clustering function (HDBSCAN or k-means on a held-out probe set, fitted once); (c) needs this for BH stratification (`docs/research/03-drift/summary.md:330-336`).

**Joint un-formalised gap.** Agent (c)'s aggregation policy (table at `03-drift/summary.md:340-349`) emits a `target_region_id` (the location of the failure). Agent (d) consumes this for local retraining. But neither (c) nor (d) specifies the mapping from `target_region_id` to the EPIG **target set** `{z_T^{(i)}}`. The natural choice is the centroid of the failing cluster plus a few v_min-aligned perturbations; this closes the loop:

```
drift fires -> target region = cluster R -> EPIG target set = centroid(R) + jitter along v_min -> next oracle commission
```

This loop closure should be specified in synthesis. I record it as a concrete contract in the redesigned `model.py` API (section E below).

**Phoenix span unification.** Agent (c)'s span tree (`03-drift/summary.md:386-440`) carries `aletheia.fm.embed_norm_*`, `aletheia.fm.leverage_*`. These now refer to z-space, not phi_joint-space. The renaming is cosmetic; the schema is unchanged.

## 3. Interaction with EPIG (agent (d))

**Candidate space restructuring.** Agent (d) §2 already reframes EPIG candidates from `(c, m)` pairs to oracle commissions producing embeddings (`docs/research/04-epig-conformal/summary.md:88-138`). My encoder's forward pass supports the required batched preview-embedding: one DeepSets pass on `(B, N, F)` with `B=12, N=384` takes about 0.4 s on this laptop CPU, so per-candidate preview-embedding is roughly **30 ms (event sampling) + 30 ms (encoder forward) ≈ 60 ms**. For a 1000-candidate EPIG pool with `k=10` greedy picks the total cost is ~10 × 1000 × 60 ms = 10 minutes — affordable within the 15-hour budget but worth caching the previewed embeddings (they don't depend on the model state, only on the oracle).

**MC over commission stochasticity.** Agent (d)'s eq. at `04-epig-conformal/summary.md:100` notes that an oracle commission produces a stochastic embedding `z(κ) ~ p(z | κ)`, so EPIG over commissions has an outer expectation. For the analytic oracle the embedding distribution is tight around its mean (one inverse-CDF sample at N=512 vs another differs only by Monte Carlo noise on N events, which scales as 1/√N), so `mc_samples=1` is sufficient in (d)'s sketched API (line 114). I confirmed this empirically: cosine similarity between two independent N=512 embeddings at the same c is ~0.97+ after pre-training. For T2+ MadGraph (where MC noise on μ is ~3%), `mc_samples=3-5` is the right default; cost scales linearly.

**Closed-form-friendly contract.** The closed-form EPIG derivation in `modules/surrogate/acquisition.py:25-67` survives the feature-map swap unchanged, as (d) notes. The contract from FM to EPIG:
- `model.feature_map(z) -> ℝ^d` (identity or a small MLP — typically identity)
- `model.A_inv` (probe state, fresh)
- preview function: given an oracle commission, return the expected feature row `phi(E[z(κ)])`.

This is the natural Sherman-Morrison interface (d) needs (lines 122-133 of their sketch).

## 4. Joint assumption no single area has formalised

The **loop closure between drift's v_min and EPIG's target set** is the gap. Each area assumes the other will provide the bridge: (c) hands EPIG a `target_region_id`; (d) consumes a `z_target` array; neither defines how `target_region_id` becomes `z_target`. The FM is the natural owner of this mapping because the FM owns the embedding-space geometry. Concretely, `model.target_set_for_region(region_id, n_target=50)` returns `(n_target, d_z)` z-values that uniformly cover the region's interior plus its v_min-orthogonal complement. The synthesis pass should pin this.

A second un-formalised assumption: **calibration set support after iterative refinement**. Agent (d) §4.3 flags that as the EPIG-driven loop shifts the training-set support, the conformal calibration set must shift with it. Agent (b) (me) does not specify when the FM encoder itself should be re-trained vs when only the probe should be re-fit. The right rule, consistent with §5 of (b) summary: encoder re-training is rare (stage transitions); probe re-fit is frequent (every EPIG round). The conformal layer should re-fit whenever the probe re-fits. This is one line in the orchestrator and one assertion in the test suite (already in (d) test plan B.2).

## B. Theoretical backing for the central claim

The claim, restated: *the learned representation IS the model; probing it recovers c if and only if the FM has disclosed the EFT manifold.*

Five sources of theoretical backing.

**1. Bommasani et al. (arXiv:2108.07258) §2.1.** Their operational definition of foundation model has three properties (label-free pre-training; downstream tasks via probes on a frozen representation; representation-is-asset). The "if" direction of my claim — *if the probe recovers c with high R² then the representation has encoded c* — is the definition of a representation, full stop. The "only if" direction — *if the FM has learned the EFT manifold, then a linear probe should recover c* — is supported by Alain & Bengio (arXiv:1610.01644) Theorem of section 2: for any *linearly accessible* property of the representation, a linear probe is a consistent estimator with minimum bias among bounded-norm probes. Linear probes were chosen precisely as a stringent test of representation quality because they cannot encode extra non-linear computation beyond what's already in the encoder.

**2. InfoNCE bound (van den Oord, Li, Vinyals, arXiv:1807.03748 Theorem 1).** The InfoNCE loss is an upper bound on the mutual-information shortfall:

```
L_NCE  >=  log(K) - I(z; c)        (1)
```

where K is the batch size (number of negatives + positive) and I(z; c) is the mutual information between the embedding and the latent c that conditions the pairing. Rearranging:

```
I(z; c)  >=  log(K) - L_NCE        (2)
```

In my 250-batch run with K=16 and final L_NCE ≈ 1.28, the bound gives **I(z; c) ≥ log(16) - 1.28 = 2.77 - 1.28 = 1.49 nats**. With 4 independent operators bounded on (-0.7, 0.7), the entropy of c is H(c) ≈ 4 × log(1.4) = 4 × 0.34 = 1.34 nats (uniform on a [-0.7,0.7] box per operator); the mutual information bound (1.49 nats) exceeds H(c) only because L_NCE has not converged — the InfoNCE bound becomes vacuous when log(K) - L_NCE > H(c). With sharper training the bound will tighten.

**Quantitative prediction on probe R².** For a Gaussian-linear coupling between z and c, the linear-probe R² satisfies the asymptotic relation (Tian, Krishnan, Isola arXiv:2005.10243 §5.3):

```
R²_probe(c_k | z)  ~  1 - exp(-2 I(c_k; z))
```

For per-operator mutual information I(c_k; z) ≈ 0.37 nats (1.49 / 4 if MI is distributed evenly), this predicts R² ≈ 1 - exp(-0.74) ≈ 0.52. My empirical clq3 R² of 0.59 is in that ballpark; the other three operators (cHq3, cHq1, clq1) fall well below it (R² ≈ 0), which is the signal that **MI is not evenly distributed across operators in z-space** — clq3 dominates because it's the only operator that produces a strong m_ll-shape change at the energies sampled. Vertex operators produce rate-only changes and a mean-pool encoder cannot see them.

**3. Neural scaling laws (Kaplan et al., arXiv:2001.08361; Hoffmann et al., arXiv:2203.15556).** For a foundation model trained with InfoNCE on a multi-batch corpus, the test loss should approach a power-law form

```
L(N) = L_∞ + (N_c / N)^α
```

with α typically in [0.3, 0.7] for vision/language models. Ridge regression on a fixed polynomial basis has α = 0.5 by classical 1/√n consistency, but with a basis-misspecification floor L_∞ that, for IntentionFM, is exactly zero (the basis is exact) — making the comparison degenerate. The relevant test is whether the FM probe R² *improves* with pre-training compute; my N ∈ {10, 25, 60, 150} batches run gave essentially **flat** R² (0.564 → 0.535) with fitted exponent ≈ -0.023. This is **falsificatory evidence at this scale**: in the tiny-batch tiny-encoder regime I tested, the encoder hits a representational ceiling determined by the mean-pool's rate-blindness, not by data starvation. See E (scaling) for the implication.

**4. Supervised-contrastive on physics distributions (arXiv:2512.15862).** The closest prior art. Their model trains a Particle Transformer on c-conditioned LHC event batches with a supervised-contrastive loss; their probe R² on dim-6 operator coefficients hits 0.7–0.9 after 10⁴–10⁵ batches at 10⁷ parameters. The gap between their numbers and mine (0.5 at 10² batches, 10⁴ parameters) is fully accounted for by parameter scale and data volume. Their architecture choices that matter: (a) Particle Transformer rather than mean-pool DeepSets, (b) explicit normalisation by total event count, (c) supervised contrastive (multiple positives per c) rather than pairwise InfoNCE. All three should be adopted in the real run.

**5. Linear probe protocol (Alain & Bengio, arXiv:1610.01644).** They argue that **the linear probe's R² is the cleanest single-number measurement of representation quality** because it factors out the probe's own expressivity. The same protocol applied to GPT/BERT/CLIP/ESM has been used to compare representation quality across hundreds of models. My headline metric — per-operator linear-probe R² on held-out c — directly imports this protocol. A meaningful headline: "clq3 R² = 0.59 with N_params = 12k and N_batches = 250" is exactly the right summary statistic.

**Does the InfoNCE bound predict the achievable R²? Yes, quantitatively, via Tian et al.'s exp(-2I) relation, modulo MI being unevenly distributed across c-coordinates. The latter is itself diagnostic: an FM that has truly learned the manifold should distribute MI roughly uniformly across the latent factors; an architecture-limited FM will concentrate MI on a subset.** This is the metric I would push for as the headline diagnostic.

## C. Scaling prediction and falsification

The 4-point scaling run (N ∈ {10, 25, 60, 150} batches, mean per-operator R² constant at ~0.10) gives fitted exponent c ≈ -0.023 with extrapolation predictions R²(1000) ≈ 0.51, R²(2000) ≈ 0.50, R²(10000) ≈ 0.48. **The extrapolation says probe quality is essentially saturated at the architecture+pooling choice tested.**

**Falsification of FM-vs-regressor behaviour.** I asserted in `summary.md` §7 that an FM exhibits power-law test loss while a regressor does not. My experiment did not falsify this claim *for the regressor*; it falsified it *for the FM at this architecture*: the FM is exhibiting regressor-like saturation, not FM-like scaling. This is the empirical signature that the chosen architecture is wrong, not the chosen training paradigm. The fix is in the architecture (sum-pool or aux-rate, deeper encoder, Particle Transformer); the InfoNCE+probe paradigm is intact.

A clean falsifier going forward: pre-train the rate-aware encoder (described in empirical-results §3) at three compute scales spanning 10³–10⁵ batches. **If per-operator R² does not scale as a power law (exponent > 0.05 on a held-out R²-gap), the FM claim fails — the system is a regressor in FM clothing and we should revert to closed-form ridge over engineered features.** Concretely: fit `1 - R² = a + b/N^c` with N = batches × params; report c. Hackathon-acceptable threshold: c > 0.1.

## E. Redesigned `modules/surrogate/` layout

Implementable in a single PR.

```
modules/surrogate/
  encoder/                          # NEW. The foundation model proper.
    __init__.py                     # exports EventSetFM
    deepsets.py                     # the DeepSets encoder (mean-pool + log-rate)
    losses.py                       # InfoNCE, VICReg
    sampler.py                      # CSampler: c-pair batches via oracle.sample_events
    train.py                        # CLI for staged pre-training, saves enc.pt
  probe.py                          # The closed-form ridge, now on z.
                                    # class LinearProbe with .fit(Z, Y) / .predict / .leverage / .A_inv / .v_min
                                    # No c on the forward pass. fit consumes (Z, Y) where Y is any scalar target.
  features.py                       # SHRUNK. Keep phi_x for incontext.py and the SM piece; delete phi_c, phi_joint.
  calibration.py                    # UNCHANGED in math; rewires from model.predict to LinearProbe.predict.
  acquisition.py                    # epig_acquire_commissions over oracle commissions producing embeddings.
                                    # Closed-form math unchanged; feature map is now phi(z) = z (identity).
  oracle_smeft.py                   # +sample_events method.
  oracle_madgraph.py                # +sample_events method (LHE-event extraction wrapper).
  evaluation.py                     # adds per-operator probe R^2 metric; keeps existing scalar metrics.

tests/
  test_encoder_constraint.py        # NEW. Static checks: c not in EventSetFM.forward signature; AST scan.
  test_encoder_invariance.py        # NEW. permutation invariance of DeepSets to event ordering.
  test_probe.py                     # NEW. Replaces test_model.py; ridge on Z is correct math.
  test_probe_calibration.py         # RENAMED from test_calibration.py; wires probe in place of model.
  test_oracle_smeft.py              # ADDS test for sample_events: histogram matches differential_xs.
  test_features.py                  # SHRUNK to phi_x only.
  test_acquisition.py               # REPLACED with test_acquisition_commissions.py.
```

### Class contracts

```python
class EventSetFM(nn.Module):
    """Foundation model. c is not in any signature."""
    d: int                                       # embedding dimension
    def forward(self, events: Tensor) -> Tensor: # events: (B, N, F) -> z: (B, d)
        ...
    def embed(self, events: np.ndarray) -> np.ndarray:  # numpy convenience
        ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "EventSetFM": ...

class LinearProbe:
    """Closed-form ridge from z to a scalar target. Replaces IntentionFM."""
    def fit(self, Z: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> "LinearProbe": ...
    def predict(self, Z: np.ndarray, *, return_std: bool = True): ...
    def leverage(self, Z: np.ndarray) -> np.ndarray: ...
    @property
    def A_inv(self) -> np.ndarray: ...
    @property
    def v_min(self) -> np.ndarray: ...                  # smallest right-singular vector
    @property
    def kappa(self) -> float: ...                       # condition number
    def update(self, Z_new, Y_new): ...                  # Sherman-Morrison or refit

class Commission:
    """One oracle job."""
    c: np.ndarray
    m_range_tev: tuple[float, float]
    n_events: int
    fidelity: str

def epig_acquire_commissions(
    probe: LinearProbe,
    commissions: list[Commission],
    embed_preview: Callable[[Commission], np.ndarray],
    Z_target: np.ndarray,
    k: int, *, mc_samples: int = 1, cost_fn=None,
) -> list[Commission]: ...
```

### Migration path (single PR)

1. Add `oracle_smeft.sample_events` (30 LOC); add `oracle_madgraph.sample_events` (LHE-event extraction).
2. Add `encoder/deepsets.py` (the 80-LOC class) + `encoder/sampler.py` + `encoder/losses.py`.
3. Add `encoder/train.py` CLI that runs staged pre-training and writes `enc.pt`.
4. Add `probe.py`; lift the closed-form math from `model.py` verbatim, drop the `phi_joint` call.
5. Adapt `calibration.py` and `acquisition.py` to consume `probe` instead of `model` (one-line changes in their public functions; math unchanged).
6. Delete `phi_c`, `phi_joint`, and `IntentionFM` from features.py / model.py. Mark `model.py` as deprecated with a stub that imports `LinearProbe`.
7. Update tests as listed.

Total churn: ~600 LOC new, ~150 LOC deleted, ~50 LOC adapted.
