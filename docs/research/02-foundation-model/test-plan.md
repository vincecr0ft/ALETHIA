# Foundation Model — test plan (agent b)

Tests that demonstrate the redesign works.

The verification suite separates representation properties (what the FM is) from regression properties (what the probe does). Tests are organised into closed-form predictions (math should give an exact answer) and empirical small-scale checks (statistical, run-against-a-seed).

## A. Constraint-enforcement tests (closed-form / static)

A1. **No `c` in encoder signature.** Inspect the encoder class (call it `EventSetFM`). Assert: `inspect.signature(EventSetFM.forward)` accepts only an events tensor, no `c` argument; the class has no attribute named after a Wilson-coefficient operator. **Closed-form prediction**: the test must pass without exception. Requires the redesigned code; flag as `pytest.mark.xfail` against the current `IntentionFM`.

A2. **Encoder output is independent of `c` given fixed events.** Generate a fixed numpy event batch `events_fixed`. Compute `z_a = enc(events_fixed)` and `z_b = enc(events_fixed)`. Assert `np.allclose(z_a, z_b)` (deterministic forward). Then introduce *any* perturbation that would have entered the old `phi_joint`: not applicable to the new encoder, by signature. The test passes by construction; its job is to be loaded as a CI fixture so a future refactor that smuggles `c` back in will fail signature-check A1.

A3. **Pairing-only label flow.** Inspect the training loop. Assert that the only tensor on the encoder's forward path is `events`. `c` may appear only in the loss assembly (positives/negatives) and in the data sampler. This is a static-analysis check — a small AST walker against the training-step function. **Closed-form prediction**: passes for the redesigned code; fails for the current `IntentionFM.fit`.

## B. Self-supervised pre-training tests (empirical, small-scale)

B1. **InfoNCE positives align under finite-`N` noise.** Sample `c`. Generate two batches of `N = 1024` events each from the analytic oracle at this `c`. Compute `z1, z2`. Compare with `z_neg` from a different `c'`. **Expected**: cosine-sim `(z1, z2) > (z1, z_neg)` with overwhelming statistical significance after pre-training (and *not* before). Empirical threshold: at convergence on stage-0 data, mean cosine `(z1, z2) > 0.7` while mean cosine over negative pairs `< 0.3`. This is the basic positive-vs-negative check.

B2. **Representation does not collapse.** Compute the standard deviation of `z` coordinates across a held-out set of measurements. Assert `min_dim_std > ε * mean_dim_std` for `ε = 0.1`. (VICReg-style covariance check.) Failure mode: BYOL-without-EMA-decay tends to collapse; this catches it.

B3. **Invariance to event ordering.** For a fixed `c`, draw events; compute `z`. Permute the event order; compute `z'`. Assert `np.allclose(z, z')` up to numerical tolerance (`1e-4`). The set/DeepSets architecture should be exactly permutation-invariant by construction; this test catches a bug where a stray ordering-dependent layer (e.g. positional encoding) leaks in.

B4. **`c` is not a needed input at inference.** Sample two `c` values that differ only in a coordinate the FM has not been pre-trained to resolve (e.g. an operator zeroed out at stage 0). The two embeddings should be statistically indistinguishable across many seeds. The test exercises the lack-of-information story — what the FM *can't* see, it shouldn't pretend to see.

## C. Probe tests

C1. **Linear-probe `R²` on stage-0 operators, held-out `|c| > 0.7`.** Protocol from § 6 of `summary.md`. **Headline threshold**: per-operator `R² > 0.5` after pre-training on stage-0 data alone. Tighter version (stretch): `R² > 0.8`.

C2. **Probe-vs-direct-regression sanity.** Run an MLP regressor directly from events to `c` (allowed, since this is what an *oracle inverter* would do — it is not the FM, it is a baseline). Compare its `R²` to that of [encoder + linear probe]. **Expected**: linear probe is within `0.1` `R²` of the direct regressor on operators it has been pre-trained to resolve; gap larger on held-out regions of `c`-space (the probe wins on generalisation because it has been trained to represent the manifold, not to memorise `c`).

C3. **Probe is calibrated.** Wrap the probe in `ConformalCalibrator`. Verify per-stratum coverage hits nominal `0.683` and `0.954` within finite-sample slack `(±2/√n)`. The conformal layer is currently `modules/surrogate/calibration.py`; carries over with `phi(z)` substituted.

## D. Iterative-refinement tests

D1. **No catastrophic forgetting under stage 0 → stage 1.** After stage-0 pre-training, freeze the probe trained on stage-0 operators. Run stage-1 pre-training (continuation, more operators activated). Re-evaluate the *frozen* probe. **Expected (testable hypothesis)**: per-operator `R²` drops by no more than 0.1. Failure indicates representation drift / forgetting; mitigation is replay buffer or EMA-teacher distillation across stages.

D2. **New-operator resolvability rises with stage.** Train a fresh probe on `c_a` where `c_a` is *added* at stage 1. **Expected**: probe `R²` is near zero after stage 0 (the operator was zeroed out — there is no information about it in the data), nonzero after stage 1, rising further by stage 2. This is the empirical content of "the embedding manifold refines".

## E. Scaling tests

E1. **Learning curve.** Plot probe `R²` vs FM pre-training compute (in batches × parameters as proxy). Fit `a · N^(-b) + c`. **Expected** (for a properly behaving FM): `b > 0` and `c` is small. For the ridge probe alone, the equivalent fit on the probe-training-set size should give `b ≈ 0.5` (the `1/√n` law).

E2. **Cost extrapolation.** Profile oracle-call cost (analytic LO, per event batch) and encoder compute (per forward+backward). Extrapolate to the full hackathon-run budget (10–20 hours). Predict total batches reachable. Verify on one mid-scale run before the full run.

## F. Backwards-compatibility / EPIG tests

F1. **EPIG closed-form on the new feature map.** The existing EPIG closed form (`acquisition.py` lines 25–67) is feature-map-agnostic. Substitute `phi(z)` for `phi_joint(c, m)` and verify: information gain `≥ 0` everywhere; positive at points that maximally reduce predictive variance; satisfies Cauchy-Schwarz (`k_Tp^2 ≤ lev_T * lev_p`). These are closed-form algebraic checks that pass for any feature map by the derivation in the docstring.

F2. **Sherman-Morrison update agrees with refit.** Compare sequential greedy via Sherman-Morrison rank-1 updates to a full refit after each acquisition. Assert agreement to numerical tolerance. Existing test `tests/test_model.py::test_update_equals_refit` (lines 38–51) is exactly this pattern; port to `phi(z)` once the new probe is wired.

## G. End-to-end smoke test

G1. **One round: oracle → events → encoder → probe.** Sample one `c`, generate one event batch, embed it, predict a held-out `c` attribute, score. The pipeline should run end-to-end in under 5 s on CPU. This is the wire-up test before any of the above are meaningful.

## Changes to existing tests

The current `tests/test_features.py` and `tests/test_model.py` are downstream of the violation. Concretely:
- `tests/test_features.py:test_phi_c_quadratic_indices` (lines 22–30) expresses the morphing assumption directly. Replace with A1, A2 above.
- `tests/test_model.py:test_in_distribution_accuracy` (lines 54–63) tests the regressor. Replace with C1, C2 above.
- `tests/test_acquisition.py` and `tests/test_calibration.py` carry over with the `phi(z)` substitution; F1, F2, C3 above are the renamed versions.
