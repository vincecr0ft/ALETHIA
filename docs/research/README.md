# ALETHIA design research

This directory contains the design research underpinning the ALETHIA
rebuild for the Google Cloud Rapid Agent Hackathon (Arize track, due
2026-06-11). The work was produced by four parallel research passes and
one synthesis pass, each focused on one component of the joint system,
all bound by a single architectural constraint.

## The constraint (read once, twice)

The foundation model's forward pass must never see a Wilson coefficient.
Not as a raw value, not as a polynomial feature, not as a one-hot embedding,
not as a conditioning vector. Wilson coefficients live exclusively in two
places: the data-generation process inside the oracle, and the
construction of the loss. The FM consumes a measured object (events,
unbinned data, or histograms over an observable) and emits a learned
representation. The science claim is that probing the representation
recovers the underlying EFT structure when, and only when, that structure
has been disclosed by the data; not that the FM regresses a parametric
ansatz onto Wilson coefficients.

The full statement is in [BRIEF.md](BRIEF.md). Every document below
assumes it.

## Directory layout

```
docs/research/
├── README.md                   you are here
├── BRIEF.md                    shared brief; the hard constraint and code map
├── 01-oracle/                  the SMEFT oracle (analytic + MadGraph)
├── 02-foundation-model/        the representation-learning FM, post-constraint
├── 03-drift/                   physics-aware drift detection for Phoenix
├── 04-epig-conformal/          EPIG acquisition and conformal calibration
└── synthesis/three-test-cases.md   the operational test plan (Tests 1, 2, 3)
```

Each of the four area subdirectories contains the same five files:

| File | Content |
| --- | --- |
| `summary.md` | First-pass research summary. The literature, the derivation, and what the area should do. |
| `test-plan.md` | First-pass test plan. Closed-form predictions where the math gives one; small-scale empirical otherwise. |
| `notable.md` | Five numbered empirical investigations with one-sentence hypothesis and expected outcome. |
| `eval.md` | Second-pass evaluation: how this area's choices interact with the other three; load-bearing contracts. |
| `empirical-results.md` | Second-pass empirical work: small-scale validation runs, debugging, scaling extrapolation to the 10–20 h budget. |

## Headline finding (added 2026-05-27)

After the first two research passes were complete, a focused empirical
comparison evaluated the **Intention closed-form-attention** architecture
(Garnelo & Czarnecki [arXiv:2305.10203](https://arxiv.org/abs/2305.10203);
the basis of fundamental.tech's foundation-model bet) against the
DeepSets-FM that the FM area had recommended on first pass. Result: a
learned-basis Intention head with 5,328 parameters reaches **held-out
median R² = 0.9997** on the polynomial-toy SMEFT oracle, vs DeepSets-FM at
**0.617** at matched parameter count — and beats even a regressor that
gets `c` directly on its forward pass (median 0.938, p5 0.36).
Sample-efficient too: Intention reaches R² > 0.96 after a single Adam
step, while DeepSets needs ~600 steps to cross zero.

The full comparison, plots, and revisions to the rebuild plan are at
[02-foundation-model/intention-vs-deepsets.md](02-foundation-model/intention-vs-deepsets.md).
The takeaway feeds back into synthesis: the recommended FM encoder
becomes a hybrid (DeepSets event encoder feeding the Intention head's
`ψ(M_ctx)` rows), with end-to-end MSE meta-learning through
`torch.linalg.solve`. The synthesis test cases in
[synthesis/three-test-cases.md](synthesis/three-test-cases.md) should be
read with this revision in mind; the Intention head is the head of choice
for Test 2.

## Reading order

If you have ten minutes, read in this order:

1. [BRIEF.md](BRIEF.md) — the constraint.
2. [02-foundation-model/intention-vs-deepsets.md](02-foundation-model/intention-vs-deepsets.md) — the headline architecture finding.
3. [synthesis/full-chain-run.md](synthesis/full-chain-run.md) — the
   end-to-end Phoenix-instrumented loop, executed: 110× before/after
   RMSE recovery on the engineered drift, 400 cycles, 500 oracle calls,
   ~17 min CPU.
4. [synthesis/three-test-cases.md](synthesis/three-test-cases.md) — the
   operational plan, with the joint architecture summary in the prologue.

If you have an hour, add the four `summary.md` files (one per area) and
each area's `eval.md`. Skip `notable.md` and the first-pass `test-plan.md`
files; they were superseded by the joint test plan in synthesis.

If you have an afternoon, read everything in numerical order.

## Load-bearing contracts (synthesised across the four areas)

Six contracts are referenced throughout the research and pinned in the
synthesis. They are the interfaces that hold the four pieces together.

1. **`Commission`** — the type of one oracle job. Carries
   `(wilson_region, kinematic_window, fidelity_tier, observable, order)`.
   See [04-epig-conformal/eval.md](04-epig-conformal/eval.md) §1.
2. **`OracleResult`** — the structured return of one commission. Identical
   top-level shape across fidelity tiers. See
   [01-oracle/summary.md](01-oracle/summary.md) §3.2.
3. **`FMHandle` protocol** — the minimal interface the drift evaluators
   and EPIG need from the FM: `predict(z) → (mu, sigma)`,
   `embed(events) → z`, `A_inv`, `v_min`, `update`. See
   [03-drift/eval.md](03-drift/eval.md) §1.
4. **`DriftedRegion`** — the payload the drift aggregator emits to EPIG
   when a `local_retrain` or `global_retrain` action fires. Embeddings,
   not Wilson coefficients. See [03-drift/eval.md](03-drift/eval.md) §2.
5. **Refit-after-update invariant** — `ConformalCalibrator.fit` must be
   called after every `model.update`. Runtime check via a model
   fingerprint that raises `OutdatedCalibratorError` on stale use. See
   [04-epig-conformal/eval.md](04-epig-conformal/eval.md) §2.
6. **Broader-probe-region calibration invariant** — the calibration set is
   drawn from the same distribution the model will be queried on, not
   from the training distribution. See
   [04-epig-conformal/summary.md](04-epig-conformal/summary.md) §4.3 and
   the empirical demonstration in
   [04-epig-conformal/empirical-results.md](04-epig-conformal/empirical-results.md) §3.

`H_T = ½ Σᵢ log(2πe σ² lev_{T_i})` is the headline span attribute
emitted on every `fm.update` span; it is the monitor for whether the
loop is making progress. See
[04-epig-conformal/eval.md](04-epig-conformal/eval.md) §3.

## Relationship to the existing repo

The research describes a rebuild that retains most of the current code
in revised roles, rather than replacing it.

**Kept verbatim.** `modules/analytic_smeft/` — the closed-form LO
Drell-Yan calculator over the 14 Warsaw-basis dim-6 operators is correct
physics and stays as the oracle's analytic backend.

**Kept with a new interpretation.** The closed-form ridge / Sherman-Morrison
/ leverage / EPIG machinery in `modules/surrogate/{model,acquisition,calibration}.py`
survives as the *probe* on the FM's embedding `z`, not as the model itself.
Its mathematics is feature-map-agnostic and stays intact.

**Rewritten.** `modules/surrogate/features.py` (`phi_c`, `phi_joint` put
`c` on the forward pass and must go).
`modules/surrogate/model.py` (`IntentionFM.fit(C, M, Y)` regresses on
`c` and must become a `LinearProbe.fit(Z, Y)`). New
`modules/surrogate/encoder/` directory for the FM proper (DeepSets first,
Particle Transformer when budget allows). See
[02-foundation-model/eval.md](02-foundation-model/eval.md) §E for the
full layout.

**New.** `modules/drift/` for the three drift detectors and the
aggregator; `modules/oracle/cache.py` for the parquet+sqlite cache;
`modules/surrogate/monitoring.py` for the `loop_monitor` function.

## Relationship to the older documents

Three older design documents pre-date this research and describe the
unconstrained design. They are kept in place with superseded markers
pointing here:

- [/aletheia_verification_plan.md](../../aletheia_verification_plan.md) —
  the previous master plan. Its §0 algebraic framing, Component 1 (Phoenix),
  and Component 2 (analytic SMEFT) survive; Component 3 (the FM) and
  Components 4–6 (drift, EPIG, end-to-end) all assume `c` on the forward
  pass and are superseded by this directory.
- [/docs/foundation-model-survey.md](../foundation-model-survey.md) — an
  earlier survey that explicitly recommends the regressor design. Its TL;DR
  ("the closed-form ridge IS the right model class") is exactly the
  pattern the constraint now forbids.
- [/docs/surrogate/](../surrogate/) — the README and architecture docs
  describe the regressor as it currently exists in `modules/surrogate/`.
  They remain accurate descriptions of running code until the rebuild PR
  lands.

## Provenance

The research was produced in three passes by independent agents:

- **First pass (four agents, parallel).** Each agent owned one area
  (oracle, FM, drift, EPIG) and produced `summary.md`, `test-plan.md`,
  `notable.md`.
- **Second pass (four agents, parallel).** Each agent read all four
  first-pass outputs, ran small empirical validations against the
  existing code, and produced `eval.md` and `empirical-results.md` for
  its own area.
- **Synthesis (one agent).** Read everything and produced
  [synthesis/three-test-cases.md](synthesis/three-test-cases.md).

Empirical work in the second pass found two non-obvious facts worth
flagging here:

- The 30× slowdown between toy and CT18NNLO PDFs in the analytic oracle
  is `mkPDF` re-instantiation per call
  ([01-oracle/empirical-results.md](01-oracle/empirical-results.md) §1).
  Cache the resolved `PDFSet` at oracle `__init__` time; per-call cost
  drops from 19 ms to 3.2 ms. One-line fix in
  `modules/surrogate/oracle_smeft.py:108`.
- The analytic oracle is not safely PDF-independent for `c_lq^(1)`:
  toy-vs-CT18 disagreement reaches 13.5 % at `m_ll = 0.5 TeV`
  ([01-oracle/empirical-results.md](01-oracle/empirical-results.md) §2).
  The cache key already discriminates on PDF; the practical consequence
  is that toy-PDF rows are not interchangeable with CT18NNLO rows for
  `clq1`-active commissions.
