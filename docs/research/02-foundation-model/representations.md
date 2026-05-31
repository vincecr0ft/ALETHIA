# What each FM representation encodes about the SMEFT structure

A representation-level comparison of the three foundation-model architectures
in the [intention-vs-deepsets](intention-vs-deepsets.md) and
[jepa-fm](jepa-fm.md) studies. The point of this study is not to rank
architectures on R² — that work is in those documents. The point is to
characterise *what each model's latent space encodes about the
generative process* `c → (M, Y)`, so we can say something physical about
the bias each architecture imposes on the learned representation.

Code: [experiments/intention-vs-deepsets/representation_analysis.py](../../../experiments/intention-vs-deepsets/representation_analysis.py),
[make_plots_representations.py](../../../experiments/intention-vs-deepsets/make_plots_representations.py).
Output: [output_representation_analysis/{summary.json,embeddings.npz}](../../../experiments/intention-vs-deepsets/output_representation_analysis/).
Plots: `docs/research/plots/repr_*.png`.

## The four representations probed

| Name              | What it is                                    | Dim  | Trained from              |
|---|---|---:|---|
| `Intention_w`     | Per-scenario ridge solve `w = (ΨᵀΨ+αI)⁻¹ ΨᵀY` | 16 | MSE meta-learning, 3000 steps, 2000 scenarios |
| `DeepSets_z`      | Mean-pooled per-event embedding `(1/K)Σ φ(m,y)` | 16 | MSE meta-learning, same |
| `JEPA_summary_m`  | Context-encoder output (matched ~6k params, DeepSets-style aggregator) | 16 | JEPA loss + 0.04·VICReg + 0.1·MSE, EMA target, same |
| `JEPA_summary_s`  | Context-encoder output (scaled ~186k params, transformer CLS) | 64 | same JEPA recipe, 6000 steps |

All four extracted on identical 200 held-out scenarios per split, identical seed.

## Finding 1: rate-blindness is the cleanest separator

Plot: [repr_y_invariance.png](../plots/repr_y_invariance.png).

For each held-out scenario, permute the y values across its K=12 context
events while keeping m fixed. Re-extract the representation; measure
cosine similarity with the unpermuted representation.

| Representation     | cos sim under y-shuffle | reading |
|---|---:|---|
| Intention_w        | **+0.30** | strongly uses the m-y alignment |
| DeepSets_z         | **+0.97** | essentially rate-blind |
| JEPA_summary_m     | **+0.92** | essentially rate-blind |
| JEPA_summary_s     | **+0.78** | partially uses alignment |

**What this says about the architecture priors.** The Intention ridge
inverts ΨᵀΨ to recover `w = ΨᵀY`, which is by construction a function
of the *joint* assignment of which `y_i` is paired with which `m_i`.
Permuting Y inside Ψᵀ produces a different w by definition — the ridge
solve is the most-aligned aggregator we have.

DeepSets and matched-budget JEPA-FM, despite both ingesting `(m_i, y_i)`
*pairs*, produce per-event embeddings `φ(m, y)` that are nearly y-shuffle
invariant after pooling. This is not a deficit of mean-pool per se; it
is a deficit of the per-event encoder under matched budget, which does
not have the capacity (or the gradient signal) to spread y into a
distinct subspace. The pretext loss in JEPA does not penalise y-blindness
because, on this generative model, a y-blind embedding still satisfies
"event embeddings within a scenario are mutually predictable" — y is a
deterministic function of (m, c), so an embedding that captures *some*
function of m alone is mutually-predictable across events.

Scaling JEPA's transformer encoder partially fixes this (+0.78 vs +0.92).
The transformer's self-attention can detect the (m, y) joint pattern
across the K context tokens, but only partially — a substantial fraction
of the representation is still y-shuffle invariant.

**Physics reading.** On the polynomial-toy oracle, the m-y per-event
correlation IS the signal: at a fixed `c`, `y(m)` is a specific function,
and the per-event ordering encodes which function. Architectures that
discard this ordering have to recover the scenario identity from
y-marginal statistics (mean, variance, kurtosis), which is the
rate-blindness failure mode flagged in
[empirical-results.md](empirical-results.md) for mean-pool DeepSets.
Intention's closed-form ridge is the only architecture that
*architecturally* prevents this discard.

## Finding 2: c-recoverability matches the R² ranking and resolves it per-coefficient

Plot: [repr_c_recoverability.png](../plots/repr_c_recoverability.png).

Fit a linear (or MLP-64) probe `repr → c ∈ ℝ⁴` on training-set
representations, evaluate R² per Wilson coefficient on held-out scenarios.

| Representation     | linear in-R²ⱼₒᵢₙₜ | linear out-R²ⱼₒᵢₙₜ | MLP in-R²ⱼₒᵢₙₜ | MLP out-R²ⱼₒᵢₙₜ |
|---|---:|---:|---:|---:|
| Intention_w        | **+0.75** | **+0.65** | +0.86 | **+0.70** |
| DeepSets_z         | +0.32 | +0.29 | +0.44 | +0.39 |
| JEPA_summary_m     | +0.33 | +0.31 | +0.37 | +0.32 |
| JEPA_summary_s     | +0.62 | +0.51 | +0.69 | +0.15 |

Per-coefficient breakdown reveals a strong hierarchy that is the same
across all four representations:

```
recoverability:   clq1  >  clq3  ≈  cHq1  >>  cHq3
                  (~0.90)  (~0.85)  (~0.85)    (~0.33 in Intention, ~0.07 in DeepSets)
```

**Physics reading.** clq1 is the dominant interference channel in
`DummyAnalyticOracle` — its effect on `y(m)` is largest in absolute
magnitude, so every representation has the easiest time isolating it.
cHq3 is the weakest channel; only Intention's ridge solve gets a
meaningful linear handle on it (+0.33 R²), DeepSets and matched JEPA
recover essentially nothing for cHq3 (+0.07). Scaled JEPA partially
recovers it (+0.25), consistent with "more capacity buys more
signal channels."

**The MLP probe out-box R² for JEPA-scaled (+0.15) is striking.** It is
*lower* than the linear probe (+0.51) on the same representation. This
is a clean signature of overfitting: JEPA-scaled's 64-d representation
has enough room for an MLP to memorise in-box scenario patterns that
don't generalise out-of-box. Intention's 16-d representation has no such
room — its linear and MLP probes give similar out-box R². The high
dimensionality of JEPA-scaled is a double-edged sword.

## Finding 3: intrinsic dimension is ~2-3 for all, but the tail differs

Plot: [repr_intrinsic_dim.png](../plots/repr_intrinsic_dim.png).

| Representation     | eff. rank (SV > 1%) | participation ratio | d (90% var) | d (99% var) |
|---|---:|---:|---:|---:|
| Intention_w        | 11 | 1.82 | 2 | 4 |
| DeepSets_z         | 6  | 1.45 | 2 | 3 |
| JEPA_summary_m     | 6  | 1.52 | 2 | 2 |
| JEPA_summary_s     | 21 | 1.86 | 3 | 6 |

The dominant 2-3 directions in each representation carry 90%+ of the
variance across scenarios. This matches expectations: although the
generative model has 4 Wilson coefficients, the polynomial oracle's
SMEFT structure has *strong* clq1/clq3 dominance, so the per-scenario
representation lives near a 2-D manifold parameterised by (clq1, clq3)
with weaker excursions along (cHq1, cHq3).

The interesting difference is in the tail. Intention uses 11 of its 16
basis directions above the 1%-of-max singular-value cutoff — it spreads
the secondary structure across many directions. JEPA-matched and
DeepSets only use 6 directions; the rest are inactive. JEPA-scaled
opens up 21 of its 64 directions but most of those are at low variance
(only 6 directions cover 99% of variance, vs 4 for Intention).

**Physics reading.** The Wilson-coefficient space is 4-D, but the
information-carrying directions on this oracle's m-distribution are
∼2-3. Intention's 11 effective directions encode physics-redundant
information: multiple basis-function projections that all carry the
same c-discriminating signal. JEPA-scaled's 21 effective directions
include physics-irrelevant variation (the cross-decoding result below
confirms 40% of JEPA-scaled's structure is not in Intention's
c-relevant subspace).

## Finding 4: representations are partially mutually contained

Plot: [repr_cross_decoding.png](../plots/repr_cross_decoding.png).

R² of fitting target representation from source representation linearly:

| source ↓ / target → | Intention_w | DeepSets_z | JEPA_m | JEPA_s |
|---|---:|---:|---:|---:|
| **Intention_w**     | —    | +0.77 | +0.75 | **+0.87** |
| **DeepSets_z**      | +0.70 | — | n/a | n/a |
| **JEPA_summary_m**  | +0.74 | n/a | — | +0.73 |
| **JEPA_summary_s**  | **+0.60** | n/a | +0.86 | — |

The two asymmetries that matter:

- **Intention → JEPA_s: +0.87, JEPA_s → Intention: +0.60.** Intention's
  16-d representation can be linearly read off from JEPA_s's 64-d
  representation with 87% accuracy. The reverse map fits only 60%.
  Reading: JEPA_s's representation *contains* most of Intention's
  information plus extra dimensions that don't map back to Intention's
  c-relevant ridge weights — i.e. the extra structure JEPA_s carries
  beyond Intention is not what Intention finds c-relevant.
- **JEPA_m → JEPA_s: +0.73, JEPA_s → JEPA_m: +0.86.** Scaling expands
  the representation but in a c-irrelevant direction (cf c-recoverability
  +0.33 → +0.62) and adds noise that's not in the matched version.

**Physics reading.** Intention's ridge weights live in a subspace that
is c-aligned by construction (they ARE the inferred function
coefficients). JEPA-scaled's representation contains a c-aligned
subspace (the part that recovers Intention's w at +0.87) plus a larger
c-orthogonal subspace (the part that doesn't reverse-fit Intention at
only +0.60). The c-orthogonal directions are what makes JEPA_s's MLP
c-probe overfit out-box: the MLP latches onto in-box-specific structure
in that subspace.

## Finding 5: the PCA scatter shows soft clustering by max|c_i|

Plot: [repr_pca_by_c.png](../plots/repr_pca_by_c.png).

PC1-PC2 of each representation, coloured by max|c_i|. Across all four
panels, brighter-yellow points (large |c|) populate the outer ring;
darker-purple points (small |c|, SM-like) cluster centrally. This
confirms the intrinsic-dim finding: 2 dimensions are enough to
qualitatively separate scenarios by Wilson-coefficient scale. The
quantitative separation is sharper in Intention than in JEPA_s than in
DeepSets, matching the c-recoverability hierarchy.

The out-box scenarios (triangles, large |c_i|) sit outside the in-box
ring in every representation — i.e. extrapolation is visible as
geometric distance in the latent space, not just as degraded R².

## What this means for the foundation-model question

The R²-on-Y comparison in [intention-vs-deepsets.md](intention-vs-deepsets.md)
and the scaling addendum in [jepa-fm.md](jepa-fm.md) settle which
architecture predicts Y best at each parameter budget. This study
gives the *why*:

1. **Intention's win is structural.** Its ridge solve is the only
   aggregator we tested that is m-y-alignment-aware by construction
   (cos sim 0.30 under y-shuffle). For a generative model where the
   m-y joint is the signal, that's a decisive architectural prior.
2. **DeepSets-matched and JEPA-matched are functionally equivalent
   at the representation level.** Both are y-shuffle invariant (0.97
   vs 0.92), both have effective rank 6, both recover ~30% of c
   linearly. The JEPA loss adds nothing at matched budget. This is
   the same conclusion as the [jepa-fm.md ablations](jepa-fm.md#ablations--what-is-actually-doing-the-work)
   but read from the embedding side instead of the R² side.
3. **JEPA-scaled discovers some m-y alignment** (0.78 cos sim) and
   roughly *doubles* the linear c-recoverability (0.62 vs 0.32) over
   DeepSets, but at 30× the parameter cost. The transformer's
   self-attention is what buys this — it can detect joint (m, y)
   patterns across the K context tokens that the per-event MLP +
   mean-pool cannot.
4. **JEPA-scaled's extra capacity is partially wasted on
   c-irrelevant directions.** 40% of its representational structure
   (the part that doesn't linearly map to Intention's w) is not
   reflected in c-recoverability, and is responsible for the MLP
   probe overfitting out-box. A 64-d transformer JEPA on K=12,
   d_input=2 is over-parametrised.

For the ALETHIA rebuild plan, the actionable findings are:

- **The "right" inductive bias for this kind of in-context posterior
  predictive on SMEFT data is m-y joint awareness in the aggregator.**
  Intention provides this exactly via the ridge solve; JEPA provides
  it partially via cross-attention; mean-pool DeepSets does not provide
  it at all.
- **If you want a *learned* aggregator (because the SMEFT oracle
  will have non-linear structure Intention's ridge can't capture),
  it has to be cross-attention based, not pooled.** The matched-budget
  JEPA result with mean-pool aggregator is direct evidence that pooling
  forfeits the m-y alignment regardless of how the per-event encoder is
  trained.
- **Probe-based representation evaluation should accompany every
  future FM iteration.** The five diagnostics in this study (c-probe,
  intrinsic dim, cross-decoding, y-shuffle, PCA-by-c) are ~50 lines of
  code each and tell you what the architecture *learned* in a way that
  pure-R² benchmarks do not. For drift detection / EPIG / oracle-call
  scheduling, what matters downstream is the representation's
  c-discriminating geometry, not just its predictive R².

## What this does NOT tell us

- The probes are trained at fixed scale (2000 train scenarios, 3000
  steps). c-recoverability and rate-blindness might both shift at a
  different scale; this study fixes scale to isolate
  representation-quality differences architecturally.
- We tested only the dummy polynomial oracle. The SMEFT analytic oracle
  with thresholds and PDF convolution will break the linear-in-ψ
  assumption Intention's ridge relies on, so the rate-blindness and
  c-recoverability rankings might re-order on that oracle. This is
  worth running as a follow-up — the same scripts work unchanged with
  `experiment_smeft.py`'s data builder.
- The y-permutation diagnostic is a one-trick pony for rate-blindness.
  Other physically-motivated perturbations (m-perturbation, additive
  noise injection, individual-event ablation) would test other
  properties of the representation.

## References

- Related representation-probing methodology: Alain & Bengio,
  "Understanding intermediate layers using linear classifier probes,"
  ICLR Workshop 2017.
- Cross-decoding as a representation-equivalence measure: Hewitt &
  Manning, "A Structural Probe for Finding Syntax in Word
  Representations," NAACL 2019.
- VICReg participation-ratio / collapse diagnostics: Bardes et al.,
  ICLR 2022.
- Underlying experiments:
  [intention-vs-deepsets.md](intention-vs-deepsets.md),
  [jepa-fm.md](jepa-fm.md).
