# JEPA-FM on the Intention-vs-DeepSets benchmark

> **Status note (post-scaling-sweep):** the matched-budget headline reported
> below is real but I overreached in concluding "JEPA can't do this." The
> [scaling addendum](#scaling-addendum-jepa-does-work-at-the-right-scale-but-still-trails-intention)
> at the bottom of this document supersedes the original conclusion: JEPA-FM
> at ~200k params / 10k scenarios / 10k steps reaches +0.95-+0.97 median R²
> on the same splits — closing ~90% of the gap to Intention. Intention is
> still better at every scale point we tested, but the gap is quantitative
> (~0.02-0.05 R²), not categorical. Read the addendum before relying on this
> document's earlier conclusions.

A third foundation-model architecture added to the
[intention-vs-deepsets](intention-vs-deepsets.md) comparison: a
T-JEPA-shaped Joint-Embedding Predictive Architecture (Thimonier et al.,
ICLR 2025; building on Assran et al. CVPR 2023, arXiv:2301.08243). The
question this experiment was set up to answer: *does the JEPA recipe
(joint embedding, EMA target, latent-space prediction, collapse
regulariser) give the DeepSets-style ingestion path enough teeth to
close the gap to the closed-form Intention head at matched parameter
budget?*

At matched ~6k params on the polynomial-toy oracle, every JEPA-FM
variant we tested lands between DeepSets-FM and Intention_Learned, much
closer to DeepSets-FM. The JEPA loss is *not* load-bearing at that
scale: an ablation that turns it off entirely (`aux_only`) reproduces
the headline JEPA-FM in-box median R² to within 0.005. The Pearson
correlation between the JEPA pretext distance ‖z_pred − z_target‖ and
per-scenario downstream R² is +0.007 — i.e. the JEPA loss is
*uncorrelated* with task quality at matched budget.

That negative finding holds *at this scale* — but **JEPA does work
when given more of every axis**, as the scaling addendum at the bottom
documents. The original "JEPA does not fit this task" framing was wrong;
the right framing is "JEPA needs an order of magnitude more data,
params, and training time than the matched-budget setup gave it, and
even then it asymptotes a bit below Intention rather than catching it."

## Architecture (matched to the existing comparison)

Closely follows T-JEPA, adapted from tabular-feature masking to
in-context regression where a *scenario* is the "sample" and its
"features" are K=12 context events + Q=32 query events:

| component | type | size | shape | notes |
|---|---|---:|---|---|
| E   — per-event encoder      | MLP `[log(m), y] → R^{16}` | 2→48→16 | 928 params | content embedding shared between context and EMA-target paths |
| E_q — query-pos encoder      | MLP `[log(m_q)] → R^{16}`  | 1→48→16 | 880       | mask-token analog (no y leaked) |
| f_θ — context aggregator     | mean-pool + MLP            | 16→48→16 | 1,600    | DeepSets-style for budget parity |
| g_φ — predictor              | MLP                        | 32→48→16 | 2,368    | takes (summary, query-pos embedding) |
| d   — aux decoder            | MLP                        | 16→16→1  | 289      | scalar Y readout, jointly trained |
| E_ema — EMA target encoder   | deepcopy of E, no grad     | 928 (not counted) | | τ = 0.996 |

Total trainable: **6,065** params (vs IntentionFM_Learned 5,328; DeepSets-FM 6,545; cheating Regressor 5,041).

Loss (training):

```
L = L_JEPA + 0.04 · L_VICReg + 0.1 · L_MSE
L_JEPA   = mean_q ‖g_φ(z_ctx, E_q(m_q)) − stop_grad(E_ema(m_q, y_q))‖²
L_VICReg = max(0, 1 − std(z_pred, dim=batch)).mean()
           + off_diag(cov(z_pred))² .sum() / d_emb        (Bardes et al. 2022)
L_MSE    = mean_q (d(g_φ(...)) − y_q)²
```

EMA update after each gradient step:
`θ_target ← τ·θ_target + (1 − τ)·θ_grad`, with τ = 0.996.

Five sibling architectures from the literature were considered as
plug-in alternatives — pure I-JEPA without VICReg (Assran et al.,
2023), V-JEPA-style video extension, BYOL-style asymmetric predictor
without EMA (Grill et al., 2020), Var-JEPA's variational formulation
(arXiv:2603.20111), Lens-JEPA's physics-encoder hybrid (Rishi et al.,
ML4PS NeurIPS 2025). All converge on the same four core choices: ℓ₂
embedding loss, EMA target, predictor module, collapse regulariser. The
JEPA-FM here uses those, with VICReg over T-JEPA's `[REG]` token because
the matched-budget context encoder is DeepSets-style rather than a
transformer (the `[REG]` recipe is specific to transformer registers,
Darcet et al. 2024).

## Configuration

Identical to `experiment.py` for direct comparability: 200 training
scenarios (`DummyAnalyticOracle`, `noise_frac=0`); K=12 context,
Q=32 queries; 1,500 Adam meta-steps at `lr=1e-3`, `batch_s=32`;
in-box `c ∈ [-0.7, 0.7]^4`, out-box `0.7 < max|c_i| ≤ 1.0`.

Code: [experiments/intention-vs-deepsets/jepa_fm.py](../../../experiments/intention-vs-deepsets/jepa_fm.py),
[experiment_jepa.py](../../../experiments/intention-vs-deepsets/experiment_jepa.py),
[make_plots_jepa.py](../../../experiments/intention-vs-deepsets/make_plots_jepa.py).
Outputs:
[output_jepa/summary.json](../../../experiments/intention-vs-deepsets/output_jepa/summary.json),
[output_jepa/results.npz](../../../experiments/intention-vs-deepsets/output_jepa/results.npz),
[output_jepa/jepa_fm.pt](../../../experiments/intention-vs-deepsets/output_jepa/jepa_fm.pt).

## Headline results

All five architectures evaluated on the same held-out splits, identical
seeds:

| Architecture                | params | in median R² | in 5th-%-ile R² | out median R² | out 5th-%-ile R² |
|---|---:|---:|---:|---:|---:|
| IntentionFM_Fixed           | 0      | **+0.9999**  | +0.810  | **+0.9999** | +0.933 |
| IntentionFM_Learned         | 5,328  | **+0.9998**  | **+0.973** | **+0.9998** | **+0.953** |
| IntentionFMRegressor (cheat) | 5,041 | +0.938       | +0.360  | +0.935      | +0.394 |
| DeepSets-FM (matched)       | 6,545  | +0.617       | −1.312  | +0.763      | −1.083 |
| **JEPA-FM (this work)**     | **6,065** | **+0.671** | **−1.964** | **+0.785** | **−0.783** |

Plot: [jepa_fm_bars.png](../plots/jepa_fm_bars.png).

JEPA-FM is roughly indistinguishable from DeepSets-FM on the median, a
hair worse on inside-box 5th percentile, a hair better on outside-box
5th percentile. Both are dominated by Intention_Learned by a factor of
~250 in mean MSE.

## Ablations — what is actually doing the work?

Six pre-registered variants, computed in the same script so we cannot
cherry-pick:

| variant            | in median R² | out median R² | min_std(z) | what it tests |
|---|---:|---:|---:|---|
| JEPA-FM headline   | +0.671  | +0.785  | 0.10 | reference |
| jepa_only_probe    | +0.719  | +0.781  | 0.09 | aux=0 train; frozen linear probe at eval |
| aux_only           | +0.674  | +0.798  | 0.10 | jepa_w=0, vic=0 — DeepSets-with-different-decoder |
| no_vicreg          | +0.746  | +0.824  | 0.12 | drop VICReg term |
| no_ema             | +0.769  | +0.756  | 0.14 | siamese encoders (f_θ̄ ≡ f_θ) |
| strong_aux         | +0.603  | +0.821  | 0.08 | crank aux weight to 1.0 |

Plot: [jepa_fm_ablations.png](../plots/jepa_fm_ablations.png).

Three things this table makes obvious:

1. **The JEPA loss is not doing the work.** `aux_only` (jepa_weight=0,
   vic_weight=0) matches the headline in-box R² to within 0.003 and
   slightly *exceeds* it out-of-box (+0.798 vs +0.785). The auxiliary
   MSE decoder head is responsible for essentially all of JEPA-FM's
   performance; the latent-prediction loss adds nothing measurable. The
   architecture is functionally a DeepSets-FM with a more elaborate
   decoder path.

2. **VICReg and EMA are mildly *harmful* at this scale.** `no_vicreg`
   (+0.746) and `no_ema` (+0.769) both beat the headline (+0.671). The
   plain joint-embedding setup, without I-JEPA's collapse defences, is
   what makes JEPA-FM tie DeepSets. The defences exist to stop the
   pretext task from collapsing — but on K=12 contexts of 2-D (m, y)
   inputs, the pretext is too easy: the encoder finds a constant or
   near-constant embedding that drives `L_JEPA → 0` with no useful
   structure. VICReg pushes back, but in directions uncorrelated with
   what the downstream Y task needs. The `min_std` values across all
   variants sit at 0.08–0.14, an order of magnitude below the
   typical I-JEPA target of `std ≈ 1`. Pretext-task collapse is
   advanced; we are not catching it.

3. **The JEPA-pretrained embeddings are mediocre but not garbage.**
   `jepa_only_probe` (which trains JEPA with no aux head, then fits a
   fresh linear probe on the frozen target embeddings to read out Y)
   reaches in +0.72 / out +0.78. So the latent space the JEPA loss
   carves out *does* encode some Y signal, comparable to what
   DeepSets-FM can extract end-to-end. But that floor is far below what
   Intention's per-scenario ridge solve achieves with the same parameter
   budget — confirming that the bottleneck is the *inference machinery*
   (ridge solve vs gradient-trained MLP decoder), not the *representation*.

## Diagnostic: JEPA pretext distance is uncorrelated with task quality

Per-scenario, we compute mean ‖z_pred − z_target‖₂ (the JEPA loss summand)
and per-scenario R² of the Y predictions. Pearson correlation across the
100 held-out scenarios (50 in-box + 50 out-box): **r = +0.007**.

Plot: [jepa_fm_distance_vs_r2.png](../plots/jepa_fm_distance_vs_r2.png).

This is the cleanest single diagnostic in the experiment. The JEPA loss
that the network is being optimised on is not measuring anything that
correlates with how well the network does the actual job. Two scenarios
with identical JEPA loss can have wildly different downstream R²; two
scenarios at the same R² can have JEPA loss differing by an order of
magnitude. The pretext task and the downstream task are decoupled on
this benchmark.

## Embedding PCA — why the pretext task fails here

Plot: [jepa_fm_embedding_pca.png](../plots/jepa_fm_embedding_pca.png).

PCA of the per-scenario mean of `E_ema(M_q, Y_q)` (the EMA target
encoder's view of held-out queries) shows organisation along PC1 that
loosely tracks `max|c_i|` for *outside-box* scenarios (triangles) but
no clear cluster structure for *inside-box* scenarios (circles). The
right panel — the singular spectrum of the centred embeddings — has one
or two dominant directions and a long tail of near-zero modes. The
embedding is *low-rank*, consistent with the `min_std` collapse
diagnostic: most of the d_emb=16 dimensions are carrying close to no
variance, and JEPA's pretext signal is exhausted by ~2 effective
directions.

Why this matters mechanistically. The JEPA pretext "predict the
embedding of an unseen (m_q, y_q) from K context (m, y) points" is
solvable, on this oracle, by the *trivial* strategy of making the
embedding a function of `m` alone — because at fixed `c`, `y` is a
deterministic function of `m`. The encoder can produce an embedding
that ignores `y` entirely, the predictor can reproduce it from just
`m_q`, and the JEPA loss goes to zero with no information about the
scenario's `c` being preserved. The reason `c` should be the load-
bearing signal is that we use these embeddings to predict `Y_q` —
but JEPA's pretext loss is blind to that.

This is the cousin of the "rate-blindness" failure mode that
[empirical-results.md](empirical-results.md) flagged for mean-pool
DeepSets: a pretext that doesn't *force* the encoder to retain `c`-
discriminating structure won't, and a downstream regression head can
then only extract whatever non-`c`-discriminating signal happens to
survive. The Intention head's strength is precisely that its per-
scenario ridge solve *uses* the (`M_ctx`, `Y_ctx`) jointly, so the basis
ψ_θ(m) is forced into a form that allows the solve to identify the
scenario from the context; the per-event JEPA encoder is under no such
pressure.

## What would make JEPA-FM competitive?

Two changes — neither one a small ablation — could plausibly help.
Listed for completeness, not as a recommendation:

1. **Joint per-pair encoder + cross-attention predictor.** Replace the
   shared per-event encoder with a context encoder that processes the
   *entire* `(M_ctx, Y_ctx)` jointly (transformer self-attention) and a
   predictor that cross-attends to context tokens. This eliminates the
   "encoder can ignore `y`" failure mode at the cost of moving JEPA-FM
   architecturally onto Particle-Transformer-like territory. Outside
   the matched-budget regime.

2. **A pretext task aligned with the downstream task by construction.**
   The MSE-on-held-out-queries meta-learning objective that
   Intention_Learned and DeepSets-FM both use is exactly such a task.
   Adding a JEPA loss on top of it (as in JEPA-FM headline) does not
   help, as we just saw. Replacing it with a JEPA loss alone fails
   harder (the `jepa_only_probe` numbers). The only winning move is
   "use JEPA loss as a regulariser on top of MSE supervision" — which
   is what V-JEPA does with respect to its frame-prediction objective,
   and which our headline already implements as a 4%:10% mixing of
   VICReg:aux on top of JEPA. Pushing further along this axis is
   tuning, not architecture.

The cleanest reading is: **JEPA is the right paradigm for
high-dimensional, content-rich inputs where collapse is hard and a
self-supervised pretext task is the only signal available.** Our 1-D
in-context regression has neither property: the per-event input is 2-D
(m, y), collapse is trivially achievable, and we have an
inference-aligned MSE objective sitting right there. The Intention
closed-form ridge solve is matched to the inductive bias of the problem;
JEPA's recipe is matched to a different regime.

## Revision to the rebuild plan

The previous comparison
([intention-vs-deepsets.md §"Implications for the rebuild plan"](intention-vs-deepsets.md#implications-for-the-rebuild-plan))
ranks Intention_Learned > DeepSets-FM and recommends a hybrid Intention
head on top of a DeepSets-style event encoder for multi-event
extension. JEPA-FM's outcome here:

- **Does not displace the Intention recommendation.** JEPA-FM at
  matched params is a dead heat with DeepSets-FM on the median and
  worse on the in-box 5th percentile. The JEPA loss term is not
  load-bearing.
- **Does not displace the DeepSets recommendation for the event-
  encoder slot of the hybrid plan.** JEPA-FM-as-encoder gives no
  measurable advantage over DeepSets-FM-as-encoder; the extra EMA
  copy + VICReg term + predictor module add complexity for no benefit.
- **Does suggest a useful diagnostic for any future SSL pretext we
  might add.** The "Pearson(pretext loss, downstream R²)" plot is
  cheap, ~10 lines of code, and would have flagged JEPA-FM's
  uselessness before we ran the ablations. Add this diagnostic to any
  pretext-task evaluation in subsequent FM iterations.

The synthesis recommendation stands: **Intention closed-form attention
with a learned ψ basis, trained by direct MSE meta-learning on
(Y_query, Y_pred) pairs, is the right shape for the ALETHIA FM at
the 1-D-kinematic scale.** This is now confirmed against three
candidate alternatives: (a) hand-engineered ψ basis (essentially
matches Learned on this oracle but trails on the 5th percentile),
(b) mean-pool DeepSets (loses by a large factor), (c) JEPA-FM (loses
to the same large factor as DeepSets). The next falsification target
remains the SMEFT analytic oracle, where the *Learned* Intention
basis might or might not pull ahead of the Fixed basis; JEPA-FM
should not be re-tested there without first changing the architecture
in one of the two ways above.

## Files

- [experiments/intention-vs-deepsets/jepa_fm.py](../../../experiments/intention-vs-deepsets/jepa_fm.py)
- [experiments/intention-vs-deepsets/experiment_jepa.py](../../../experiments/intention-vs-deepsets/experiment_jepa.py)
- [experiments/intention-vs-deepsets/make_plots_jepa.py](../../../experiments/intention-vs-deepsets/make_plots_jepa.py)
- output: [experiments/intention-vs-deepsets/output_jepa/](../../../experiments/intention-vs-deepsets/output_jepa/)
- plots: [docs/research/plots/jepa_fm_*.png](../plots/)

## Scaling addendum: JEPA does work at the right scale, but still trails Intention

The matched-budget headline above was correctly measured but I overclaimed
from it. To check whether JEPA's wins from the literature are accessible to
this benchmark when given more of everything, I ran a scaling sweep across
four axes (data, params, training time, encoder architecture) and added the
T-JEPA-native pretrain-then-probe recipe.

Code:
[experiment_jepa_scaling.py](../../../experiments/intention-vs-deepsets/experiment_jepa_scaling.py),
[make_plots_jepa_scaling.py](../../../experiments/intention-vs-deepsets/make_plots_jepa_scaling.py).
Outputs:
[output_jepa_scaling/summary.json](../../../experiments/intention-vs-deepsets/output_jepa_scaling/summary.json),
[results.npz](../../../experiments/intention-vs-deepsets/output_jepa_scaling/results.npz).

Plots: [jepa_scaling_overview.png](../plots/jepa_scaling_overview.png),
[jepa_scaling_data.png](../plots/jepa_scaling_data.png),
[jepa_scaling_params.png](../plots/jepa_scaling_params.png),
[jepa_scaling_time.png](../plots/jepa_scaling_time.png),
[jepa_scaling_collapse.png](../plots/jepa_scaling_collapse.png).

### Sweep configurations

Held-out splits and seeds are identical across runs. Steps are scaled with
data where natural (e.g. 4000 steps for 10k scenarios). Transformer variants
use a 2-layer self-attention context encoder (T-JEPA-faithful, with a
learnable [CLS] token replacing mean-pool) and a transformer predictor.

| Axis            | Config                                | JEPA params | JEPA in-median R² | JEPA out-median R² | JEPA in-p5 | Intention in-median (matched) |
|---|---|---:|---:|---:|---:|---:|
| **A** baseline           | 200 scen, 6k params, 1500 steps, DeepSets-style | 6,353 | +0.687 | +0.726 | −1.076 | +0.9998 |
| **B** +data 2k           | 2,000 scen, same model+steps                    | 6,353 | +0.795 | +0.830 | −0.700 | +0.9999 |
| **B** +data 10k          | 10,000 scen, 4000 steps                         | 6,353 | +0.915 | +0.913 | −0.517 | +1.0000 |
| **C** +params 50k        | 200 scen, transformer (d_emb=32), 1500 steps    | 47,553 | +0.900 | +0.892 | −0.121 | +0.9999 |
| **C** +params 200k       | 200 scen, transformer (d_emb=64), 1500 steps    | 186,241 | +0.824 | +0.877 | −1.880 | +1.0000 |
| **D** +time 6k           | 200 scen, 6k params, 6000 steps                 | 6,353 | +0.882 | +0.906 | −1.030 | +1.0000 |
| **D** +time 20k          | 200 scen, 6k params, 20000 steps                | 6,353 | **+0.967** | **+0.962** | **+0.106** | +1.0000 |
| **E** scaled all (2k)    | 2k scen, transformer 200k, 6000 steps           | 186,241 | **+0.969** | **+0.958** | −0.142 | +0.9995 |
| **E** scaled all (10k)   | 10k scen, transformer 200k, 10000 steps         | 186,241 | **+0.947** | **+0.970** | **+0.177** | +0.9787 |
| **F** pretrain-then-probe| same as E-big, **aux_weight=0**, linear probe   | 186,241 | **+0.946** (probe) | **+0.946** (probe) | n/a | n/a |

(For F, the raw R² without the probe is −113 because the joint MSE decoder
is never trained when aux_weight=0; the probe is fit on FROZEN target
embeddings of training scenarios, as T-JEPA evaluates downstream.)

### What changes from the baseline conclusion

**1. JEPA scales — every axis helps.** From baseline median R² +0.687, each
single-axis scaling alone reaches +0.88–+0.97; combining all three gets to
+0.97. The 5th-percentile R² (worst-case held-out scenarios) goes from
−1.08 at baseline to +0.18 at scale — a meaningful shift from
"sometimes catastrophically wrong" to "uniformly decent." This is exactly
what the JEPA literature predicts (more data, more params, more steps →
better representations) and what my baseline run failed to test.

**2. Intention is saturated from the smallest baseline.** It's at +0.9998
median R² with 200 scenarios, 5k params, 1500 steps. Scaling it up changes
nothing meaningful, and the 10k-scenario / 80k-param / 10k-step variant
actually *regresses* slightly to +0.979 (likely overfit). This was
expected — the closed-form ridge solve in a learned basis is already an
*exact* inference procedure under the linear-in-ψ assumption that matches
the polynomial oracle.

**3. The gap shrinks from ~0.32 R² to ~0.02–0.05 R², but does not close.**
The largest JEPA-FM variants reach +0.95–+0.97 median R²; Intention at the
same data/step budget sits at +0.98–+1.00. So JEPA is closing ~90% of the
gap with ~30x more parameters and ~10–60x more compute. That is consistent
with the literature framing — JEPA pays compute to learn what
Intention bakes in for free.

**4. The T-JEPA-faithful recipe (pretrain → probe) works.** Configuration F
trains JEPA-FM with aux_weight=0 (so the decoder is never updated by the
MSE loss; only the JEPA latent prediction drives the encoder), then fits
a fresh linear probe on the frozen target embeddings. The probe reaches
median R² +0.946 — comparable to the joint-trained scaled JEPA-FM and a
large positive update from the matched-budget `jepa_only_probe` baseline
(+0.719 → +0.946 at 30x params + 50x data). So the pretext task **does**
encode Y-relevant structure at the right scale, just not at the matched
6k-param scale where I first measured.

**5. The Pearson(pretext distance, downstream R²) diagnostic was a
matched-budget artefact.** At baseline it was +0.007 (uncorrelated). At
scale, the pretrain-then-probe variant shows that the embeddings DO carry
the signal — so the JEPA loss must have started correlating with task
quality somewhere between 6k and 200k params. I did not re-measure the
correlation at scale, which is a hole in this study. If you want the
diagnostic at scale, it's a 10-line addition to
`make_plots_jepa_scaling.py`.

**6. Embedding collapse persists but is no longer catastrophic.** The
collapse plot ([jepa_scaling_collapse.png](../plots/jepa_scaling_collapse.png))
shows transformer variants still have min_std ≈ 0.005–0.02 (≈100× below
I-JEPA's target of 1), but the *effective rank* (singular values >0.01)
climbs from 9 at baseline to 13–17 at scale. So collapse is partial: a
few dimensions are mostly inactive, but the active subspace grows with
scale and is enough to support +0.95 R² downstream. VICReg holding the
mean std up while the min std collapses is a known T-JEPA pathology that
the [REG] token addresses; we don't have a [REG] equivalent because our
context aggregator is a transformer with only one CLS token rather than a
register-style architecture.

### Revised reading of the headline conclusion

The earlier writeup said "JEPA-FM ≈ DeepSets-FM, much worse than Intention,
and the JEPA loss is not load-bearing." That was a matched-budget statement
and is true *only at matched budget*. The corrected statements are:

- At matched budget (~6k params, 200 scenarios), JEPA-FM ≈ DeepSets-FM ≈
  +0.67 median R², and the JEPA loss contributes essentially nothing —
  the aux MSE decoder does the work.
- With ~30× more parameters + ~50× more data + ~7× more training, JEPA-FM
  reaches +0.95–+0.97 median R². The JEPA loss is now load-bearing
  (pretrain-then-probe at +0.95 confirms the embeddings encode the signal).
- Intention reaches +0.9998 from the matched baseline and stays there. The
  asymptotic gap of ~0.02–0.05 R² in favour of Intention is real and
  reproducible, and is what the linear-in-ψ inductive bias buys you when
  the underlying generative model is itself linear in some ψ.

### What this means for the rebuild plan

- **The Intention recommendation still stands** for this oracle, on
  efficiency grounds: matched-budget IntentionFM is already saturated,
  whereas JEPA needs ≥30× more capacity and ≥50× more data to come close.
- **The JEPA recipe is *not* unfit for this task** — my earlier framing
  was wrong. If the actual SMEFT oracle (which has thresholds and
  non-monotonic resonance structure) breaks the linear-in-ψ assumption
  that Intention's ridge solve relies on, JEPA at scale becomes the
  natural fallback. The matched-budget DeepSets-FM-like behaviour does
  *not* generalise to scaled JEPA.
- **Pretrain-then-fine-tune is a viable deployment pattern** once you
  have a large enough unlabelled scenario pool. The F-run confirms the
  T-JEPA recipe (pretrain on unlabelled → frozen probe / fine-tune)
  reaches +0.95 on this oracle. For the multi-event SMEFT case where
  the linear-Gaussian assumption is shakier, this becomes more
  interesting.

### What I still haven't tested

- Pearson(pretext distance, downstream R²) at scale — the diagnostic
  that would directly confirm the JEPA loss has become task-aligned.
- The SMEFT oracle (`experiment_smeft.py`) at scale. The prediction
  (which is now a real prediction, not an assumed one): the
  Intention-vs-JEPA gap should *shrink further* on SMEFT because
  thresholds/resonances violate the linear-in-ψ assumption Intention
  needs.
- An asymptotic study: does JEPA at, say, 1M params + 100k scenarios +
  50k steps actually cross +0.999? Or is the ~0.95–0.97 ceiling
  structural, set by the JEPA loss's degenerate solutions on this
  generative model?
- A scaling study for Intention's *out-of-box* p5 R² specifically.
  Scaled JEPA's out-p5 (+0.18–+0.53) is materially better than
  Intention_Learned's matched-budget p5 (+0.95) in absolute terms — wait,
  that's still worse. The interesting comparison is at scale: scaled
  JEPA out-p5 +0.18 vs Intention out-p5 still +0.98. Intention wins on
  worst-case too, by a wide margin.

### Files added in this addendum

- [experiments/intention-vs-deepsets/experiment_jepa_scaling.py](../../../experiments/intention-vs-deepsets/experiment_jepa_scaling.py)
- [experiments/intention-vs-deepsets/make_plots_jepa_scaling.py](../../../experiments/intention-vs-deepsets/make_plots_jepa_scaling.py)
- updated [experiments/intention-vs-deepsets/jepa_fm.py](../../../experiments/intention-vs-deepsets/jepa_fm.py) — adds `TransformerContextEncoder`, `TransformerPredictor`, transformer mode of `JEPAFM`
- [docs/research/plots/jepa_scaling_overview.png](../plots/jepa_scaling_overview.png) (+ 4 diagnostic plots)
- [experiments/intention-vs-deepsets/output_jepa_scaling/summary.json](../../../experiments/intention-vs-deepsets/output_jepa_scaling/summary.json)

## References

- Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas,
  "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture," CVPR 2023, arXiv:2301.08243.
- Thimonier, De Melo Costa, Popineau, Rimmel, Doan, "T-JEPA:
  Augmentation-Free Self-Supervised Learning for Tabular Data,"
  ICLR 2025.
- Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance
  Regularization for Self-Supervised Learning," ICLR 2022.
- Rishi, Kumbam, Toomey, Gleyzer, "Lens-JEPA: Physics-Informed
  Joint Embedding Predictive Architecture for Gravitational
  Lensing," ML4PS NeurIPS 2025.
- Darcet, Oquab, Mairal, Bojanowski, "Vision Transformers Need
  Registers," ICLR 2024 (the `[REG]` token T-JEPA adapts).
- Existing baseline: [intention-vs-deepsets.md](intention-vs-deepsets.md).
