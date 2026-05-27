# Intention vs DeepSets foundation-model comparison

Side-by-side empirical evaluation of two FM architectures for SMEFT regression
under the ALETHIA constraint that *the Wilson coefficient `c` never enters
the foundation model's forward pass*:

1. **IntentionFM_Fixed** — closed-form linear attention (Garnelo & Czarnecki,
   arXiv:2305.10203) with a hand-engineered polynomial-in-`log(m)` basis
   (no training). The structural template lives in
   `docs/historical/smeft_surrogate-2026-05-25.tar.gz` as `incontext.py`.
2. **IntentionFM_Learned** — same closed-form attention but the basis
   `psi_theta(m)` is a small MLP trained end-to-end through
   `torch.linalg.solve` on a meta-learning MSE loss.
3. **DeepSets-FM** — per-event MLP, mean-pool, decoder MLP. Parameter count
   matched to within ~1.2x of IntentionFM_Learned.
4. **IntentionFM_Regressor (ceiling, cheating)** — a direct `(c, m) -> y`
   MLP. Forbidden by the BRIEF, kept here only to mark "the architecture
   that consumes `c` on the forward pass" as a quantitative reference.

## Layout

```
data.py               scenario sampler (built on DummyAnalyticOracle)
intention_learned.py  PsiMLP, IntentionFMLearned, IntentionFMFixed
deepsets_matched.py   DeepSetsFM, IntentionFMRegressor
experiment.py         training loop + held-out evaluation
make_plots.py         the four PNG diagnostics
```

Results land under `docs/research/plots/` and the writeup is at
`docs/research/02-foundation-model/intention-vs-deepsets.md`.

## Configuration

- 200 training scenarios; 50 in-box + 50 outside-box test scenarios.
- Each scenario: K=12 context points, Q=32 query points,
  `m` sampled uniformly in [0.3, 2.3] TeV.
- Inside-box: `c ~ U([-0.7, 0.7]^4)`. Outside-box: rejection-sampled shell
  `0.7 < max|c_i| <= 1.0`.
- 1500 Adam meta-steps at `lr=1e-3`, `batch_s=32` scenarios per step.
- `DummyAnalyticOracle` from
  [modules/surrogate/ground_truth.py](../../modules/surrogate/ground_truth.py),
  `noise_frac=0`, so the comparison is architectural rather than
  denoising-driven.

## Reproducing

This experiment requires PyTorch, which is not in the project's main `uv`
environment. Create a small side-env once:

```bash
export PATH="$HOME/snap/code/240/.local/bin:$PATH"
uv venv --python 3.12 /tmp/fm_check
uv pip install --python /tmp/fm_check/bin/python torch \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/fm_check/bin/python numpy scipy matplotlib
```

Then from the repo root:

```bash
cd experiments/intention-vs-deepsets
PYTHONPATH=../.. /tmp/fm_check/bin/python experiment.py
PYTHONPATH=../.. /tmp/fm_check/bin/python make_plots.py
```

Wall-time on CPU: ~6 s training + ~3 s plotting. Both scripts emit
`results.npz`, `summary.json`, three `.pt` checkpoints, and four PNGs.

## Headline numbers

| Architecture | params | in median R² | in p5 R² | out median R² | out p5 R² |
|---|---:|---:|---:|---:|---:|
| IntentionFM_Fixed       | 0     | +0.99994 | +0.810 | +0.99994 | +0.932 |
| IntentionFM_Learned     | 5,328 | +0.99973 | **+0.964** | +0.99962 | **+0.927** |
| DeepSets-FM (matched)   | 6,545 | +0.617   | -1.312 | +0.763   | -1.083 |
| Regressor (cheat)       | 5,041 | +0.938   | +0.360 | +0.935   | +0.394 |

Full discussion in
[../../docs/research/02-foundation-model/intention-vs-deepsets.md](../../docs/research/02-foundation-model/intention-vs-deepsets.md).
