# Demo results

## Run status

Executed in the main session (numpy 2.4.4 + scikit-learn 1.8.0). Runs in a few seconds.

To reproduce:

```
python3 experiments/fm-taxonomy/01-fm-surrogate/demo.py
```

## What the demo measures

The script holds an unseen target function fixed and compares two modes given only
`k` labelled points from it:

- **(A) single-task surrogate** — a small MLP fit from scratch on the `k` points;
  amortized over inputs only.
- **(B) foundation-style representation** — an MLP trunk pretrained on 300 sibling
  functions from the family `f_c(x) = sin(w x + p) + a x`, its penultimate layer frozen
  as a feature basis `phi(x)`, with adaptation done as a ridge linear probe on the `k`
  points; amortized over the family.

## Verified output

```
Few-shot generalisation MSE on unseen family members
(median over 40 target functions; lower is better)

k (labelled pts) |  surrogate (A) |  foundation (B) |  ratio A/B
------------------------------------------------------------------
               3 |         1.0085 |          0.9396 |       1.07
               5 |         0.7343 |          0.3806 |       1.93
              10 |         0.3256 |          0.0445 |       7.32
              20 |         0.0769 |          0.0089 |       8.64
              50 |         0.0981 |          0.0009 |     106.11
```

## Interpretation

The frozen family basis lets the linear probe reconstruct an unseen family member from
few points: the foundation-style representation (B) wins at every `k`, and the advantage
*grows* with `k` here (ratio A/B rising from 1.07 at k=3 to ~106 at k=50). The probe
sits in a low-dimensional family-aligned feature space, so added points drive its error
toward zero, while the from-scratch surrogate plateaus near MSE ≈ 0.08–0.10 — it has to
relearn the family structure from scratch every time and cannot exploit the shared basis.
This is the numerical signature of *amortization over the family* (Axis 1 of the writeup):
reuse of pretrained structure, not just more per-task data, is what closes the gap.

(Note: the original authored prediction guessed the ratio would *shrink* toward 1 as `k`
grew. The run shows the opposite — the probe's family-aligned basis keeps paying off — so
that prediction is corrected here against the measured numbers.)
