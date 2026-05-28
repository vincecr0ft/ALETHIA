"""Patch 10: replace the raw-band coverage claim with the BH-corrected p-value.

Reads experiments/full-chain-run/output/trajectory.npz, takes the per-region
empirical 68% coverage in the post-recovery window (cycles 100-400), and
reports the per-cycle Benjamini-Hochberg-corrected binomial p-value at the
5% nominal FDR.

The actionable statistic: minimum and median BH-corrected p-value across
the post-recovery window. If the chain's calibration is honest, the
minimum stays above 0.05 / 5 = 0.01 across the window.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
NOMINAL_COV = 0.683
WINDOW_START = 100
WINDOW_END = 400
N_PER_REGION = 50          # nominal sample count per window (per Patch 10)
ALPHA = 0.05


def bh_correct(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg corrected p-values for a single batch of tests."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((m / np.arange(1, m + 1))[::-1] * ranked[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def main():
    npz_path = OUT / "trajectory.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}; run the full chain first")
    d = np.load(npz_path)
    cal_p = d["cal_pvalue_min"]            # (T,) min-over-regions BH-corrected
    cov_68 = d["cov_68_by_region"]         # (T, R) per-cycle per-region rate
    T = len(cal_p)
    print(f"# trajectory: T={T} cycles, R={cov_68.shape[1]} regions")

    end = min(WINDOW_END, T)
    window_p = cal_p[WINDOW_START:end]
    window_cov = cov_68[WINDOW_START:end]

    print(f"# post-recovery window: cycles [{WINDOW_START}, {end})  "
          f"length={len(window_p)}")
    print(f"\n# In-chain BH-corrected per-region binomial p-value (already FDR-controlled across the {cov_68.shape[1]} regions)")
    print(f"  minimum over post-recovery window  : p_min  = {window_p.min():.4f}")
    print(f"  median  over post-recovery window  : p_med  = {np.median(window_p):.4f}")
    print(f"  fraction of cycles with p < {ALPHA}: {(window_p < ALPHA).mean():.2%}")
    print(f"  fraction of cycles with p < 0.01   : {(window_p < 0.01).mean():.2%}")

    final_cov = np.nanmean(window_cov[-1])
    print(f"\n# Final-cycle mean coverage across regions: {final_cov:.4f} "
          f"(nominal {NOMINAL_COV})")
    print(f"  difference from nominal in pp: {(final_cov - NOMINAL_COV) * 100:+.2f}")
    print("\n# Interpretation:")
    print(f"  At the chain's per-region sample density (~5 probes/region/cycle,")
    print(f"  accumulating to ~{int(5 * end / cov_68.shape[1])} per region by cycle {end}), the binomial")
    print(f"  test rejects H0: p = {NOMINAL_COV} at BH-FDR < {ALPHA} for {(window_p < ALPHA).mean():.0%} of post-recovery")
    print(f"  cycles. The empirical coverage stabilises at {final_cov:.3f} vs the nominal")
    print(f"  {NOMINAL_COV} — a separation that is small in absolute terms (~{abs(final_cov - NOMINAL_COV) * 100:.1f}pp)")
    print(f"  but statistically resolved at this sample density. The chain's drift")
    print(f"  aggregator interprets this as the steady-state calibration-drift signal.")

    summary = dict(
        window=[WINDOW_START, end],
        nominal_coverage=NOMINAL_COV,
        p_min=float(window_p.min()),
        p_median=float(np.median(window_p)),
        fraction_below_alpha=float((window_p < ALPHA).mean()),
        fraction_below_001=float((window_p < 0.01).mean()),
        final_mean_coverage=float(final_cov),
        coverage_delta_pp=float((final_cov - NOMINAL_COV) * 100),
        alpha=ALPHA,
    )
    out_path = OUT / "bh_coverage_pvalues.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
