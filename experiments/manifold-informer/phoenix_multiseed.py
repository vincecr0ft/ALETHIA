r"""Multi-seed band version of the span-completeness trace (paper Figure 2).

The seed context (initial working points) and the random arm's picks are
stochastic, so a single trajectory is not representative. Here we run both arms
over many seeds and plot median +/- a 10-90 percentile band.

Speed: the per-event log-ratio cache is keyed by working point, not by seed, so
we precompute every candidate's log_w ONCE (the only oracle cost) and every seed
of both arms is then pure linear algebra.

PYTHONPATH=/home/vince/ALETHIA python3 phoenix_multiseed.py [--n-probe 4000] [--seeds 40]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
os.environ.setdefault("PHOENIX_TRACING", "0")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phoenix_span_completeness import (Config, run_loop, tier_pool, TIERS, psi_mass_only)
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import sample_events, event_log_likelihood_ratio
from modules.surrogate.features import N_WC


def pad(trace, L):
    """Pad a (completed) trace to length L with its final value."""
    t = list(trace)
    return t + [t[-1]] * (L - len(t)) if len(t) < L else t[:L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probe", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="output_phoenix_loop_n4000")
    args = ap.parse_args()
    out = HERE / args.out; out.mkdir(exist_ok=True)

    cfg0 = Config(n_probe=args.n_probe)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    probe = sample_events(oracle, np.zeros(N_WC), cfg0.n_probe, seed=7)

    # ---- precompute log_w for EVERY candidate once (the only oracle cost) ----
    rng = np.random.default_rng(0)
    cands = []
    for t in TIERS:
        cands += tier_pool(t["dims"], rng)
    shared_lw: dict = {}
    print(f"precomputing log_w for {len(cands)} candidates at n_probe={cfg0.n_probe} ...")
    for c in cands:
        key = c.tobytes()
        if key not in shared_lw:
            shared_lw[key] = event_log_likelihood_ratio(oracle, c, probe)
    print(f"  cached {len(shared_lw)} unique working points")

    # ---- run both arms over many seeds, reusing the cache (instant) ----
    L = cfg0.max_cycles
    traces = {"curvature": [], "random": []}
    ext_cycles = {"curvature": [], "random": []}
    for s in range(args.seeds):
        for arm in ("curvature", "random"):
            cfg = Config(n_probe=cfg0.n_probe, seed=1000 + s)
            res = run_loop(cfg, oracle, probe, arm, dict(shared_lw))  # copy: don't mutate
            traces[arm].append(pad(res["sigma1_trace"], L))
            if res["extensions"]:
                ext_cycles[arm].append(res["extensions"][-1]["cycle"])
    for arm in traces:
        traces[arm] = np.array(traces[arm])

    # ---- plot: median + 10-90 percentile band ----
    cyc = np.arange(L)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    colors = {"curvature": "C3", "random": "C0"}
    for arm in ("curvature", "random"):
        T = np.maximum(traces[arm], 1e-6)
        med = np.median(T, axis=0)
        lo, hi = np.percentile(T, 10, axis=0), np.percentile(T, 90, axis=0)
        ax.fill_between(cyc, lo, hi, color=colors[arm], alpha=0.20, lw=0)
        ax.semilogy(cyc, med, "-o", c=colors[arm], ms=4,
                    label=f"{arm} (median, 10-90\\% band)")
        # completion cycle: first cycle after the last psi-extension (manifold stops growing)
        comp = int(np.median([e + 1 for e in ext_cycles[arm]])) if ext_cycles[arm] else None
        if comp is not None:
            ax.axvline(comp, ls="-", c=colors[arm], alpha=0.7, lw=1.4)
            ax.annotate(f"{arm}\\ncompletes (cyc {comp})", xy=(comp, 0.5), xytext=(comp + 0.3, 0.5),
                        fontsize=7, c=colors[arm], rotation=90, va="center")
    ax.axhline(cfg0.completion_floor, ls="--", c="k", alpha=0.45, label="completion floor $\\tau=0.30$")
    ax.set_xlabel("loop cycle"); ax.set_ylabel(r"residual $\sigma_1$")
    ax.set_title(f"Span-completeness trace over {args.seeds} seeds "
                 f"(n_probe={cfg0.n_probe})\n"
                 "AL completes (curvature) vs random; band = seed-to-seed spread")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out / "spectra_band.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- summary numbers for the caption ----
    def at(arm, k):
        T = traces[arm]
        return np.median(T[:, k]), np.percentile(T[:, k], 10), np.percentile(T[:, k], 90)
    print(f"\n{'cyc':>4} | {'curvature med [10-90]':>28} | {'random med [10-90]':>28}")
    for k in [0, 1, 2, 4, 6, 10]:
        cm, cl, ch = at("curvature", k); rm, rl, rh = at("random", k)
        print(f"{k:>4} | {cm:8.3f} [{cl:.3f}, {ch:.3f}]      | {rm:8.3f} [{rl:.3f}, {rh:.3f}]")
    print(f"\nmedian last-extension cycle: curvature={np.median(ext_cycles['curvature']):.0f} "
          f"random={np.median(ext_cycles['random']):.0f}")
    print(f"final sigma1 (cyc {L-1}): curvature med={np.median(traces['curvature'][:,-1]):.3f}  "
          f"random med={np.median(traces['random'][:,-1]):.3f}")
    print(f"# wrote {out}/spectra_band.png")


if __name__ == "__main__":
    main()
