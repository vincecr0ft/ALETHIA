r"""Multi-seed wrapper for campaigns_physics.run — statistical significance for
the cross-architecture comparison on the (m, mu) physics interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaigns_physics as cp

OUT = HERE / "output_matrix"
CAMPS = list(cp.CAMPAIGN_DIR)


def main(seeds=(0, 1, 2, 3, 4)):
    runs = [cp.run(seed=s) for s in seeds]
    names = list(runs[0]["results"])
    agg = {}
    for n in names:
        agg[n] = {}
        for camp in CAMPS:
            xs = np.array([r["results"][n][camp] for r in runs])
            agg[n][camp] = {"mean": float(xs.mean()), "std": float(xs.std())}
        mr = np.array([r["mean_rank"][n] for r in runs])
        agg[n]["mean_rank"] = {"mean": float(mr.mean()), "std": float(mr.std())}
    # rank on per-campaign means
    ranks = {n: {} for n in names}
    for camp in CAMPS:
        d = cp.CAMPAIGN_DIR[camp]
        order = sorted(names, key=lambda n: agg[n][camp]["mean"], reverse=(d > 0))
        for r, n in enumerate(order, 1):
            ranks[n][camp] = r
    overall = {n: float(np.mean(list(ranks[n].values()))) for n in names}
    out = {"seeds": list(seeds), "agg": agg, "ranks": ranks,
           "overall_mean_rank": overall}
    (OUT / "campaigns_physics_multiseed.json").write_text(json.dumps(out, indent=2))
    print("\n=== (m,mu) physics interface, multi-seed mean rank (1=best) ===")
    for n, mr in sorted(overall.items(), key=lambda kv: kv[1]):
        cells = "  ".join(f"{c[:4]}={agg[n][c]['mean']:.2f}±{agg[n][c]['std']:.2f}"
                          for c in CAMPS)
        print(f"  {n:20s} rank {mr:.2f}  | {cells}")
    return out


if __name__ == "__main__":
    main()
