"""Compare BCRB efficiency between v1 and v2 psi_theta.

v1 is the original IntentionFM trained on plain MSE (no c-recoverability).
v2 is the IntentionFM retrained with the c-recoverability auxiliary loss
(retrain_with_c_recovery.py).

Reads:
  output/bcrb_efficiency_mass_only.json    (v1)
  output/bcrb_efficiency_mass_only_v2.json (v2)

Emits a short table to stdout and a JSON file at
output/bcrb_efficiency_v1_v2_comparison.json.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"


def load(p: Path) -> dict:
    if not p.exists():
        print(f"WARNING: {p} missing.")
        return {}
    return json.loads(p.read_text())


def main():
    v1 = load(OUT_DIR / "bcrb_efficiency_mass_only.json")
    v2 = load(OUT_DIR / "bcrb_efficiency_mass_only_v2.json")
    if not v1 or not v2:
        print("ERROR: missing one or both BCRB summaries.")
        sys.exit(1)

    out = {"pools": {}}
    for pool in ("inbox", "withheld_band"):
        rows_v1 = v1["pools"][pool]["per_direction"]
        rows_v2 = v2["pools"][pool]["per_direction"]
        out["pools"][pool] = {"per_direction": []}
        print(f"\n=== Pool {pool} ===")
        print(f"{'dir':<5} {'lambda_K(v1)':>14} {'lambda_K(v2)':>14}  "
              f"{'eta v1':>8} {'eta v2':>8}  "
              f"{'MSE v1':>10} {'MSE v2':>10}  "
              f"{'MLE var v1':>11} {'MLE var v2':>11}  "
              f"{'verdict v1':<25} {'verdict v2':<25}")
        for a, (r1, r2) in enumerate(zip(rows_v1, rows_v2)):
            print(f"c~{a+1:<3} "
                  f"{r1['lambda_K_mean']:>14.3g} {r2['lambda_K_mean']:>14.3g}  "
                  f"{r1['efficiency_eta']:>8.3f} {r2['efficiency_eta']:>8.3f}  "
                  f"{r1['mse_mean']:>10.4f} {r2['mse_mean']:>10.4f}  "
                  f"{r1['mle_variance_mean']:>11.4g} "
                  f"{r2['mle_variance_mean']:>11.4g}  "
                  f"{r1['verdict']:<25} {r2['verdict']:<25}")
            out["pools"][pool]["per_direction"].append({
                "c_tilde_index": a + 1,
                "v1": {
                    "lambda_K_mean": r1["lambda_K_mean"],
                    "eta": r1["efficiency_eta"],
                    "mse_mean": r1["mse_mean"],
                    "mle_var_mean": r1["mle_variance_mean"],
                    "verdict": r1["verdict"],
                },
                "v2": {
                    "lambda_K_mean": r2["lambda_K_mean"],
                    "eta": r2["efficiency_eta"],
                    "mse_mean": r2["mse_mean"],
                    "mle_var_mean": r2["mle_variance_mean"],
                    "verdict": r2["verdict"],
                },
            })
    summary_path = OUT_DIR / "bcrb_efficiency_v1_v2_comparison.json"
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved comparison summary to {summary_path}.")

    # Headline numbers for resolved directions.
    print("\n=== Headline: resolved directions (inbox) ===")
    inbox_rows = out["pools"]["inbox"]["per_direction"]
    for a in range(2):
        r = inbox_rows[a]
        change = r["v2"]["mse_mean"] / max(r["v1"]["mse_mean"], 1e-30)
        change_eta = r["v2"]["eta"] / max(r["v1"]["eta"], 1e-30) if r["v1"]["eta"] > 0 else float("inf")
        print(f"  c_tilde_{a+1}: MSE  v1 = {r['v1']['mse_mean']:.4f}  "
              f"v2 = {r['v2']['mse_mean']:.4f}  (factor {change:.2f})")
        print(f"  c_tilde_{a+1}: eta  v1 = {r['v1']['eta']:.3f}  "
              f"v2 = {r['v2']['eta']:.3f}  "
              f"(factor {change_eta if change_eta != float('inf') else 'inf':}{'' if change_eta == float('inf') else ''})")


if __name__ == "__main__":
    main()
