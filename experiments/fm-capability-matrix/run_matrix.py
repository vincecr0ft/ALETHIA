r"""Driver + assembler for the FM capability matrix.

Runs every capability cell (or just collects their cached JSON) and assembles
the fitness-for-purpose matrix into matrix.json + RESULTS.md. The thesis (see
ALETHEIA_ablative_FM_survey_plan.md): on ONE fixed SMEFT Drell-Yan substrate,
each FM family is scored on the task its pretraining objective targets, with a
metric native to that objective — not all on one shared probe. The matrix
therefore reads as a diagonal of fitness-for-purpose, the opposite of the prior
single-probe survey.

Each cell writes output_matrix/cell_<name>.json with a common schema:
  cell, fm_family, hep_task, metric_primary{name,value}, metrics{...},
  ablation_isolated, n_params, wall_seconds, config.

Usage:
  python run_matrix.py            # run any missing cells, then assemble
  python run_matrix.py --rerun    # re-run every cell from scratch
  python run_matrix.py --assemble # assemble from existing JSON only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"
PY = str(HERE.parent.parent / ".venv" / "bin" / "python")

# Display order = the survey plan's capability rows.
CELLS = [
    "generative",   # autoregressive
    "ssl",          # masked / denoising SSL
    "jepa",         # contrastive / JEPA
    "pfn",          # in-context / amortized
    "quantum_kernel",  # quantum-kernel sibling of the Intention ridge
    "operator",     # neural-operator / DeepONet
    "diffusion",    # diffusion / flow
    "anomaly",      # anomaly detection
]


def run_cell(name: str) -> bool:
    script = HERE / f"cell_{name}.py"
    if not script.exists():
        print(f"  ! {script.name} missing")
        return False
    print(f"  running cell_{name}.py ...", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, str(script)], cwd=str(HERE),
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! cell_{name} failed ({time.time() - t0:.0f}s):\n{r.stderr[-1500:]}")
        return False
    print(f"  ok cell_{name} ({time.time() - t0:.0f}s)")
    return True


def load_cells() -> dict:
    rows = {}
    for name in CELLS:
        p = OUT / f"cell_{name}.json"
        if p.exists():
            rows[name] = json.loads(p.read_text())
    return rows


def fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def assemble(rows: dict) -> None:
    matrix = {"cells": rows, "order": CELLS,
              "substrate": "analytic SMEFT Drell-Yan (pp->ll), x=(log m_ll, cos θ*), "
                           "4 Wilson coeffs (cHq3,cHq1,clq3,clq1)"}
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2))

    lines = []
    lines.append("# FM capability matrix — results\n")
    lines.append(
        "One fixed SMEFT Drell-Yan substrate; each foundation-model family "
        "scored on the task its pretraining objective targets, with a metric "
        "native to that objective. This is the fitness-for-purpose grid the "
        "prior single-probe survey "
        "(`ALETHEIA_manifold_informer_landscape_practice.md`) lacked. See "
        "`ALETHEIA_ablative_FM_survey_plan.md` for the design.\n")

    # Main matrix table.
    lines.append("## Matrix (each cell = a different FM family on its native task)\n")
    lines.append("| FM family | HEP task | Primary metric | Value | Ablation isolated |")
    lines.append("|---|---|---|---|---|")
    for name in CELLS:
        if name not in rows:
            lines.append(f"| _{name}_ | (not run) | — | — | — |")
            continue
        r = rows[name]
        mp = r["metric_primary"]
        lines.append(
            f"| {r['fm_family']} | {r['hep_task']} | {mp['name']} | "
            f"**{fmt(mp['value'])}** | {r['ablation_isolated']} |")
    lines.append("")

    # Per-cell detail.
    lines.append("## Per-cell metrics\n")
    for name in CELLS:
        if name not in rows:
            continue
        r = rows[name]
        lines.append(f"### {name} — {r['fm_family']}")
        lines.append(f"*Task:* {r['hep_task']}  ")
        lines.append(f"*Params:* {r.get('n_params','?')}  *Wall:* "
                     f"{fmt(r.get('wall_seconds',0),1)}s\n")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                lines.append(f"| {k} | {fmt(v, 4)} |")
            else:
                vs = ", ".join(fmt(x, 3) if isinstance(x, float) else str(x)
                               for x in v) if isinstance(v, list) else str(v)
                lines.append(f"| {k} | {vs} |")
        lines.append("")

    # Capacity-control column (the P3/P4 fix), where cells report it.
    lines.append("## Capacity-controlled probe column (the P3/P4 fix)\n")
    lines.append(
        "Where a cell recovers Wilson coefficients via a frozen-representation "
        "probe, the probe is ridge, held-out, and benchmarked against a "
        "random-feature floor of the *same* width. `margin = r2 - floor_r2 > 0` "
        "means the learned representation beats a random projection of the raw "
        "events — the test the prior in-sample P3/P4 gates could not pose.\n")
    lines.append("| cell | probe r2 | floor r2 | margin |")
    lines.append("|---|---|---|---|")
    for name in CELLS:
        if name not in rows:
            continue
        m = rows[name]["metrics"]
        r2 = m.get("probe_r2_heldout", m.get("transfer_probe_r2",
                   m.get("jepa_r2_box")))
        fl = m.get("probe_floor_r2", m.get("transfer_floor_r2",
                   m.get("jepa_floor_r2")))
        mg = m.get("probe_margin", m.get("transfer_margin",
                   m.get("jepa_margin")))
        if r2 is None and fl is None and mg is None:
            continue
        lines.append(f"| {name} | {fmt(r2,3) if r2 is not None else '—'} | "
                     f"{fmt(fl,3) if fl is not None else '—'} | "
                     f"{fmt(mg,3) if mg is not None else '—'} |")
    lines.append("")

    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print(f"\nassembled {OUT / 'matrix.json'} and {OUT / 'RESULTS.md'}")
    print(f"cells present: {sorted(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="re-run every cell")
    ap.add_argument("--assemble", action="store_true",
                    help="assemble from existing JSON only")
    args = ap.parse_args()

    if not args.assemble:
        for name in CELLS:
            if args.rerun or not (OUT / f"cell_{name}.json").exists():
                run_cell(name)
    assemble(load_cells())


if __name__ == "__main__":
    main()
