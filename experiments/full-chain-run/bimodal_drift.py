"""Bimodal drift experiment (U6): discriminate EPIG from random.

The unimodal drift event in run.py (only c_lq^(3) withheld) produces a
single information-rich region of m_ll, and Section 6.4 of the paper
honestly notes that EPIG and random acquisition recover at similar
rates on it — the recovery-curve gap is not statistically meaningful.
The plan's audit identifies this as a failure-mode of the experiment
design rather than of EPIG: with a unimodal information landscape,
random acquisition is already near-optimal because anywhere-in-the-
gap is informative.

This script runs a *bimodal* drift event: simultaneously withhold
``c_lq^(3) in [0.6, 1.0]`` AND ``c_Hq^(3) in [0.4, 0.8]``, with a
target scenario that activates both operators
(``c = (-0.5, 0, 0.8, 0)``). The four-fermion operator drives a tail
at high m_ll; the vertex operator drives a rate-like shift across the
low-m region. The two regions of high information are at opposite
ends of the kinematic spectrum, and random acquisition can no longer
hit both with a tight oracle budget.

Three sub-runs under shared seeds:

  ACQUISITION=random        # uniform pool, uniform pick
  ACQUISITION=epig          # closed-form EPIG over the pool
  ACQUISITION=epig_eigen    # importance-resample the pool from the
                            # low-eigen subspace, then EPIG

Outputs land in ``experiments/full-chain-run/output_bimodal_<acq>/``
and a comparison summary at
``experiments/full-chain-run/output_bimodal_summary.json``.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/full-chain-run/bimodal_drift.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ACQUISITIONS = ("random", "epig", "epig_eigen")


def _env_for(acq: str) -> dict[str, str]:
    e = os.environ.copy()
    e["BIMODAL"] = "1"
    e["ACQUISITION"] = acq
    e.setdefault("PHOENIX_PROJECT_NAME", "alethia-bimodal")
    return e


def _output_dir(acq: str) -> Path:
    suffix = f"bimodal_{acq}" if acq != "epig" else "bimodal"
    return HERE / f"output_{suffix}"


def _summary_path(acq: str) -> Path:
    return _output_dir(acq) / "summary.json"


def _trajectory_path(acq: str) -> Path:
    return _output_dir(acq) / "trajectory.npz"


def run_one(acq: str, *, force: bool = False) -> dict:
    """Run a single ACQUISITION sub-run of run.py with BIMODAL=1.

    Returns {summary: parsed summary.json, rmse_trace: list[float]}.
    The RMSE trace lives in trajectory.npz (not the JSON summary).

    Resumability: if the sub-run's summary.json already exists and
    ``force`` is False, the existing artefacts are loaded without
    re-running the chain. Delete the output dir or pass ``force=True``
    to redo.
    """
    summary_path = _summary_path(acq)
    if summary_path.exists() and not force:
        print(f"\n=== bimodal sub-run: ACQUISITION={acq}  [CACHED — reusing] ===")
    else:
        print(f"\n=== bimodal sub-run: ACQUISITION={acq} ===")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(HERE / "run.py")],
            env=_env_for(acq),
            check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        wall = time.time() - t0
        print(f"  finished in {wall:.1f}s, returncode={proc.returncode}")
        if proc.returncode != 0:
            sys.stdout.buffer.write(proc.stdout)
            raise RuntimeError(f"bimodal sub-run for {acq} failed")
        if not summary_path.exists():
            sys.stdout.buffer.write(proc.stdout)
            raise FileNotFoundError(
                f"expected {summary_path} after {acq} sub-run")
    with open(summary_path) as f:
        summary = json.load(f)
    traj_path = _trajectory_path(acq)
    if traj_path.exists():
        with np.load(traj_path) as data:
            rmse_trace = data["rmse_band_trace"].astype(float).tolist()
    else:
        rmse_trace = []
    return {"summary": summary, "rmse_trace": rmse_trace}


def collate(results: dict[str, dict]) -> dict:
    """Distil the three sub-runs into a single comparison record.

    Per acquisition: final RMSE on the target band, cycle index at
    which RMSE first crosses thresholds, area-under-curve over the
    recovery trajectory, oracle calls used, final context size.
    """
    out = {}
    for acq, payload in results.items():
        summ = payload["summary"]
        rmse_trace = payload["rmse_trace"]
        rmse_initial = rmse_trace[0] if rmse_trace else float("nan")
        rmse_final = rmse_trace[-1] if rmse_trace else float("nan")
        # cycle to first cross each ratio of initial RMSE
        crossings = {}
        for ratio in (0.5, 0.1, 0.01):
            target = rmse_initial * ratio
            idx = next(
                (i for i, v in enumerate(rmse_trace) if v <= target),
                None,
            )
            crossings[f"cycle_to_{ratio:g}x"] = idx
        auc = float(sum(rmse_trace)) if rmse_trace else float("nan")
        out[acq] = {
            "rmse_initial": rmse_initial,
            "rmse_final": rmse_final,
            "rmse_auc": auc,
            "rmse_trace_length": len(rmse_trace),
            "crossings": crossings,
            "oracle_calls": summ.get("oracle_calls"),
            "n_local_retrain": summ.get("n_local_retrain"),
            "before_band_rmse": summ.get("before_band_rmse"),
            "after_band_rmse": summ.get("after_band_rmse"),
            "final_kappa": summ.get("final_kappa"),
        }
    return out


def plot_recovery(results: dict[str, dict], out_path: Path) -> None:
    """Recovery-curve plot: RMSE on the target band per cycle, three
    acquisitions overlaid on a single log-y axis."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = {"random": "C0", "epig": "C1", "epig_eigen": "C2"}
    for acq in ACQUISITIONS:
        trace = results[acq]["rmse_trace"]
        if not trace:
            continue
        ax.plot(np.arange(len(trace)), trace,
                label=acq, color=colors.get(acq), lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("cycle")
    ax.set_ylabel("RMSE on target band $(c_{Hq}^{(3)}, c_{\\ell q}^{(3)})$"
                  " withheld")
    ax.set_title("Bimodal drift recovery: random vs EPIG vs EPIG-eigen")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    print("# Bimodal drift experiment (U6)")
    results = {}
    for acq in ACQUISITIONS:
        results[acq] = run_one(acq)
    compare = collate(results)
    out_path = HERE / "output_bimodal_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {"per_acquisition": compare,
             "ordering": list(ACQUISITIONS)},
            f, indent=2,
        )
    print(f"\nWrote {out_path}")
    print(json.dumps(compare, indent=2))

    plot_dir = HERE.parent.parent / "docs" / "research" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "bimodal_recovery.png"
    plot_recovery(results, plot_path)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
