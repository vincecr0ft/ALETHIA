"""Parallel multi-seed bimodal drift experiment.

Runs the bimodal drift event of ``bimodal_drift.py`` across 5 seeds for
each of 3 acquisition modes (random / epig / epig_eigen), in parallel,
and aggregates the recovery-curve trajectory into a mean +/- std
ribbon per acquisition. This replaces the single-seed comparison in
``bimodal_drift.py``, whose noise floor was large enough that the
acquisition ordering was not stable across seeds.

Resource budget: 15 chains, each constrained to 2 BLAS / OMP threads,
total 30 threads on a 32-logical-core box. Each chain runs at roughly
the same wall time as a single sequential run because the BLAS calls
share cache.

Each sub-run writes to ``output_bimodal_<acq>_s<seed>/`` and the
aggregated result lands at
``output_bimodal_parallel/aggregate.json`` plus
``docs/research/plots/bimodal_recovery_avg.png``.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/full-chain-run/bimodal_parallel.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Acquisition modes. The INV-3 parameter-space EPIG modes are wired by
# default so the angular-observable sweep covers
# {random, epig, leverage, param_epig_d, param_epig_a}; override via
# ACQUISITIONS="random,epig,leverage,param_epig_d,param_epig_a" (comma-
# separated). Default keeps backwards-compatible mass-only set.
_DEFAULT_ACQ = "random,epig,epig_eigen"
ACQUISITIONS = tuple(
    a.strip() for a in os.environ.get("ACQUISITIONS", _DEFAULT_ACQ).split(",")
    if a.strip())

# Seeds: configurable via env var. Two layout options for power:
#   SEEDS="2026,2027,2028"   -> explicit list
#   N_SEEDS=20               -> default 2026..2026+N-1
# Default is the legacy 5-seed list so existing runs are unchanged.
_seeds_env = os.environ.get("SEEDS", "").strip()
if _seeds_env:
    SEEDS = tuple(int(s) for s in _seeds_env.split(",") if s.strip())
else:
    _n = int(os.environ.get("N_SEEDS", "5"))
    SEEDS = tuple(2026 + i for i in range(_n))

# Observable / stressed-budget pass-through. These flow into each
# subprocess via _env_for and propagate to run.py's OBSERVABLE /
# STRESSED_BUDGET handling, which controls both the oracle truth
# function (mass vs mu_afb) and the (k, pool-size) tuple.
OBSERVABLE = os.environ.get("OBSERVABLE", "mass").lower()
STRESSED_BUDGET = os.environ.get("STRESSED_BUDGET", "0").lower() in {
    "1", "true", "yes"}

THREADS_PER_CHAIN = int(os.environ.get("THREADS_PER_CHAIN", "2"))
# Cap on concurrent processes. Default is len(jobs) so single sweeps run
# everything in parallel. For larger sweeps (e.g. 5 acqs x 20 seeds = 100
# chains), set MAX_WORKERS to (cores / THREADS_PER_CHAIN) to avoid CPU
# oversubscription.
MAX_WORKERS_ENV = os.environ.get("MAX_WORKERS", "").strip()

# Output dir suffixes match run.py's OUT_BASENAME convention so the
# aggregator finds the per-seed sub-runs.
_TAG = ""
if OBSERVABLE != "mass":
    _TAG += f"_{OBSERVABLE}"
if STRESSED_BUDGET:
    _TAG += "_stressed"
OUT = HERE / f"output_bimodal_parallel{_TAG}"
OUT.mkdir(parents=True, exist_ok=True)


def _output_dir(acq: str, seed: int) -> Path:
    # Mirror run.py's OUT_BASENAME logic. With BIMODAL=1 the suffix
    # always starts with "bimodal"; the acquisition label is appended
    # unless it's the default epig. OBSERVABLE/STRESSED_BUDGET tags are
    # then appended in the same order as run.py.
    suffix = f"bimodal_{acq}" if acq != "epig" else "bimodal"
    if OBSERVABLE != "mass":
        suffix = f"{suffix}_{OBSERVABLE}"
    if STRESSED_BUDGET:
        suffix = f"{suffix}_stressed"
    seed_suffix = "" if seed == 2026 else f"_s{seed}"
    return HERE / f"output_{suffix}{seed_suffix}"


def _summary_path(acq: str, seed: int) -> Path:
    return _output_dir(acq, seed) / "summary.json"


def _trajectory_path(acq: str, seed: int) -> Path:
    return _output_dir(acq, seed) / "trajectory.npz"


def _env_for(acq: str, seed: int) -> dict[str, str]:
    e = os.environ.copy()
    e["BIMODAL"] = "1"
    e["ACQUISITION"] = acq
    e["SEED"] = str(seed)
    # OBSERVABLE / STRESSED_BUDGET pass-through so the sweep stays in
    # sync with the harness controls; run.py reads them from the env.
    e["OBSERVABLE"] = OBSERVABLE
    e["STRESSED_BUDGET"] = "1" if STRESSED_BUDGET else "0"
    # Tracing off — many parallel processes streaming to one Phoenix
    # project add nothing the unimodal single-seed run did not already
    # contribute, and the HTTP exporter contention slows pretraining.
    e["PHOENIX_TRACING"] = "0"
    # Constrain per-process threads so chains coexist on the box.
    # Without this, each torch process tries to use every core.
    t = str(THREADS_PER_CHAIN)
    e["OMP_NUM_THREADS"] = t
    e["MKL_NUM_THREADS"] = t
    e["OPENBLAS_NUM_THREADS"] = t
    e["NUMEXPR_NUM_THREADS"] = t
    e["TORCH_NUM_THREADS"] = t
    return e


def _run_one(args: tuple[str, int]) -> dict:
    acq, seed = args
    summary_path = _summary_path(acq, seed)
    if summary_path.exists():
        return {
            "acq": acq, "seed": seed,
            "status": "cached",
            "summary_path": str(summary_path),
        }
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(HERE / "run.py")],
        env=_env_for(acq, seed),
        check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    wall = time.time() - t0
    if proc.returncode != 0 or not summary_path.exists():
        return {
            "acq": acq, "seed": seed,
            "status": "failed",
            "returncode": proc.returncode, "wall": wall,
            "tail": proc.stdout.decode(
                "utf-8", errors="replace")[-2000:],
        }
    return {
        "acq": acq, "seed": seed,
        "status": "ok", "wall": wall,
    }


def aggregate() -> dict:
    """Load every (acq, seed) result and compute mean/std trajectories.

    The chain in run.py has a convergence-aware early-exit (oracle
    budget reached + coverage recovered, or two consecutive recovered
    windows past 50 oracle calls), so seeds that recover quickly end
    early. We pad shorter traces with their final value out to the
    maximum trace length seen across seeds — semantically: "this seed
    has converged and would remain near this RMSE if the loop
    continued".
    """
    def _pad(traces: list[np.ndarray]) -> np.ndarray:
        L = max(len(t) for t in traces)
        return np.stack(
            [np.concatenate([t, np.full(L - len(t), t[-1])]) for t in traces],
            axis=0,
        )

    def _stats(stacked: np.ndarray) -> dict:
        L = int(stacked.shape[1])
        return {
            "trace_mean": stacked.mean(axis=0).tolist(),
            "trace_std": stacked.std(axis=0, ddof=1).tolist()
                if stacked.shape[0] > 1 else [0.0] * L,
            "trace_p25": np.percentile(stacked, 25, axis=0).tolist(),
            "trace_p75": np.percentile(stacked, 75, axis=0).tolist(),
            "trace_min": stacked.min(axis=0).tolist(),
            "trace_max": stacked.max(axis=0).tolist(),
        }

    per_acq: dict[str, dict] = {}
    for acq in ACQUISITIONS:
        traces = []
        H_traces: list[np.ndarray] = []
        err1_traces: list[np.ndarray] = []
        err2_traces: list[np.ndarray] = []
        raw_lengths = []
        finals = []
        afters = []
        for seed in SEEDS:
            traj_path = _trajectory_path(acq, seed)
            summ_path = _summary_path(acq, seed)
            if not traj_path.exists() or not summ_path.exists():
                continue
            with np.load(traj_path) as data:
                traces.append(data["rmse_band_trace"].astype(float))
                raw_lengths.append(int(len(traces[-1])))
                # INV-3 c̃-space scoring traces (added by run.py's
                # trajectory.npz). Old runs without these keys keep
                # the rmse-only output; the new traces are appended
                # only when present.
                if "H_ctilde" in data.files:
                    H_traces.append(data["H_ctilde"].astype(float))
                if "c_tilde_err_d1" in data.files:
                    err1_traces.append(data["c_tilde_err_d1"].astype(float))
                if "c_tilde_err_d2" in data.files:
                    err2_traces.append(data["c_tilde_err_d2"].astype(float))
            with open(summ_path) as f:
                summ = json.load(f)
            finals.append(traces[-1][-1])
            afters.append(float(summ.get("after_band_rmse", float("nan"))))
        if not traces:
            per_acq[acq] = {"n_seeds": 0}
            continue
        stacked = _pad(traces)
        record: dict = {
            "n_seeds": int(stacked.shape[0]),
            "trace_length_padded_to": int(stacked.shape[1]),
            "trace_lengths_raw": raw_lengths,
            **_stats(stacked),
            "final_rmse_mean": float(np.mean(finals)),
            "final_rmse_std": float(np.std(finals, ddof=1))
                if len(finals) > 1 else 0.0,
            "after_band_rmse_mean": float(np.mean(afters)),
            "after_band_rmse_std": float(np.std(afters, ddof=1))
                if len(afters) > 1 else 0.0,
            "after_band_rmse_per_seed": [float(x) for x in afters],
            "seeds_used": [int(s) for s in SEEDS[: stacked.shape[0]]],
        }
        # Append c̃-space trace stats with explicit name prefixes so the
        # downstream plotters can distinguish them from the µ-RMSE
        # trace_* keys. Each new stack is padded the same way.
        if H_traces:
            Hs = _pad(H_traces)
            for k, v in _stats(Hs).items():
                record[k.replace("trace_", "H_ctilde_")] = v
            record["H_ctilde_final_mean"] = float(np.mean(
                [t[-1] for t in H_traces]))
            record["H_ctilde_initial_mean"] = float(np.mean(
                [t[0] for t in H_traces]))
        if err1_traces:
            E1 = _pad(err1_traces)
            for k, v in _stats(E1).items():
                record[k.replace("trace_", "c_tilde_err_d1_")] = v
            record["c_tilde_err_d1_final_mean"] = float(np.mean(
                [t[-1] for t in err1_traces]))
        if err2_traces:
            E2 = _pad(err2_traces)
            for k, v in _stats(E2).items():
                record[k.replace("trace_", "c_tilde_err_d2_")] = v
            record["c_tilde_err_d2_final_mean"] = float(np.mean(
                [t[-1] for t in err2_traces]))
        per_acq[acq] = record
    return per_acq


_COLOR_TABLE = {
    "random": "C0", "epig": "C1", "epig_eigen": "C2",
    "leverage": "C3", "param_epig_d": "C4", "param_epig_a": "C5",
}


def plot(per_acq: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for i, acq in enumerate(ACQUISITIONS):
        a = per_acq.get(acq, {})
        if a.get("n_seeds", 0) == 0:
            continue
        color = _COLOR_TABLE.get(acq, f"C{i % 10}")
        mean = np.asarray(a["trace_mean"])
        p25 = np.asarray(a["trace_p25"])
        p75 = np.asarray(a["trace_p75"])
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, lw=1.6,
                label=f"{acq}  (n={a['n_seeds']})")
        ax.fill_between(x, p25, p75, color=color, alpha=0.20)
    ax.set_yscale("log")
    ax.set_xlabel("cycle")
    ax.set_ylabel("RMSE on bimodal target band")
    ax.set_title("Bimodal drift recovery: mean trace, IQR ribbon")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    jobs = [(acq, seed) for acq in ACQUISITIONS for seed in SEEDS]
    max_workers = int(MAX_WORKERS_ENV) if MAX_WORKERS_ENV else len(jobs)
    max_workers = max(1, min(max_workers, len(jobs)))
    print(f"# Bimodal parallel sweep: {len(jobs)} chains, "
          f"{max_workers} concurrent, {THREADS_PER_CHAIN} threads each")
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"  {r['acq']:11s} seed={r['seed']:5d}  "
                  f"status={r['status']:7s}  "
                  f"wall={r.get('wall', 0):.0f}s")
            results.append(r)
    wall_total = time.time() - t0
    print(f"\nAll done in {wall_total:.0f}s")

    n_failed = sum(1 for r in results if r["status"] == "failed")
    if n_failed:
        print(f"\nWARNING: {n_failed} sub-runs failed")
        for r in results:
            if r["status"] == "failed":
                print(f"  {r['acq']} seed={r['seed']}")
                print("  --- tail ---")
                print("  " + r.get("tail", "").replace("\n", "\n  "))

    per_acq = aggregate()
    with open(OUT / "aggregate.json", "w") as f:
        json.dump(
            {"per_acquisition": per_acq,
             "ordering": list(ACQUISITIONS),
             "seeds": list(SEEDS),
             "n_seeds_target": len(SEEDS),
             "observable": OBSERVABLE,
             "stressed_budget": STRESSED_BUDGET,
             "wall_seconds_total": wall_total,
             "failures": [r for r in results if r["status"] == "failed"]},
            f, indent=2,
        )
    plot_path = HERE.parent.parent / "docs" / "research" / "plots" / \
        f"bimodal_recovery_avg{_TAG}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot(per_acq, plot_path)
    print(f"\nWrote {OUT / 'aggregate.json'}")
    print(f"Wrote {plot_path}")

    # Headline table.
    print("\n# Summary (mean +/- std over seeds)")
    for acq in ACQUISITIONS:
        a = per_acq.get(acq, {})
        if a.get("n_seeds", 0) == 0:
            print(f"  {acq:14s}  no data")
            continue
        line = (
            f"  {acq:14s}  after_band = "
            f"{a['after_band_rmse_mean']:8.3f} +/- "
            f"{a['after_band_rmse_std']:.3f}   "
            f"final_trace = "
            f"{a['final_rmse_mean']:8.3f} +/- "
            f"{a['final_rmse_std']:.3f}"
        )
        if "H_ctilde_final_mean" in a:
            line += (f"   H_init={a.get('H_ctilde_initial_mean', float('nan')):.3f}"
                     f"  H_fin={a['H_ctilde_final_mean']:.3f}")
        if "c_tilde_err_d1_final_mean" in a:
            line += (f"   err_d1={a['c_tilde_err_d1_final_mean']:.3f}"
                     f"  err_d2={a.get('c_tilde_err_d2_final_mean', float('nan')):.3f}")
        line += f"  (n={a['n_seeds']})"
        print(line)


if __name__ == "__main__":
    main()
