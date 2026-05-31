"""Parallel driver for the Phoenix-instrumented closed-loop AL runs.

Mirrors experiments/full-chain-run/bimodal_parallel.py: spawns subprocesses for
the cartesian product of (acquisition × pool × K_ctx × seed), runs them with a
concurrency cap that respects the BLAS/OMP thread budget, and aggregates per
(pool, K_ctx, acq) into a JSON summary + a plot.

Defaults are tuned for the 12-hour budget: 20 seeds, 5 acquisitions, P0/P1/P2,
K_ctx 12/24, N_CYCLES=100. = 600 chains. Each chain should take ~10s wall on
this box (no pretrain — the chains share the frozen checkpoint), so total wall
is well under an hour with 16-way concurrency.

Override via env:
  POOLS=P0,P1,P2   K_CTX_GRID=12,24   N_SEEDS=20   N_CYCLES=100   ACQS=random,...
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
OUT = HERE / "output"

POOLS = tuple(s.strip() for s in os.environ.get("POOLS", "P0,P1,P2").split(",") if s.strip())
K_CTX_GRID = tuple(int(s) for s in os.environ.get("K_CTX_GRID", "12,24").split(",") if s.strip())
ACQS = tuple(s.strip() for s in os.environ.get("ACQS",
    "random,leverage,epig,param_epig_d,param_epig_a").split(",") if s.strip())
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
SEED0 = int(os.environ.get("SEED0", "2026"))
N_CYCLES = int(os.environ.get("N_CYCLES", "100"))
THREADS_PER_CHAIN = int(os.environ.get("THREADS_PER_CHAIN", "2"))
MAX_WORKERS_ENV = os.environ.get("MAX_WORKERS", "").strip()


def _chain_dir(acq: str, pool: str, K: int, seed: int) -> Path:
    return OUT / f"closed_loop_{pool}_K{K}_N{N_CYCLES}_{acq}_s{seed}"


def _env_for(acq: str, pool: str, K: int, seed: int) -> dict[str, str]:
    e = os.environ.copy()
    e["ACQ"] = acq
    e["POOL"] = pool
    e["K_CTX"] = str(K)
    e["SEED"] = str(seed)
    e["N_CYCLES"] = str(N_CYCLES)
    # Tracing OFF for parallel sweep; we'll re-run a small representative
    # subset with tracing ON afterwards so Phoenix shows the dashboard view
    # without contention slowing down the 600-chain sweep.
    e["PHOENIX_TRACING"] = "0"
    t = str(THREADS_PER_CHAIN)
    e["OMP_NUM_THREADS"] = t
    e["MKL_NUM_THREADS"] = t
    e["OPENBLAS_NUM_THREADS"] = t
    e["NUMEXPR_NUM_THREADS"] = t
    e["TORCH_NUM_THREADS"] = t
    return e


def _run_one(args: tuple[str, str, int, int]) -> dict:
    acq, pool, K, seed = args
    summary_path = _chain_dir(acq, pool, K, seed) / "summary.json"
    if summary_path.exists():
        return {"acq": acq, "pool": pool, "K": K, "seed": seed, "status": "cached"}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(HERE / "stage_c_closed_loop.py")],
        env=_env_for(acq, pool, K, seed),
        check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    wall = time.time() - t0
    if proc.returncode != 0 or not summary_path.exists():
        return {"acq": acq, "pool": pool, "K": K, "seed": seed,
                "status": "failed", "wall": wall,
                "tail": proc.stdout.decode("utf-8", errors="replace")[-1500:]}
    return {"acq": acq, "pool": pool, "K": K, "seed": seed,
            "status": "ok", "wall": wall}


def _welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "md": float("nan"), "se": float("nan")}
    va = a.var(ddof=1) / len(a); vb = b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    md = float(a.mean() - b.mean())
    return {"t": md / max(se, 1e-30), "md": md, "se": se}


def aggregate() -> dict:
    per_cfg: dict[tuple, dict] = {}
    for acq in ACQS:
        for pool in POOLS:
            for K in K_CTX_GRID:
                contractions, mles_d0, mles_d1, contractions_d0 = [], [], [], []
                seeds_ok = []
                for s in range(N_SEEDS):
                    seed = SEED0 + s
                    p = _chain_dir(acq, pool, K, seed) / "summary.json"
                    if not p.is_file():
                        continue
                    with open(p) as f:
                        d = json.load(f)
                    contractions.append(d["contraction_final_d1"])
                    contractions_d0.append(d["contraction_final_d0"])
                    mles_d0.append(d["mle_err_d0"])
                    mles_d1.append(d["mle_err_d1"])
                    seeds_ok.append(seed)
                if not contractions:
                    continue
                per_cfg[(pool, K, acq)] = {
                    "n_seeds": len(contractions),
                    "seeds": seeds_ok,
                    "contraction_d0_mean": float(np.mean(contractions_d0)),
                    "contraction_d0_std": float(np.std(contractions_d0, ddof=1)) if len(contractions_d0) > 1 else 0.0,
                    "contraction_d1_mean": float(np.mean(contractions)),
                    "contraction_d1_std": float(np.std(contractions, ddof=1)) if len(contractions) > 1 else 0.0,
                    "mle_err_d0_mean": float(np.mean(mles_d0)),
                    "mle_err_d0_std": float(np.std(mles_d0, ddof=1)) if len(mles_d0) > 1 else 0.0,
                    "mle_err_d1_mean": float(np.mean(mles_d1)),
                    "mle_err_d1_std": float(np.std(mles_d1, ddof=1)) if len(mles_d1) > 1 else 0.0,
                }

    # Welch vs random for each (pool, K).
    by_pool_K: dict[tuple, dict[str, dict]] = {}
    for (pool, K, acq), entry in per_cfg.items():
        by_pool_K.setdefault((pool, K), {})[acq] = entry

    for (pool, K), entries in by_pool_K.items():
        if "random" not in entries:
            continue
        ref_contraction = np.array([
            json.load(open(_chain_dir("random", pool, K, s).joinpath("summary.json")))["contraction_final_d1"]
            for s in entries["random"]["seeds"]])
        ref_mle = np.array([
            json.load(open(_chain_dir("random", pool, K, s).joinpath("summary.json")))["mle_err_d1"]
            for s in entries["random"]["seeds"]])
        for acq, e in entries.items():
            if acq == "random":
                continue
            arr_c = np.array([
                json.load(open(_chain_dir(acq, pool, K, s).joinpath("summary.json")))["contraction_final_d1"]
                for s in e["seeds"]])
            arr_m = np.array([
                json.load(open(_chain_dir(acq, pool, K, s).joinpath("summary.json")))["mle_err_d1"]
                for s in e["seeds"]])
            e["welch_contraction_d1_vs_random"] = _welch(arr_c, ref_contraction)
            e["welch_mle_d1_vs_random"] = _welch(arr_m, ref_mle)

    return {"per_cfg": [{"pool": k[0], "K": k[1], "acq": k[2], **v}
                         for k, v in per_cfg.items()]}


def main():
    jobs = [(acq, pool, K, SEED0 + s)
            for acq in ACQS for pool in POOLS for K in K_CTX_GRID
            for s in range(N_SEEDS)]
    max_workers = int(MAX_WORKERS_ENV) if MAX_WORKERS_ENV else max(1, min(16, len(jobs)))
    print(f"# Stage C closed-loop parallel sweep: {len(jobs)} chains, "
          f"{max_workers} concurrent, {THREADS_PER_CHAIN} threads each")
    t0 = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            if r["status"] == "failed":
                print(f"  FAILED {r['acq']}/{r['pool']}/K{r['K']}/s{r['seed']}: "
                      f"wall={r.get('wall', 0):.1f}s — tail:\n{r.get('tail', '')[:500]}")
            elif n_done % 25 == 0 or n_done == len(jobs):
                el = time.time() - t0
                print(f"  {n_done}/{len(jobs)} done in {el:.0f}s "
                      f"(est total {el / n_done * len(jobs):.0f}s)")
    wall = time.time() - t0
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n# done in {wall:.0f}s  (ok={len(results) - n_failed}, failed={n_failed})")

    agg = aggregate()
    out_path = OUT / "stage_c_closed_loop_aggregate.json"
    with open(out_path, "w") as f:
        json.dump({
            "wall_seconds": wall, "n_jobs": len(jobs),
            "config": {
                "POOLS": list(POOLS), "K_CTX_GRID": list(K_CTX_GRID),
                "ACQS": list(ACQS), "N_SEEDS": N_SEEDS,
                "SEED0": SEED0, "N_CYCLES": N_CYCLES},
            "aggregate": agg,
        }, f, indent=2, default=float)
    print(f"# wrote {out_path}")

    # Headline table.
    by_pool_K: dict[tuple, list[dict]] = {}
    for row in agg["per_cfg"]:
        by_pool_K.setdefault((row["pool"], row["K"]), []).append(row)
    for (pool, K), rows in sorted(by_pool_K.items()):
        print(f"\n# pool={pool} K={K} (n_seeds per acq printed in parens)")
        ref = next((r for r in rows if r["acq"] == "random"), None)
        rows.sort(key=lambda r: r["contraction_d1_mean"])
        for r in rows:
            t_c = r.get("welch_contraction_d1_vs_random", {}).get("t", float("nan"))
            t_m = r.get("welch_mle_d1_vs_random", {}).get("t", float("nan"))
            print(f"  {r['acq']:14s}  contraction_d1 = {r['contraction_d1_mean']:.4f} "
                  f"± {r['contraction_d1_std']:.4f}   "
                  f"mle_d1 = {r['mle_err_d1_mean']:.4f} "
                  f"± {r['mle_err_d1_std']:.4f}   "
                  f"t_c={t_c:+.2f}  t_m={t_m:+.2f}   (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
