"""Parallel driver for the FM-representation Stage C sweep (T0.2).

Mirrors stage_c_parallel.py but drives stage_c_fm.py over the cartesian
product (rep x acq x pool x K_ctx x seed), aggregates per (rep, pool, K, acq)
with Welch-t vs the matched random arm on BOTH contraction and MLE, and carries
the FM-representation |cos_{A^-1}| collinearity number through to the aggregate.

Override via env:
  REPS=full,def_ext   POOLS=P0   K_CTX_GRID=12   N_SEEDS=5   N_CYCLES=50
  ACQS=random,leverage,epig,param_epig_d,param_epig_a   MAX_WORKERS=8

Validation default (fast end-to-end): REPS=full POOLS=P0 K_CTX_GRID=12
N_SEEDS=5 N_CYCLES=50. Scale-up: REPS=full,def_ext POOLS=P0,P2
K_CTX_GRID=12,48 N_SEEDS=15 N_CYCLES=200.
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

REPS = tuple(s.strip() for s in os.environ.get("REPS", "full").split(",") if s.strip())
POOLS = tuple(s.strip() for s in os.environ.get("POOLS", "P0").split(",") if s.strip())
K_CTX_GRID = tuple(int(s) for s in os.environ.get("K_CTX_GRID", "12").split(",") if s.strip())
ACQS = tuple(s.strip() for s in os.environ.get("ACQS",
    "random,leverage,epig,param_epig_d,param_epig_a").split(",") if s.strip())
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
SEED0 = int(os.environ.get("SEED0", "2026"))
N_CYCLES = int(os.environ.get("N_CYCLES", "50"))
THREADS_PER_CHAIN = int(os.environ.get("THREADS_PER_CHAIN", "1"))
MAX_WORKERS_ENV = os.environ.get("MAX_WORKERS", "8").strip()


def _chain_dir(rep, acq, pool, K, seed) -> Path:
    return OUT / f"fm_{rep}_{pool}_K{K}_N{N_CYCLES}_{acq}_s{seed}"


def _env_for(rep, acq, pool, K, seed) -> dict:
    e = os.environ.copy()
    e.update(REP=rep, ACQ=acq, POOL=pool, K_CTX=str(K), SEED=str(seed),
             N_CYCLES=str(N_CYCLES), PHOENIX_TRACING="0")
    t = str(THREADS_PER_CHAIN)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        e[k] = t
    return e


def _run_one(args) -> dict:
    rep, acq, pool, K, seed = args
    sp = _chain_dir(rep, acq, pool, K, seed) / "summary.json"
    if sp.exists():
        return {"rep": rep, "acq": acq, "pool": pool, "K": K, "seed": seed,
                "status": "cached"}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(HERE / "stage_c_fm.py")],
        env=_env_for(rep, acq, pool, K, seed),
        check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if proc.returncode != 0 or not sp.exists():
        return {"rep": rep, "acq": acq, "pool": pool, "K": K, "seed": seed,
                "status": "failed", "wall": wall,
                "tail": proc.stdout.decode("utf-8", errors="replace")[-2000:]}
    return {"rep": rep, "acq": acq, "pool": pool, "K": K, "seed": seed,
            "status": "ok", "wall": wall}


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "md": float("nan"), "se": float("nan"),
                "rel_pct": float("nan")}
    va = a.var(ddof=1) / len(a); vb = b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb)); md = float(a.mean() - b.mean())
    rel = 100.0 * md / max(abs(float(b.mean())), 1e-30)
    return {"t": md / max(se, 1e-30), "md": md, "se": se, "rel_pct": rel}


def _load(rep, acq, pool, K, seeds, key):
    out = []
    for s in seeds:
        p = _chain_dir(rep, acq, pool, K, s) / "summary.json"
        if p.is_file():
            out.append(json.load(open(p))[key])
    return np.array(out, dtype=float)


def aggregate() -> dict:
    per_cfg = {}
    for rep in REPS:
        for acq in ACQS:
            for pool in POOLS:
                for K in K_CTX_GRID:
                    seeds = [SEED0 + s for s in range(N_SEEDS)
                             if (_chain_dir(rep, acq, pool, K, SEED0 + s)
                                 / "summary.json").is_file()]
                    if not seeds:
                        continue
                    cd0 = _load(rep, acq, pool, K, seeds, "contraction_final_d0")
                    cd1 = _load(rep, acq, pool, K, seeds, "contraction_final_d1")
                    m0 = _load(rep, acq, pool, K, seeds, "mle_err_d0")
                    m1 = _load(rep, acq, pool, K, seeds, "mle_err_d1")
                    cos = _load(rep, acq, pool, K, seeds, "cos_Ainv_top_mean")
                    per_cfg[(rep, pool, K, acq)] = {
                        "n_seeds": len(seeds), "seeds": seeds,
                        "contraction_d0_mean": float(cd0.mean()),
                        "contraction_d1_mean": float(cd1.mean()),
                        "contraction_d1_std": float(cd1.std(ddof=1)) if len(cd1) > 1 else 0.0,
                        "mle_err_d0_mean": float(m0.mean()),
                        "mle_err_d1_mean": float(m1.mean()),
                        "mle_err_d1_std": float(m1.std(ddof=1)) if len(m1) > 1 else 0.0,
                        "cos_Ainv_top_mean": float(cos.mean()),
                        "cos_Ainv_top_std": float(cos.std(ddof=1)) if len(cos) > 1 else 0.0,
                    }
    # Welch vs random per (rep, pool, K).
    groups = {}
    for (rep, pool, K, acq), e in per_cfg.items():
        groups.setdefault((rep, pool, K), {})[acq] = e
    for (rep, pool, K), entries in groups.items():
        if "random" not in entries:
            continue
        rseeds = entries["random"]["seeds"]
        ref_c = _load(rep, "random", pool, K, rseeds, "contraction_final_d1")
        ref_m = _load(rep, "random", pool, K, rseeds, "mle_err_d1")
        for acq, e in entries.items():
            if acq == "random":
                continue
            ac = _load(rep, acq, pool, K, e["seeds"], "contraction_final_d1")
            am = _load(rep, acq, pool, K, e["seeds"], "mle_err_d1")
            e["welch_contraction_d1_vs_random"] = _welch(ac, ref_c)
            e["welch_mle_d1_vs_random"] = _welch(am, ref_m)
    return {"per_cfg": [{"rep": k[0], "pool": k[1], "K": k[2], "acq": k[3], **v}
                        for k, v in per_cfg.items()]}


def main():
    jobs = [(rep, acq, pool, K, SEED0 + s)
            for rep in REPS for acq in ACQS for pool in POOLS
            for K in K_CTX_GRID for s in range(N_SEEDS)]
    mw = int(MAX_WORKERS_ENV) if MAX_WORKERS_ENV else max(1, min(8, len(jobs)))
    print(f"# FM Stage C sweep: {len(jobs)} chains, {mw} concurrent, "
          f"{THREADS_PER_CHAIN} threads each  (REPS={REPS} POOLS={POOLS} "
          f"K={K_CTX_GRID} N_SEEDS={N_SEEDS} N_CYCLES={N_CYCLES})")
    t0 = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            if r["status"] == "failed":
                print(f"  FAILED {r['rep']}/{r['acq']}/{r['pool']}/K{r['K']}/"
                      f"s{r['seed']} — tail:\n{r.get('tail','')[:1200]}")
            elif n_done % 10 == 0 or n_done == len(jobs):
                el = time.time() - t0
                print(f"  {n_done}/{len(jobs)} in {el:.0f}s "
                      f"(est {el/n_done*len(jobs):.0f}s)")
    wall = time.time() - t0
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n# done in {wall:.0f}s (ok={len(results)-n_failed}, "
          f"failed={n_failed})")

    agg = aggregate()
    out_path = OUT / "stage_c_fm_aggregate.json"
    with open(out_path, "w") as f:
        json.dump({"wall_seconds": wall, "n_jobs": len(jobs),
                   "config": {"REPS": list(REPS), "POOLS": list(POOLS),
                              "K_CTX_GRID": list(K_CTX_GRID), "ACQS": list(ACQS),
                              "N_SEEDS": N_SEEDS, "SEED0": SEED0,
                              "N_CYCLES": N_CYCLES},
                   "aggregate": agg}, f, indent=2, default=float)
    print(f"# wrote {out_path}")

    by = {}
    for row in agg["per_cfg"]:
        by.setdefault((row["rep"], row["pool"], row["K"]), []).append(row)
    for (rep, pool, K), rows in sorted(by.items()):
        print(f"\n# rep={rep} pool={pool} K={K}")
        rows.sort(key=lambda r: r["contraction_d1_mean"])
        for r in rows:
            tc = r.get("welch_contraction_d1_vs_random", {})
            tm = r.get("welch_mle_d1_vs_random", {})
            print(f"  {r['acq']:14s} contr_d1={r['contraction_d1_mean']:.4f}"
                  f"±{r['contraction_d1_std']:.4f}  "
                  f"mle_d1={r['mle_err_d1_mean']:.4f}±{r['mle_err_d1_std']:.4f}  "
                  f"cos={r['cos_Ainv_top_mean']:.4f}  "
                  f"t_c={tc.get('t', float('nan')):+.2f} "
                  f"(md={tc.get('md', float('nan')):+.4f},{tc.get('rel_pct', float('nan')):+.1f}%)  "
                  f"t_m={tm.get('t', float('nan')):+.2f} "
                  f"(md={tm.get('md', float('nan')):+.4f},{tm.get('rel_pct', float('nan')):+.1f}%)  "
                  f"(n={r['n_seeds']})")


if __name__ == "__main__":
    main()
