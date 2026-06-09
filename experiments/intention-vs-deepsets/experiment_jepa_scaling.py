"""JEPA-FM scaling study.

Tests whether JEPA closes the gap to IntentionFM_Learned when scaled along
each of four axes independently, and all together:

  - more DATA           (N_train scenarios: 200 -> 2000 -> 10000)
  - more PARAMS         (small DeepSets -> transformer small -> transformer large)
  - more TRAINING TIME  (1500 -> 6000 -> 20000 meta-steps)
  - ARCHITECTURE        (DeepSets-style vs transformer; T-JEPA-faithful)

Plus the T-JEPA-native recipe:
  - PRETRAIN-then-PROBE (train JEPA with aux=0, freeze, fit linear probe on
                         frozen embeddings — this is what T-JEPA actually does)

For each JEPA configuration, the same data/step budget is used to train
IntentionFM_Learned (matched-scale) as the comparison reference.

Writes:
  output_jepa_scaling/summary.json    config + metrics for every run
  output_jepa_scaling/results.npz     predictions per run (for plotting)
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import (make_oracle, sample_c, sample_c_shell, make_dataset,
                  scenarios_to_tensors, N_WC)
from intention_learned import IntentionFMLearned
from deepsets_matched import DeepSetsFM
from jepa_fm import JEPAFM


# ---- fixed scenario / split config (shared with experiment.py) ----------
K_CTX = 12
Q_QUERY = 32
N_TEST_IN = 100
N_TEST_OUT = 100
C_MAX_TRAIN = 0.7
C_OUTER = 1.0
LR = 1e-3
BATCH_S = 32
SEED = 0

OUT_DIR = HERE / "output_jepa_scaling"
OUT_DIR.mkdir(exist_ok=True)


# ---- shared helpers ------------------------------------------------------
def r2_per_scenario(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def summarise(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    r2 = r2_per_scenario(y_pred, y_true)
    return {
        "name": name,
        "r2_median": float(np.median(r2)),
        "r2_p5": float(np.percentile(r2, 5)),
        "r2_mean": float(np.mean(r2)),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "n_scenarios": int(len(r2)),
    }


# ---- dataset cache: avoid regenerating scenarios across runs ------------
_DATASET_CACHE: dict[int, dict] = {}


def get_datasets(n_train: int):
    """Returns (train_t, test_in_t, test_out_t, test_in_scen, test_out_scen,
    train_scen). Held-out splits are fixed across runs; train scales with
    n_train. Cached by n_train."""
    if n_train in _DATASET_CACHE:
        return _DATASET_CACHE[n_train]
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train_scen = make_dataset(n_train,
                              lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
                              oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=1)
    test_in_scen = make_dataset(N_TEST_IN,
                                lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
                                oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=42)
    test_out_scen = make_dataset(N_TEST_OUT,
                                 lambda r: sample_c_shell(r, 1,
                                                          c_inner=C_MAX_TRAIN,
                                                          c_outer=C_OUTER)[0],
                                 oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=43)
    bundle = {
        "train_scen": train_scen, "test_in_scen": test_in_scen,
        "test_out_scen": test_out_scen,
        "train_t": scenarios_to_tensors(train_scen),
        "test_in_t": scenarios_to_tensors(test_in_scen),
        "test_out_t": scenarios_to_tensors(test_out_scen),
    }
    _DATASET_CACHE[n_train] = bundle
    return bundle


# ---- training loops ------------------------------------------------------
def train_jepa(model: JEPAFM, train_t, n_steps, lr, batch_s, seed=0,
               label="jepa", log_every=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    log_steps = set()
    if log_every:
        log_steps = {n_steps * k // 5 for k in range(1, 6)}
    last_jepa = 0.0
    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=min(batch_s, S_train), replace=False)
        M_ctx = train_t["M_ctx"][idx]; Y_ctx = train_t["Y_ctx"][idx]
        M_q = train_t["M_query"][idx]; Y_q = train_t["Y_query"][idx]
        model.train()
        y_pred, jepa_loss, vic_loss, mse_loss, _ = model(
            M_ctx, Y_ctx, M_q, Y_q, return_losses=True)
        total = (model.jepa_weight * jepa_loss
                 + model.vicreg_weight * vic_loss
                 + model.aux_weight * mse_loss)
        opt.zero_grad(); total.backward(); opt.step()
        model.update_target()
        last_jepa = float(jepa_loss.detach())
        if step in log_steps:
            print(f"    [{label}] step {step:6d}  jepa={last_jepa:.4f}  "
                  f"vic={float(vic_loss.detach()):.3f}  "
                  f"mse={float(mse_loss.detach()):.4f}", flush=True)
    return time.time() - t0


def train_intention(model: IntentionFMLearned, train_t, n_steps, lr, batch_s,
                    seed=0, label="int"):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=min(batch_s, S_train), replace=False)
        yp = model(train_t["M_ctx"][idx], train_t["Y_ctx"][idx],
                   train_t["M_query"][idx])
        loss = ((yp - train_t["Y_query"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return time.time() - t0


def eval_jepa(model, t):
    model.eval()
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


def eval_int(model, t):
    model.eval()
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


def fit_probe_and_eval(model: JEPAFM, train_t, eval_in_t, eval_out_t,
                       lr=1e-2, n_steps=1000, seed=0):
    """T-JEPA-style downstream evaluation: fit a fresh linear probe
    d': R^{d_emb} -> R on FROZEN target embeddings of training scenarios,
    apply to PREDICTED embeddings on held-out scenarios."""
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        z_tr = model.encode_target(train_t["M_query"],
                                   train_t["Y_query"]).reshape(-1, model.d_emb)
        y_tr = train_t["Y_query"].reshape(-1)
    probe = torch.nn.Linear(model.d_emb, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        loss = ((probe(z_tr).squeeze(-1) - y_tr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        z_in = model.predict_embedding(
            eval_in_t["M_ctx"], eval_in_t["Y_ctx"], eval_in_t["M_query"])
        z_out = model.predict_embedding(
            eval_out_t["M_ctx"], eval_out_t["Y_ctx"], eval_out_t["M_query"])
        yp_in = probe(z_in).squeeze(-1).cpu().numpy()
        yp_out = probe(z_out).squeeze(-1).cpu().numpy()
    return yp_in, yp_out


def collapse_stats(model: JEPAFM, t) -> dict:
    model.eval()
    with torch.no_grad():
        z = model.encode_target(t["M_query"], t["Y_query"])
    z = z.reshape(-1, z.shape[-1]).cpu().numpy()
    return {"min_std": float(z.std(axis=0).min()),
            "mean_std": float(z.std(axis=0).mean()),
            "rank99": int(np.sum(np.linalg.svd(z - z.mean(0), compute_uv=False) > 0.01))}


# ---- run definitions ------------------------------------------------------
def run_jepa(cfg: dict) -> dict:
    bundle = get_datasets(cfg["n_train"])
    model = JEPAFM(
        d_emb=cfg.get("d_emb", 16),
        hidden=cfg.get("hidden", 48),
        encoder=cfg.get("encoder", "deepsets"),
        n_layers=cfg.get("n_layers", 2),
        n_heads=cfg.get("n_heads", 4),
        ema_momentum=cfg.get("ema_momentum", 0.996),
        vicreg_weight=cfg.get("vicreg_weight", 0.04),
        aux_weight=cfg.get("aux_weight", 0.1),
        jepa_weight=cfg.get("jepa_weight", 1.0),
        use_ema=cfg.get("use_ema", True),
    )
    n_params = model.n_params
    wall = train_jepa(model, bundle["train_t"], cfg["n_steps"], LR, BATCH_S,
                      seed=SEED, label=cfg["name"], log_every=True)
    yp_in = eval_jepa(model, bundle["test_in_t"])
    yp_out = eval_jepa(model, bundle["test_out_t"])
    Yq_in = bundle["test_in_t"]["Y_query"].numpy()
    Yq_out = bundle["test_out_t"]["Y_query"].numpy()
    res = {
        "kind": "jepa",
        "config": cfg,
        "n_params": n_params,
        "wall_s": wall,
        "in": summarise(cfg["name"], yp_in, Yq_in),
        "out": summarise(cfg["name"], yp_out, Yq_out),
        "collapse_in": collapse_stats(model, bundle["test_in_t"]),
    }
    # If aux_weight == 0, also do the T-JEPA probe evaluation.
    if cfg.get("aux_weight", 0.1) == 0.0:
        yp_in_p, yp_out_p = fit_probe_and_eval(
            model, bundle["train_t"],
            bundle["test_in_t"], bundle["test_out_t"])
        res["in_probe"] = summarise(cfg["name"] + "_probe", yp_in_p, Yq_in)
        res["out_probe"] = summarise(cfg["name"] + "_probe", yp_out_p, Yq_out)
    res["yp_in"] = yp_in; res["yp_out"] = yp_out
    return res


def run_intention(cfg: dict) -> dict:
    bundle = get_datasets(cfg["n_train"])
    model = IntentionFMLearned(d_psi=cfg.get("d_psi", 16),
                               hidden=cfg.get("hidden", 64),
                               alpha=cfg.get("alpha", 1e-3))
    n_params = model.n_params
    wall = train_intention(model, bundle["train_t"], cfg["n_steps"], LR, BATCH_S,
                           seed=SEED, label=cfg["name"])
    yp_in = eval_int(model, bundle["test_in_t"])
    yp_out = eval_int(model, bundle["test_out_t"])
    Yq_in = bundle["test_in_t"]["Y_query"].numpy()
    Yq_out = bundle["test_out_t"]["Y_query"].numpy()
    return {
        "kind": "intention",
        "config": cfg, "n_params": n_params, "wall_s": wall,
        "in": summarise(cfg["name"], yp_in, Yq_in),
        "out": summarise(cfg["name"], yp_out, Yq_out),
        "yp_in": yp_in, "yp_out": yp_out,
    }


# ---- the actual sweep ----------------------------------------------------
JEPA_RUNS = [
    # ---------- baseline (reference, same as experiment_jepa.py headline) -
    dict(name="A_baseline_jepa",
         n_train=200, d_emb=16, hidden=48, encoder="deepsets", n_steps=1500),
    # ---------- +DATA --------------------------------------------------------
    dict(name="B_data_2k_jepa",
         n_train=2000, d_emb=16, hidden=48, encoder="deepsets", n_steps=1500),
    dict(name="B_data_10k_jepa",
         n_train=10000, d_emb=16, hidden=48, encoder="deepsets", n_steps=4000),
    # ---------- +PARAMS  ----------------------------------------------------
    dict(name="C_params_50k_jepa_tx",
         n_train=200, d_emb=32, hidden=64, encoder="transformer",
         n_layers=2, n_heads=4, n_steps=1500),
    dict(name="C_params_200k_jepa_tx",
         n_train=200, d_emb=64, hidden=128, encoder="transformer",
         n_layers=2, n_heads=4, n_steps=1500),
    # ---------- +TIME -------------------------------------------------------
    dict(name="D_time_6k_jepa",
         n_train=200, d_emb=16, hidden=48, encoder="deepsets", n_steps=6000),
    dict(name="D_time_20k_jepa",
         n_train=200, d_emb=16, hidden=48, encoder="deepsets", n_steps=20000),
    # ---------- ALL THREE: data + params + time  ---------------------------
    dict(name="E_scale_all_jepa_tx",
         n_train=2000, d_emb=64, hidden=128, encoder="transformer",
         n_layers=2, n_heads=4, n_steps=6000),
    dict(name="E_scale_all_jepa_tx_big",
         n_train=10000, d_emb=64, hidden=128, encoder="transformer",
         n_layers=2, n_heads=4, n_steps=10000),
    # ---------- PRETRAIN-then-probe (T-JEPA's native recipe at scale) ------
    dict(name="F_pretrain_probe_jepa_tx",
         n_train=10000, d_emb=64, hidden=128, encoder="transformer",
         n_layers=2, n_heads=4, n_steps=10000, aux_weight=0.0),
]

INTENTION_RUNS = [
    dict(name="A_baseline_int",     n_train=200,   d_psi=16, hidden=64,  n_steps=1500),
    dict(name="B_data_2k_int",      n_train=2000,  d_psi=16, hidden=64,  n_steps=1500),
    dict(name="B_data_10k_int",     n_train=10000, d_psi=16, hidden=64,  n_steps=4000),
    dict(name="C_params_50k_int",   n_train=200,   d_psi=32, hidden=128, n_steps=1500),
    dict(name="C_params_200k_int",  n_train=200,   d_psi=64, hidden=256, n_steps=1500),
    dict(name="D_time_6k_int",      n_train=200,   d_psi=16, hidden=64,  n_steps=6000),
    dict(name="D_time_20k_int",     n_train=200,   d_psi=16, hidden=64,  n_steps=20000),
    dict(name="E_scale_all_int",    n_train=2000,  d_psi=64, hidden=256, n_steps=6000),
    dict(name="E_scale_all_int_big",n_train=10000, d_psi=64, hidden=256, n_steps=10000),
]


def _save_incremental(results: dict, t_start: float) -> None:
    """Write summary.json and results.npz after each run, so a kill mid-sweep
    doesn't lose all progress."""
    summary = {}
    npz_dict = {}
    for name, r in results.items():
        summary[name] = {k: v for k, v in r.items()
                         if k not in ("yp_in", "yp_out")}
        if "yp_in" in r and "yp_out" in r:
            npz_dict[name + "__yp_in"] = r["yp_in"]
            npz_dict[name + "__yp_out"] = r["yp_out"]
    if results:
        # use the largest n_train bundle for the held-out grid (same splits for all)
        any_cfg = next(iter(results.values()))["config"]
        bundle = get_datasets(any_cfg["n_train"])
        npz_dict["test_in_M_q"] = bundle["test_in_t"]["M_query"].numpy()
        npz_dict["test_in_Y_q"] = bundle["test_in_t"]["Y_query"].numpy()
        npz_dict["test_out_M_q"] = bundle["test_out_t"]["M_query"].numpy()
        npz_dict["test_out_Y_q"] = bundle["test_out_t"]["Y_query"].numpy()
        np.savez(OUT_DIR / "results.npz", **npz_dict)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _load_existing() -> dict:
    """Resume support: read prior summary.json if present (drops the yp_*
    arrays, but those will be re-derived from results.npz if needed)."""
    p = OUT_DIR / "summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    existing = _load_existing()
    if existing:
        print(f"### Resuming: {len(existing)} runs already in summary.json")
    results = {}
    # Re-hydrate prior results' scalar parts (yp_* arrays come back from .npz if needed)
    for k, v in existing.items():
        results[k] = dict(v)
    t_start = time.time()
    total = len(JEPA_RUNS) + len(INTENTION_RUNS)
    print(f"### Targeting {total} runs ({len(JEPA_RUNS)} JEPA + {len(INTENTION_RUNS)} Intention)")

    for cfg in JEPA_RUNS:
        if cfg["name"] in results and "in" in results[cfg["name"]]:
            print(f"\n## SKIP (done): {cfg['name']}")
            continue
        print(f"\n## {cfg['name']}  "
              f"(n_train={cfg['n_train']}, encoder={cfg.get('encoder','deepsets')}, "
              f"d_emb={cfg.get('d_emb')}, hidden={cfg.get('hidden')}, "
              f"steps={cfg['n_steps']})")
        t0 = time.time()
        r = run_jepa(cfg)
        elapsed = time.time() - t0
        print(f"  -> params={r['n_params']:>8d}  "
              f"in_med={r['in']['r2_median']:+.4f}  "
              f"in_p5={r['in']['r2_p5']:+.4f}  "
              f"out_med={r['out']['r2_median']:+.4f}  "
              f"out_p5={r['out']['r2_p5']:+.4f}  "
              f"min_std={r['collapse_in']['min_std']:.3f}  "
              f"wall={elapsed:.1f}s  cumulative={time.time()-t_start:.0f}s",
              flush=True)
        results[cfg["name"]] = r
        _save_incremental(results, t_start)

    for cfg in INTENTION_RUNS:
        if cfg["name"] in results and "in" in results[cfg["name"]]:
            print(f"\n## SKIP (done): {cfg['name']}")
            continue
        print(f"\n## {cfg['name']}  (n_train={cfg['n_train']}, d_psi={cfg['d_psi']}, "
              f"hidden={cfg['hidden']}, steps={cfg['n_steps']})")
        t0 = time.time()
        r = run_intention(cfg)
        elapsed = time.time() - t0
        print(f"  -> params={r['n_params']:>8d}  "
              f"in_med={r['in']['r2_median']:+.4f}  "
              f"out_med={r['out']['r2_median']:+.4f}  "
              f"wall={elapsed:.1f}s  cumulative={time.time()-t_start:.0f}s",
              flush=True)
        results[cfg["name"]] = r
        _save_incremental(results, t_start)

    print(f"\n# wrote {OUT_DIR}/summary.json + results.npz  "
          f"(total {time.time()-t_start:.0f}s)")
    return results


if __name__ == "__main__":
    main()
