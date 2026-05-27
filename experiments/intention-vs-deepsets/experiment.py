"""End-to-end Intention-vs-DeepSets FM comparison experiment.

Pipeline:
  1. Build train / inside-test / outside-test scenario sets.
  2. Train IntentionFM_Learned (closed-form attention with learned psi).
  3. Train DeepSets-FM (mean-pool encoder + decoder MLP).
  4. Train IntentionFMRegressor (c on the forward pass) for ceiling.
  5. Evaluate all four (including the fixed-psi closed-form baseline) on
     the two held-out sets.
  6. Save results to /tmp/fm_compare/results.npz and a JSON dump.

Run::
   /tmp/fm_check/bin/python /tmp/fm_compare/experiment.py
"""
from __future__ import annotations
import json
import os
import sys
import time
sys.path.insert(0, "/tmp/fm_compare")

import numpy as np
import torch

from data import (
    make_oracle, sample_c, sample_c_shell,
    make_dataset, scenarios_to_tensors,
    N_WC, M_RANGE,
)
from intention_learned import IntentionFMLearned, IntentionFMFixed
from deepsets_matched import DeepSetsFM, IntentionFMRegressor


# --------- configuration ---------
N_TRAIN_SCENARIOS = 200
N_TEST_IN = 50
N_TEST_OUT = 50
K_CTX = 12
Q_QUERY = 32
C_MAX_TRAIN = 0.7
C_OUTER = 1.0

N_META_STEPS = 1500
LR = 1e-3
BATCH_S = 32           # # scenarios per gradient step
SEED = 0

LOG_STEPS = [1, 10, 25, 50, 100, 200, 400, 600, 1000, 1500]


# --------- metric helpers ---------
def r2_per_scenario(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """y_pred, y_true: (S, Q). Returns (S,) per-scenario R^2."""
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def mse_total(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def summarise(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    r2 = r2_per_scenario(y_pred, y_true)
    return {
        "name": name,
        "r2_median": float(np.median(r2)),
        "r2_p5": float(np.percentile(r2, 5)),
        "r2_mean": float(np.mean(r2)),
        "mse": mse_total(y_pred, y_true),
        "n_scenarios": int(len(r2)),
        "r2_per_scenario": r2.tolist(),
    }


# --------- training: Intention learned-psi ---------
def train_intention(model: IntentionFMLearned,
                    train_t: dict, val_t: dict,
                    n_steps: int, lr: float, batch_s: int,
                    log_steps=LOG_STEPS, seed: int = 0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)

    losses = []
    val_r2_history = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=batch_s, replace=False)
        M_ctx = train_t["M_ctx"][idx]
        Y_ctx = train_t["Y_ctx"][idx]
        M_q = train_t["M_query"][idx]
        Y_q = train_t["Y_query"][idx]
        y_pred = model(M_ctx, Y_ctx, M_q)
        loss = ((y_pred - Y_q) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step in log_steps:
            model.eval()
            with torch.no_grad():
                yp = model(val_t["M_ctx"], val_t["Y_ctx"], val_t["M_query"]).cpu().numpy()
            r2 = r2_per_scenario(yp, val_t["Y_query"].cpu().numpy())
            val_r2_history.append((step, float(np.median(r2))))
            print(f"  [intention] step {step:5d}  loss={loss.item():.6f}  val_R2_median={np.median(r2):+.4f}")
            model.train()
    wall = time.time() - t0
    return losses, val_r2_history, wall


# --------- training: DeepSets ---------
def train_deepsets(model: DeepSetsFM,
                   train_t: dict, val_t: dict,
                   n_steps: int, lr: float, batch_s: int,
                   log_steps=LOG_STEPS, seed: int = 0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)

    losses = []
    val_r2_history = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=batch_s, replace=False)
        M_ctx = train_t["M_ctx"][idx]
        Y_ctx = train_t["Y_ctx"][idx]
        M_q = train_t["M_query"][idx]
        Y_q = train_t["Y_query"][idx]
        y_pred = model(M_ctx, Y_ctx, M_q)
        loss = ((y_pred - Y_q) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step in log_steps:
            model.eval()
            with torch.no_grad():
                yp = model(val_t["M_ctx"], val_t["Y_ctx"], val_t["M_query"]).cpu().numpy()
            r2 = r2_per_scenario(yp, val_t["Y_query"].cpu().numpy())
            val_r2_history.append((step, float(np.median(r2))))
            print(f"  [deepsets ] step {step:5d}  loss={loss.item():.6f}  val_R2_median={np.median(r2):+.4f}")
            model.train()
    wall = time.time() - t0
    return losses, val_r2_history, wall


# --------- training: ceiling regressor (c on forward pass) ---------
def train_regressor(model: IntentionFMRegressor,
                    train_scenarios: list[dict],
                    n_steps: int, lr: float, batch_s: int = 256,
                    seed: int = 0):
    """Flatten training scenarios into (c, m, y) triples and fit."""
    torch.manual_seed(seed)
    # gather all (c, M, Y) from contexts AND queries
    Cs, Ms, Ys = [], [], []
    for s in train_scenarios:
        c = s["c"]
        for k in range(len(s["M_ctx"])):
            Cs.append(c); Ms.append(s["M_ctx"][k]); Ys.append(s["Y_ctx"][k])
        for k in range(len(s["M_query"])):
            Cs.append(c); Ms.append(s["M_query"][k]); Ys.append(s["Y_query"][k])
    C = torch.from_numpy(np.stack(Cs))
    M = torch.from_numpy(np.array(Ms, dtype=np.float32))
    Y = torch.from_numpy(np.array(Ys, dtype=np.float32))
    N = len(M)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(N, size=batch_s, replace=False)
        yp = model(C[idx], M[idx])
        loss = ((yp - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  [ceil    ] step {step:5d}  loss={loss.item():.6f}")
    wall = time.time() - t0
    return wall


def eval_regressor(model: IntentionFMRegressor, scenarios: list[dict]) -> np.ndarray:
    """For each scenario, predict at M_query using the *known* c (cheating).
    Returns (S, Q)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in scenarios:
            c = torch.from_numpy(np.tile(s["c"], (len(s["M_query"]), 1)))
            M = torch.from_numpy(s["M_query"])
            yp = model(c, M).cpu().numpy()
            preds.append(yp)
    return np.stack(preds)


def eval_fixed(scenarios: list[dict]) -> np.ndarray:
    fm = IntentionFMFixed()
    preds = []
    for s in scenarios:
        yp = fm.predict(s["M_ctx"], s["Y_ctx"], s["M_query"])
        preds.append(yp)
    return np.stack(preds)


def eval_fm(model, t: dict) -> np.ndarray:
    """For batched evaluator. Returns (S, Q)."""
    model.eval()
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


# --------- main ---------
def main():
    print("# Build datasets")
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train_scen = make_dataset(N_TRAIN_SCENARIOS,
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
    train_t = scenarios_to_tensors(train_scen)
    test_in_t = scenarios_to_tensors(test_in_scen)
    test_out_t = scenarios_to_tensors(test_out_scen)

    # ---- 1. IntentionFM_Fixed (no training) ----
    print("\n# IntentionFM_Fixed (no training)")
    yp_fixed_in = eval_fixed(test_in_scen)
    yp_fixed_out = eval_fixed(test_out_scen)
    res_fixed = {
        "in": summarise("IntentionFM_Fixed", yp_fixed_in, test_in_t["Y_query"].numpy()),
        "out": summarise("IntentionFM_Fixed", yp_fixed_out, test_out_t["Y_query"].numpy()),
        "n_params": 0,
        "wall": 0.0,
    }
    print(f"  in : median R2 = {res_fixed['in']['r2_median']:+.4f}  p5 = {res_fixed['in']['r2_p5']:+.4f}")
    print(f"  out: median R2 = {res_fixed['out']['r2_median']:+.4f}  p5 = {res_fixed['out']['r2_p5']:+.4f}")

    # ---- 2. IntentionFM_Learned ----
    print("\n# IntentionFM_Learned (closed-form attention, learned psi)")
    int_model = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    print(f"  n_params = {int_model.n_params}")
    int_losses, int_hist, int_wall = train_intention(
        int_model, train_t, test_in_t, N_META_STEPS, LR, BATCH_S, seed=SEED)
    yp_int_in = eval_fm(int_model, test_in_t)
    yp_int_out = eval_fm(int_model, test_out_t)
    res_int = {
        "in": summarise("IntentionFM_Learned", yp_int_in, test_in_t["Y_query"].numpy()),
        "out": summarise("IntentionFM_Learned", yp_int_out, test_out_t["Y_query"].numpy()),
        "n_params": int_model.n_params,
        "wall": int_wall,
        "scaling_history": int_hist,
        "losses": int_losses,
    }
    print(f"  in : median R2 = {res_int['in']['r2_median']:+.4f}  p5 = {res_int['in']['r2_p5']:+.4f}")
    print(f"  out: median R2 = {res_int['out']['r2_median']:+.4f}  p5 = {res_int['out']['r2_p5']:+.4f}")

    # ---- 3. DeepSets-FM ----
    print("\n# DeepSets-FM (mean-pool encoder, decoder MLP)")
    ds_model = DeepSetsFM(d_set=16, hidden=48)
    print(f"  n_params = {ds_model.n_params}")
    ds_losses, ds_hist, ds_wall = train_deepsets(
        ds_model, train_t, test_in_t, N_META_STEPS, LR, BATCH_S, seed=SEED)
    yp_ds_in = eval_fm(ds_model, test_in_t)
    yp_ds_out = eval_fm(ds_model, test_out_t)
    res_ds = {
        "in": summarise("DeepSets-FM", yp_ds_in, test_in_t["Y_query"].numpy()),
        "out": summarise("DeepSets-FM", yp_ds_out, test_out_t["Y_query"].numpy()),
        "n_params": ds_model.n_params,
        "wall": ds_wall,
        "scaling_history": ds_hist,
        "losses": ds_losses,
    }
    print(f"  in : median R2 = {res_ds['in']['r2_median']:+.4f}  p5 = {res_ds['in']['r2_p5']:+.4f}")
    print(f"  out: median R2 = {res_ds['out']['r2_median']:+.4f}  p5 = {res_ds['out']['r2_p5']:+.4f}")

    # ---- 4. Ceiling: IntentionFMRegressor (c on forward pass) ----
    print("\n# Ceiling: IntentionFMRegressor (c on forward pass) [the FORBIDDEN architecture]")
    reg_model = IntentionFMRegressor(n_wc=N_WC, hidden=48)
    print(f"  n_params = {reg_model.n_params}")
    reg_wall = train_regressor(reg_model, train_scen, n_steps=N_META_STEPS, lr=LR, seed=SEED)
    yp_reg_in = eval_regressor(reg_model, test_in_scen)
    yp_reg_out = eval_regressor(reg_model, test_out_scen)
    res_reg = {
        "in": summarise("IntentionFM_Regressor_cheat", yp_reg_in, test_in_t["Y_query"].numpy()),
        "out": summarise("IntentionFM_Regressor_cheat", yp_reg_out, test_out_t["Y_query"].numpy()),
        "n_params": reg_model.n_params,
        "wall": reg_wall,
    }
    print(f"  in : median R2 = {res_reg['in']['r2_median']:+.4f}  p5 = {res_reg['in']['r2_p5']:+.4f}")
    print(f"  out: median R2 = {res_reg['out']['r2_median']:+.4f}  p5 = {res_reg['out']['r2_p5']:+.4f}")

    # ---- save raw predictions for the plotting script ----
    np.savez("/tmp/fm_compare/results.npz",
             test_in_M_ctx=test_in_t["M_ctx"].numpy(),
             test_in_Y_ctx=test_in_t["Y_ctx"].numpy(),
             test_in_M_query=test_in_t["M_query"].numpy(),
             test_in_Y_query=test_in_t["Y_query"].numpy(),
             test_in_c=test_in_t["c"].numpy(),
             test_out_M_ctx=test_out_t["M_ctx"].numpy(),
             test_out_Y_ctx=test_out_t["Y_ctx"].numpy(),
             test_out_M_query=test_out_t["M_query"].numpy(),
             test_out_Y_query=test_out_t["Y_query"].numpy(),
             test_out_c=test_out_t["c"].numpy(),
             yp_fixed_in=yp_fixed_in, yp_fixed_out=yp_fixed_out,
             yp_int_in=yp_int_in, yp_int_out=yp_int_out,
             yp_ds_in=yp_ds_in, yp_ds_out=yp_ds_out,
             yp_reg_in=yp_reg_in, yp_reg_out=yp_reg_out,
             )
    # Save psi-basis snapshot for plot 4.
    torch.save(int_model.state_dict(), "/tmp/fm_compare/intention_learned.pt")
    torch.save(ds_model.state_dict(), "/tmp/fm_compare/deepsets.pt")
    torch.save(reg_model.state_dict(), "/tmp/fm_compare/regressor.pt")
    # Save scalar summary.
    summary = {
        "config": {
            "N_TRAIN_SCENARIOS": N_TRAIN_SCENARIOS,
            "N_TEST_IN": N_TEST_IN, "N_TEST_OUT": N_TEST_OUT,
            "K_CTX": K_CTX, "Q_QUERY": Q_QUERY,
            "N_META_STEPS": N_META_STEPS, "BATCH_S": BATCH_S, "LR": LR,
            "C_MAX_TRAIN": C_MAX_TRAIN, "C_OUTER": C_OUTER,
        },
        "IntentionFM_Fixed": res_fixed,
        "IntentionFM_Learned": {k: v for k, v in res_int.items() if k != "losses"},
        "DeepSets_FM": {k: v for k, v in res_ds.items() if k != "losses"},
        "IntentionFM_Regressor_cheat": res_reg,
    }
    with open("/tmp/fm_compare/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n# wrote /tmp/fm_compare/results.npz, summary.json, *.pt")
    return summary


if __name__ == "__main__":
    main()
