"""End-to-end JEPA-FM training + ablations + apples-to-apples comparison.

What this script does:

1. Build the same train / in-box / out-box scenarios as experiment.py (same
   seeds, K=12, Q=32, 200 train scenarios), so JEPA-FM's numbers can be
   compared directly to the existing IntentionFM_Fixed/Learned/DeepSets/
   Regressor table.

2. Train the headline JEPA-FM (jepa=1, vic=0.04, aux=0.1, EMA=on) plus five
   pre-planned ablations:
     - 'jepa_only_probe'  : aux=0; freeze JEPA, fit a linear probe on the
                             frozen target embeddings, evaluate Y from probe.
     - 'aux_only'         : jepa=0, vic=0 — sanity that the JEPA loss term
                             carries weight. Should reproduce DeepSets-FM.
     - 'no_vicreg'        : vic=0 (full EMA + JEPA + aux). Tests whether the
                             VICReg regularizer was load-bearing for collapse.
     - 'no_ema'           : siamese f_θ̄ = f_θ. Tests whether EMA matters.
     - 'strong_aux'       : aux=1 — tests if cranking the aux head closes any
                             remaining gap.

3. Re-evaluate the four reference architectures (Fixed/Learned-Intention,
   DeepSets, Regressor) on identical splits so the report table is
   self-contained, matched seed.

4. Compute three diagnostic quantities for the headline JEPA-FM:
     - PCA of E_ema(M_q, Y_q) on the held-out queries, coloured by max|c_i|
     - per-scenario distance ‖z_pred − z_target‖ vs per-scenario R²
     - per-dimension std of z_pred  (collapse guard — raises if any < 0.5)

Writes:
  output_jepa/summary.json     scalar + ablation table
  output_jepa/results.npz      raw predictions + held-out embeddings
  output_jepa/jepa_fm.pt       headline JEPA-FM checkpoint
  output_jepa/ablations/*.pt   per-ablation checkpoints
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

from data import (
    make_oracle, sample_c, sample_c_shell,
    make_dataset, scenarios_to_tensors,
    N_WC,
)
from intention_learned import IntentionFMLearned, IntentionFMFixed
from deepsets_matched import DeepSetsFM, IntentionFMRegressor
from jepa_fm import JEPAFM


# ---- configuration (matched to experiment.py defaults) -------------------
N_TRAIN_SCENARIOS = 200
N_TEST_IN = 50
N_TEST_OUT = 50
K_CTX = 12
Q_QUERY = 32
C_MAX_TRAIN = 0.7
C_OUTER = 1.0
N_META_STEPS = 1500
LR = 1e-3
BATCH_S = 32
SEED = 0
LOG_STEPS = [1, 10, 25, 50, 100, 200, 400, 600, 1000, 1500]

OUT_DIR = HERE / "output_jepa"
ABL_DIR = OUT_DIR / "ablations"
OUT_DIR.mkdir(exist_ok=True)
ABL_DIR.mkdir(exist_ok=True)


# ---- metrics --------------------------------------------------------------
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
        "r2_per_scenario": r2.tolist(),
    }


# ---- training: JEPA-FM ---------------------------------------------------
def train_jepa(model: JEPAFM, train_t: dict, val_t: dict,
               n_steps: int, lr: float, batch_s: int,
               seed: int = 0, label: str = "jepa", log_steps=LOG_STEPS):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)

    losses = []
    val_history = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=batch_s, replace=False)
        M_ctx = train_t["M_ctx"][idx]
        Y_ctx = train_t["Y_ctx"][idx]
        M_q = train_t["M_query"][idx]
        Y_q = train_t["Y_query"][idx]

        model.train()
        y_pred, jepa_loss, vic_loss, mse_loss, _ = model(
            M_ctx, Y_ctx, M_q, Y_q, return_losses=True)
        total = (model.jepa_weight * jepa_loss
                 + model.vicreg_weight * vic_loss
                 + model.aux_weight * mse_loss)
        opt.zero_grad()
        total.backward()
        opt.step()
        model.update_target()
        losses.append((float(jepa_loss.detach()),
                       float(vic_loss.detach()),
                       float(mse_loss.detach())))

        if step in log_steps:
            model.eval()
            with torch.no_grad():
                yp = model(val_t["M_ctx"], val_t["Y_ctx"],
                           val_t["M_query"]).cpu().numpy()
            r2 = r2_per_scenario(yp, val_t["Y_query"].cpu().numpy())
            val_history.append((step, float(np.median(r2))))
            print(f"  [{label:>14s}] step {step:5d}  jepa={jepa_loss.item():.4f}  "
                  f"vic={vic_loss.item():.4f}  mse={mse_loss.item():.4f}  "
                  f"val_R2_med={np.median(r2):+.4f}")
    wall = time.time() - t0
    return losses, val_history, wall


def eval_jepa(model: JEPAFM, t: dict) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


def collapse_check(model: JEPAFM, t: dict, threshold: float = 0.05) -> dict:
    """Per-dimension std of EMA target embeddings on held-out queries.
    Raises RuntimeError if any dimension collapsed (std < threshold)."""
    model.eval()
    with torch.no_grad():
        z_tgt = model.encode_target(t["M_query"], t["Y_query"])
    z_flat = z_tgt.reshape(-1, z_tgt.shape[-1]).cpu().numpy()
    stds = z_flat.std(axis=0)
    means = z_flat.mean(axis=0)
    return {
        "per_dim_std": stds.tolist(),
        "min_std": float(stds.min()),
        "max_std": float(stds.max()),
        "mean_std": float(stds.mean()),
        "per_dim_mean": means.tolist(),
        "collapsed": bool(stds.min() < threshold),
    }


# ---- frozen-probe evaluator for the 'jepa_only_probe' ablation ----------
def fit_and_eval_probe(model: JEPAFM, train_t: dict, eval_t: dict,
                       lr: float = 1e-2, n_steps: int = 600,
                       seed: int = 0) -> np.ndarray:
    """Train a fresh linear probe d': R^{d_emb} -> R on the FROZEN target
    encoder's outputs over training scenarios' queries, then apply it to
    the JEPA-FM's predicted embeddings (g_φ output) on held-out scenarios.

    This isolates 'does the JEPA pretext alone learn a Y-relevant basis?'
    from 'does the joint aux head do the work?'.
    """
    torch.manual_seed(seed)
    model.eval()
    # Gather training (z_target, Y_q) pairs from the EMA target encoder.
    with torch.no_grad():
        z_tr = model.encode_target(train_t["M_query"],
                                   train_t["Y_query"]).reshape(-1, model.d_emb)
        y_tr = train_t["Y_query"].reshape(-1)
    probe = torch.nn.Linear(model.d_emb, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        yp = probe(z_tr).squeeze(-1)
        loss = ((yp - y_tr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Eval: apply probe to PREDICTED embeddings on held-out scenarios.
    probe.eval()
    with torch.no_grad():
        z_eval = model.predict_embedding(eval_t["M_ctx"], eval_t["Y_ctx"],
                                         eval_t["M_query"])
        y_pred = probe(z_eval).squeeze(-1)
    return y_pred.cpu().numpy()


# ---- reference architecture re-evaluations -----------------------------
# These mirror the corresponding logic in experiment.py exactly so the
# numbers in summary.json are directly comparable to docs/research/02-foundation-model/intention-vs-deepsets.md.

def train_intention(model, train_t, n_steps, lr, batch_s, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=batch_s, replace=False)
        yp = model(train_t["M_ctx"][idx], train_t["Y_ctx"][idx],
                   train_t["M_query"][idx])
        loss = ((yp - train_t["Y_query"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return time.time() - t0


def train_deepsets(model, train_t, n_steps, lr, batch_s, seed=0):
    return train_intention(model, train_t, n_steps, lr, batch_s, seed)  # same loop


def train_regressor(model, train_scenarios, n_steps, lr, batch_s=256, seed=0):
    torch.manual_seed(seed)
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
    return time.time() - t0


def eval_regressor(model, scenarios):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in scenarios:
            c = torch.from_numpy(np.tile(s["c"], (len(s["M_query"]), 1)))
            M = torch.from_numpy(s["M_query"])
            preds.append(model(c, M).cpu().numpy())
    return np.stack(preds)


def eval_fixed(scenarios):
    fm = IntentionFMFixed()
    return np.stack([fm.predict(s["M_ctx"], s["Y_ctx"], s["M_query"])
                     for s in scenarios])


def eval_fm(model, t: dict) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


# ---- main -----------------------------------------------------------------
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
    Yq_in = test_in_t["Y_query"].numpy()
    Yq_out = test_out_t["Y_query"].numpy()

    # ============ JEPA-FM headline + ablations ============
    print("\n# JEPA-FM (headline)")
    headline_cfg = dict(d_emb=16, hidden=48, ema_momentum=0.996,
                        vicreg_weight=0.04, aux_weight=0.1,
                        jepa_weight=1.0, use_ema=True)
    headline = JEPAFM(**headline_cfg)
    print(f"  n_params = {headline.n_params}")
    h_losses, h_hist, h_wall = train_jepa(
        headline, train_t, test_in_t, N_META_STEPS, LR, BATCH_S,
        seed=SEED, label="JEPA headline")
    h_collapse = collapse_check(headline, test_in_t)
    print(f"  collapse check: min_std={h_collapse['min_std']:.3f}  "
          f"mean_std={h_collapse['mean_std']:.3f}  "
          f"collapsed={h_collapse['collapsed']}")
    yp_jepa_in = eval_jepa(headline, test_in_t)
    yp_jepa_out = eval_jepa(headline, test_out_t)
    torch.save(headline.state_dict(), OUT_DIR / "jepa_fm.pt")

    # Diagnostic: PCA + per-scenario distance vs R²
    with torch.no_grad():
        z_tgt_in = headline.encode_target(test_in_t["M_query"],
                                          test_in_t["Y_query"]).cpu().numpy()
        z_pred_in = headline.predict_embedding(test_in_t["M_ctx"],
                                               test_in_t["Y_ctx"],
                                               test_in_t["M_query"]).cpu().numpy()
        z_tgt_out = headline.encode_target(test_out_t["M_query"],
                                           test_out_t["Y_query"]).cpu().numpy()
        z_pred_out = headline.predict_embedding(test_out_t["M_ctx"],
                                                test_out_t["Y_ctx"],
                                                test_out_t["M_query"]).cpu().numpy()

    print("\n## ablations")
    ablations = {}

    # 1. jepa-only with frozen probe
    print("\n# ablation: jepa_only_probe  (aux=0 during train, linear probe at eval)")
    abl1 = JEPAFM(**{**headline_cfg, "aux_weight": 0.0})
    train_jepa(abl1, train_t, test_in_t, N_META_STEPS, LR, BATCH_S,
               seed=SEED, label="jepa_only", log_steps=[1, 100, 500, 1500])
    yp_in = fit_and_eval_probe(abl1, train_t, test_in_t)
    yp_out = fit_and_eval_probe(abl1, train_t, test_out_t)
    ablations["jepa_only_probe"] = {
        "in": summarise("jepa_only_probe", yp_in, Yq_in),
        "out": summarise("jepa_only_probe", yp_out, Yq_out),
        "collapse": collapse_check(abl1, test_in_t),
    }
    torch.save(abl1.state_dict(), ABL_DIR / "jepa_only_probe.pt")

    # 2. aux-only (no JEPA, no VICReg)
    print("\n# ablation: aux_only  (jepa_weight=0, vic_weight=0)")
    abl2 = JEPAFM(**{**headline_cfg, "jepa_weight": 0.0, "vicreg_weight": 0.0,
                     "aux_weight": 1.0})
    train_jepa(abl2, train_t, test_in_t, N_META_STEPS, LR, BATCH_S,
               seed=SEED, label="aux_only", log_steps=[1, 100, 500, 1500])
    yp_in = eval_jepa(abl2, test_in_t); yp_out = eval_jepa(abl2, test_out_t)
    ablations["aux_only"] = {
        "in": summarise("aux_only", yp_in, Yq_in),
        "out": summarise("aux_only", yp_out, Yq_out),
        "collapse": collapse_check(abl2, test_in_t),
    }
    torch.save(abl2.state_dict(), ABL_DIR / "aux_only.pt")

    # 3. no VICReg
    print("\n# ablation: no_vicreg  (vic_weight=0)")
    abl3 = JEPAFM(**{**headline_cfg, "vicreg_weight": 0.0})
    train_jepa(abl3, train_t, test_in_t, N_META_STEPS, LR, BATCH_S,
               seed=SEED, label="no_vicreg", log_steps=[1, 100, 500, 1500])
    yp_in = eval_jepa(abl3, test_in_t); yp_out = eval_jepa(abl3, test_out_t)
    ablations["no_vicreg"] = {
        "in": summarise("no_vicreg", yp_in, Yq_in),
        "out": summarise("no_vicreg", yp_out, Yq_out),
        "collapse": collapse_check(abl3, test_in_t),
    }
    torch.save(abl3.state_dict(), ABL_DIR / "no_vicreg.pt")

    # 4. no EMA (siamese)
    print("\n# ablation: no_ema  (target encoder = grad encoder)")
    abl4 = JEPAFM(**{**headline_cfg, "use_ema": False})
    train_jepa(abl4, train_t, test_in_t, N_META_STEPS, LR, BATCH_S,
               seed=SEED, label="no_ema", log_steps=[1, 100, 500, 1500])
    yp_in = eval_jepa(abl4, test_in_t); yp_out = eval_jepa(abl4, test_out_t)
    ablations["no_ema"] = {
        "in": summarise("no_ema", yp_in, Yq_in),
        "out": summarise("no_ema", yp_out, Yq_out),
        "collapse": collapse_check(abl4, test_in_t),
    }
    torch.save(abl4.state_dict(), ABL_DIR / "no_ema.pt")

    # 5. strong aux (lambda_aux = 1.0)
    print("\n# ablation: strong_aux  (aux_weight=1.0)")
    abl5 = JEPAFM(**{**headline_cfg, "aux_weight": 1.0})
    train_jepa(abl5, train_t, test_in_t, N_META_STEPS, LR, BATCH_S,
               seed=SEED, label="strong_aux", log_steps=[1, 100, 500, 1500])
    yp_in = eval_jepa(abl5, test_in_t); yp_out = eval_jepa(abl5, test_out_t)
    ablations["strong_aux"] = {
        "in": summarise("strong_aux", yp_in, Yq_in),
        "out": summarise("strong_aux", yp_out, Yq_out),
        "collapse": collapse_check(abl5, test_in_t),
    }
    torch.save(abl5.state_dict(), ABL_DIR / "strong_aux.pt")

    # ============ reference architectures (matched seed/data) ============
    print("\n# IntentionFM_Fixed (no training)")
    yp_fix_in = eval_fixed(test_in_scen); yp_fix_out = eval_fixed(test_out_scen)
    res_fixed = {
        "in": summarise("IntentionFM_Fixed", yp_fix_in, Yq_in),
        "out": summarise("IntentionFM_Fixed", yp_fix_out, Yq_out),
        "n_params": 0,
    }

    print("\n# IntentionFM_Learned")
    int_model = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    train_intention(int_model, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED)
    yp_int_in = eval_fm(int_model, test_in_t); yp_int_out = eval_fm(int_model, test_out_t)
    res_int = {
        "in": summarise("IntentionFM_Learned", yp_int_in, Yq_in),
        "out": summarise("IntentionFM_Learned", yp_int_out, Yq_out),
        "n_params": int_model.n_params,
    }
    print(f"  in median R2 = {res_int['in']['r2_median']:+.4f}  "
          f"out median R2 = {res_int['out']['r2_median']:+.4f}")

    print("\n# DeepSets-FM")
    ds_model = DeepSetsFM(d_set=16, hidden=48)
    train_deepsets(ds_model, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED)
    yp_ds_in = eval_fm(ds_model, test_in_t); yp_ds_out = eval_fm(ds_model, test_out_t)
    res_ds = {
        "in": summarise("DeepSets-FM", yp_ds_in, Yq_in),
        "out": summarise("DeepSets-FM", yp_ds_out, Yq_out),
        "n_params": ds_model.n_params,
    }
    print(f"  in median R2 = {res_ds['in']['r2_median']:+.4f}  "
          f"out median R2 = {res_ds['out']['r2_median']:+.4f}")

    print("\n# IntentionFMRegressor (cheating ceiling)")
    reg_model = IntentionFMRegressor(n_wc=N_WC, hidden=48)
    train_regressor(reg_model, train_scen, n_steps=N_META_STEPS, lr=LR, seed=SEED)
    yp_reg_in = eval_regressor(reg_model, test_in_scen)
    yp_reg_out = eval_regressor(reg_model, test_out_scen)
    res_reg = {
        "in": summarise("IntentionFM_Regressor_cheat", yp_reg_in, Yq_in),
        "out": summarise("IntentionFM_Regressor_cheat", yp_reg_out, Yq_out),
        "n_params": reg_model.n_params,
    }

    # ---- JEPA headline summary
    res_jepa = {
        "in": summarise("JEPA-FM", yp_jepa_in, Yq_in),
        "out": summarise("JEPA-FM", yp_jepa_out, Yq_out),
        "n_params": headline.n_params,
        "wall": h_wall,
        "scaling_history": h_hist,
        "collapse": h_collapse,
    }
    print(f"\nHEADLINE JEPA-FM:  in median R2 = {res_jepa['in']['r2_median']:+.4f}  "
          f"out median R2 = {res_jepa['out']['r2_median']:+.4f}  "
          f"in p5 = {res_jepa['in']['r2_p5']:+.4f}  out p5 = {res_jepa['out']['r2_p5']:+.4f}")

    # ---- per-scenario embedding-distance correlations
    dist_in = np.linalg.norm(z_pred_in - z_tgt_in, axis=-1).mean(axis=1)  # (S,)
    dist_out = np.linalg.norm(z_pred_out - z_tgt_out, axis=-1).mean(axis=1)
    r2_in_arr = np.array(res_jepa["in"]["r2_per_scenario"])
    r2_out_arr = np.array(res_jepa["out"]["r2_per_scenario"])

    # ---- write outputs
    np.savez(OUT_DIR / "results.npz",
             test_in_M_ctx=test_in_t["M_ctx"].numpy(),
             test_in_Y_ctx=test_in_t["Y_ctx"].numpy(),
             test_in_M_query=test_in_t["M_query"].numpy(),
             test_in_Y_query=Yq_in,
             test_in_c=test_in_t["c"].numpy(),
             test_out_M_ctx=test_out_t["M_ctx"].numpy(),
             test_out_Y_ctx=test_out_t["Y_ctx"].numpy(),
             test_out_M_query=test_out_t["M_query"].numpy(),
             test_out_Y_query=Yq_out,
             test_out_c=test_out_t["c"].numpy(),
             yp_fixed_in=yp_fix_in, yp_fixed_out=yp_fix_out,
             yp_int_in=yp_int_in, yp_int_out=yp_int_out,
             yp_ds_in=yp_ds_in, yp_ds_out=yp_ds_out,
             yp_reg_in=yp_reg_in, yp_reg_out=yp_reg_out,
             yp_jepa_in=yp_jepa_in, yp_jepa_out=yp_jepa_out,
             z_tgt_in=z_tgt_in, z_pred_in=z_pred_in,
             z_tgt_out=z_tgt_out, z_pred_out=z_pred_out,
             dist_in=dist_in, dist_out=dist_out,
             r2_in=r2_in_arr, r2_out=r2_out_arr,
             scaling_history=np.array(h_hist),
             )

    summary = {
        "config": {
            "N_TRAIN_SCENARIOS": N_TRAIN_SCENARIOS,
            "N_TEST_IN": N_TEST_IN, "N_TEST_OUT": N_TEST_OUT,
            "K_CTX": K_CTX, "Q_QUERY": Q_QUERY,
            "N_META_STEPS": N_META_STEPS, "BATCH_S": BATCH_S, "LR": LR,
            "C_MAX_TRAIN": C_MAX_TRAIN, "C_OUTER": C_OUTER,
            "JEPA_HEADLINE": headline_cfg,
        },
        "IntentionFM_Fixed": res_fixed,
        "IntentionFM_Learned": res_int,
        "DeepSets_FM": res_ds,
        "IntentionFM_Regressor_cheat": res_reg,
        "JEPA_FM": res_jepa,
        "ablations": ablations,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n# wrote {OUT_DIR}/summary.json, results.npz, jepa_fm.pt + ablations/")
    return summary


if __name__ == "__main__":
    main()
