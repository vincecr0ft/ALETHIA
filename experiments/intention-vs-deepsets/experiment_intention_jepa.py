"""IntentionJEPAFM vs the existing FM zoo, on identical data splits.

Trains IntentionJEPAFM at two scales (matched, scaled) and compares against
the four established architectures (IntentionFM_Fixed, IntentionFM_Learned,
DeepSets_FM, JEPA_FM matched) plus the optional JEPA-scaled config. All runs
use the same seed, splits, K, Q and step budget, so the numbers go directly
into the architecture-comparison table.

Writes:
  output_intention_jepa/summary.json   per-architecture metrics + ablations
  output_intention_jepa/results.npz    raw predictions + held-out tensors
  output_intention_jepa/*.pt           model checkpoints (one per architecture)
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
from intention_jepa_fm import IntentionJEPAFM


# ----------------------------------------------------------------------------
# Configuration — matched to experiment_jepa.py defaults so the numbers join
# the existing comparison table without re-running anything that already ran.
# ----------------------------------------------------------------------------
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
LOG_STEPS = [1, 25, 100, 400, 1000, 1500]

OUT_DIR = HERE / "output_intention_jepa"
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Metrics — same per-scenario R² definition as experiment_jepa.py.
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Training loops.
# ----------------------------------------------------------------------------
def train_jepa_like(model, train_t, n_steps, lr, batch_s, seed=0,
                    label="ijepa", log_steps=LOG_STEPS):
    """Train any JEPA-FM-style model (JEPAFM or IntentionJEPAFM)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    S_train = train_t["M_ctx"].size(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    val_history = []
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
        if step in log_steps:
            print(f"    [{label:>22s}] step {step:5d}  "
                  f"jepa={jepa_loss.item():.4f}  vic={vic_loss.item():.4f}  "
                  f"mse={mse_loss.item():.4f}", flush=True)
            val_history.append((step,
                                float(jepa_loss.detach()),
                                float(mse_loss.detach())))
    return time.time() - t0, val_history


def train_intention_supervised(model, train_t, n_steps, lr, batch_s, seed=0,
                                label="int"):
    """Closed-form ridge head trained by MSE through linalg.solve."""
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
        if step in LOG_STEPS:
            print(f"    [{label:>22s}] step {step:5d}  mse={loss.item():.4f}",
                  flush=True)
    return time.time() - t0


def train_regressor(model, train_scenarios, n_steps, lr, batch_s=256, seed=0):
    """Per-event MLP that gets c on the forward pass — the cheat ceiling."""
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


# ----------------------------------------------------------------------------
# Evaluation.
# ----------------------------------------------------------------------------
def eval_fm(model, t):
    model.eval()
    with torch.no_grad():
        return model(t["M_ctx"], t["Y_ctx"], t["M_query"]).cpu().numpy()


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


def collapse_check(model, t, threshold=0.05):
    """Per-dim std of EMA target embeddings — guard against representation
    collapse (the standard JEPA pathology)."""
    model.eval()
    with torch.no_grad():
        z = model.encode_target(t["M_query"], t["Y_query"])
    z = z.reshape(-1, z.shape[-1]).cpu().numpy()
    stds = z.std(axis=0)
    return {
        "min_std": float(stds.min()),
        "mean_std": float(stds.mean()),
        "max_std": float(stds.max()),
        "per_dim_std": stds.tolist(),
        "collapsed": bool(stds.min() < threshold),
    }


# ----------------------------------------------------------------------------
# IntentionJEPAFM configs.
# ----------------------------------------------------------------------------
IJEPA_MATCHED = dict(
    d_psi=16, d_emb=16, psi_hidden=32, encode_hidden=24,
    n_layers=1, n_heads=4, ff_mult=2, predictor_hidden=24,
    decoder_hidden=16,
    ema_momentum=0.996, vicreg_weight=0.04, aux_weight=0.1,
    jepa_weight=1.0, use_ema=True,
)
IJEPA_SCALED = dict(
    d_psi=32, d_emb=64, psi_hidden=64, encode_hidden=64,
    n_layers=2, n_heads=8, ff_mult=4, predictor_hidden=64,
    decoder_hidden=32,
    ema_momentum=0.996, vicreg_weight=0.04, aux_weight=0.1,
    jepa_weight=1.0, use_ema=True,
)


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    print(f"# Build datasets (n_train={N_TRAIN_SCENARIOS}, K={K_CTX}, Q={Q_QUERY})")
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

    results = {}

    # ----- IntentionFM_Fixed (no training) --------------------------------
    print("\n# IntentionFM_Fixed")
    yp_in = eval_fixed(test_in_scen); yp_out = eval_fixed(test_out_scen)
    results["IntentionFM_Fixed"] = {
        "n_params": 0, "wall_s": 0.0,
        "in": summarise("IntentionFM_Fixed", yp_in, Yq_in),
        "out": summarise("IntentionFM_Fixed", yp_out, Yq_out),
    }

    # ----- IntentionFM_Learned (closed-form ridge + supervised MSE) -------
    print("\n# IntentionFM_Learned (closed-form ridge + supervised MSE)")
    int_model = IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3)
    print(f"  n_params = {int_model.n_params}")
    wall = train_intention_supervised(
        int_model, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED, label="int")
    yp_in = eval_fm(int_model, test_in_t); yp_out = eval_fm(int_model, test_out_t)
    results["IntentionFM_Learned"] = {
        "n_params": int_model.n_params, "wall_s": wall,
        "in": summarise("IntentionFM_Learned", yp_in, Yq_in),
        "out": summarise("IntentionFM_Learned", yp_out, Yq_out),
    }
    torch.save(int_model.state_dict(), OUT_DIR / "intention_fm_learned.pt")
    print(f"  -> in med={results['IntentionFM_Learned']['in']['r2_median']:+.4f}  "
          f"out med={results['IntentionFM_Learned']['out']['r2_median']:+.4f}")

    # ----- DeepSets_FM -----------------------------------------------------
    print("\n# DeepSets_FM (mean-pool aggregator, supervised MSE)")
    ds_model = DeepSetsFM(d_set=16, hidden=48)
    print(f"  n_params = {ds_model.n_params}")
    wall = train_intention_supervised(
        ds_model, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED, label="deepsets")
    yp_in = eval_fm(ds_model, test_in_t); yp_out = eval_fm(ds_model, test_out_t)
    results["DeepSets_FM"] = {
        "n_params": ds_model.n_params, "wall_s": wall,
        "in": summarise("DeepSets_FM", yp_in, Yq_in),
        "out": summarise("DeepSets_FM", yp_out, Yq_out),
    }
    torch.save(ds_model.state_dict(), OUT_DIR / "deepsets_fm.pt")
    print(f"  -> in med={results['DeepSets_FM']['in']['r2_median']:+.4f}  "
          f"out med={results['DeepSets_FM']['out']['r2_median']:+.4f}")

    # ----- JEPA_FM matched ------------------------------------------------
    print("\n# JEPA_FM matched (T-JEPA with generic phi(m,y) per-event encoder)")
    jepa = JEPAFM(d_emb=16, hidden=48, ema_momentum=0.996,
                  vicreg_weight=0.04, aux_weight=0.1, jepa_weight=1.0,
                  use_ema=True, encoder="deepsets")
    print(f"  n_params = {jepa.n_params}")
    wall, _ = train_jepa_like(
        jepa, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED, label="jepa-matched")
    yp_in = eval_fm(jepa, test_in_t); yp_out = eval_fm(jepa, test_out_t)
    results["JEPA_FM_matched"] = {
        "n_params": jepa.n_params, "wall_s": wall,
        "in": summarise("JEPA_FM_matched", yp_in, Yq_in),
        "out": summarise("JEPA_FM_matched", yp_out, Yq_out),
        "collapse": collapse_check(jepa, test_in_t),
    }
    torch.save(jepa.state_dict(), OUT_DIR / "jepa_matched.pt")
    print(f"  -> in med={results['JEPA_FM_matched']['in']['r2_median']:+.4f}  "
          f"out med={results['JEPA_FM_matched']['out']['r2_median']:+.4f}  "
          f"min_std={results['JEPA_FM_matched']['collapse']['min_std']:.3f}")

    # ----- JEPA_FM scaled (transformer encoder, ~200k params) -------------
    print("\n# JEPA_FM scaled (transformer encoder over (m,y) pairs)")
    jepa_s = JEPAFM(d_emb=64, hidden=128, ema_momentum=0.996,
                    vicreg_weight=0.04, aux_weight=0.1, jepa_weight=1.0,
                    use_ema=True, encoder="transformer",
                    n_layers=2, n_heads=4, ff_mult=4)
    print(f"  n_params = {jepa_s.n_params}")
    wall, _ = train_jepa_like(
        jepa_s, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED, label="jepa-scaled")
    yp_in = eval_fm(jepa_s, test_in_t); yp_out = eval_fm(jepa_s, test_out_t)
    results["JEPA_FM_scaled"] = {
        "n_params": jepa_s.n_params, "wall_s": wall,
        "in": summarise("JEPA_FM_scaled", yp_in, Yq_in),
        "out": summarise("JEPA_FM_scaled", yp_out, Yq_out),
        "collapse": collapse_check(jepa_s, test_in_t),
    }
    torch.save(jepa_s.state_dict(), OUT_DIR / "jepa_scaled.pt")
    print(f"  -> in med={results['JEPA_FM_scaled']['in']['r2_median']:+.4f}  "
          f"out med={results['JEPA_FM_scaled']['out']['r2_median']:+.4f}  "
          f"min_std={results['JEPA_FM_scaled']['collapse']['min_std']:.3f}")

    # ----- IntentionJEPAFM matched ----------------------------------------
    print("\n# IntentionJEPAFM matched (T-JEPA with Intention-basis per-event encoder)")
    ijepa = IntentionJEPAFM(**IJEPA_MATCHED)
    print(f"  n_params = {ijepa.n_params}")
    wall, hist = train_jepa_like(
        ijepa, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED, label="ijepa-matched")
    yp_in = eval_fm(ijepa, test_in_t); yp_out = eval_fm(ijepa, test_out_t)
    results["IntentionJEPAFM_matched"] = {
        "n_params": ijepa.n_params, "wall_s": wall,
        "in": summarise("IntentionJEPAFM_matched", yp_in, Yq_in),
        "out": summarise("IntentionJEPAFM_matched", yp_out, Yq_out),
        "collapse": collapse_check(ijepa, test_in_t),
        "config": IJEPA_MATCHED,
        "loss_history": hist,
    }
    torch.save(ijepa.state_dict(), OUT_DIR / "intention_jepa_matched.pt")
    print(f"  -> in med={results['IntentionJEPAFM_matched']['in']['r2_median']:+.4f}  "
          f"out med={results['IntentionJEPAFM_matched']['out']['r2_median']:+.4f}  "
          f"min_std={results['IntentionJEPAFM_matched']['collapse']['min_std']:.3f}")

    # ----- IntentionJEPAFM scaled -----------------------------------------
    print("\n# IntentionJEPAFM scaled (~200k params)")
    ijepa_s = IntentionJEPAFM(**IJEPA_SCALED)
    print(f"  n_params = {ijepa_s.n_params}")
    wall, hist = train_jepa_like(
        ijepa_s, train_t, N_META_STEPS, LR, BATCH_S, seed=SEED, label="ijepa-scaled")
    yp_in = eval_fm(ijepa_s, test_in_t); yp_out = eval_fm(ijepa_s, test_out_t)
    results["IntentionJEPAFM_scaled"] = {
        "n_params": ijepa_s.n_params, "wall_s": wall,
        "in": summarise("IntentionJEPAFM_scaled", yp_in, Yq_in),
        "out": summarise("IntentionJEPAFM_scaled", yp_out, Yq_out),
        "collapse": collapse_check(ijepa_s, test_in_t),
        "config": IJEPA_SCALED,
        "loss_history": hist,
    }
    torch.save(ijepa_s.state_dict(), OUT_DIR / "intention_jepa_scaled.pt")
    print(f"  -> in med={results['IntentionJEPAFM_scaled']['in']['r2_median']:+.4f}  "
          f"out med={results['IntentionJEPAFM_scaled']['out']['r2_median']:+.4f}  "
          f"min_std={results['IntentionJEPAFM_scaled']['collapse']['min_std']:.3f}")

    # ----- IntentionFMRegressor (cheat ceiling) ---------------------------
    print("\n# IntentionFM_Regressor_cheat (sees c on forward pass)")
    reg_model = IntentionFMRegressor(n_wc=N_WC, hidden=48)
    wall = train_regressor(reg_model, train_scen, n_steps=N_META_STEPS, lr=LR, seed=SEED)
    yp_in = eval_regressor(reg_model, test_in_scen)
    yp_out = eval_regressor(reg_model, test_out_scen)
    results["IntentionFM_Regressor_cheat"] = {
        "n_params": reg_model.n_params, "wall_s": wall,
        "in": summarise("IntentionFM_Regressor_cheat", yp_in, Yq_in),
        "out": summarise("IntentionFM_Regressor_cheat", yp_out, Yq_out),
    }
    print(f"  -> in med={results['IntentionFM_Regressor_cheat']['in']['r2_median']:+.4f}  "
          f"out med={results['IntentionFM_Regressor_cheat']['out']['r2_median']:+.4f}")

    # ----- Save -----------------------------------------------------------
    summary = {
        "config": {
            "N_TRAIN_SCENARIOS": N_TRAIN_SCENARIOS,
            "N_TEST_IN": N_TEST_IN, "N_TEST_OUT": N_TEST_OUT,
            "K_CTX": K_CTX, "Q_QUERY": Q_QUERY,
            "N_META_STEPS": N_META_STEPS, "BATCH_S": BATCH_S, "LR": LR,
            "C_MAX_TRAIN": C_MAX_TRAIN, "C_OUTER": C_OUTER,
            "SEED": SEED,
            "IJEPA_MATCHED": IJEPA_MATCHED,
            "IJEPA_SCALED": IJEPA_SCALED,
        },
        **results,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Sorted table.
    print("\n\n## Summary (sorted by out-box median R^2)")
    rows = []
    for name, r in results.items():
        rows.append((
            name, r["n_params"],
            r["in"]["r2_median"], r["in"]["r2_p5"],
            r["out"]["r2_median"], r["out"]["r2_p5"],
            r["wall_s"],
        ))
    rows.sort(key=lambda x: -x[4])
    print(f"  {'architecture':30s}  {'params':>8s}  "
          f"{'in_med':>8s}  {'in_p5':>8s}  "
          f"{'out_med':>8s}  {'out_p5':>8s}  {'wall_s':>8s}")
    for name, np_, im, ip5, om, op5, w in rows:
        print(f"  {name:30s}  {np_:>8d}  "
              f"{im:+8.4f}  {ip5:+8.4f}  {om:+8.4f}  {op5:+8.4f}  {w:>8.1f}")
    return summary


if __name__ == "__main__":
    main()
