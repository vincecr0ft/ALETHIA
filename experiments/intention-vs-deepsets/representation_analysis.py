"""Representation analysis: what does each FM architecture's latent space
actually encode about the SMEFT generative structure?

This is the physics-driven complement to experiment_jepa_scaling.py. Rather
than measuring R² on Y_query, we extract per-scenario representations from
each architecture and ask:

  1. How much of the Wilson coefficient c is recoverable from the
     representation alone? (Train a small c-regressor on (representation, c)
     pairs; report R² per c-component and joint.)
  2. What is the intrinsic dimension of each representation?
     (Effective rank from the singular spectrum; participation ratio;
     dimension to explain 90%/99% variance.)
  3. Do the embedding-bearing representations (Intention's ridge weights
     vs JEPA's context summary) linearly contain each other?
     (Cross-decoding R² in both directions.)
  4. Does each representation use the m-y per-event alignment, or is it
     y-shuffle-invariant (rate-blind)? (Per-scenario y-permutation cosine
     similarity.)

Crash-resilient: trains models one-by-one and checkpoints to disk
(output_representation_analysis/<name>.pt); skips already-trained models
on rerun. Metric blocks also incrementally update summary.json so a kill
mid-analysis loses at most one block.
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

OUT_DIR = HERE / "output_representation_analysis"
OUT_DIR.mkdir(exist_ok=True)
SUM_PATH = OUT_DIR / "summary.json"

K_CTX = 12
Q_QUERY = 32
N_TRAIN = 2000        # enough for a fair representation
N_EVAL = 200          # 200 held-out scenarios per split
SEED = 0
N_META_STEPS = 3000   # matched-budget models
N_META_STEPS_SCALED = 6000   # scaled JEPA transformer
LR = 1e-3
BATCH_S = 32


# ---- summary IO ----------------------------------------------------------
def load_summary() -> dict:
    if SUM_PATH.exists():
        with open(SUM_PATH) as f:
            return json.load(f)
    return {}


def save_summary(s: dict) -> None:
    with open(SUM_PATH, "w") as f:
        json.dump(s, f, indent=2)


# ---- dataset (cached on disk) --------------------------------------------
_DATA_CACHE_PATH = OUT_DIR / "splits.pt"


def build_data():
    """Build or load the train / eval_in / eval_out splits."""
    if _DATA_CACHE_PATH.exists():
        d = torch.load(_DATA_CACHE_PATH, weights_only=False)
        return d
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train = make_dataset(N_TRAIN, lambda r: sample_c(r, 1)[0],
                         oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=1)
    eval_in = make_dataset(N_EVAL, lambda r: sample_c(r, 1)[0],
                           oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=42)
    eval_out = make_dataset(N_EVAL,
                            lambda r: sample_c_shell(r, 1)[0],
                            oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=43)
    bundle = {
        "train_scen": train, "eval_in_scen": eval_in, "eval_out_scen": eval_out,
        "train_t": scenarios_to_tensors(train),
        "eval_in_t": scenarios_to_tensors(eval_in),
        "eval_out_t": scenarios_to_tensors(eval_out),
    }
    torch.save(bundle, _DATA_CACHE_PATH)
    return bundle


# ---- training (lean, no logging) -----------------------------------------
def train_intention_or_deepsets(model, train_t, n_steps, lr=LR,
                                batch_s=BATCH_S, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    S = train_t["M_ctx"].size(0)
    for _ in range(n_steps):
        idx = rng.choice(S, size=min(batch_s, S), replace=False)
        yp = model(train_t["M_ctx"][idx], train_t["Y_ctx"][idx],
                   train_t["M_query"][idx])
        loss = ((yp - train_t["Y_query"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def train_jepa(model, train_t, n_steps, lr=LR, batch_s=BATCH_S, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    S = train_t["M_ctx"].size(0)
    for step in range(1, n_steps + 1):
        idx = rng.choice(S, size=min(batch_s, S), replace=False)
        yp, jl, vl, ml, _ = model(train_t["M_ctx"][idx], train_t["Y_ctx"][idx],
                                  train_t["M_query"][idx], train_t["Y_query"][idx],
                                  return_losses=True)
        total = (model.jepa_weight * jl + model.vicreg_weight * vl
                 + model.aux_weight * ml)
        opt.zero_grad(); total.backward(); opt.step()
        model.update_target()


# ---- model factories + checkpoint helpers ------------------------------
def model_path(name: str) -> Path:
    return OUT_DIR / f"{name}.pt"


def get_or_train(name: str, factory, trainer, train_t) -> torch.nn.Module:
    p = model_path(name)
    model = factory()
    if p.exists():
        print(f"  [load] {name} from {p.name}", flush=True)
        sd = torch.load(p, weights_only=True)
        model.load_state_dict(sd)
        return model
    print(f"  [train] {name}", flush=True)
    t0 = time.time()
    trainer(model, train_t)
    print(f"    trained in {time.time()-t0:.1f}s", flush=True)
    torch.save(model.state_dict(), p)
    return model


# ---- representation extractors -------------------------------------------
@torch.no_grad()
def repr_intention(model: IntentionFMLearned, t: dict) -> np.ndarray:
    """The Intention ridge solve gives a per-scenario coefficient vector
    w in R^{d_psi}. That IS the scenario representation."""
    M_ctx, Y_ctx = t["M_ctx"], t["Y_ctx"]
    S = M_ctx.size(0); K = M_ctx.size(1); d = model.d_psi
    Psi = model.psi(M_ctx.reshape(-1)).reshape(S, K, d)
    A = torch.einsum("skd,ske->sde", Psi, Psi)
    A = A + model.alpha * torch.eye(d).expand_as(A)
    b = torch.einsum("skd,sk->sd", Psi, Y_ctx)
    w = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    return w.cpu().numpy()


@torch.no_grad()
def repr_deepsets(model: DeepSetsFM, t: dict) -> np.ndarray:
    """The mean-pooled summary z is DeepSets' scenario representation."""
    M_ctx, Y_ctx = t["M_ctx"], t["Y_ctx"]
    S, K = M_ctx.shape
    m_ctx_s = torch.log(M_ctx)
    evt_in = torch.stack([m_ctx_s, Y_ctx], dim=-1)
    h = model.phi_event(evt_in.reshape(S * K, 2)).reshape(S, K, -1)
    return h.mean(dim=1).cpu().numpy()


@torch.no_grad()
def repr_jepa(model: JEPAFM, t: dict) -> np.ndarray:
    """JEPA's scenario representation is the context-encoder summary."""
    return model.encode_context(t["M_ctx"], t["Y_ctx"]).cpu().numpy()


# ---- c-recoverability probe ----------------------------------------------
def fit_c_probe(repr_train, c_train, repr_eval, c_eval,
                lr=1e-2, n_steps=2000, hidden=None, seed=0) -> dict:
    torch.manual_seed(seed)
    D = repr_train.shape[1]
    if hidden is None:
        probe = torch.nn.Linear(D, N_WC)
    else:
        probe = torch.nn.Sequential(
            torch.nn.Linear(D, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, N_WC))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    Xt = torch.from_numpy(repr_train).float()
    yt = torch.from_numpy(c_train).float()
    Xe = torch.from_numpy(repr_eval).float()
    ye = c_eval
    for _ in range(n_steps):
        loss = ((probe(Xt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        yp_eval = probe(Xe).cpu().numpy()
    r2_per = []
    for k in range(N_WC):
        ss_res = np.sum((yp_eval[:, k] - ye[:, k]) ** 2)
        ss_tot = np.sum((ye[:, k] - ye[:, k].mean()) ** 2)
        r2_per.append(float(1.0 - ss_res / max(ss_tot, 1e-12)))
    ss_res_total = np.sum((yp_eval - ye) ** 2)
    ss_tot_total = np.sum((ye - ye.mean(axis=0, keepdims=True)) ** 2)
    r2_joint = float(1.0 - ss_res_total / max(ss_tot_total, 1e-12))
    return {"r2_per_c": r2_per, "r2_joint": r2_joint,
            "c_pred_eval": yp_eval.tolist()}


def representation_diagnostics(R: np.ndarray) -> dict:
    Rc = R - R.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(Rc, full_matrices=False)
    var = (s ** 2) / max(1, len(R) - 1)
    total = var.sum()
    eff_rank = int(np.sum(s > 0.01 * s.max())) if s.max() > 0 else 0
    pr = float(total ** 2 / (var ** 2).sum()) if total > 0 else 0.0
    cum = np.cumsum(var) / max(total, 1e-12)
    d90 = int(np.searchsorted(cum, 0.9)) + 1
    d99 = int(np.searchsorted(cum, 0.99)) + 1
    return {"effective_rank_1pct": eff_rank,
            "participation_ratio": pr,
            "d_90pct_var": d90, "d_99pct_var": d99,
            "singular_values": s.tolist(),
            "per_dim_std": Rc.std(axis=0).tolist(),
            "shape": list(R.shape)}


def cross_decode(src_train, src_eval, tgt_train, tgt_eval,
                 lr=1e-2, n_steps=2000, hidden=None, seed=0) -> float:
    torch.manual_seed(seed)
    D_in = src_train.shape[1]; D_out = tgt_train.shape[1]
    if hidden is None:
        probe = torch.nn.Linear(D_in, D_out)
    else:
        probe = torch.nn.Sequential(
            torch.nn.Linear(D_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, D_out))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    Xt = torch.from_numpy(src_train).float()
    yt = torch.from_numpy(tgt_train).float()
    Xe = torch.from_numpy(src_eval).float()
    ye = tgt_eval
    for _ in range(n_steps):
        loss = ((probe(Xt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        yp = probe(Xe).cpu().numpy()
    ss_res = np.sum((yp - ye) ** 2)
    ss_tot = np.sum((ye - ye.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def y_permutation_invariance(model, t, extractor, n_perms=5, seed=0) -> float:
    rng = np.random.default_rng(seed)
    R0 = extractor(model, t)
    R0n = R0 / (np.linalg.norm(R0, axis=1, keepdims=True) + 1e-12)
    sims = []
    for _ in range(n_perms):
        Y = t["Y_ctx"].clone()
        for s_i in range(Y.size(0)):
            perm = rng.permutation(Y.size(1))
            Y[s_i] = Y[s_i, perm]
        t_perm = {**t, "Y_ctx": Y}
        Rp = extractor(model, t_perm)
        Rpn = Rp / (np.linalg.norm(Rp, axis=1, keepdims=True) + 1e-12)
        sims.append(float(np.mean(np.sum(R0n * Rpn, axis=1))))
    return float(np.mean(sims))


# ---- main ----------------------------------------------------------------
def main():
    print("# Datasets")
    bundle = build_data()
    train_t = bundle["train_t"]; eval_in_t = bundle["eval_in_t"]; eval_out_t = bundle["eval_out_t"]
    c_train = np.stack([s["c"] for s in bundle["train_scen"]])
    c_in = np.stack([s["c"] for s in bundle["eval_in_scen"]])
    c_out = np.stack([s["c"] for s in bundle["eval_out_scen"]])

    summary = load_summary()
    summary.setdefault("config", {
        "N_TRAIN": N_TRAIN, "N_EVAL": N_EVAL,
        "K_CTX": K_CTX, "Q_QUERY": Q_QUERY,
        "N_META_STEPS": N_META_STEPS,
        "N_META_STEPS_SCALED": N_META_STEPS_SCALED,
        "BATCH_S": BATCH_S, "LR": LR})
    save_summary(summary)

    # ---- Train models -------------------------------------------------
    print("\n# Train models (checkpoints under output_representation_analysis/)")
    int_model = get_or_train(
        "Intention_w",
        lambda: IntentionFMLearned(d_psi=16, hidden=64, alpha=1e-3),
        lambda m, t: train_intention_or_deepsets(m, t, N_META_STEPS),
        train_t)
    ds_model = get_or_train(
        "DeepSets_z",
        lambda: DeepSetsFM(d_set=16, hidden=48),
        lambda m, t: train_intention_or_deepsets(m, t, N_META_STEPS),
        train_t)
    jepa_m = get_or_train(
        "JEPA_summary_m",
        lambda: JEPAFM(d_emb=16, hidden=48, encoder="deepsets",
                       vicreg_weight=0.04, aux_weight=0.1, jepa_weight=1.0,
                       use_ema=True),
        lambda m, t: train_jepa(m, t, N_META_STEPS),
        train_t)
    jepa_s = get_or_train(
        "JEPA_summary_s",
        lambda: JEPAFM(d_emb=64, hidden=128, encoder="transformer",
                       n_layers=2, n_heads=4,
                       vicreg_weight=0.04, aux_weight=0.1, jepa_weight=1.0,
                       use_ema=True),
        lambda m, t: train_jepa(m, t, N_META_STEPS_SCALED),
        train_t)

    extractors = {
        "Intention_w":      (int_model, repr_intention),
        "DeepSets_z":       (ds_model, repr_deepsets),
        "JEPA_summary_m":   (jepa_m, repr_jepa),
        "JEPA_summary_s":   (jepa_s, repr_jepa),
    }

    # ---- Extract representations -------------------------------------
    print("\n# Extract per-scenario representations")
    reps_train, reps_in, reps_out = {}, {}, {}
    for name, (m, ex) in extractors.items():
        reps_train[name] = ex(m, train_t)
        reps_in[name] = ex(m, eval_in_t)
        reps_out[name] = ex(m, eval_out_t)
        print(f"  {name:20s} shape={reps_train[name].shape}", flush=True)
    np.savez(OUT_DIR / "embeddings.npz",
             **{f"{k}__train": v for k, v in reps_train.items()},
             **{f"{k}__in":    v for k, v in reps_in.items()},
             **{f"{k}__out":   v for k, v in reps_out.items()},
             c_train=c_train, c_in=c_in, c_out=c_out)

    # ---- Block 1: diagnostics ---------------------------------------
    if "diagnostics" not in summary:
        print("\n# Intrinsic-dimension diagnostics (in-box held-out)")
        diag = {}
        for name in extractors:
            d = representation_diagnostics(reps_in[name])
            diag[name] = d
            print(f"  {name:20s} eff_rank={d['effective_rank_1pct']:>3d}  "
                  f"PR={d['participation_ratio']:.2f}  "
                  f"d_90={d['d_90pct_var']:>3d}  d_99={d['d_99pct_var']:>3d}",
                  flush=True)
        summary["diagnostics"] = diag
        save_summary(summary)
    else:
        print("\n# [skip] diagnostics already in summary.json")

    # ---- Block 2: c-recoverability ----------------------------------
    if "c_recoverability" not in summary:
        print("\n# c-recoverability probe (linear + MLP)")
        c_probe = {}
        for name in extractors:
            c_probe[name] = {}
            c_probe[name]["linear_in"] = fit_c_probe(
                reps_train[name], c_train, reps_in[name], c_in, hidden=None)
            c_probe[name]["linear_out"] = fit_c_probe(
                reps_train[name], c_train, reps_out[name], c_out, hidden=None)
            c_probe[name]["mlp_in"] = fit_c_probe(
                reps_train[name], c_train, reps_in[name], c_in, hidden=64)
            c_probe[name]["mlp_out"] = fit_c_probe(
                reps_train[name], c_train, reps_out[name], c_out, hidden=64)
            lr_in = c_probe[name]["linear_in"]["r2_joint"]
            lr_out = c_probe[name]["linear_out"]["r2_joint"]
            mlp_in = c_probe[name]["mlp_in"]["r2_joint"]
            mlp_out = c_probe[name]["mlp_out"]["r2_joint"]
            print(f"  {name:20s} "
                  f"linear in_R²={lr_in:+.3f} out_R²={lr_out:+.3f}  "
                  f"mlp in_R²={mlp_in:+.3f} out_R²={mlp_out:+.3f}",
                  flush=True)
        summary["c_recoverability"] = c_probe
        save_summary(summary)
    else:
        print("\n# [skip] c_recoverability already in summary.json")

    # ---- Block 3: cross-decoding ------------------------------------
    if "cross_decoding" not in summary:
        print("\n# Cross-decoding (linear-fit one representation from another)")
        cross = {}
        pairs = [
            ("Intention_w", "JEPA_summary_m"),
            ("Intention_w", "JEPA_summary_s"),
            ("JEPA_summary_m", "Intention_w"),
            ("JEPA_summary_s", "Intention_w"),
            ("DeepSets_z", "Intention_w"),
            ("Intention_w", "DeepSets_z"),
            ("JEPA_summary_s", "JEPA_summary_m"),
            ("JEPA_summary_m", "JEPA_summary_s"),
        ]
        for src, tgt in pairs:
            r2 = cross_decode(reps_train[src], reps_in[src],
                              reps_train[tgt], reps_in[tgt], hidden=None)
            cross[f"{src}->{tgt}"] = r2
            print(f"  {src:20s} -> {tgt:20s}  linear R²={r2:+.3f}", flush=True)
        summary["cross_decoding"] = cross
        save_summary(summary)
    else:
        print("\n# [skip] cross_decoding already in summary.json")

    # ---- Block 4: y-permutation invariance --------------------------
    if "y_permutation_invariance" not in summary:
        print("\n# y-permutation invariance (cosine sim under per-scenario y-shuffle)")
        y_invar = {}
        for name, (m, ex) in extractors.items():
            sim = y_permutation_invariance(m, eval_in_t, ex, n_perms=5, seed=7)
            y_invar[name] = sim
            note = "invariant (rate-blind)" if sim > 0.99 else "uses m-y alignment"
            print(f"  {name:20s}  cos_sim under y-shuffle = {sim:+.3f}   ({note})",
                  flush=True)
        summary["y_permutation_invariance"] = y_invar
        save_summary(summary)
    else:
        print("\n# [skip] y_permutation_invariance already in summary.json")

    print(f"\n# wrote {SUM_PATH} + embeddings.npz + per-model .pt checkpoints")
    return summary


if __name__ == "__main__":
    main()
