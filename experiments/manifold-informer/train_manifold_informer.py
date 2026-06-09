r"""Train the ManifoldInformer on event-level SMEFT data.

Generates an event-level training dataset once and caches it; trains the
ManifoldInformer with the full JEPA + VICReg + view-invariance + density-
anchor loss; saves the trained model + loss trajectories for the P1-P4
latent-geometry gates downstream.

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    uv run python experiments/manifold-informer/train_manifold_informer.py

Outputs:
    output_manifold_informer/
        train_dataset.npz    (cached event sets, scenarios, log_w_q)
        manifold_informer.pt model state dict
        train_summary.json   final losses, R² on Y if a decoder is included
        loss_curves.png      JEPA / VICReg / view / density traces
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import (
    sample_events, event_log_likelihood_ratio,
)
from modules.surrogate.features import N_WC

from manifold_informer import ManifoldInformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Sized so the dataset generation completes in ~10 min on the per-event
# oracle.truth path. Larger runs are appropriate once the oracle is
# vectorised (see oracle_events.py TODO); the architecture verification
# only needs modest scenario / event counts to exercise the four loss
# terms and produce a P1-P4 testable trained model.
N_SCENARIOS_TRAIN = 60
N_SCENARIOS_VAL = 12
N_EVENTS_PER_VIEW = 250       # per-view context+query event count
SPLIT_QUERY_FRAC = 0.25       # 25% of events as query, 75% as context
C_BOX_TRAIN = 0.6
SEED = 2026

D_EMB = 16
D_HIDDEN = 32
ALPHA_RIDGE = 1e-3

EMA_MOMENTUM = 0.996
VICREG_W = 0.04
VIEW_W = 0.1
DENSITY_W = 0.05
LR = 1e-3
BATCH_S = 8
N_STEPS = 400

OUT_DIR = HERE / "output_manifold_informer"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def _generate_dataset(n_scenarios: int, c_box: float,
                       n_events_per_view: int,
                       seed: int) -> dict:
    """Two independent event sets per scenario + per-scenario log_w on
    the second view's events. Returns numpy arrays for fast torch loading."""
    rng = np.random.default_rng(seed)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    cs = np.empty((n_scenarios, N_WC), dtype=np.float32)
    X_view1 = np.empty((n_scenarios, n_events_per_view, 2), dtype=np.float32)
    X_view2 = np.empty((n_scenarios, n_events_per_view, 2), dtype=np.float32)
    log_w_view1 = np.empty((n_scenarios, n_events_per_view), dtype=np.float32)
    log_w_view2 = np.empty((n_scenarios, n_events_per_view), dtype=np.float32)
    for s in range(n_scenarios):
        c = rng.uniform(-c_box, c_box, N_WC).astype(np.float32)
        cs[s] = c
        # Two independent event sets at the same c.
        ev1 = sample_events(oracle, c, n_events_per_view,
                             seed=int(rng.integers(0, 2**31 - 1)))
        ev2 = sample_events(oracle, c, n_events_per_view,
                             seed=int(rng.integers(0, 2**31 - 1)))
        X_view1[s] = ev1.astype(np.float32)
        X_view2[s] = ev2.astype(np.float32)
        log_w_view1[s] = event_log_likelihood_ratio(
            oracle, c, ev1).astype(np.float32)
        log_w_view2[s] = event_log_likelihood_ratio(
            oracle, c, ev2).astype(np.float32)
        if (s + 1) % 20 == 0:
            print(f"    [gen] {s+1}/{n_scenarios}", flush=True)
    return {
        "c": cs, "X_view1": X_view1, "X_view2": X_view2,
        "log_w_view1": log_w_view1, "log_w_view2": log_w_view2,
    }


def get_or_make_dataset() -> tuple[dict, dict]:
    """Cache the dataset so repeat runs don't re-sample events."""
    cache_train = OUT_DIR / f"train_dataset_n{N_SCENARIOS_TRAIN}_ev{N_EVENTS_PER_VIEW}.npz"
    cache_val = OUT_DIR / f"val_dataset_n{N_SCENARIOS_VAL}_ev{N_EVENTS_PER_VIEW}.npz"
    if cache_train.exists() and cache_val.exists():
        print(f"  loading cached datasets from {OUT_DIR}")
        train = dict(np.load(cache_train))
        val = dict(np.load(cache_val))
        return train, val
    print(f"  generating training dataset "
          f"({N_SCENARIOS_TRAIN} scenarios × 2 views × {N_EVENTS_PER_VIEW} events)")
    t0 = time.time()
    train = _generate_dataset(N_SCENARIOS_TRAIN, C_BOX_TRAIN,
                                N_EVENTS_PER_VIEW, seed=SEED)
    print(f"    train gen: {time.time() - t0:.1f}s")
    print(f"  generating validation dataset")
    t1 = time.time()
    val = _generate_dataset(N_SCENARIOS_VAL, C_BOX_TRAIN,
                              N_EVENTS_PER_VIEW, seed=SEED + 1)
    print(f"    val gen:   {time.time() - t1:.1f}s")
    np.savez(cache_train, **train)
    np.savez(cache_val, **val)
    return train, val


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def _split_ctx_q(events_torch, query_frac, generator):
    """Random per-scenario split into (X_ctx, X_q) + matched log_w split."""
    S, N, d = events_torch.shape
    N_q = int(round(N * query_frac))
    perm = torch.argsort(torch.rand(S, N, generator=generator,
                                      device=events_torch.device), dim=1)
    idx_ctx = perm[:, N_q:]                                            # (S, N - N_q)
    idx_q = perm[:, :N_q]                                              # (S, N_q)
    return idx_ctx, idx_q


def train_model(model: ManifoldInformer, train: dict, val: dict,
                 n_steps: int, lr: float, batch_s: int,
                 seed: int = 0) -> dict:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    Xt = torch.from_numpy(train["X_view1"])                            # (S, N, 2)
    Xt2 = torch.from_numpy(train["X_view2"])
    logw_t = torch.from_numpy(train["log_w_view1"])                    # (S, N)
    S_train, N_train, _ = Xt.shape
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    history = {"step": [], "jepa": [], "vicreg": [], "view": [],
                "density": [], "total": [], "val_jepa": []}
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = rng.choice(S_train, size=min(batch_s, S_train), replace=False)
        X = Xt[idx]                                                     # (B, N, 2)
        X2 = Xt2[idx]
        logw = logw_t[idx]                                              # (B, N)
        # Per-scenario context/query split.
        i_ctx, i_q = _split_ctx_q(X, SPLIT_QUERY_FRAC, gen)
        X_ctx = torch.gather(X, 1, i_ctx.unsqueeze(-1).expand(-1, -1, 2))
        X_q = torch.gather(X, 1, i_q.unsqueeze(-1).expand(-1, -1, 2))
        log_w_q = torch.gather(logw, 1, i_q)
        # Second view is the full ev2 set used for the RS3L term.
        out = model(X_ctx, X_q, X_ctx_view2=X2,
                     log_w_q=log_w_q, return_losses=True)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        model.update_target()

        if step % max(1, n_steps // 40) == 0 or step == 1:
            with torch.no_grad():
                Xv = torch.from_numpy(val["X_view1"])
                logwv = torch.from_numpy(val["log_w_view1"])
                iv_ctx, iv_q = _split_ctx_q(Xv, SPLIT_QUERY_FRAC, gen)
                Xv_ctx = torch.gather(Xv, 1, iv_ctx.unsqueeze(-1).expand(-1, -1, 2))
                Xv_q = torch.gather(Xv, 1, iv_q.unsqueeze(-1).expand(-1, -1, 2))
                logwv_q = torch.gather(logwv, 1, iv_q)
                v_out = model(Xv_ctx, Xv_q, log_w_q=logwv_q,
                                return_losses=True)
                val_jepa = float(v_out["jepa"].item())
            history["step"].append(step)
            history["jepa"].append(float(out["jepa"].item()))
            history["vicreg"].append(float(out["vicreg"].item()))
            history["view"].append(float(out["view"].item()))
            history["density"].append(float(out["density"].item()))
            history["total"].append(float(out["total"].item()))
            history["val_jepa"].append(val_jepa)
            print(f"  step {step:5d}/{n_steps}  "
                  f"jepa={out['jepa'].item():.4f}  "
                  f"vic={out['vicreg'].item():.3f}  "
                  f"view={out['view'].item():.4f}  "
                  f"dens={out['density'].item():.4f}  "
                  f"|  val jepa={val_jepa:.4f}",
                  flush=True)
    history["wall_seconds"] = time.time() - t0
    return history


def plot_history(history: dict, out_path: Path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    s = np.array(history["step"])
    ax.semilogy(s, history["jepa"], "-o", c="C0", label="train jepa", ms=3)
    ax.semilogy(s, history["val_jepa"], "-s", c="C0", alpha=0.6, label="val jepa", ms=3)
    ax.semilogy(s, history["vicreg"], "-^", c="C2", label="vicreg", ms=3)
    ax.semilogy(s, history["view"], "-v", c="C3", label="view", ms=3)
    ax.semilogy(s, history["density"], "-d", c="C4", label="density", ms=3)
    ax.semilogy(s, history["total"], "-x", c="k", label="total", lw=2.0, ms=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("ManifoldInformer training curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("# ManifoldInformer training (Task 3)")
    print(f"  config: d_emb={D_EMB} hidden={D_HIDDEN} alpha={ALPHA_RIDGE}")
    print(f"          n_steps={N_STEPS} batch={BATCH_S} lr={LR}")
    print(f"          n_train_scen={N_SCENARIOS_TRAIN} "
          f"n_events_per_view={N_EVENTS_PER_VIEW}")
    print(f"          loss weights: jepa=1, vicreg={VICREG_W}, "
          f"view={VIEW_W}, density={DENSITY_W}")

    train, val = get_or_make_dataset()
    print(f"  train X_view1 shape = {train['X_view1'].shape}")

    model = ManifoldInformer(
        d_event=2, d_emb=D_EMB, hidden=D_HIDDEN, alpha=ALPHA_RIDGE,
        ema_momentum=EMA_MOMENTUM, vicreg_weight=VICREG_W,
        view_weight=VIEW_W, density_anchor_weight=DENSITY_W,
        use_ema=True,
    )
    print(f"  model n_params = {model.n_params}")

    history = train_model(model, train, val, n_steps=N_STEPS, lr=LR,
                            batch_s=BATCH_S, seed=SEED)
    print(f"\n  training wall = {history['wall_seconds']:.1f}s")

    torch.save(model.state_dict(), OUT_DIR / "manifold_informer.pt")
    plot_history(history, OUT_DIR / "loss_curves.png")
    summary = {
        "config": {
            "d_emb": D_EMB, "hidden": D_HIDDEN, "alpha": ALPHA_RIDGE,
            "n_steps": N_STEPS, "batch_s": BATCH_S, "lr": LR,
            "vicreg_w": VICREG_W, "view_w": VIEW_W,
            "density_w": DENSITY_W,
            "n_train_scen": N_SCENARIOS_TRAIN,
            "n_events_per_view": N_EVENTS_PER_VIEW,
            "split_query_frac": SPLIT_QUERY_FRAC,
            "c_box_train": C_BOX_TRAIN,
            "seed": SEED,
        },
        "n_params": model.n_params,
        "final_jepa": history["jepa"][-1],
        "final_val_jepa": history["val_jepa"][-1],
        "final_vicreg": history["vicreg"][-1],
        "final_view": history["view"][-1],
        "final_density": history["density"][-1],
        "wall_seconds": history["wall_seconds"],
        "history": history,
    }
    with open(OUT_DIR / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {OUT_DIR}/{{manifold_informer.pt, train_summary.json, loss_curves.png}}")
    return summary


if __name__ == "__main__":
    main()
