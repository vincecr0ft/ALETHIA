r"""T0.1-residual + T1.2 on the TRAINED ManifoldInformer encoders.

Reproduces the two headline residual-SVD numbers (Pearson r(u1, cos theta*)
and held-out RMS reduction from psi-extension) on the *trained* deficient
encoder psi_theta, not on the hand-engineered polynomial basis, and tests
the leverage blind-spot claim at fixed (unit) embedding norm.

Polynomial-basis baseline (output_smeft_residual_svd/summary.json):
    Pearson r(u1, cos theta*) = -0.885
    held-out RMS reduction     = 13.4x
    leverage_def_mean = 7.15e-4,  leverage_full_mean = 2.22e-3

Trained encoders:
    deficient psi_theta : d_event=1, output_integrated_loop/manifold_informer_deficient.pt
    full      psi_theta : d_event=2, output_manifold_informer/manifold_informer.pt

Run:
    export PATH="$HOME/snap/code/240/.local/bin:$PATH"
    OMP_NUM_THREADS=1 uv run python experiments/manifold-informer/fm_residual_svd.py
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

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import (
    sample_events, event_log_likelihood_ratio,
)
from modules.surrogate.features import N_WC

from manifold_informer import ManifoldInformer


# ---------------------------------------------------------------------------
# Configuration (matches integrated_loop / smeft_residual_svd where sensible)
# ---------------------------------------------------------------------------
N_PROBE_EVENTS = 6000
N_TEST_EVENTS = 1000
SEED_PROBE = 2222
SEED_TEST = 5678
RIDGE_ALPHA = 1e-6                     # same projector ridge as polynomial study

FULL_CKPT = HERE / "output_manifold_informer" / "manifold_informer.pt"
DEF_CKPT = HERE / "output_integrated_loop" / "manifold_informer_deficient.pt"
OUT_DIR = HERE / "output_fm_residual_svd"
OUT_DIR.mkdir(exist_ok=True)

D_EMB = 16
D_HIDDEN = 32

# Same training-sweep / held-out working points as the polynomial study,
# so the comparison is on the same operator-space trajectory.
C_TRAIN = np.array([
    [0.0, 0.0, 0.3, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.7, 0.0],
    [0.2, 0.0, 0.4, 0.0],
    [0.4, 0.0, 0.4, 0.0],
    [0.0, 0.2, 0.4, 0.0],
    [0.0, 0.0, 0.4, 0.2],
    [0.3, 0.0, 0.3, 0.0],
])
C_HELD = np.array([
    [0.0, 0.0, 0.6, 0.0],
    [0.3, 0.0, 0.5, 0.0],
    [0.0, 0.3, 0.3, 0.0],
])
C_STRONG_ANGULAR = np.array([0.0, 0.0, 0.4, 0.0])    # T1.2 leverage probe


def load_full() -> ManifoldInformer:
    m = ManifoldInformer(d_event=2, d_emb=D_EMB, hidden=D_HIDDEN)
    m.load_state_dict(torch.load(FULL_CKPT, weights_only=True))
    m.eval()
    return m


def load_def() -> ManifoldInformer:
    m = ManifoldInformer(d_event=1, d_emb=D_EMB, hidden=D_HIDDEN)
    m.load_state_dict(torch.load(DEF_CKPT, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def psi_from_encoder(model: ManifoldInformer, events: np.ndarray) -> np.ndarray:
    """Per-event trained-encoder embeddings, (N, d_emb)."""
    d = model.d_event
    Xt = torch.from_numpy(events[:, :d]).float()
    return model.event_encoder(Xt).cpu().numpy().astype(np.float64)


def projector_residual(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    A = psi.T @ psi + RIDGE_ALPHA * np.eye(psi.shape[1])
    w = np.linalg.solve(A, psi.T @ target)
    return target - psi @ w


def leverage_per_event(psi: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
    return np.einsum("nd,de,ne->n", psi, A_inv, psi)


# ---------------------------------------------------------------------------
# T0.1-residual: residual SVD on the trained deficient encoder
# ---------------------------------------------------------------------------
def t01_residual_svd(oracle, probe_events: np.ndarray) -> dict:
    print("\n# T0.1-residual: residual SVD on TRAINED deficient encoder psi_theta")
    def_model = load_def()
    full_model = load_full()
    psi_def = psi_from_encoder(def_model, probe_events)        # (N, 16)
    psi_full = psi_from_encoder(full_model, probe_events)      # (N, 16)
    print(f"  D_psi_def = {psi_def.shape[1]}, D_psi_full = {psi_full.shape[1]}")

    R_def_cols, R_full_cols = [], []
    for k, c in enumerate(C_TRAIN):
        t0 = time.time()
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        R_def_cols.append(projector_residual(psi_def, log_w))
        R_full_cols.append(projector_residual(psi_full, log_w))
        print(f"    c={c.tolist()}  ({time.time()-t0:.1f}s)")
    R_def = np.stack(R_def_cols, axis=1)
    R_full = np.stack(R_full_cols, axis=1)

    U_def, S_def, _ = np.linalg.svd(R_def, full_matrices=False)
    U_full, S_full, _ = np.linalg.svd(R_full, full_matrices=False)
    print(f"  sigma_def top5 = {S_def[:5].round(4).tolist()}")
    print(f"  sigma_full top5 = {S_full[:5].round(4).tolist()}")

    # (1) Pearson r(u1, cos theta*).
    u1 = U_def[:, 0]
    u1c = u1 - u1.mean()
    cos_th = probe_events[:, 1] - probe_events[:, 1].mean()
    rho = float(np.dot(u1c, cos_th)
                / (np.linalg.norm(u1c) * np.linalg.norm(cos_th) + 1e-30))
    print(f"  Pearson r(u1, cos theta*) = {rho:.4f}   (polynomial: -0.885)")

    # (2) Held-out RMS reduction from appending u1 to deficient psi and refit.
    psi_def_aug = np.concatenate([psi_def, u1.reshape(-1, 1)], axis=1)
    rb, ra = [], []
    for c in C_HELD:
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        rb.append(projector_residual(psi_def, log_w))
        ra.append(projector_residual(psi_def_aug, log_w))
    rms_before = float(np.sqrt(np.mean(np.stack(rb) ** 2)))
    rms_after = float(np.sqrt(np.mean(np.stack(ra) ** 2)))
    reduction = rms_before / max(rms_after, 1e-30)
    print(f"  held-out RMS before={rms_before:.4f} after={rms_after:.4f} "
          f"reduction={reduction:.2f}x   (polynomial: 13.4x)")

    return {
        "n_probe_events": int(probe_events.shape[0]),
        "n_train_working_points": int(len(C_TRAIN)),
        "n_held_working_points": int(len(C_HELD)),
        "sigma_top5_def": S_def[:5].tolist(),
        "sigma_top5_full": S_full[:5].tolist(),
        "sigma1_over_sigma2_def": float(S_def[0] / max(S_def[1], 1e-12)),
        "sigma1_over_sigma2_full": float(S_full[0] / max(S_full[1], 1e-12)),
        "pearson_u1_costheta_trained": rho,
        "pearson_u1_costheta_polynomial_baseline": -0.8853484729119309,
        "rms_residual_before": rms_before,
        "rms_residual_after": rms_after,
        "held_out_reduction_trained": reduction,
        "held_out_reduction_polynomial_baseline": 13.365674484334209,
    }


# ---------------------------------------------------------------------------
# T1.2: leverage at fixed (unit) embedding norm, deficient vs full
# ---------------------------------------------------------------------------
def unit_norm_rows(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi, axis=1, keepdims=True)
    return psi / np.clip(n, 1e-12, None)


def t12_leverage(probe_events: np.ndarray, test_events: np.ndarray) -> dict:
    print("\n# T1.2: leverage at fixed (unit) embedding norm")
    def_model = load_def()
    full_model = load_full()

    # Raw (un-normalised) embeddings, reproducing the reported gate.
    psi_def = psi_from_encoder(def_model, probe_events)
    psi_full = psi_from_encoder(full_model, probe_events)
    psi_def_t = psi_from_encoder(def_model, test_events)
    psi_full_t = psi_from_encoder(full_model, test_events)

    def lev_mean(psi_design, psi_test):
        A = psi_design.T @ psi_design + 1e-6 * np.eye(psi_design.shape[1])
        return leverage_per_event(psi_test, np.linalg.inv(A))

    lev_def_raw = lev_mean(psi_def, psi_def_t)
    lev_full_raw = lev_mean(psi_full, psi_full_t)
    print(f"  RAW   leverage_def_mean = {lev_def_raw.mean():.3e}  "
          f"leverage_full_mean = {lev_full_raw.mean():.3e}")

    # Fixed unit-norm embeddings.
    psi_def_u = unit_norm_rows(psi_def)
    psi_full_u = unit_norm_rows(psi_full)
    psi_def_tu = unit_norm_rows(psi_def_t)
    psi_full_tu = unit_norm_rows(psi_full_t)
    lev_def_unit = lev_mean(psi_def_u, psi_def_tu)
    lev_full_unit = lev_mean(psi_full_u, psi_full_tu)
    print(f"  UNIT  leverage_def_mean = {lev_def_unit.mean():.3e}  "
          f"leverage_full_mean = {lev_full_unit.mean():.3e}")

    # Embedding-norm diagnostic (the named confound).
    norm_def = float(np.linalg.norm(psi_def, axis=1).mean())
    norm_full = float(np.linalg.norm(psi_full, axis=1).mean())
    print(f"  mean ||psi_def|| = {norm_def:.4f}  mean ||psi_full|| = {norm_full:.4f}")

    def_blind = lev_def_unit.mean() <= lev_full_unit.mean()
    print(f"  fixed-norm deficient leverage {'<=' if def_blind else '>'} full "
          f"-> formal blind-spot claim {'SURVIVES' if def_blind else 'FAILS'}")

    return {
        "c_strong_angular": C_STRONG_ANGULAR.tolist(),
        "leverage_def_mean_raw": float(lev_def_raw.mean()),
        "leverage_full_mean_raw": float(lev_full_raw.mean()),
        "leverage_def_mean_unitnorm": float(lev_def_unit.mean()),
        "leverage_full_mean_unitnorm": float(lev_full_unit.mean()),
        "leverage_def_mean_raw_paper": 7.15e-4,
        "leverage_full_mean_raw_paper": 2.22e-3,
        "mean_embedding_norm_def": norm_def,
        "mean_embedding_norm_full": norm_full,
        "fixed_norm_def_blind_or_lower": bool(def_blind),
    }


def main():
    print("# T0.1-residual + T1.2 on the trained ManifoldInformer encoders")
    t0 = time.time()
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    print(f"  sampling {N_PROBE_EVENTS} probe events at SM ...")
    probe_events = sample_events(oracle, np.zeros(N_WC), N_PROBE_EVENTS,
                                 seed=SEED_PROBE)
    test_events = sample_events(oracle, np.zeros(N_WC), N_TEST_EVENTS,
                                seed=SEED_TEST)

    t01 = t01_residual_svd(oracle, probe_events)
    t12 = t12_leverage(probe_events, test_events)

    summary = {
        "description": ("Trained-encoder analogues of the polynomial-basis "
                        "residual-SVD headline numbers (T0.1-residual) and "
                        "the fixed-norm leverage blind-spot test (T1.2)."),
        "config": {
            "n_probe_events": N_PROBE_EVENTS,
            "n_test_events": N_TEST_EVENTS,
            "seed_probe": SEED_PROBE,
            "seed_test": SEED_TEST,
            "ridge_alpha_projector": RIDGE_ALPHA,
            "full_ckpt": str(FULL_CKPT.relative_to(HERE.parent.parent)),
            "def_ckpt": str(DEF_CKPT.relative_to(HERE.parent.parent)),
            "d_emb": D_EMB,
        },
        "T0_1_residual_svd": t01,
        "T1_2_leverage_fixed_norm": t12,
        "wall_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {OUT_DIR}/summary.json  (wall {summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
