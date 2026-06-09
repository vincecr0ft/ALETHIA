r"""Tightened manifold-identity certificate (held-out, vector targets).

Fixes the flaw `compare_aggregators_gates.py` surfaced: the original P3/P4
fit the probe map in-sample with 16 latent dims against ≤10 scalar targets,
so R²≈1.0 was near-automatic and the gate did not discriminate aggregators.

Tightened construction:
  1. Fit ONE linear decoder D: w_θ -> μ(c, m) profile over a 40-bin m-grid,
     on TRAIN scenarios; report its held-out profile R² (the latent must
     linearly carry the full m-resolved c-dependence of the observable).
  2. With D frozen (it never saw the analytic derivatives), require the SAME
     map to turn the latent's finite-difference tangent into A_i(m) and its
     curvature into B_ij(m), scored as VECTORS over the m-grid:
        Â_i(m)   = (∂w/∂c_i)        @ Dᵀ   vs   A_i(m) = ∂μ/∂c_i |_SM
        B̂_ij(m)  = (∂²w/∂c_i∂c_j)   @ Dᵀ   vs   B_ij(m) = ∂²μ/∂c_i∂c_j |_SM
     Target dim (N_WC·40 for P3) ≫ the frozen map's freedom, and D was fit on
     function values not derivatives, so these R² can genuinely fall below 1.

A latent that predicts well but encodes a rate-blind / lossy summary will
have a low profile R² and its frozen-decoder derivatives will miss A_i(m)/
B_ij(m). Run on the three aggregators trained identically in
compare_aggregators_gates.py (checkpoints reused).

Run:
    uv run --no-sync python experiments/manifold-informer/compare_aggregators_gates_v2.py
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import sample_events
from modules.surrogate.features import N_WC
from modules.surrogate.intention.fisher import empirical_fisher_c, fisher_basis

from manifold_informer import ManifoldInformer
from manifold_informer_variants import MeanPoolInformer, AttnInformer
from train_manifold_informer import (
    D_EMB, D_HIDDEN, ALPHA_RIDGE, EMA_MOMENTUM, VICREG_W, VIEW_W, DENSITY_W,
)

OUT_DIR = HERE / "output_aggregator_comparison"
RIDGE_CKPT = HERE / "output_manifold_informer" / "manifold_informer.pt"

M_GRID = np.linspace(0.3, 2.3, 40)
N_TRAIN = 120
N_TEST = 60
C_BOX = 0.6
N_EVENTS = 250
H = 0.05
SEED_TANGENT = 9999
SEED_CURV = 7777


def _common_kw():
    return dict(d_event=2, d_emb=D_EMB, hidden=D_HIDDEN,
                ema_momentum=EMA_MOMENTUM, vicreg_weight=VICREG_W,
                view_weight=VIEW_W, density_anchor_weight=DENSITY_W,
                use_ema=True)


def _w(model, events_list):
    Xt = torch.from_numpy(np.stack(events_list)).float()
    with torch.no_grad():
        return model.manifold_coord(Xt).cpu().numpy()


def _profile(oracle, c):
    """Analytic μ(c, m) over the m-grid -> (40,)."""
    c_tiled = np.tile(np.asarray(c, float), (len(M_GRID), 1))
    return oracle.truth(c_tiled, M_GRID)


def _analytic_A(oracle):
    """A_i(m) = ∂μ/∂c_i |_SM over the m-grid -> (N_WC, 40)."""
    A = np.empty((N_WC, len(M_GRID)))
    for i in range(N_WC):
        cp = np.zeros((len(M_GRID), N_WC)); cp[:, i] = +H
        cm = np.zeros((len(M_GRID), N_WC)); cm[:, i] = -H
        A[i] = (oracle.truth(cp, M_GRID) - oracle.truth(cm, M_GRID)) / (2 * H)
    return A


def _analytic_B(oracle):
    """B_ij(m) = ∂²μ/∂c_i∂c_j |_SM over the m-grid, upper triangle stacked."""
    iu = np.triu_indices(N_WC)
    B = np.empty((len(iu[0]), len(M_GRID)))
    for k, (i, j) in enumerate(zip(*iu)):
        cpp = np.zeros((len(M_GRID), N_WC)); cpp[:, i] += H; cpp[:, j] += H
        cpm = np.zeros((len(M_GRID), N_WC)); cpm[:, i] += H; cpm[:, j] += -H
        cmp = np.zeros((len(M_GRID), N_WC)); cmp[:, i] += -H; cmp[:, j] += H
        cmm = np.zeros((len(M_GRID), N_WC)); cmm[:, i] += -H; cmm[:, j] += -H
        B[k] = (oracle.truth(cpp, M_GRID) + oracle.truth(cmm, M_GRID)
                - oracle.truth(cpm, M_GRID) - oracle.truth(cmp, M_GRID)) / (4 * H * H)
    return B, iu


def _latent_tangent(model, oracle):
    """∂w/∂c_i |_SM by shared-seed central differences -> (N_WC, d_emb)."""
    J = np.empty((N_WC, model.d_emb))
    for i in range(N_WC):
        cp = np.zeros(N_WC, np.float32); cp[i] = +H
        cm = np.zeros(N_WC, np.float32); cm[i] = -H
        ev_p = sample_events(oracle, cp, N_EVENTS, seed=SEED_TANGENT)
        ev_m = sample_events(oracle, cm, N_EVENTS, seed=SEED_TANGENT)
        J[i] = (_w(model, [ev_p])[0] - _w(model, [ev_m])[0]) / (2 * H)
    return J


def _latent_curvature(model, oracle, iu):
    """∂²w/∂c_i∂c_j |_SM -> (n_pairs, d_emb)."""
    J2 = np.empty((len(iu[0]), model.d_emb))
    for k, (i, j) in enumerate(zip(*iu)):
        cpp = np.zeros(N_WC, np.float32); cpp[i] += H; cpp[j] += H
        cpm = np.zeros(N_WC, np.float32); cpm[i] += H; cpm[j] += -H
        cmp = np.zeros(N_WC, np.float32); cmp[i] += -H; cmp[j] += H
        cmm = np.zeros(N_WC, np.float32); cmm[i] += -H; cmm[j] += -H
        wpp = _w(model, [sample_events(oracle, cpp, N_EVENTS, seed=SEED_CURV)])[0]
        wpm = _w(model, [sample_events(oracle, cpm, N_EVENTS, seed=SEED_CURV)])[0]
        wmp = _w(model, [sample_events(oracle, cmp, N_EVENTS, seed=SEED_CURV)])[0]
        wmm = _w(model, [sample_events(oracle, cmm, N_EVENTS, seed=SEED_CURV)])[0]
        J2[k] = (wpp + wmm - wpm - wmp) / (4 * H * H)
    return J2


# --- directional (Fisher-eigenvector) derivatives, for the resolved subspace ---
def _lat_tan_dir(model, oracle, v):
    cp = (+H * v).astype(np.float32); cm = (-H * v).astype(np.float32)
    ep = sample_events(oracle, cp, N_EVENTS, seed=SEED_TANGENT)
    em = sample_events(oracle, cm, N_EVENTS, seed=SEED_TANGENT)
    return (_w(model, [ep])[0] - _w(model, [em])[0]) / (2 * H)


def _an_tan_dir(oracle, v):
    cp = np.tile(+H * v, (len(M_GRID), 1)); cm = np.tile(-H * v, (len(M_GRID), 1))
    return (oracle.truth(cp, M_GRID) - oracle.truth(cm, M_GRID)) / (2 * H)


def _lat_curv_dir(model, oracle, vk, vl):
    cpp = (+H * vk + H * vl).astype(np.float32); cmm = (-H * vk - H * vl).astype(np.float32)
    cpm = (+H * vk - H * vl).astype(np.float32); cmp = (-H * vk + H * vl).astype(np.float32)
    wpp = _w(model, [sample_events(oracle, cpp, N_EVENTS, seed=SEED_CURV)])[0]
    wmm = _w(model, [sample_events(oracle, cmm, N_EVENTS, seed=SEED_CURV)])[0]
    wpm = _w(model, [sample_events(oracle, cpm, N_EVENTS, seed=SEED_CURV)])[0]
    wmp = _w(model, [sample_events(oracle, cmp, N_EVENTS, seed=SEED_CURV)])[0]
    return (wpp + wmm - wpm - wmp) / (4 * H * H)


def _an_curv_dir(oracle, vk, vl):
    cpp = np.tile(+H * vk + H * vl, (len(M_GRID), 1))
    cmm = np.tile(-H * vk - H * vl, (len(M_GRID), 1))
    cpm = np.tile(+H * vk - H * vl, (len(M_GRID), 1))
    cmp = np.tile(-H * vk + H * vl, (len(M_GRID), 1))
    return (oracle.truth(cpp, M_GRID) + oracle.truth(cmm, M_GRID)
            - oracle.truth(cpm, M_GRID) - oracle.truth(cmp, M_GRID)) / (4 * H * H)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tightened_gate(model, oracle, rng, resolved_dirs=None) -> dict:
    # --- scenarios + latent + analytic profiles ---
    c_tr = rng.uniform(-C_BOX, C_BOX, (N_TRAIN, N_WC)).astype(np.float32)
    c_te = rng.uniform(-C_BOX, C_BOX, (N_TEST, N_WC)).astype(np.float32)
    ev_tr = [sample_events(oracle, c, N_EVENTS, seed=int(rng.integers(0, 2**31 - 1)))
             for c in c_tr]
    ev_te = [sample_events(oracle, c, N_EVENTS, seed=int(rng.integers(0, 2**31 - 1)))
             for c in c_te]
    w_tr, w_te = _w(model, ev_tr), _w(model, ev_te)
    P_tr = np.stack([_profile(oracle, c) for c in c_tr])      # (N_TRAIN, 40)
    P_te = np.stack([_profile(oracle, c) for c in c_te])

    # --- one linear decoder D: w -> μ(m) profile, fit on TRAIN ---
    D = RidgeCV(alphas=np.logspace(-6, 2, 25)).fit(w_tr, P_tr)
    profile_r2_train = float(r2_score(P_tr, D.predict(w_tr)))
    profile_r2_heldout = float(r2_score(P_te, D.predict(w_te)))
    C = D.coef_                                               # (40, d_emb)

    # --- frozen-decoder derivative match (vector targets over m) ---
    # R² is scale-sensitive (a global linear decoder's gradient is a poor
    # local-derivative estimate when profile R²<1), so we also report the
    # scale-free shape match: mean per-direction cosine between the predicted
    # and analytic A_i(m) / B_ij(m) profiles.
    def _row_cos(P, Q):
        num = (P * Q).sum(axis=1)
        den = np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1) + 1e-12
        return float(np.mean(num / den))

    A_true = _analytic_A(oracle)                             # (N_WC, 40)
    A_pred = _latent_tangent(model, oracle) @ C.T            # (N_WC, 40)
    p3_r2 = float(r2_score(A_true.ravel(), A_pred.ravel()))
    p3_cos = _row_cos(A_pred, A_true)

    B_true, iu = _analytic_B(oracle)                        # (n_pairs, 40)
    B_pred = _latent_curvature(model, oracle, iu) @ C.T      # (n_pairs, 40)
    p4_r2 = float(r2_score(B_true.ravel(), B_pred.ravel()))
    p4_cos = _row_cos(B_pred, B_true)

    out = {
        "profile_r2_train": profile_r2_train,
        "profile_r2_heldout": profile_r2_heldout,
        "P3_tangent_r2_heldout": p3_r2,
        "P4_curvature_r2_heldout": p4_r2,
        "P3_tangent_cos": p3_cos,
        "P4_curvature_cos": p4_cos,
    }

    # --- resolved Fisher-subspace shape match (the identifiable part) ---
    if resolved_dirs is not None:
        tan_cos, curv_cos = [], []
        for v in resolved_dirs:
            pred = _lat_tan_dir(model, oracle, v) @ C.T
            tan_cos.append(_cos(pred, _an_tan_dir(oracle, v)))
        for a in range(len(resolved_dirs)):
            for b in range(a, len(resolved_dirs)):
                pred = _lat_curv_dir(model, oracle, resolved_dirs[a],
                                     resolved_dirs[b]) @ C.T
                curv_cos.append(_cos(pred, _an_curv_dir(
                    oracle, resolved_dirs[a], resolved_dirs[b])))
        out["P3_tangent_cos_resolved"] = float(np.mean(tan_cos))
        out["P4_curvature_cos_resolved"] = float(np.mean(curv_cos))
        out["n_resolved"] = len(resolved_dirs)
    return out


def main():
    print("# Tightened certificate (held-out decoder, vector A_i(m)/B_ij(m) targets)\n")
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)

    # Fisher basis to isolate the identifiable (resolved) c-subspace.
    frng = np.random.default_rng(2026)
    F = empirical_fisher_c(oracle, frng.uniform(-C_BOX, C_BOX, (400, N_WC)),
                           frng.uniform(0.3, 2.3, 400), fd_step=1e-3)
    D_eig, V_fisher = fisher_basis(F)
    n_res = int((D_eig > 1e-2).sum())
    resolved_dirs = [V_fisher[:, k] for k in range(n_res)]
    print(f"  Fisher eigenvalues: {D_eig.round(4).tolist()}  "
          f"(resolved: {n_res}/{N_WC})\n")

    models = {}
    ridge = ManifoldInformer(**_common_kw())
    ridge.load_state_dict(torch.load(RIDGE_CKPT, weights_only=True)); ridge.eval()
    models["Ridge (ManifoldInformer)"] = ridge
    for name, cls in [("MeanPool", MeanPoolInformer), ("Attention", AttnInformer)]:
        ck = OUT_DIR / f"{name}.pt"
        if not ck.exists():
            print(f"  ERROR: missing {ck}; run compare_aggregators_gates.py first")
            sys.exit(2)
        m = cls(**_common_kw())
        m.load_state_dict(torch.load(ck, weights_only=True)); m.eval()
        models[name] = m

    rows = {}
    for name, model in models.items():
        t0 = time.time()
        rows[name] = tightened_gate(model, oracle, np.random.default_rng(20260605),
                                    resolved_dirs=resolved_dirs)
        rows[name]["n_params"] = int(model.n_params)
        r = rows[name]
        print(f"{name:<26} profile R² heldout={r['profile_r2_heldout']:+.4f}   "
              f"all-dir cos P3={r['P3_tangent_cos']:+.3f} P4={r['P4_curvature_cos']:+.3f}   "
              f"resolved cos P3={r['P3_tangent_cos_resolved']:+.3f} "
              f"P4={r['P4_curvature_cos_resolved']:+.3f}   ({time.time()-t0:.0f}s)")

    (OUT_DIR / "certificate_comparison_v2.json").write_text(
        json.dumps({"m_grid_bins": len(M_GRID), "n_train": N_TRAIN,
                    "n_test": N_TEST, "fisher_eigenvalues": D_eig.tolist(),
                    "n_resolved": n_res, "rows": rows}, indent=2))

    print(f"\n{'#'*78}\n# Tightened manifold-identity certificate (all held-out) "
          f"— resolved = {n_res}/{N_WC} dirs\n{'#'*78}")
    print(f"{'aggregator':<26}{'profile R²':>11}{'P3 cos res':>12}"
          f"{'P4 cos res':>12}{'(P3 all)':>10}{'(P4 all)':>10}")
    for name, r in rows.items():
        print(f"{name:<26}{r['profile_r2_heldout']:>11.4f}"
              f"{r['P3_tangent_cos_resolved']:>12.4f}"
              f"{r['P4_curvature_cos_resolved']:>12.4f}"
              f"{r['P3_tangent_cos']:>10.3f}{r['P4_curvature_cos']:>10.3f}")
    print(f"\n# wrote {(OUT_DIR / 'certificate_comparison_v2.json').relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
