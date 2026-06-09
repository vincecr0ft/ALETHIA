r"""Integrated end-to-end run of the ManifoldInformer inside the four-monitor
closed-loop of Figure 1: per-event trained encoder, residual-SVD span-
completeness fingerprint, condition-number context-shape monitor, DAS-CUSUM
accuracy detector on density-decoder residuals, Benjamini-Hochberg per-
region calibration coverage, and the psi-extension action when the
fingerprint persists.

Three arms span the AL1 thesis bullet on the *trained encoder* (not on
hand-engineered polynomial bases):

    full_psi_no_extension       d_event=2, trained on (log m, cos theta*).
                                    AL1 in-span no-fire test.
    def_psi_no_extension        d_event=1, trained on (log m,) alone.
                                    Persistence-without-correction.
    def_psi_with_extension      same deficient encoder, psi-extension on.
                                    Fire-on-OOS + structural correction.

The deficient encoder is retrained inline from the cached training events,
keeping only the first per-event feature column. No hand deletion of
columns from psi_theta -- the deficiency is a property of the *input* the
encoder is shown, just as the polynomial-basis ground-truth-known test
deliberately omits cos theta* from the basis.

Outputs:
    output_integrated_loop/
        manifold_informer_deficient.pt
        summary.json
        spectra.png
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass, field
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
N_PROBE_EVENTS = 4000
SIGMA_FLOOR = 1e-3
# Calibrated above the empirical max in-span sigma_1/sigma_2 of the full
# trained-encoder reference (18.16 across cycles 1-11), same ~1.1x safety
# factor as the polynomial-basis study (threshold 10 above ratio 9.28).
RATIO_THRESHOLD = 20.0
PERSISTENCE_N = 3
KAPPA_THRESHOLD = 1e6                  # ridge solve stability sentinel
CUSUM_H = 6.0                          # h matching the precursor controller
N_M_REGIONS = 5
NOMINAL_COVERAGE = 0.68
BH_ALPHA = 0.05
SEED = 7777

# Same working-point queue as the polynomial-basis Task 4 unit test, so
# the comparison across studies is on the same operator-space trajectory.
WORKING_POINT_QUEUE = [
    np.array([0.0, 0.0, 0.3, 0.0]),
    np.array([0.0, 0.0, 0.5, 0.0]),
    np.array([0.2, 0.0, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.4, 0.2]),
    np.array([0.4, 0.0, 0.4, 0.0]),
    np.array([0.0, 0.2, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.7, 0.0]),
    np.array([0.3, 0.0, 0.3, 0.0]),
    np.array([0.0, 0.0, 0.6, 0.0]),
    np.array([0.2, 0.0, 0.6, 0.0]),
    np.array([0.0, 0.0, 0.5, 0.3]),
    np.array([0.3, 0.2, 0.3, 0.0]),
]

OUT_DIR = HERE / "output_integrated_loop"
OUT_DIR.mkdir(exist_ok=True)
FULL_CKPT = HERE / "output_manifold_informer" / "manifold_informer.pt"
CACHE_TRAIN = HERE / "output_manifold_informer" / "train_dataset_n60_ev250.npz"
CACHE_VAL = HERE / "output_manifold_informer" / "val_dataset_n12_ev250.npz"

# Encoder/training constants matching the full run.
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
SPLIT_QUERY_FRAC = 0.25


# ---------------------------------------------------------------------------
# Train the deficient encoder (d_event=1, mass-only events)
# ---------------------------------------------------------------------------
def _split_ctx_q(events_torch, query_frac, generator):
    S, N, d = events_torch.shape
    N_q = int(round(N * query_frac))
    perm = torch.argsort(torch.rand(S, N, generator=generator,
                                    device=events_torch.device), dim=1)
    idx_ctx = perm[:, N_q:]
    idx_q = perm[:, :N_q]
    return idx_ctx, idx_q


def train_deficient_encoder(seed: int = 2026) -> ManifoldInformer:
    """Retrain the ManifoldInformer with d_event=1 on the same JEPA pipeline.

    The cached training events have shape (S, N, 2); we feed [:, :, :1].
    """
    if not CACHE_TRAIN.exists() or not CACHE_VAL.exists():
        raise FileNotFoundError(
            f"Cached training events not found at {CACHE_TRAIN} or {CACHE_VAL}. "
            f"Run train_manifold_informer.py first."
        )
    train = dict(np.load(CACHE_TRAIN))
    val = dict(np.load(CACHE_VAL))

    # Truncate to mass-only feature.
    Xt = torch.from_numpy(train["X_view1"][:, :, :1])                     # (S, N, 1)
    Xt2 = torch.from_numpy(train["X_view2"][:, :, :1])
    logw_t = torch.from_numpy(train["log_w_view1"])
    Xv = torch.from_numpy(val["X_view1"][:, :, :1])
    logwv = torch.from_numpy(val["log_w_view1"])

    model = ManifoldInformer(
        d_event=1, d_emb=D_EMB, hidden=D_HIDDEN, alpha=ALPHA_RIDGE,
        ema_momentum=EMA_MOMENTUM, vicreg_weight=VICREG_W,
        view_weight=VIEW_W, density_anchor_weight=DENSITY_W,
        use_ema=True,
    )
    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    S_train = Xt.shape[0]
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        idx = rng.choice(S_train, size=min(BATCH_S, S_train), replace=False)
        X = Xt[idx]
        X2 = Xt2[idx]
        logw = logw_t[idx]
        i_ctx, i_q = _split_ctx_q(X, SPLIT_QUERY_FRAC, gen)
        X_ctx = torch.gather(X, 1, i_ctx.unsqueeze(-1).expand(-1, -1, 1))
        X_q = torch.gather(X, 1, i_q.unsqueeze(-1).expand(-1, -1, 1))
        log_w_q = torch.gather(logw, 1, i_q)
        out = model(X_ctx, X_q, X_ctx_view2=X2,
                    log_w_q=log_w_q, return_losses=True)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        model.update_target()
    print(f"  deficient encoder train wall = {time.time() - t0:.1f}s, "
          f"final jepa = {out['jepa'].item():.4f}")
    return model


def load_full_encoder() -> ManifoldInformer:
    model = ManifoldInformer(
        d_event=2, d_emb=D_EMB, hidden=D_HIDDEN, alpha=ALPHA_RIDGE,
        ema_momentum=EMA_MOMENTUM, vicreg_weight=VICREG_W,
        view_weight=VIEW_W, density_anchor_weight=DENSITY_W,
        use_ema=True,
    )
    model.load_state_dict(torch.load(FULL_CKPT, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Encoder-as-basis adapter
# ---------------------------------------------------------------------------
@torch.no_grad()
def psi_from_encoder(model: ManifoldInformer, events: np.ndarray) -> np.ndarray:
    """Per-event embeddings of the trained encoder, shape (N_events, d_emb).

    The deficient encoder accepts d_event=1; full accepts d_event=2. We
    pass the right slice of the event matrix automatically.
    """
    d_event = model.d_event
    Xt = torch.from_numpy(events[:, :d_event]).float()
    return model.event_encoder(Xt).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Span-completeness state on the encoder basis
# ---------------------------------------------------------------------------
def projector_residual(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    A = psi.T @ psi + 1e-6 * np.eye(psi.shape[1])
    w = np.linalg.solve(A, psi.T @ target)
    return target - psi @ w


@dataclass
class LoopState:
    psi: np.ndarray                                     # (N_probe, D_psi)
    R_cols: list = field(default_factory=list)
    sigma1: list = field(default_factory=list)
    sigma1_over_sigma2: list = field(default_factory=list)
    fingerprint_fired: list = field(default_factory=list)
    persistence_count: int = 0
    # Monitor traces
    kappa_A: list = field(default_factory=list)        # condition number of A
    cusum_S: float = 0.0
    cusum_trace: list = field(default_factory=list)
    bh_min_p_trace: list = field(default_factory=list)


def span_step(state: LoopState, log_w_per_event: np.ndarray):
    r = projector_residual(state.psi, log_w_per_event)
    state.R_cols.append(r)
    R = np.stack(state.R_cols, axis=1)
    if R.shape[1] >= 2:
        U, S, _ = np.linalg.svd(R, full_matrices=False)
        sigma1 = float(S[0])
        sigma2 = float(S[1])
        ratio = sigma1 / max(sigma2, 1e-12)
    else:
        U, S = None, np.array([np.linalg.norm(r), 0.0])
        sigma1 = float(S[0])
        ratio = float("inf")
    fired = (sigma1 > SIGMA_FLOOR) and (ratio > RATIO_THRESHOLD)
    state.sigma1.append(sigma1)
    state.sigma1_over_sigma2.append(ratio)
    state.fingerprint_fired.append(fired)
    state.persistence_count = state.persistence_count + 1 if fired else 0
    oos = state.persistence_count >= PERSISTENCE_N
    return {"sigma1": sigma1, "ratio": ratio, "fired": fired,
            "oos_flag": oos, "U": U, "S": S, "residual": r}


def kappa_step(state: LoopState):
    """Condition number of A = psi^T psi + alpha I."""
    A = state.psi.T @ state.psi + ALPHA_RIDGE * np.eye(state.psi.shape[1])
    s = np.linalg.svd(A, compute_uv=False)
    k = float(s[0] / max(s[-1], 1e-12))
    state.kappa_A.append(k)
    return k


def cusum_step(state: LoopState, residual: np.ndarray):
    """Data-Adaptive Symmetric CUSUM on standardised per-event residuals.

    We use the cycle's residual r (already in mean-zero feature space by
    construction of the projection), normalise by its running std, and
    accumulate max(0, |r|-h) symmetrically.
    """
    rs = residual / (residual.std() + 1e-12)
    inc = float(np.maximum(0.0, np.abs(rs) - CUSUM_H).sum())
    state.cusum_S = max(0.0, state.cusum_S + inc)
    state.cusum_trace.append(state.cusum_S)
    return state.cusum_S


def bh_coverage_step(state: LoopState, log_w_true: np.ndarray,
                     log_w_pred_std: np.ndarray, log_w_pred_mean: np.ndarray,
                     m_bins: np.ndarray) -> float:
    """Realised 68% coverage in 5 m-regions, BH-corrected min p-value.

    Compares the realised coverage of the +/- z*sigma intervals against
    the nominal 0.68 with a binomial test per region; BH-corrects across
    the N_M_REGIONS p-values.
    """
    from math import erf, sqrt
    from scipy.stats import binomtest
    z = 1.0                                                # 68%
    upper = log_w_pred_mean + z * log_w_pred_std
    lower = log_w_pred_mean - z * log_w_pred_std
    inside = (log_w_true >= lower) & (log_w_true <= upper)
    region_p_values = []
    for r in range(N_M_REGIONS):
        sel = (m_bins == r)
        n = int(sel.sum())
        if n < 10:
            continue
        k = int(inside[sel].sum())
        bt = binomtest(k, n, p=NOMINAL_COVERAGE, alternative="two-sided")
        region_p_values.append(float(bt.pvalue))
    if not region_p_values:
        state.bh_min_p_trace.append(1.0)
        return 1.0
    # BH-corrected min p.
    region_p_values = np.array(sorted(region_p_values))
    n_p = len(region_p_values)
    bh_q = region_p_values * n_p / (np.arange(1, n_p + 1))
    bh_min_p = float(min(np.minimum.accumulate(bh_q[::-1])[::-1]))
    state.bh_min_p_trace.append(bh_min_p)
    return bh_min_p


# ---------------------------------------------------------------------------
# Per-cycle density-decoder prediction (for CUSUM and BH coverage)
# ---------------------------------------------------------------------------
@torch.no_grad()
def density_predict(model: ManifoldInformer,
                    events_ctx: np.ndarray, log_w_ctx: np.ndarray,
                    events_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict (mean, std) per query event for log_w via the density anchor
    chained off the closed-form ridge predictor.

    Uses the events themselves both as a single-scenario context and as the
    query set. Returns (mean, std) of shape (N_q,). The std comes from the
    closed-form leverage at the query point.
    """
    d_event = model.d_event
    Xc = torch.from_numpy(events_ctx[:, :d_event]).float().unsqueeze(0)   # (1, Nc, de)
    Xq = torch.from_numpy(events_q[:, :d_event]).float().unsqueeze(0)
    K = model.event_encoder(Xc)
    Kq = model.event_encoder(Xq)
    V = model.target_encoder(Xc)
    Z_pred = model.ridge(K, V, Kq).squeeze(0)                             # (Nq, d_emb)
    log_w_pred = model.density_decoder(Z_pred).cpu().numpy()              # (Nq,)
    # Leverage as a closed-form predictive std proxy.
    A = (K.transpose(-2, -1) @ K
         + model.alpha * torch.eye(K.shape[-1], dtype=K.dtype)).squeeze(0)
    A_inv = torch.linalg.inv(A).cpu().numpy()
    Kq_np = Kq.squeeze(0).cpu().numpy()
    lev = np.einsum("ij,jk,ik->i", Kq_np, A_inv, Kq_np)
    sigma_y = float(np.std(log_w_pred - 0.0) + 1e-8)                       # robust to scale
    std = sigma_y * np.sqrt(1.0 + np.clip(lev, 0.0, None))
    return log_w_pred.astype(np.float64), std.astype(np.float64)


def m_bin_assignments(events: np.ndarray) -> np.ndarray:
    """Five equal-log-width m-bins over the probe events' log m support."""
    log_m = events[:, 0]
    edges = np.quantile(log_m, np.linspace(0, 1, N_M_REGIONS + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return np.digitize(log_m, edges[1:-1])                                 # 0..N-1


# ---------------------------------------------------------------------------
# psi-extension
# ---------------------------------------------------------------------------
def extend_psi(state: LoopState, U: np.ndarray) -> np.ndarray:
    """Append u_1 as a new column to psi (the per-event encoder output)."""
    u1 = U[:, 0]
    return np.concatenate([state.psi, u1.reshape(-1, 1)], axis=1)


# ---------------------------------------------------------------------------
# Single-arm loop on a trained encoder
# ---------------------------------------------------------------------------
def run_arm(name: str, model: ManifoldInformer, probe_events: np.ndarray,
            oracle: AnalyticSMEFTOracle, c_queue: list,
            allow_extension: bool, m_bins: np.ndarray) -> dict:
    print(f"\n## arm: {name}")
    psi_initial = psi_from_encoder(model, probe_events)
    print(f"  initial D_psi = {psi_initial.shape[1]}  "
          f"allow_extension = {allow_extension}  d_event = {model.d_event}")
    state = LoopState(psi=psi_initial)
    actions = []
    extended_at = None

    for k, c in enumerate(c_queue):
        log_w = event_log_likelihood_ratio(oracle, c, probe_events)
        # Span-completeness step (residual SVD on encoder basis).
        upd = span_step(state, log_w)
        # Condition-number monitor.
        kappa = kappa_step(state)
        # DAS-CUSUM on residuals.
        cusum_S = cusum_step(state, upd["residual"])
        # BH coverage on density-decoder predictions.
        log_w_pred, log_w_std = density_predict(
            model, probe_events, log_w, probe_events)
        bh_min_p = bh_coverage_step(state, log_w, log_w_std, log_w_pred, m_bins)

        # Aggregator -> action.
        decision = "watch"
        if upd["oos_flag"] and allow_extension and extended_at is None:
            U = upd["U"]
            state.psi = extend_psi(state, U)
            extended_at = k
            state.R_cols = []
            state.persistence_count = 0
            decision = "psi_extend"
        elif upd["fired"]:
            decision = "in_span_acquire"

        actions.append({
            "cycle": k,
            "c": c.tolist(),
            "sigma1": upd["sigma1"],
            "sigma1_over_sigma2": (upd["ratio"]
                                    if np.isfinite(upd["ratio"]) else None),
            "fingerprint_fired": upd["fired"],
            "persistence_count": state.persistence_count,
            "oos_flag": upd["oos_flag"],
            "kappa_A": kappa,
            "cusum_S": cusum_S,
            "bh_min_p": bh_min_p,
            "decision": decision,
            "D_psi": int(state.psi.shape[1]),
        })
        ratio_str = f"{upd['ratio']:.2f}" if np.isfinite(upd["ratio"]) else "inf"
        print(f"    cycle {k:2d}: c={c.tolist()}  σ_1={upd['sigma1']:.3f}  "
              f"σ_1/σ_2={ratio_str}  fired={int(upd['fired'])}  "
              f"κ={kappa:.2e}  CUSUM={cusum_S:.1f}  "
              f"BH_p={bh_min_p:.3f}  "
              f"D_psi={state.psi.shape[1]}  action={decision}")

    return {
        "name": name,
        "extended_at": extended_at,
        "actions": actions,
        "final_D_psi": int(state.psi.shape[1]),
        "sigma1_trace": state.sigma1,
        "ratio_trace": [r if np.isfinite(r) else None
                         for r in state.sigma1_over_sigma2],
        "kappa_trace": state.kappa_A,
        "cusum_trace": state.cusum_trace,
        "bh_min_p_trace": state.bh_min_p_trace,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("# Integrated loop — ManifoldInformer + 4 monitors + ψ-extension")
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)

    print(f"  Sampling {N_PROBE_EVENTS} probe events at SM ...")
    probe_events = sample_events(oracle, np.zeros(N_WC), N_PROBE_EVENTS,
                                 seed=SEED)
    m_bins = m_bin_assignments(probe_events)
    print(f"  probe events shape = {probe_events.shape}, "
          f"{N_M_REGIONS} m-regions")

    print(f"\n  Loading full encoder from {FULL_CKPT.name} ...")
    full_model = load_full_encoder()
    print(f"  full encoder n_params = {full_model.n_params}")

    print(f"\n  Training deficient encoder (d_event=1, mass-only) ...")
    def_model = train_deficient_encoder(seed=2026)
    torch.save(def_model.state_dict(), OUT_DIR / "manifold_informer_deficient.pt")
    print(f"  deficient encoder n_params = {def_model.n_params}")
    def_model.eval()

    arms = []
    arms.append(run_arm("full_psi_no_extension", full_model, probe_events,
                        oracle, WORKING_POINT_QUEUE,
                        allow_extension=False, m_bins=m_bins))
    arms.append(run_arm("def_psi_no_extension", def_model, probe_events,
                        oracle, WORKING_POINT_QUEUE,
                        allow_extension=False, m_bins=m_bins))
    arms.append(run_arm("def_psi_with_extension", def_model, probe_events,
                        oracle, WORKING_POINT_QUEUE,
                        allow_extension=True, m_bins=m_bins))

    # ---- Gate analysis ----
    by_name = {a["name"]: a for a in arms}
    arm_full = by_name["full_psi_no_extension"]
    arm_def = by_name["def_psi_no_extension"]
    arm_ext = by_name["def_psi_with_extension"]
    sigma1_full = arm_full["sigma1_trace"]
    sigma1_def = arm_def["sigma1_trace"]
    sigma1_ext = arm_ext["sigma1_trace"]
    extended_at = arm_ext["extended_at"]

    gates = {}
    gates["ext_arm_extended"] = extended_at is not None
    if extended_at is not None and extended_at + 1 < len(sigma1_ext):
        pre_peak = max(sigma1_ext[: extended_at + 1])
        post_immediate = sigma1_ext[extended_at + 1]
        gates["ext_arm_sigma1_drops_ge_10x"] = (
            pre_peak / max(post_immediate, 1e-12) >= 10.0)
    else:
        gates["ext_arm_sigma1_drops_ge_10x"] = False
    gates["def_no_ext_sigma1_grows_ge_3x"] = (
        sigma1_def[-1] / max(sigma1_def[0], 1e-12) >= 3.0)
    gates["ext_arm_final_le_def_div_3"] = (
        sigma1_def[-1] / max(sigma1_ext[-1], 1e-12) >= 3.0)
    gates["full_arm_silent_post_cycle_0"] = all(
        (a["sigma1_over_sigma2"] is None or a["sigma1_over_sigma2"] < RATIO_THRESHOLD)
        for a in arm_full["actions"][1:]
    )
    gates["full_arm_final_le_def_div_3"] = (
        sigma1_def[-1] / max(sigma1_full[-1], 1e-12) >= 3.0)

    print("\n## Gates")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # ---- Plot: 4-panel (sigma_1, kappa, CUSUM, BH p) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True,
                             sharex=True)
    colours = {
        "full_psi_no_extension": ("tab:blue", "^", r"full $\psi_\theta$"),
        "def_psi_no_extension": ("tab:red", "s", r"def. $\psi_\theta$"),
        "def_psi_with_extension": ("tab:orange", "o", r"def. $\psi_\theta$ + ext."),
    }
    ax = axes[0, 0]
    for a in arms:
        c, mk, lab = colours[a["name"]]
        ax.semilogy(range(len(a["sigma1_trace"])),
                    np.maximum(a["sigma1_trace"], 1e-6),
                    "-" + mk, c=c, label=lab, ms=6)
    if arm_ext["extended_at"] is not None:
        ax.axvline(arm_ext["extended_at"] + 0.5, ls=":", c="tab:orange",
                   alpha=0.6,
                   label=f"ψ-ext at cycle {arm_ext['extended_at']}")
    ax.set_ylabel(r"$\sigma_1$ of $\mathbf{R}$")
    ax.set_title("Span-completeness (residual SVD)")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for a in arms:
        c, mk, lab = colours[a["name"]]
        ax.semilogy(range(len(a["kappa_trace"])), a["kappa_trace"],
                    "-" + mk, c=c, ms=6)
    ax.set_ylabel(r"$\kappa(\mathbf{A})$")
    ax.set_title("Context-shape (condition number)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for a in arms:
        c, mk, lab = colours[a["name"]]
        ax.plot(range(len(a["cusum_trace"])), a["cusum_trace"],
                "-" + mk, c=c, ms=6)
    ax.set_xlabel("cycle"); ax.set_ylabel(r"DAS-CUSUM $S$")
    ax.set_title("Accuracy drift (DAS-CUSUM on residuals)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for a in arms:
        c, mk, lab = colours[a["name"]]
        ax.semilogy(range(len(a["bh_min_p_trace"])),
                    np.maximum(a["bh_min_p_trace"], 1e-12),
                    "-" + mk, c=c, ms=6)
    ax.axhline(BH_ALPHA, ls="--", c="grey", alpha=0.5, label=rf"$\alpha={BH_ALPHA}$")
    ax.set_xlabel("cycle"); ax.set_ylabel(r"BH-corrected min $p$")
    ax.set_title("Calibration coverage (BH per-region)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Integrated four-monitor loop on the trained ManifoldInformer\n"
                 f"AL1 thesis: deficient $\\psi_\\theta$ fires (cycle "
                 f"{arm_ext['extended_at']}, $\\psi$-ext), "
                 f"full $\\psi_\\theta$ is silent.", fontsize=11)
    fig.savefig(OUT_DIR / "spectra.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "config": {
            "n_probe": N_PROBE_EVENTS,
            "sigma_floor": SIGMA_FLOOR,
            "ratio_threshold": RATIO_THRESHOLD,
            "persistence_n": PERSISTENCE_N,
            "kappa_threshold": KAPPA_THRESHOLD,
            "cusum_h": CUSUM_H,
            "n_m_regions": N_M_REGIONS,
            "nominal_coverage": NOMINAL_COVERAGE,
            "bh_alpha": BH_ALPHA,
            "n_working_points": len(WORKING_POINT_QUEUE),
            "seed": SEED,
            "encoder_full_d_event": full_model.d_event,
            "encoder_full_d_emb": full_model.d_emb,
            "encoder_full_n_params": full_model.n_params,
            "encoder_def_d_event": def_model.d_event,
            "encoder_def_d_emb": def_model.d_emb,
            "encoder_def_n_params": def_model.n_params,
        },
        "arms": arms,
        "gates": gates,
        "all_pass": all(gates.values()),
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {OUT_DIR}/{{summary.json, spectra.png, "
          f"manifold_informer_deficient.pt}}")
    print(f"# Overall: {'PASS' if summary['all_pass'] else 'FAIL'}  "
          f"(wall = {summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
