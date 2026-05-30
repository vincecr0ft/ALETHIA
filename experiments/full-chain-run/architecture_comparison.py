"""INV-6 consolidator — architecture_comparison.json (Claim A artifact).

Consolidates existing numbers from the Section 6 / Appendix C tables and the
matched-budget / scaled-JEPA runs into a single architecture-comparison row set.
The four closed-form properties (CFV, CIG, CSM, EYS) are set by architecture
definition (each carries a `reason` string). Other columns are loaded from the
artefacts listed below; missing values are explicitly marked.

No new training. The only fresh compute is for the fixed-ψ Intention row,
which has no trained parameters: the y-shuffle cosine and c-recoverability
linear probe are evaluated against the closed-form ridge weight w on the
existing repr_analysis splits.

Sources:
  experiments/intention-vs-deepsets/output_smeft/summary.json
  experiments/intention-vs-deepsets/output_jepa_scaling/summary.json
  experiments/intention-vs-deepsets/output_representation_analysis/summary.json
  experiments/intention-vs-deepsets/output_representation_analysis/splits.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
IVD = REPO / "experiments" / "intention-vs-deepsets"
OUT = HERE / "output" / "architecture_comparison.json"

# Make IntentionFMFixed importable for the fixed-ψ fresh compute.
sys.path.insert(0, str(IVD))


# ------- closed-form property definitions (architecture-defined) -------------
CFV_YES_REASON = (
    "ridge solve A = Ψᵀ Ψ + α I gives leverage(q) = ψ(q)ᵀ A⁻¹ ψ(q); "
    "predictive variance is σ_y² (1 + leverage)"
)
CFV_NO_REASON_POOL = (
    "no design matrix — encoder + pooling aggregator has no analytic "
    "leverage / predictive variance"
)
CFV_NO_REASON_REG = (
    "regressor consumes c on the forward pass; not c-agnostic, no "
    "context-conditional posterior over c"
)
CIG_YES_REASON = (
    "EPIG closed form follows from leverage via the matrix determinant "
    "lemma (paper Eq. 7 / INV-3 ΔH_D)"
)
CIG_NO_REASON_POOL = "no analytic information-gain functional over the pooled summary"
CIG_NO_REASON_REG = (
    "no posterior over c to define information gain against; c is the input"
)
CSM_YES_REASON = (
    "Sherman-Morrison rank-one update A_new⁻¹ = A⁻¹ − (A⁻¹ψ ψᵀ A⁻¹)/(1+lev) "
    "(paper Eq. 8)"
)
CSM_NO_REASON_POOL = (
    "encoder + pooler is opaque to a rank-one context update; no closed "
    "sequential refit"
)
CSM_NO_REASON_REG = (
    "no inverted design matrix to update; the regressor consumes c, not a "
    "ridge solve over context"
)
EYS_YES_REASON = (
    "Eckart-Young bounds the sensitivity of A⁻¹ via its condition number; "
    "drift detector uses κ(A) directly"
)
EYS_NO_REASON_POOL = (
    "no design matrix and no analytic condition number; sensitivity is "
    "encoder-Jacobian dependent, not closed form"
)
EYS_NO_REASON_REG = (
    "no design matrix; the regressor has no condition-number sensitivity tied "
    "to a closed-form posterior"
)


# ------- I/O ----------------------------------------------------------------
def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# ------- fixed-ψ fresh compute (cheap, additive, no training) ---------------
def _compute_fixed_psi_alignment_and_recoverability() -> dict:
    """Fixed-ψ Intention: closed-form ridge solve with hand-engineered
    polynomial basis. No trainable parameters. Compute on the same
    train/in/out splits used by repr_analysis so the numbers are
    directly comparable to the other rows."""
    from intention_learned import IntentionFMFixed

    splits_path = IVD / "output_representation_analysis" / "splits.pt"
    bundle = torch.load(splits_path, weights_only=False)

    fixed = IntentionFMFixed(K_X=5, M_REF=1.0, alpha=1e-3)
    D = fixed.K_X

    def ridge_weights(t: dict) -> np.ndarray:
        """Return w in R^{S, D} from the closed-form ridge solve."""
        M = t["M_ctx"].cpu().numpy()
        Y = t["Y_ctx"].cpu().numpy()
        S, K = M.shape
        W = np.zeros((S, D), dtype=np.float64)
        for s in range(S):
            Psi = fixed.psi(M[s])  # (K, D)
            A = Psi.T @ Psi + fixed.alpha * np.eye(D)
            W[s] = np.linalg.solve(A, Psi.T @ Y[s].astype(np.float64))
        return W

    W_train = ridge_weights(bundle["train_t"])
    W_in = ridge_weights(bundle["eval_in_t"])
    W_out = ridge_weights(bundle["eval_out_t"])

    # y-permutation invariance (cosine sim under per-scenario Y shuffle).
    rng = np.random.default_rng(7)
    R0 = W_in
    R0n = R0 / (np.linalg.norm(R0, axis=1, keepdims=True) + 1e-12)
    sims = []
    n_perms = 5
    for _ in range(n_perms):
        Y_perm = bundle["eval_in_t"]["Y_ctx"].clone()
        for s_i in range(Y_perm.size(0)):
            perm = rng.permutation(Y_perm.size(1))
            Y_perm[s_i] = Y_perm[s_i, perm]
        t_perm = {**bundle["eval_in_t"], "Y_ctx": Y_perm}
        Rp = ridge_weights(t_perm)
        Rpn = Rp / (np.linalg.norm(Rp, axis=1, keepdims=True) + 1e-12)
        sims.append(float(np.mean(np.sum(R0n * Rpn, axis=1))))
    y_perm_cos = float(np.mean(sims))

    # c-recoverability via linear probe trained on (W_train, c_train).
    c_train = np.stack([s["c"] for s in bundle["train_scen"]]).astype(np.float64)
    c_in = np.stack([s["c"] for s in bundle["eval_in_scen"]]).astype(np.float64)
    c_out = np.stack([s["c"] for s in bundle["eval_out_scen"]]).astype(np.float64)

    # Closed-form least-squares linear probe to match the linear setting
    # used in repr_analysis (the MLP probe is not relevant for INV-6).
    X = np.hstack([W_train, np.ones((W_train.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(X, c_train, rcond=None)

    def joint_r2(W_eval: np.ndarray, c_eval: np.ndarray) -> float:
        X_eval = np.hstack([W_eval, np.ones((W_eval.shape[0], 1))])
        yp = X_eval @ coef
        ss_res = float(np.sum((yp - c_eval) ** 2))
        ss_tot = float(np.sum((c_eval - c_eval.mean(axis=0, keepdims=True)) ** 2))
        return float(1.0 - ss_res / max(ss_tot, 1e-12))

    return {
        "y_perm_cos_in": y_perm_cos,
        "alignment_retained": 1.0 - y_perm_cos,
        "c_recov_linear_in": joint_r2(W_in, c_in),
        "c_recov_linear_out": joint_r2(W_out, c_out),
    }


# ------- assemble rows ------------------------------------------------------
def build_rows() -> list[dict[str, Any]]:
    smeft = load_json(IVD / "output_smeft" / "summary.json")
    jepa_scale = load_json(IVD / "output_jepa_scaling" / "summary.json")
    repr_sum = load_json(IVD / "output_representation_analysis" / "summary.json")

    # repr_analysis-side numbers (c-recoverability, y-shuffle cosine).
    yperm = repr_sum["y_permutation_invariance"]
    crec = repr_sum["c_recoverability"]
    eff_rank = {
        k: v["effective_rank_1pct"] for k, v in repr_sum["diagnostics"].items()
    }

    rows: list[dict[str, Any]] = []

    # ---- 1. closed-form Intention, learned ψ ------------------------------
    int_l = smeft["IntentionFM_Learned"]
    rows.append({
        "architecture": "Intention (closed-form, learned ψ_θ)",
        "row_id": "intention_learned",
        "closed_form_predictive_variance": {
            "yes": True, "reason": CFV_YES_REASON},
        "closed_form_information_gain": {
            "yes": True, "reason": CIG_YES_REASON},
        "closed_form_sequential_update": {
            "yes": True, "reason": CSM_YES_REASON},
        "eckart_young_sensitivity": {
            "yes": True, "reason": EYS_YES_REASON},
        "alignment_retained": 1.0 - yperm["Intention_w"],
        "y_shuffle_cosine": yperm["Intention_w"],
        "alignment_source": (
            "experiments/intention-vs-deepsets/output_representation_analysis/"
            "summary.json (Intention_w; architecturally identical to the "
            "output_smeft row)"),
        "r2_median_in": int_l["in"]["r2_median"],
        "r2_p5_in": int_l["in"]["r2_p5"],
        "r2_median_out": int_l["out"]["r2_median"],
        "r2_p5_out": int_l["out"]["r2_p5"],
        "c_recoverability_in_linear": crec["Intention_w"]["linear_in"]["r2_joint"],
        "c_recoverability_out_linear": crec["Intention_w"]["linear_out"]["r2_joint"],
        "effective_rank": eff_rank["Intention_w"],
        "n_params": int_l["n_params"],
        "caveats": [],
    })

    # ---- 2. fixed-ψ Intention (closed-form, hand-engineered polynomial) ---
    print("# computing fixed-ψ alignment and c-recoverability (no training)",
          flush=True)
    fixed_calc = _compute_fixed_psi_alignment_and_recoverability()
    int_f = smeft["IntentionFM_Fixed"]
    rows.append({
        "architecture": "Intention (closed-form, fixed ψ — polynomial basis)",
        "row_id": "intention_fixed",
        "closed_form_predictive_variance": {
            "yes": True, "reason": CFV_YES_REASON},
        "closed_form_information_gain": {
            "yes": True, "reason": CIG_YES_REASON},
        "closed_form_sequential_update": {
            "yes": True, "reason": CSM_YES_REASON},
        "eckart_young_sensitivity": {
            "yes": True, "reason": EYS_YES_REASON},
        "alignment_retained": fixed_calc["alignment_retained"],
        "y_shuffle_cosine": fixed_calc["y_perm_cos_in"],
        "alignment_source": (
            "computed fresh by architecture_comparison.py on "
            "output_representation_analysis/splits.pt — the model has no "
            "trainable parameters, so no checkpoint is required"),
        "r2_median_in": int_f["in"]["r2_median"],
        "r2_p5_in": int_f["in"]["r2_p5"],
        "r2_median_out": int_f["out"]["r2_median"],
        "r2_p5_out": int_f["out"]["r2_p5"],
        "c_recoverability_in_linear": fixed_calc["c_recov_linear_in"],
        "c_recoverability_out_linear": fixed_calc["c_recov_linear_out"],
        "effective_rank": None,
        "n_params": int_f["n_params"],  # 0
        "caveats": [
            "fixed-ψ effective rank not computed (the polynomial basis is "
            "5-dimensional by construction)",
        ],
    })

    # ---- 3. Deep Sets matched-budget --------------------------------------
    ds = smeft["DeepSets_FM"]
    rows.append({
        "architecture": "Deep Sets (matched-budget pooling FM)",
        "row_id": "deepsets_matched",
        "closed_form_predictive_variance": {
            "yes": False, "reason": CFV_NO_REASON_POOL},
        "closed_form_information_gain": {
            "yes": False, "reason": CIG_NO_REASON_POOL},
        "closed_form_sequential_update": {
            "yes": False, "reason": CSM_NO_REASON_POOL},
        "eckart_young_sensitivity": {
            "yes": False, "reason": EYS_NO_REASON_POOL},
        "alignment_retained": 1.0 - yperm["DeepSets_z"],
        "y_shuffle_cosine": yperm["DeepSets_z"],
        "alignment_source": (
            "experiments/intention-vs-deepsets/output_representation_analysis/"
            "summary.json (DeepSets_z)"),
        "r2_median_in": ds["in"]["r2_median"],
        "r2_p5_in": ds["in"]["r2_p5"],
        "r2_median_out": ds["out"]["r2_median"],
        "r2_p5_out": ds["out"]["r2_p5"],
        "c_recoverability_in_linear": crec["DeepSets_z"]["linear_in"]["r2_joint"],
        "c_recoverability_out_linear": crec["DeepSets_z"]["linear_out"]["r2_joint"],
        "effective_rank": eff_rank["DeepSets_z"],
        "n_params": ds["n_params"],
        "caveats": [],
    })

    # ---- 4. constraint-violating regressor --------------------------------
    reg = smeft["IntentionFM_Regressor_cheat"]
    rows.append({
        "architecture": "Constraint-violating regressor (c on forward pass)",
        "row_id": "regressor_cheat",
        "closed_form_predictive_variance": {
            "yes": False, "reason": CFV_NO_REASON_REG},
        "closed_form_information_gain": {
            "yes": False, "reason": CIG_NO_REASON_REG},
        "closed_form_sequential_update": {
            "yes": False, "reason": CSM_NO_REASON_REG},
        "eckart_young_sensitivity": {
            "yes": False, "reason": EYS_NO_REASON_REG},
        "alignment_retained": None,
        "y_shuffle_cosine": None,
        "alignment_source": (
            "N/A — the regressor consumes c on the forward pass and has no "
            "Y_ctx slot; y-shuffle on the context is undefined"),
        "r2_median_in": reg["in"]["r2_median"],
        "r2_p5_in": reg["in"]["r2_p5"],
        "r2_median_out": reg["out"]["r2_median"],
        "r2_p5_out": reg["out"]["r2_p5"],
        "c_recoverability_in_linear": None,
        "c_recoverability_out_linear": None,
        "effective_rank": None,
        "n_params": reg["n_params"],
        "caveats": [
            "alignment_retained = N/A: no (M_ctx, Y_ctx) context summary",
            ("c_recoverability = N/A: c is the architecture's input, not "
             "something to probe a context for"),
        ],
    })

    # ---- 5. scaled JEPA-FM (surviving INV-5 point) ------------------------
    jepa = jepa_scale["E_scale_all_jepa_tx_big"]
    rows.append({
        "architecture": "Scaled JEPA-FM (transformer, 186k params, 10k scenarios)",
        "row_id": "jepa_scaled",
        "closed_form_predictive_variance": {
            "yes": False, "reason": CFV_NO_REASON_POOL},
        "closed_form_information_gain": {
            "yes": False, "reason": CIG_NO_REASON_POOL},
        "closed_form_sequential_update": {
            "yes": False, "reason": CSM_NO_REASON_POOL},
        "eckart_young_sensitivity": {
            "yes": False, "reason": EYS_NO_REASON_POOL},
        "alignment_retained": 1.0 - yperm["JEPA_summary_s"],
        "y_shuffle_cosine": yperm["JEPA_summary_s"],
        "alignment_source": (
            "experiments/intention-vs-deepsets/output_representation_analysis/"
            "summary.json (JEPA_summary_s) — same transformer architecture "
            "(d_emb=64, hidden=128, 2 layers, 4 heads) as the 186k/10k "
            "scaled-JEPA headline; the repr_analysis variant trains on 2k "
            "scenarios / 6k steps rather than 10k/10k"),
        "r2_median_in": jepa["in"]["r2_median"],
        "r2_p5_in": jepa["in"]["r2_p5"],
        "r2_median_out": jepa["out"]["r2_median"],
        "r2_p5_out": jepa["out"]["r2_p5"],
        "c_recoverability_in_linear":
            crec["JEPA_summary_s"]["linear_in"]["r2_joint"],
        "c_recoverability_out_linear":
            crec["JEPA_summary_s"]["linear_out"]["r2_joint"],
        "effective_rank": eff_rank["JEPA_summary_s"],
        "n_params": jepa["n_params"],
        "caveats": [
            ("alignment_retained, c_recoverability, effective_rank sourced "
             "from architecturally identical JEPA_summary_s in repr_analysis "
             "(2k scenarios / 6k steps); the 186k/10k headline run "
             "(E_scale_all_jepa_tx_big) does not carry a y-shuffle or "
             "c-probe measurement of its own"),
        ],
    })

    return rows


def order_rows_by_alignment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort descending by alignment_retained. Rows with None alignment
    (architecturally undefined / unknown) are placed last."""
    def key(row: dict[str, Any]) -> tuple[int, float]:
        a = row.get("alignment_retained")
        if a is None:
            return (1, 0.0)  # last
        return (0, -float(a))  # higher alignment first
    return sorted(rows, key=key)


# ------- main ---------------------------------------------------------------
def main() -> None:
    rows = build_rows()
    rows = order_rows_by_alignment(rows)

    artifact = {
        "claim": (
            "Claim A — the system compares different foundation-model "
            "readings of the same (M_ctx, Y_ctx) context; the four analytic "
            "properties (closed-form predictive variance, information gain, "
            "sequential update, Eckart-Young sensitivity) are not generic "
            "across them, and track M–Y-alignment retention together with "
            "c-recoverability."
        ),
        "columns": {
            "closed_form_predictive_variance":
                "yes/no by architecture definition + reason",
            "closed_form_information_gain":
                "yes/no by architecture definition + reason",
            "closed_form_sequential_update":
                "yes/no by architecture definition + reason "
                "(Sherman-Morrison)",
            "eckart_young_sensitivity":
                "yes/no by architecture definition + reason",
            "alignment_retained":
                "1 − cosine similarity under per-scenario y-shuffle "
                "(higher = uses M-Y joint assignment more)",
            "r2_median_in / r2_p5_in / r2_median_out / r2_p5_out":
                "held-out R² on 50/50 in/out scenarios (Table 4)",
            "c_recoverability_in_linear / c_recoverability_out_linear":
                "joint R² of a linear c-probe (4 Wilson directions) "
                "fitted on training-scenario representations",
            "effective_rank":
                "singular values above 1% of max, from repr_analysis "
                "diagnostics",
            "n_params": "trained parameter count (0 for fixed-ψ)",
        },
        "sources": [
            "experiments/intention-vs-deepsets/output_smeft/summary.json",
            "experiments/intention-vs-deepsets/output_jepa_scaling/summary.json",
            "experiments/intention-vs-deepsets/output_representation_analysis/summary.json",
            "experiments/intention-vs-deepsets/output_representation_analysis/splits.pt",
        ],
        "ordering": "rows sorted descending by alignment_retained; None last",
        "global_caveats": [
            ("fixed-ψ alignment and c-recoverability are computed fresh by "
             "this script (the model has no trainable parameters, so this "
             "is closed-form evaluation, not training)"),
            ("scaled-JEPA alignment / c-recoverability / effective rank are "
             "from JEPA_summary_s in repr_analysis (same transformer "
             "architecture as the 186k/10k headline E_scale_all_jepa_tx_big "
             "but trained on 2k/6k); the 186k/10k held-out R² and parameter "
             "count are from output_jepa_scaling"),
            ("constraint-violating regressor has architecturally undefined "
             "alignment and c-recoverability (it consumes c on the forward "
             "pass and has no Y_ctx slot); these cells are N/A by design"),
        ],
        "rows": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(artifact, f, indent=2)
    print(f"wrote {OUT.relative_to(REPO)}", flush=True)

    # Brief stdout summary.
    print()
    print(f"{'rank':<5}{'architecture':<55}{'align':>8}"
          f"{'R²_med_in':>10}{'C_in':>8}{'params':>10}")
    for i, r in enumerate(rows):
        a = r["alignment_retained"]
        a_s = f"{a:.3f}" if a is not None else "N/A"
        c = r["c_recoverability_in_linear"]
        c_s = f"{c:.3f}" if c is not None else "N/A"
        print(f"{i+1:<5}{r['architecture'][:54]:<55}{a_s:>8}"
              f"{r['r2_median_in']:>10.4f}{c_s:>8}{r['n_params']:>10d}")


if __name__ == "__main__":
    main()
