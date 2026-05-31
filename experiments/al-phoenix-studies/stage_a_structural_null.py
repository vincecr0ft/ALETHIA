"""Stage A — structural-null proofs + the four pre-sweep diagnostics from the
postmortem (active-learning-postmortem.md §4). Cheap; runs in under a minute.

Emits Phoenix spans into project ``alethia-al-studies``:
  - chain.al.stage_a.label_independence
  - chain.al.stage_a.eig_spread_original
  - chain.al.stage_a.order_irrelevance
  - chain.al.stage_a.fisher_eigvals
  - chain.al.stage_a.single_context_contraction
  - chain.al.stage_a.probe_vs_mle

Writes a JSON summary to output/stage_a_summary.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import (  # type: ignore  # noqa: E402
    K_CTX, M_RANGE, N_WC, OUT, SIGMA_Y,
    build_oracle, build_pool_P0, ig_per_candidate, load_pretrained_model,
    load_probe, sigma_ctilde, target_context_mu_fb, truth_mu_fb, tracer,
)
from modules.surrogate.intention import (  # noqa: E402
    epig_acquire_m, param_epig_d_acquire, param_epig_a_acquire,
)

SEED = int(os.environ.get("SEED", "2026"))


def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


def _set_attrs(span, prefix: str, payload: dict) -> None:
    for k, v in _flatten(payload).items():
        if isinstance(v, (list, tuple, np.ndarray)):
            v = [float(x) for x in np.asarray(v).ravel()]
            span.set_attribute(f"{prefix}.{k}", v)
        elif isinstance(v, (int, float, bool, str)):
            span.set_attribute(f"{prefix}.{k}", v)
        elif isinstance(v, np.generic):
            span.set_attribute(f"{prefix}.{k}", float(v))


# ---------- A.1 Label independence ----------

def stage_a1_label_independence(model, oracle, probe) -> dict:
    rng = np.random.default_rng(SEED)
    target_c, M_ctx, Y_ctx_true = target_context_mu_fb(rng, oracle)

    Y_perm = rng.permutation(Y_ctx_true)
    Y_noise = rng.standard_normal(len(M_ctx)) * float(np.std(Y_ctx_true))

    S_true = sigma_ctilde(model, M_ctx, Y_ctx_true, probe["P"],
                          probe["sigma_y"], resolved_dim=2)
    S_perm = sigma_ctilde(model, M_ctx, Y_perm, probe["P"],
                          probe["sigma_y"], resolved_dim=2)
    S_noise = sigma_ctilde(model, M_ctx, Y_noise, probe["P"],
                           probe["sigma_y"], resolved_dim=2)
    diff_perm = float(np.max(np.abs(S_true - S_perm)))
    diff_noise = float(np.max(np.abs(S_true - S_noise)))

    return {
        "Sigma_true_diag": np.diag(S_true).tolist(),
        "max_abs_diff_perm": diff_perm,
        "max_abs_diff_noise": diff_noise,
        "passes_label_independence": diff_perm < 1e-10 and diff_noise < 1e-10,
    }


# ---------- A.2 EIG-spread on the original sweep pool ----------

def stage_a2_eig_spread_p0(model, oracle, probe) -> dict:
    rng = np.random.default_rng(SEED)
    target_c, M_ctx, Y_ctx = target_context_mu_fb(rng, oracle)
    pool = build_pool_P0(rng, size=30)
    ig = ig_per_candidate(model, M_ctx, Y_ctx, pool)
    mu = float(np.mean(ig))
    sd = float(np.std(ig))
    med = float(np.median(ig))
    cv = sd / mu if mu > 0 else float("inf")
    top5 = float(np.sort(ig)[-5:].mean())
    return {
        "n_pool": int(len(pool)),
        "ig_mean": mu, "ig_std": sd, "ig_median": med,
        "ig_min": float(ig.min()), "ig_max": float(ig.max()),
        "ig_cv": cv,
        "ig_top5_mean": top5,
        "spread_topk_over_median": (top5 - med) / med if med > 0 else float("inf"),
        "ratio_max_minus_min_over_median": (float(ig.max()) - float(ig.min())) / med if med > 0 else float("inf"),
        # AL_separation §A.2 prediction: CV << 1 on the homogeneous in-region
        # pool. Recorded so the falsification is auditable from the trace.
        "al_separation_prediction": "CV(IG) << 1",
        "al_separation_prediction_holds": cv < 0.3,
        "ig_distribution": ig.tolist(),
    }


# ---------- A.3 Order irrelevance ----------

def stage_a3_order_irrelevance(model, oracle, probe) -> dict:
    rng = np.random.default_rng(SEED)
    target_c, M_ctx, Y_ctx = target_context_mu_fb(rng, oracle)
    pool = build_pool_P0(rng, size=30)
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)

    # Greedy EPIG forwards.
    picks_fwd = epig_acquire_m(model, M_ctx, Y_ctx, pool, M_target, k=8)
    # Greedy EPIG over the reverse-order pool.
    pool_rev = pool[::-1].copy()
    picks_rev_idx = epig_acquire_m(model, M_ctx, Y_ctx, pool_rev, M_target, k=8)
    picks_rev = (len(pool) - 1) - picks_rev_idx  # map back to forward indices

    # Terminal contexts and Σ_c̃ on the resolved subspace.
    M_fwd_ctx = np.concatenate([M_ctx, pool[picks_fwd]])
    M_rev_ctx = np.concatenate([M_ctx, pool[picks_rev]])
    Y_fwd_ctx = truth_mu_fb(oracle, target_c, M_fwd_ctx)
    Y_rev_ctx = truth_mu_fb(oracle, target_c, M_rev_ctx)
    S_fwd = sigma_ctilde(model, M_fwd_ctx, Y_fwd_ctx, probe["P"],
                         probe["sigma_y"], resolved_dim=2)
    S_rev = sigma_ctilde(model, M_rev_ctx, Y_rev_ctx, probe["P"],
                         probe["sigma_y"], resolved_dim=2)
    set_fwd = set(picks_fwd.tolist())
    set_rev = set(picks_rev.tolist())
    same_set = set_fwd == set_rev
    diff_S = float(np.max(np.abs(S_fwd - S_rev))) if same_set else float("nan")
    return {
        "picks_forward": picks_fwd.tolist(),
        "picks_reverse_mapped": picks_rev.tolist(),
        "same_set": same_set,
        "max_abs_diff_Sigma": diff_S,
        "Sigma_fwd_diag": np.diag(S_fwd).tolist(),
        "Sigma_rev_diag": np.diag(S_rev).tolist(),
        # Order-irrelevance holds when the *set* of picks is identical;
        # greedy with a tie can pick a different element from a tie class.
        "passes_order_irrelevance": same_set and (diff_S is not float("nan")),
    }


# ---------- A.4 Fisher eigenvalues on mu_FB at K=12 (postmortem §4.1) ----------

def stage_a4_fisher_eigvals(probe) -> dict:
    lam = np.asarray(probe["lam_fisher"], dtype=float)
    # The postmortem uses "data-dominated iff λ_K · σ²_prior >> 1" with
    # σ_prior = 0.7 / sqrt(3) (uniform [-0.7, 0.7] per direction). We report
    # both the per-trace product and the K=12 multiplier.
    sigma_prior_sq = (0.7 ** 2) / 3.0
    ratio_per_trace = lam * sigma_prior_sq
    ratio_at_K = K_CTX * ratio_per_trace
    n_data_dominated = int((ratio_at_K > 1.0).sum())
    return {
        "fisher_eigenvalues_desc": lam.tolist(),
        "sigma_prior_sq": sigma_prior_sq,
        "K_ctx": K_CTX,
        "lambda_sigmaprior2_per_trace": ratio_per_trace.tolist(),
        "lambda_sigmaprior2_at_K": ratio_at_K.tolist(),
        "n_data_dominated_at_K": n_data_dominated,
        "postmortem_check_passes": n_data_dominated >= 4,
        "note": ("postmortem §4.1: c̃_a is data-dominated at context size K "
                 "iff K · λ_a · σ²_prior >> 1; here K=12 and σ²_prior=0.7²/3."),
    }


# ---------- A.5 Single-context contraction per acquisition (postmortem §4.2) ----------

def stage_a5_single_context_contraction(model, oracle, probe) -> dict:
    rng = np.random.default_rng(SEED)
    target_c, M_ctx, Y_ctx = target_context_mu_fb(rng, oracle)
    pool = build_pool_P0(rng, size=30)
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    S_pre = sigma_ctilde(model, M_ctx, Y_ctx, probe["P"], probe["sigma_y"],
                          resolved_dim=2)
    diag_pre = np.diag(S_pre)

    def with_pick(pick_m: np.ndarray) -> np.ndarray:
        Y_pick = truth_mu_fb(oracle, target_c, pick_m)
        M_new = np.concatenate([M_ctx, pick_m])
        Y_new = np.concatenate([Y_ctx, Y_pick])
        return sigma_ctilde(model, M_new, Y_new, probe["P"], probe["sigma_y"],
                            resolved_dim=2)

    methods: dict[str, np.ndarray] = {}
    # k=1 pick per method.
    methods["random"] = pool[rng.integers(0, len(pool), size=1)]
    Psi_pool = model.psi_np(pool)
    A_inv, _, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    lev = np.einsum("pd,de,pe->p", Psi_pool, A_inv, Psi_pool)
    methods["leverage"] = pool[[int(np.argmax(lev))]]
    methods["epig"] = pool[epig_acquire_m(model, M_ctx, Y_ctx, pool, M_target, k=1)]
    methods["param_epig_d"] = pool[
        param_epig_d_acquire(model, M_ctx, Y_ctx, pool, probe["P"], k=1,
                             sigma_y=probe["sigma_y"], resolved_dim=2)]
    methods["param_epig_a_d1"] = pool[
        param_epig_a_acquire(model, M_ctx, Y_ctx, pool, probe["P"], k=1,
                              target_direction=1, sigma_y=probe["sigma_y"])]

    contraction = {}
    for name, pick_m in methods.items():
        S_post = with_pick(pick_m)
        diag_post = np.diag(S_post)
        contraction[name] = {
            "pick_m": pick_m.tolist(),
            "Sigma_pre_diag": diag_pre.tolist(),
            "Sigma_post_diag": diag_post.tolist(),
            "ratio_post_over_pre_per_direction": (diag_post / diag_pre).tolist(),
        }

    # Postmortem §4.2 gate: ratios should differ across methods on the targeted
    # direction by more than ~0.1. We compute it on direction 1 (c̃_2 in the
    # 1-indexed table — the second-most-resolved, where param_epig_a targets).
    ratios_d1 = np.array(
        [contraction[m]["ratio_post_over_pre_per_direction"][1]
         for m in methods])
    spread_d1 = float(ratios_d1.max() - ratios_d1.min())
    return {
        "contraction": contraction,
        "spread_ratio_d1": spread_d1,
        "passes_postmortem_4_2": spread_d1 > 0.1,
        "note": ("postmortem §4.2: if the post/pre ratios on the targeted "
                 "direction don't differ by >0.1 across methods, no sweep will "
                 "separate them at any seed count."),
    }


# ---------- A.6 Probe vs MLE on the targeted direction (postmortem §4.3) ----------

def stage_a6_probe_vs_mle(model, oracle, probe) -> dict:
    """Reports the per-direction posterior std (the probe's ceiling for
    contraction-based scoring) alongside the analytic MLE residual std on
    the same context. The postmortem flags a 2x gap as the bottleneck."""
    rng = np.random.default_rng(SEED)
    target_c, M_ctx, Y_ctx = target_context_mu_fb(rng, oracle)
    pool = build_pool_P0(rng, size=30)
    # Grow the context with 8 EPIG picks so we measure the "after some
    # acquisition" probe vs MLE — the regime the postmortem cares about.
    M_target = np.linspace(M_RANGE[0] + 0.05, M_RANGE[1] - 0.05, 50)
    picks = epig_acquire_m(model, M_ctx, Y_ctx, pool,
                            M_target, k=min(8, len(pool) - 1))
    M_aug = np.concatenate([M_ctx, pool[picks]])
    Y_aug = truth_mu_fb(oracle, target_c, M_aug)

    S = sigma_ctilde(model, M_aug, Y_aug, probe["P"], probe["sigma_y"],
                      resolved_dim=2)
    posterior_std = np.sqrt(np.maximum(np.diag(S), 0.0))

    # Analytic MLE std via Fisher info on the actual context: σ_a ≈ σ_y /
    # sqrt(K * λ_a) for the per-direction Fisher eigenvalue λ_a.
    lam = np.asarray(probe["lam_fisher"], dtype=float)
    K = len(M_aug)
    mle_std = probe["sigma_y"] / np.sqrt(K * np.maximum(lam[:2], 1e-30))

    ratio = posterior_std / np.maximum(mle_std, 1e-30)
    return {
        "context_size_after_picks": K,
        "posterior_std_per_direction": posterior_std.tolist(),
        "mle_std_per_direction_approx": mle_std.tolist(),
        "probe_over_mle_ratio": ratio.tolist(),
        "passes_postmortem_4_3": bool(np.all(ratio < 2.0)),
        "note": ("postmortem §4.3: if probe σ / MLE σ > 2 on the target "
                 "direction, the probe is the bottleneck, not the acquisition "
                 "rule."),
    }


# ---------- Driver ----------

def main() -> dict:
    model = load_pretrained_model()
    oracle = build_oracle(seed=SEED)
    probe = load_probe(oracle)

    summary: dict[str, Any] = {"seed": SEED, "probe_path": probe["probe_path"]}
    tr = tracer()

    with tr.start_as_current_span("chain.al.stage_a.run") as root:
        root.set_attribute("aletheia.al.stage", "A")
        root.set_attribute("aletheia.al.seed", SEED)

        with tr.start_as_current_span("chain.al.stage_a.label_independence") as sp:
            res = stage_a1_label_independence(model, oracle, probe)
            _set_attrs(sp, "aletheia.al.label_independence", res)
            summary["A1_label_independence"] = res

        with tr.start_as_current_span("chain.al.stage_a.eig_spread_original") as sp:
            res = stage_a2_eig_spread_p0(model, oracle, probe)
            # The full distribution is large; emit summary stats only as span
            # attributes, keep the full list in JSON.
            stats = {k: v for k, v in res.items() if k != "ig_distribution"}
            _set_attrs(sp, "aletheia.al.eig_spread_original", stats)
            summary["A2_eig_spread_p0"] = res

        with tr.start_as_current_span("chain.al.stage_a.order_irrelevance") as sp:
            res = stage_a3_order_irrelevance(model, oracle, probe)
            _set_attrs(sp, "aletheia.al.order_irrelevance", res)
            summary["A3_order_irrelevance"] = res

        with tr.start_as_current_span("chain.al.stage_a.fisher_eigvals") as sp:
            res = stage_a4_fisher_eigvals(probe)
            _set_attrs(sp, "aletheia.al.fisher", res)
            summary["A4_fisher_eigvals"] = res

        with tr.start_as_current_span("chain.al.stage_a.single_context_contraction") as sp:
            res = stage_a5_single_context_contraction(model, oracle, probe)
            # Flatten only top-level summary attrs.
            sp.set_attribute("aletheia.al.contraction.spread_ratio_d1",
                             res["spread_ratio_d1"])
            sp.set_attribute("aletheia.al.contraction.passes_postmortem_4_2",
                             res["passes_postmortem_4_2"])
            for name, payload in res["contraction"].items():
                sp.set_attribute(
                    f"aletheia.al.contraction.{name}.ratio_d0",
                    float(payload["ratio_post_over_pre_per_direction"][0]))
                sp.set_attribute(
                    f"aletheia.al.contraction.{name}.ratio_d1",
                    float(payload["ratio_post_over_pre_per_direction"][1]))
            summary["A5_single_context_contraction"] = res

        with tr.start_as_current_span("chain.al.stage_a.probe_vs_mle") as sp:
            res = stage_a6_probe_vs_mle(model, oracle, probe)
            _set_attrs(sp, "aletheia.al.probe_vs_mle", res)
            summary["A6_probe_vs_mle"] = res

        # One-line headline for the trace viewer.
        verdict_bits = [
            ("label_independence", summary["A1_label_independence"]["passes_label_independence"]),
            ("eig_spread_P0_CV<<1", summary["A2_eig_spread_p0"]["al_separation_prediction_holds"]),
            ("order_irrelevance", summary["A3_order_irrelevance"]["passes_order_irrelevance"]),
            ("fisher_all_data_dominated", summary["A4_fisher_eigvals"]["postmortem_check_passes"]),
            ("contraction_spread>0.1", summary["A5_single_context_contraction"]["passes_postmortem_4_2"]),
            ("probe_close_to_mle", summary["A6_probe_vs_mle"]["passes_postmortem_4_3"]),
        ]
        root.set_attribute("aletheia.al.stage_a.verdict",
                           ", ".join(f"{k}={v}" for k, v in verdict_bits))

    out_path = OUT / "stage_a_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[stage_a] wrote {out_path}")
    for k, v in verdict_bits:
        print(f"  - {k}: {v}")
    return summary


if __name__ == "__main__":
    main()
