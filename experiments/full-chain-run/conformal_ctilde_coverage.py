"""T1.1 remediation: extend the leverage-stratified split-conformal
procedure from the mu-predictions to the c_tilde (Wilson-readout)
directions, and recompute per-direction coverage at nominal 68%.

Context (paper Section 7.1, Table 7 / tab:coverage):
The closed-form head's Wilson readout c_tilde = W (w, log M_ctx) + b is
sold as "the mean of a calibrated Gaussian posterior per rotated
direction". Its intervals are currently
    [c_hat_a - z68 * sd_a , c_hat_a + z68 * sd_a]
with sd_a fixed per direction by propagating the OLS residual covariance
diag(sigma_resid_sq) through the Fisher rotation V. That propagation is
the SAME parametric Gaussian as probe_efficiency_augmented.py. Table 7
reports per-direction 68% coverage of {0.88, 0.48, 0.58, 0.52} in-box and
{0.82, 0.48, 0.70, 0.58} withheld -- no direction at nominal except by
accident; c_tilde_1 over-covers, c_tilde_2 under-covers.

The conformal machinery in modules/surrogate/{calibration,intention/
calibration}.py calibrates mu-space predictions: nonconformity score
|Y - mu| / (sigma_y sqrt(1+lev)), Mondrian-stratified by leverage,
multiplier = corrected quantile of calibration scores. This script
applies the IDENTICAL split-conformal recipe to the c_tilde readout:

  - nonconformity score per direction a:
        s_a = |c_tilde_a_true - c_tilde_a_hat| / sd_raw_a
    where sd_raw_a is the same parametric SD the paper quotes (the raw
    Gaussian predictive that Table 7 uses).
  - Mondrian stratification by the natural leverage analogue for the
    c_tilde readout: the per-context ridge leverage of the augmented
    probe feature x = (w, log M_ctx, 1) in the OLS design,
        h(x) = x^T (X^T X)^{-1} x ,
    which is exactly the quantity the parametric sd_raw already carries
    (Var(c_hat_a) = sigma_resid_sq_a (1 + h)). We bin h into n_strata
    quantile strata, exactly as the mu-space calibrator bins leverage.
  - per-(direction, stratum) multiplier:
        f = quantile(scores_in_stratum, ceil((n+1)*0.683)/n)
    the standard finite-sample-corrected split-conformal quantile.
  - conformal interval: c_hat_a +/- f * sd_raw_a, recompute coverage on a
    DISJOINT held-out test pool.

Calibration / test pools are drawn fresh and disjoint. We report raw
(parametric) vs conformalised coverage, in-box and withheld, per
direction, plus the marginal (pooled-over-strata) conformal coverage so
the marginal-vs-conditional distinction is explicit.

Scientific question (per the work order): split-conformal restores
MARGINAL coverage by widening intervals; it does NOT remove the bias in
a biased point estimate. So it should fix c_tilde_1 over-coverage and the
aggregate c_tilde_2 under-coverage, but conditional coverage on the
bias-dominated c_tilde_2 (per-context |beta| ~ 0.39 in-box, 0.47
withheld) need not be restored. We report exactly which.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch
from scipy.stats import norm

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM
from modules.surrogate.intention.fisher import (
    empirical_fisher_c, fisher_basis, sample_c_prior_inbox,
)

from probe_efficiency import (
    N_WC, WITHHOLD_DIM, WITHHOLD_BAND, C_TRAIN_BOX, M_RANGE, K_CTX,
    SIGMA_Y, N_FISHER_PRIOR,
    sample_c_inbox, sample_c_withheld,
    MorphingTable, mu_from_morphing,
)
from probe_efficiency_augmented import context_features, draw_context

SEED = 7771
N_TRAIN_PROBE = 600        # probe (W, b) fit pool -- matches augmented probe
N_CAL = 400                # split-conformal calibration pool (in-box)
N_TEST_INBOX = 200         # held-out test pool, in-box
N_TEST_WITHHELD = 200      # held-out test pool, withheld band
N_REP = 100                # fresh contexts per held-out scenario (matches Table 7)
N_STRATA = 5               # Mondrian strata, same count as the mu-space calibrator
NOMINAL = 0.683
OUT_DIR = HERE / "output"


def build_probe(model, oracle, morphing, rng, d_feat):
    """Fit the augmented-OLS probe c_tilde = W (w, log M_ctx) + b exactly as
    probe_efficiency_augmented.py does. Returns W_lin, b_lin, XtX_inv,
    sigma_resid_sq, and the Warsaw->c_tilde rotation is applied by caller."""
    c_train = sample_c_inbox(N_TRAIN_PROBE, rng)
    feat_train = np.empty((N_TRAIN_PROBE, d_feat))
    for i, c in enumerate(c_train):
        M_ctx, Y_ctx = draw_context(rng, oracle, c, morphing)
        feat_train[i] = context_features(model, M_ctx, Y_ctx)
    X = np.hstack([feat_train, np.ones((N_TRAIN_PROBE, 1))])
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Wb = XtX_inv @ X.T @ c_train
    W_lin = Wb[:d_feat].T          # (n_wc, d_feat)  Warsaw basis
    b_lin = Wb[d_feat]             # (n_wc,)
    resid = c_train - X @ Wb
    dof = N_TRAIN_PROBE - (d_feat + 1)
    sigma_resid_sq = (resid ** 2).sum(axis=0) / dof   # (n_wc,) Warsaw
    return W_lin, b_lin, XtX_inv, sigma_resid_sq


def predict_ctilde(feat, W_lin, b_lin, V):
    """Augmented-probe Warsaw prediction rotated into c_tilde."""
    c_hat_warsaw = W_lin @ feat + b_lin
    return V.T @ c_hat_warsaw


def sd_raw_ctilde(feat, XtX_inv, sigma_resid_sq, V):
    """Parametric predictive SD per c_tilde direction -- the SAME Gaussian
    Table 7 uses. Var(c_hat_warsaw_a) = sigma_resid_sq_a (1 + lev), the
    diagonal Warsaw covariance rotated into c_tilde. lev = x^T XtX_inv x."""
    x = np.concatenate([feat, [1.0]])
    lev = float(x @ XtX_inv @ x)
    Sigma_warsaw = np.diag(sigma_resid_sq * (1.0 + lev))
    Sigma_tilde = V.T @ Sigma_warsaw @ V
    sd = np.sqrt(np.maximum(np.diag(Sigma_tilde), 0.0))
    return sd, lev


def collect_scenarios(cs_pool, model, oracle, morphing, rng, d_feat,
                      W_lin, b_lin, XtX_inv, sigma_resid_sq, V):
    """For each scenario, average the probe c_tilde prediction and the raw
    SD over N_REP fresh contexts (matching the Table 7 estimator), and
    record the per-context leverage (averaged). Returns arrays:
      c_hat_tilde_mean (n,4), sd_raw_mean (n,4), lev_mean (n,), c_tilde_true (n,4)."""
    n = len(cs_pool)
    c_hat_mean = np.zeros((n, N_WC))
    sd_mean = np.zeros((n, N_WC))
    lev_mean = np.zeros(n)
    c_tilde_true = np.zeros((n, N_WC))
    for i, c_true in enumerate(cs_pool):
        c_tilde_true[i] = V.T @ c_true
        c_hat_reps = np.zeros((N_REP, N_WC))
        sd_reps = np.zeros((N_REP, N_WC))
        lev_reps = np.zeros(N_REP)
        for r in range(N_REP):
            M_ctx, Y_ctx = draw_context(rng, oracle, c_true, morphing)
            feat = context_features(model, M_ctx, Y_ctx)
            c_hat_reps[r] = predict_ctilde(feat, W_lin, b_lin, V)
            sd, lev = sd_raw_ctilde(feat, XtX_inv, sigma_resid_sq, V)
            sd_reps[r] = sd
            lev_reps[r] = lev
        c_hat_mean[i] = c_hat_reps.mean(axis=0)
        sd_mean[i] = sd_reps.mean(axis=0)
        lev_mean[i] = lev_reps.mean()
    return c_hat_mean, sd_mean, lev_mean, c_tilde_true


def main():
    rng = np.random.default_rng(SEED)
    print(f"Conformal c_tilde calibration. sigma_y={SIGMA_Y}, K={K_CTX}, "
          f"N_rep={N_REP}, n_strata={N_STRATA}, nominal={NOMINAL}.")

    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    m_grid = np.linspace(M_RANGE[0], M_RANGE[1], 240)
    morphing = MorphingTable(oracle, m_grid)

    ckpt = OUT_DIR / "intention_fm.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    d_psi = state["psi.net.4.bias"].shape[0]
    model = IntentionFM(d_psi=d_psi, hidden=64, alpha=1e-3)
    model.load_state_dict(state)
    model.eval()
    d_feat = d_psi + K_CTX
    print(f"Loaded psi_theta (d_psi={d_psi}); augmented feature dim={d_feat}.")

    # Fisher eigenbasis V (rotation Warsaw -> c_tilde).
    c_fisher = sample_c_prior_inbox(
        n=N_FISHER_PRIOR, n_wc=N_WC, box=C_TRAIN_BOX, rng=rng,
        withhold_dim=WITHHOLD_DIM, withhold_band=WITHHOLD_BAND)
    m_fisher = rng.uniform(*M_RANGE, size=len(c_fisher))
    F_emp = empirical_fisher_c(oracle, c_fisher, m_fisher, fd_step=1e-3)
    D_eig, V = fisher_basis(F_emp)
    print(f"Fisher eigenvalues: {D_eig.round(6).tolist()}")

    # Fit the augmented OLS probe exactly as the paper's readout.
    W_lin, b_lin, XtX_inv, sigma_resid_sq = build_probe(
        model, oracle, morphing, rng, d_feat)
    print(f"OLS sigma_resid^2 (Warsaw) = {sigma_resid_sq.round(5)}")

    # ---- Split-conformal calibration on a fresh in-box pool ----
    cal_pool = sample_c_inbox(N_CAL, rng)
    cal_chat, cal_sd, cal_lev, cal_true = collect_scenarios(
        cal_pool, model, oracle, morphing, rng, d_feat,
        W_lin, b_lin, XtX_inv, sigma_resid_sq, V)
    # Nonconformity scores per direction.
    cal_scores = np.abs(cal_true - cal_chat) / np.maximum(cal_sd, 1e-12)  # (N_CAL,4)

    # Mondrian strata over the c_tilde-readout leverage h(x).
    edges = np.quantile(cal_lev, np.linspace(0, 1, N_STRATA + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    def stratum_of(lev):
        return np.clip(np.searchsorted(edges[1:-1], lev), 0, N_STRATA - 1)

    cal_strat = stratum_of(cal_lev)
    # Per-(direction, stratum) conformal multiplier.
    factors = np.full((N_WC, N_STRATA), np.nan)
    for a in range(N_WC):
        for s in range(N_STRATA):
            mask = cal_strat == s
            ns = int(mask.sum())
            if ns >= 8:
                q = min(1.0, float(np.ceil((ns + 1) * NOMINAL)) / ns)
                factors[a, s] = float(np.quantile(cal_scores[mask, a], q))
    # Also a marginal (single-stratum) multiplier per direction, for the
    # explicit marginal-vs-conditional comparison.
    qN = min(1.0, float(np.ceil((N_CAL + 1) * NOMINAL)) / N_CAL)
    marg_factor = np.array([float(np.quantile(cal_scores[:, a], qN))
                            for a in range(N_WC)])
    print(f"Marginal conformal multipliers per direction: "
          f"{marg_factor.round(3)} (parametric z68={norm.ppf(0.5+NOMINAL/2):.3f})")

    z68 = float(norm.ppf(0.5 + NOMINAL / 2.0))
    out = {
        "nominal": NOMINAL,
        "sigma_y": SIGMA_Y,
        "K": K_CTX,
        "n_rep": N_REP,
        "n_strata": N_STRATA,
        "n_cal": N_CAL,
        "fisher_eigenvalues": D_eig.tolist(),
        "z68_parametric": z68,
        "marginal_conformal_multiplier_per_direction": marg_factor.tolist(),
        "stratum_conformal_multipliers_per_direction": factors.tolist(),
        "raw_table7_reference": {
            "inbox": [0.88, 0.48, 0.58, 0.52],
            "withheld_band": [0.82, 0.48, 0.70, 0.58],
        },
        "pools": {},
    }

    test_pools = {
        "inbox": sample_c_inbox(N_TEST_INBOX, rng),
        "withheld_band": sample_c_withheld(N_TEST_WITHHELD, rng),
    }

    for pool_name, cs_pool in test_pools.items():
        chat, sd, lev, ctrue = collect_scenarios(
            cs_pool, model, oracle, morphing, rng, d_feat,
            W_lin, b_lin, XtX_inv, sigma_resid_sq, V)
        n = len(cs_pool)
        strat = stratum_of(lev)

        # Raw parametric coverage (reproduces Table 7's estimator on this pool).
        lo_raw = chat - z68 * sd
        hi_raw = chat + z68 * sd
        cov_raw = ((ctrue >= lo_raw) & (ctrue <= hi_raw)).mean(axis=0)

        # Conformalised (Mondrian, per-stratum) coverage.
        f_per_pt = np.zeros((n, N_WC))
        for a in range(N_WC):
            fa = factors[a].copy()
            # Fall back to the marginal multiplier for any empty stratum.
            fa[np.isnan(fa)] = marg_factor[a]
            f_per_pt[:, a] = fa[strat]
        lo_c = chat - f_per_pt * sd
        hi_c = chat + f_per_pt * sd
        cov_conf = ((ctrue >= lo_c) & (ctrue <= hi_c)).mean(axis=0)

        # Marginal-conformal coverage (single multiplier, no stratification).
        lo_m = chat - marg_factor[None, :] * sd
        hi_m = chat + marg_factor[None, :] * sd
        cov_marg = ((ctrue >= lo_m) & (ctrue <= hi_m)).mean(axis=0)

        # Bias diagnostic: mean signed residual per direction (this is the
        # quantity conformal cannot remove).
        mean_bias = (chat - ctrue).mean(axis=0)
        mean_abs_bias = np.abs(chat - ctrue).mean(axis=0)
        mean_sd = sd.mean(axis=0)

        print(f"\n=== {pool_name} (n={n}) ===")
        print(f"  dir  cov_raw  cov_marg_conf  cov_strat_conf  "
              f"mean|bias|  mean_sd  bias/sd")
        for a in range(N_WC):
            print(f"  c_tilde_{a+1}  {cov_raw[a]:.2f}     "
                  f"{cov_marg[a]:.2f}          {cov_conf[a]:.2f}          "
                  f"{mean_abs_bias[a]:.3f}      {mean_sd[a]:.3f}    "
                  f"{mean_abs_bias[a]/max(mean_sd[a],1e-9):.2f}")

        out["pools"][pool_name] = {
            "n_scenarios": int(n),
            "n_rep": int(N_REP),
            "per_direction": [
                {
                    "c_tilde_index": a + 1,
                    "coverage_raw_parametric": float(cov_raw[a]),
                    "coverage_marginal_conformal": float(cov_marg[a]),
                    "coverage_stratified_conformal": float(cov_conf[a]),
                    "mean_signed_bias": float(mean_bias[a]),
                    "mean_abs_bias": float(mean_abs_bias[a]),
                    "mean_raw_sd": float(mean_sd[a]),
                    "bias_to_sd_ratio": float(mean_abs_bias[a] /
                                              max(mean_sd[a], 1e-9)),
                }
                for a in range(N_WC)
            ],
        }

    out_path = OUT_DIR / "conformal_ctilde_coverage.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
