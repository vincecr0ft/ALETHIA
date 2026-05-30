"""INV-2 with the augmented disclosure probe (w, M_ctx) -> c_tilde.

Diagnostic: three retrain attempts (v2 fine-tune, v3 from-scratch with
d_psi=32) confirmed that no psi_theta retrain at d_psi <= 32 closes the
MSE(probe) -> MLE-variance gap on c_tilde_1. The ridge weight
w = A^-1 Psi^T Y_ctx is a 16-dim summary of (M_ctx, Y_ctx) that retains
sufficient statistics for predicting mu at new queries but discards the
M-conditional sensitivity that the MLE uses to resolve c_tilde_1.

The principled fix per the same Bayesian-linear-regression framework
that defines the head: augment the disclosure-probe input from w to
(w, M_ctx). Given (M_ctx, w), one can recover Y_ctx ~ Psi(M_ctx) w up to
the ridge regularisation, so the augmented probe has essentially the
same sufficient information as MLE-from-(M_ctx, Y_ctx). The probe stays
linear (Bayesian-linear-regression), so the disclosure framing is
preserved.

Expectation: the augmented probe should approach MLE variance on
resolved directions (c_tilde_1, c_tilde_2), confirming the
representation-extraction gap is closed when the probe has the same
sufficient statistic as MLE.

Reads v1's psi_theta (intention_fm.pt), fits the augmented probe on
fresh contexts, and re-runs the efficiency comparison.
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares
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
    MorphingTable, fisher_K_analytic,
    mle_c_from_morphing, mu_from_morphing,
)


SEED = 7771
N_TRAIN_PROBE = 600
N_TEST_INBOX = 50
N_TEST_WITHHELD = 50
N_REP = 100              # reduced from 200 to keep bounded-multi-restart MLE in budget
N_MLE_RESTARTS = 2       # initial c0=0 plus one random; enough to catch trivial divergence
OUT_DIR = HERE / "output"


class MLPProbeAug(nn.Module):
    """MLP probe over (w, log M_ctx) features.

    Used only as the disclosure-integrity gate: if the linear and MLP
    probes agree on the withheld band, the representation carries the
    structure and the disclosure framing holds; if the MLP pulls ahead
    OOD, work has migrated from the representation into the probe.
    """

    def __init__(self, d_in: int, n_wc: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_wc),
        )

    def forward(self, x):
        return self.net(x)

    def fit(self, feat_train: np.ndarray, c_train: np.ndarray,
            rng: np.random.Generator, n_steps: int = 2500,
            batch: int = 64, lr: float = 1e-3) -> None:
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        x_t = torch.from_numpy(feat_train).float()
        c_t = torch.from_numpy(c_train).float()
        N = len(x_t)
        for _ in range(n_steps):
            idx = rng.choice(N, size=batch, replace=False)
            pred = self(x_t[idx])
            loss = F.mse_loss(pred, c_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x[None, :]
            return self(torch.from_numpy(x).float()).numpy()


def context_features(model: IntentionFM, M_ctx: np.ndarray,
                     Y_ctx: np.ndarray) -> np.ndarray:
    """Augmented probe feature: [w (d_psi), sorted log(M_ctx) (K)].

    M_ctx is sorted ascending and rescaled to log(m/M_ref) to match the
    psi_theta input convention. This vector together with w is jointly
    a sufficient statistic for c at fixed (M_ctx, Y_ctx) ridge geometry.
    """
    _, w, _ = model.A_inv_and_w(M_ctx, Y_ctx)
    M_sorted = np.sort(M_ctx)
    log_M = np.log(M_sorted / 1.0)            # M_REF=1.0 TeV in psi_theta
    return np.concatenate([w, log_M])


def draw_context(rng: np.random.Generator, oracle, c: np.ndarray,
                 morphing: MorphingTable) -> tuple[np.ndarray, np.ndarray]:
    """Draw a K=12 context with additive Gaussian noise on Y."""
    M_ctx = rng.uniform(*M_RANGE, size=K_CTX)
    mu_SM_k, A_k, B_k = morphing.at(M_ctx)
    Y_truth = mu_from_morphing(c, mu_SM_k, A_k, B_k)
    Y_ctx = Y_truth + rng.normal(0.0, SIGMA_Y, size=K_CTX)
    return M_ctx, Y_ctx


def main():
    rng = np.random.default_rng(SEED)
    print(f"Augmented probe (w, log(M_ctx)) -> c. sigma_y = {SIGMA_Y}, "
          f"K = {K_CTX}.")

    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    print("Pre-tabulating morphing on m-grid...")
    m_grid = np.linspace(M_RANGE[0], M_RANGE[1], 240)
    morphing = MorphingTable(oracle, m_grid)

    ckpt = OUT_DIR / "intention_fm.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    inferred_d_psi = state["psi.net.4.bias"].shape[0]
    model = IntentionFM(d_psi=inferred_d_psi, hidden=64, alpha=1e-3)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded v1 psi_theta (d_psi = {inferred_d_psi}).")

    # Fisher and prior
    c_fisher = sample_c_prior_inbox(
        n=N_FISHER_PRIOR, n_wc=N_WC, box=C_TRAIN_BOX, rng=rng,
        withhold_dim=WITHHOLD_DIM, withhold_band=WITHHOLD_BAND)
    m_fisher = rng.uniform(*M_RANGE, size=len(c_fisher))
    F_emp = empirical_fisher_c(oracle, c_fisher, m_fisher, fd_step=1e-3)
    D_eig, V = fisher_basis(F_emp)
    print(f"Fisher eigenvalues: {D_eig.round(6).tolist()}")
    c_prior_for_cov = sample_c_prior_inbox(
        n=N_FISHER_PRIOR, n_wc=N_WC, box=C_TRAIN_BOX, rng=rng,
        withhold_dim=WITHHOLD_DIM, withhold_band=WITHHOLD_BAND)
    sigma_prior_diag_sq = (c_prior_for_cov @ V).var(axis=0)

    # Build augmented training features
    print(f"\n=== Augmented OLS probe (input = w | log(M_ctx); "
          f"dim = {inferred_d_psi + K_CTX}) ===")
    c_train = sample_c_inbox(N_TRAIN_PROBE, rng)
    feat_train = np.empty((N_TRAIN_PROBE, inferred_d_psi + K_CTX))
    for i, c in enumerate(c_train):
        M_ctx, Y_ctx = draw_context(rng, oracle, c, morphing)
        feat_train[i] = context_features(model, M_ctx, Y_ctx)

    # OLS fit with intercept; the augmented probe is a linear map from
    # (w, log M_ctx) -> c, fit by ordinary least squares.
    X = np.hstack([feat_train, np.ones((N_TRAIN_PROBE, 1))])
    Wb = np.linalg.lstsq(X, c_train, rcond=None)[0]
    n_feat = inferred_d_psi + K_CTX
    W_lin = Wb[:n_feat].T                  # (n_wc, n_feat)
    b_lin = Wb[n_feat]                     # (n_wc,)
    c_hat_train = X @ Wb
    resid = c_train - c_hat_train
    dof = N_TRAIN_PROBE - (n_feat + 1)
    sigma_resid_sq = (resid ** 2).sum(axis=0) / dof
    print(f"  OLS sigma_resid^2 per Warsaw direction = "
          f"{sigma_resid_sq.round(5)}")

    # Disclosure-integrity gate: fit a parameter-matched MLP probe on
    # the same (w, log M_ctx) features. If linear and MLP agree on the
    # withheld band, the representation carries the structure; if MLP
    # pulls ahead OOD, work has moved from psi_theta into the probe.
    print(f"\n=== MLP augmented probe (same input) ===")
    torch.manual_seed(SEED)
    mlp_probe = MLPProbeAug(d_in=n_feat, n_wc=N_WC, hidden=64)
    mlp_probe.fit(feat_train, c_train, rng=rng)
    c_hat_train_mlp = mlp_probe.predict(feat_train)
    rmse_train_mlp = np.sqrt(((c_hat_train_mlp - c_train) ** 2).mean(axis=0))
    print(f"  MLP in-sample Warsaw c-RMSE = {rmse_train_mlp.round(4)}")

    # Held-out evaluation.
    z68 = float(norm.ppf(0.5 + 0.683 / 2.0))
    pools = {
        "inbox": sample_c_inbox(N_TEST_INBOX, rng),
        "withheld_band": sample_c_withheld(N_TEST_WITHHELD, rng),
    }
    out = {"pools": {}, "sigma_y": SIGMA_Y, "K": K_CTX,
           "n_train": N_TRAIN_PROBE, "n_rep": N_REP,
           "fisher_eigenvalues": D_eig.tolist(),
           "augmented": True,
           "feature_layout": "[w, log(M_ctx_sorted)]"}

    for pool_name, cs_pool in pools.items():
        print(f"\n=== Pool {pool_name} (N={len(cs_pool)}, "
              f"N_rep={N_REP}) ===")
        n_pool = len(cs_pool)
        biases = np.zeros((n_pool, N_WC))
        variances = np.zeros((n_pool, N_WC))
        biases_mlp = np.zeros((n_pool, N_WC))
        variances_mlp = np.zeros((n_pool, N_WC))
        bcrb_diag = np.zeros((n_pool, N_WC))
        mle_biases = np.zeros((n_pool, N_WC))
        mle_variances = np.zeros((n_pool, N_WC))
        cov68 = np.zeros((n_pool, N_WC))
        post_var_tilde = np.zeros((n_pool, N_WC))

        # Inv-prior for BCRB
        inv_prior_diag = 1.0 / sigma_prior_diag_sq

        # Per-context MLE convergence diagnostics
        mle_converged = np.zeros((n_pool, N_REP), dtype=bool)
        mle_on_edge = np.zeros((n_pool, N_REP), dtype=bool)
        mle_grad_norm = np.full((n_pool, N_REP), np.nan)
        mle_cost = np.full((n_pool, N_REP), np.nan)

        for i, c_true in enumerate(cs_pool):
            c_tilde_true = V.T @ c_true
            c_hat_reps = np.zeros((N_REP, N_WC))
            c_hat_reps_mlp = np.zeros((N_REP, N_WC))
            mle_reps = np.zeros((N_REP, N_WC))
            F_K_avg = np.zeros((N_WC, N_WC))
            saved_M, saved_Y, saved_feat = None, None, None
            for r in range(N_REP):
                M_ctx, Y_ctx = draw_context(rng, oracle, c_true, morphing)
                feat = context_features(model, M_ctx, Y_ctx)
                c_hat = W_lin @ feat + b_lin
                c_hat_reps[r] = c_hat
                c_hat_reps_mlp[r] = mlp_probe.predict(feat)[0]
                mu_SM_k, A_k, B_k = morphing.at(M_ctx)
                mle_val, info = mle_c_from_morphing(
                    Y_ctx, mu_SM_k, A_k, B_k, SIGMA_Y,
                    c0=np.zeros(N_WC), n_restarts=N_MLE_RESTARTS,
                    rng=rng, return_info=True,
                    # Widen bounds to 5x the training prior so the
                    # bound only catches genuinely-flat likelihood
                    # directions, not a tight constraint.
                    bounds=((-5.0,) * 4, (5.0,) * 4))
                mle_reps[r] = mle_val
                mle_converged[i, r] = info.get("converged", False)
                mle_on_edge[i, r] = info.get("on_box_edge", False)
                mle_grad_norm[i, r] = info.get("grad_norm", np.nan)
                mle_cost[i, r] = info.get("cost", np.nan)
                F_K_avg += fisher_K_analytic(c_true, mu_SM_k, A_k, B_k,
                                              SIGMA_Y)
                if r == 0:
                    saved_M = M_ctx.copy()
                    saved_Y = Y_ctx.copy()
                    saved_feat = feat.copy()
            F_K_avg /= N_REP
            F_K_tilde = V.T @ F_K_avg @ V
            M_tilde = F_K_tilde + np.diag(inv_prior_diag)
            bcrb_diag[i] = np.diag(np.linalg.inv(M_tilde))

            c_hat_tilde = c_hat_reps @ V
            c_hat_tilde_mlp = c_hat_reps_mlp @ V
            mle_tilde = mle_reps @ V
            biases[i] = c_hat_tilde.mean(axis=0) - c_tilde_true
            variances[i] = c_hat_tilde.var(axis=0)
            biases_mlp[i] = c_hat_tilde_mlp.mean(axis=0) - c_tilde_true
            variances_mlp[i] = c_hat_tilde_mlp.var(axis=0)
            mle_biases[i] = mle_tilde.mean(axis=0) - c_tilde_true
            mle_variances[i] = mle_tilde.var(axis=0)

            # Probe predictive variance from OLS (frequentist):
            # Var(c_hat_a) = sigma_resid_sq[a] * (1 + x^T (X^T X)^-1 x)
            # In feature space; we just store the marginal in c_tilde via rotation.
            Sigma_warsaw = np.diag(sigma_resid_sq * 1.0)  # marginal per Warsaw dir
            Sigma_tilde = V.T @ Sigma_warsaw @ V
            post_var_tilde[i] = np.diag(Sigma_tilde)
            sd_post = np.sqrt(np.maximum(post_var_tilde[i], 0.0))
            c_hat_mean_tilde = c_hat_tilde.mean(axis=0)
            lo = c_hat_mean_tilde - z68 * sd_post
            hi = c_hat_mean_tilde + z68 * sd_post
            cov68[i] = ((c_tilde_true >= lo) & (c_tilde_true <= hi)).astype(float)

        mean_bcrb = bcrb_diag.mean(axis=0)
        mean_bias = biases.mean(axis=0)
        mean_var = variances.mean(axis=0)
        mean_mse = (biases ** 2 + variances).mean(axis=0)
        mean_bias_mlp = biases_mlp.mean(axis=0)
        mean_var_mlp = variances_mlp.mean(axis=0)
        mean_mse_mlp = (biases_mlp ** 2 + variances_mlp).mean(axis=0)
        mean_mle_bias = mle_biases.mean(axis=0)
        mean_mle_var = mle_variances.mean(axis=0)
        mean_mle_mse = (mle_biases ** 2 + mle_variances).mean(axis=0)
        # Probe is biased toward the prior mean; MLE is approximately
        # unbiased. Comparing MSE-to-MSE (not variance-to-variance) is
        # the apples-to-apples comparison. variance-to-variance would
        # reward the probe for shrinkage at the cost of bias.
        ratio_mse_to_mle = mean_mse / np.maximum(mean_mle_mse, 1e-30)
        ratio_var_to_mle_var = mean_var / np.maximum(mean_mle_var, 1e-30)
        ratio_mse_mlp_to_mle = mean_mse_mlp / np.maximum(mean_mle_mse, 1e-30)
        # Disclosure-integrity gate: linear-vs-MLP MSE ratio per direction.
        # ratio close to 1 means the representation, not the probe family,
        # carries the c structure. Ratio >> 1 means MLP wins (probe family
        # matters; representation weaker).
        linear_vs_mlp = mean_mse / np.maximum(mean_mse_mlp, 1e-30)
        cov_rate = cov68.mean(axis=0)

        # MLE convergence diagnostics: report what fraction of contexts
        # converged, what fraction sit on a box edge, and the median
        # gradient norm at the solution.
        frac_conv = mle_converged.mean()
        frac_edge = mle_on_edge.mean()
        med_grad = float(np.nanmedian(mle_grad_norm))
        med_cost = float(np.nanmedian(mle_cost))
        print(f"  MLE convergence: converged {frac_conv:.2%}, "
              f"on box edge {frac_edge:.2%}, "
              f"median gradient norm at solution = {med_grad:.3e}, "
              f"median cost = {med_cost:.3e}")

        print(f"  c_tilde direction summary (probe vs finite-sample MLE):")
        print(f"    a   MSE_lin   MSE_mlp   MSE_MLE   "
              f"MSE_lin/MLE   MSE_mlp/MLE   lin/mlp   BCRB        cov68")
        for a in range(N_WC):
            print(f"    {a}   {mean_mse[a]:.4f}    {mean_mse_mlp[a]:.4f}    "
                  f"{mean_mle_mse[a]:.4f}    "
                  f"{ratio_mse_to_mle[a]:6.2f}        {ratio_mse_mlp_to_mle[a]:6.2f}        "
                  f"{linear_vs_mlp[a]:5.2f}     "
                  f"{mean_bcrb[a]:.3e}   {cov_rate[a]:.2f}")

        out["pools"][pool_name] = {
            "n_scenarios": int(n_pool),
            "n_rep": int(N_REP),
            "per_direction": [
                {
                    "c_tilde_index": a + 1,
                    "bcrb_mean": float(mean_bcrb[a]),
                    "linear_probe_mse_mean": float(mean_mse[a]),
                    "linear_probe_bias_mean": float(mean_bias[a]),
                    "linear_probe_variance_mean": float(mean_var[a]),
                    "mlp_probe_mse_mean": float(mean_mse_mlp[a]),
                    "mlp_probe_bias_mean": float(mean_bias_mlp[a]),
                    "mlp_probe_variance_mean": float(mean_var_mlp[a]),
                    "mle_mse_mean": float(mean_mle_mse[a]),
                    "mle_bias_mean": float(mean_mle_bias[a]),
                    "mle_variance_mean": float(mean_mle_var[a]),
                    "ratio_linear_mse_over_mle_mse": float(ratio_mse_to_mle[a]),
                    "ratio_mlp_mse_over_mle_mse": float(ratio_mse_mlp_to_mle[a]),
                    "ratio_linear_mse_over_mlp_mse": float(linear_vs_mlp[a]),
                    "ratio_linear_var_over_mle_var": float(ratio_var_to_mle_var[a]),
                    "coverage_68": float(cov_rate[a]),
                }
                for a in range(N_WC)
            ],
        }

    out_path = OUT_DIR / "bcrb_efficiency_augmented.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved augmented-probe BCRB summary to {out_path}.")


if __name__ == "__main__":
    main()
