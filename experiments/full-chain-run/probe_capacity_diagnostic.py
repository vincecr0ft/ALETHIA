"""INV-2 diagnostic: is the probe (W) the bottleneck, or psi_theta?

The mass-only probe efficiency run gave eta ~ 0 on c_tilde_1 and c_tilde_2
while the oracle MLE on the same K=12 contexts resolved those directions
to variance 0.002 / 0.117. Per the INV-2 spec: "If the MLE resolves a
direction the probe cannot, the probe is the bottleneck -- debug W (rank,
regularisation) and psi_theta before any efficiency claim."

This script runs the same protocol with three probe families on the SAME
frozen psi_theta:
  1. The OLS probe (baseline; matches the FIX result).
  2. A degree-2 polynomial probe in w (since mu is exactly quadratic in c,
     w should be quadratic in c; a linear probe is structurally wrong).
  3. A small MLP probe (16 -> 64 -> 64 -> 4 with GELU).

We compare per-direction realised MSE (in c_tilde) against the MLE
variance from the original run. If the polynomial / MLP probe closes the
gap to MLE, the bottleneck is W (linear probe inadequacy) and we switch
probes. If it does NOT close the gap, the bottleneck is psi_theta and
INV-2's next step is to retrain psi_theta with a c-recoverability
auxiliary loss.
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
from scipy.interpolate import CubicSpline

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.intention import IntentionFM
from modules.surrogate.intention.fisher import (
    empirical_fisher_c, fisher_basis, sample_c_prior_inbox,
)

from probe_efficiency import (
    SEED, N_WC, WITHHOLD_DIM, WITHHOLD_BAND, C_TRAIN_BOX, M_RANGE, K_CTX,
    SIGMA_Y, N_FISHER_PRIOR,
    sample_c_inbox, sample_c_withheld,
    MorphingTable, implicit_w, batch_implicit_w, fisher_K_analytic,
    mle_c_from_morphing, mu_from_morphing,
)

# Smaller pools than INV-2 main run -- this is a diagnostic.
N_TRAIN_PROBE = 600        # train all three probes on the same data
N_TEST_INBOX = 50
N_TEST_WITHHELD = 50
N_REP = 100                # smaller; diagnostic only
OUT_DIR = HERE / "output"

# -----------------------------------------------------------------
# Probe families
# -----------------------------------------------------------------
class LinearProbe:
    """w (D,) -> c (4,) via OLS (+ intercept)."""
    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, w_train: np.ndarray, c_train: np.ndarray) -> None:
        X = np.hstack([w_train, np.ones((len(w_train), 1))])
        Wb = np.linalg.lstsq(X, c_train, rcond=None)[0]
        self.W = Wb[:-1].T               # (n_wc, D)
        self.b = Wb[-1]                  # (n_wc,)

    def predict(self, w: np.ndarray) -> np.ndarray:
        return w @ self.W.T + self.b


class QuadraticProbe:
    """w (D,) -> c (4,) via [w, w*w outer-product upper-triangle] -> linear.

    The closed-form ridge weight is a (noisy) sufficient statistic for
    the SMEFT mu = mu_SM + A c + c^T B c at fixed m. Since mu is
    quadratic in c, the ridge weight (which is linear in Y, hence linear
    in mu, hence quadratic in c) should be quadratic in c -- and a
    polynomial probe of degree 2 in w should capture more of the
    structure than a linear probe. We use [w, vech(w w^T)] as features.
    """
    def __init__(self):
        self.W = None
        self.b = None
        self.D = None

    @staticmethod
    def _phi(w: np.ndarray) -> np.ndarray:
        if w.ndim == 1:
            w = w[None, :]
        D = w.shape[1]
        outer = np.einsum("ni,nj->nij", w, w)
        idx_i, idx_j = np.triu_indices(D)
        quad = outer[:, idx_i, idx_j]                   # (n, D*(D+1)/2)
        return np.hstack([w, quad])                     # (n, D + D(D+1)/2)

    def fit(self, w_train: np.ndarray, c_train: np.ndarray) -> None:
        self.D = w_train.shape[1]
        X = self._phi(w_train)
        X = np.hstack([X, np.ones((len(X), 1))])
        # ridge-regularise: the quadratic block has D(D+1)/2 = 136
        # columns at D=16, plus D=16 linear, plus 1 intercept, total 153.
        # With N=600 we have 4 samples/coef -- regularisation needed.
        lam = 1e-3
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        Wb = np.linalg.solve(XtX, X.T @ c_train)
        self.W = Wb[:-1].T                              # (n_wc, P)
        self.b = Wb[-1]                                 # (n_wc,)

    def predict(self, w: np.ndarray) -> np.ndarray:
        X = self._phi(w)
        return X @ self.W.T + self.b


class MLPProbe(nn.Module):
    """w -> 64 GELU -> 64 GELU -> c. ~5k parameters."""
    def __init__(self, d_in: int = 16, n_wc: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_wc),
        )

    def forward(self, w):
        return self.net(w)

    def fit(self, w_train: np.ndarray, c_train: np.ndarray,
            *, n_steps: int = 2500, batch: int = 64, lr: float = 1e-3,
            rng: np.random.Generator) -> None:
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        w_t = torch.from_numpy(w_train).float()
        c_t = torch.from_numpy(c_train).float()
        N = len(w_t)
        for step in range(n_steps):
            idx = rng.choice(N, size=batch, replace=False)
            pred = self(w_t[idx])
            loss = F.mse_loss(pred, c_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, w: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(torch.from_numpy(w).float()).numpy()


# -----------------------------------------------------------------
# Driver
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    print(f"Probe-capacity diagnostic. sigma_y = {SIGMA_Y}.")

    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    print("Pre-tabulating morphing on m-grid...")
    m_grid = np.linspace(M_RANGE[0], M_RANGE[1], 240)
    morphing = MorphingTable(oracle, m_grid)

    ckpt = OUT_DIR / "intention_fm.pt"
    if not ckpt.exists():
        print(f"ERROR: pretrained IntentionFM missing at {ckpt}. "
              "Run experiments/full-chain-run/run.py first.")
        sys.exit(1)
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Fisher rotation V (same as INV-2 main run).
    c_fisher = sample_c_prior_inbox(
        n=N_FISHER_PRIOR, n_wc=N_WC, box=C_TRAIN_BOX, rng=rng,
        withhold_dim=WITHHOLD_DIM, withhold_band=WITHHOLD_BAND)
    m_fisher = rng.uniform(*M_RANGE, size=len(c_fisher))
    F_emp = empirical_fisher_c(oracle, c_fisher, m_fisher, fd_step=1e-3)
    D_eig, V = fisher_basis(F_emp)
    print(f"Fisher eigenvalues: {D_eig.round(6).tolist()}")

    # Training data for the probes (in-box).
    print(f"\nGenerating {N_TRAIN_PROBE} training (c, w) pairs (one fresh K=12 "
          "context per c)...")
    c_train = sample_c_inbox(N_TRAIN_PROBE, rng)
    w_train = batch_implicit_w(model, c_train, oracle, rng, SIGMA_Y, morphing)

    probes = {
        "linear-OLS": LinearProbe(),
        "quadratic-vech": QuadraticProbe(),
        "MLP-2x64": MLPProbe(d_in=16, n_wc=N_WC, hidden=64),
    }
    print("\nFitting probes on the same (w_train, c_train)...")
    for name, p in probes.items():
        if isinstance(p, MLPProbe):
            p.fit(w_train, c_train, rng=rng)
        else:
            p.fit(w_train, c_train)
        # In-sample c-RMSE for sanity (per-direction, c-space).
        c_pred_train = p.predict(w_train)
        rmse_train = np.sqrt(((c_pred_train - c_train) ** 2).mean(axis=0))
        print(f"  {name:20s}  in-sample Warsaw c-RMSE = {rmse_train.round(4)}")

    # Held-out pools.
    pools = {
        "inbox": sample_c_inbox(N_TEST_INBOX, rng),
        "withheld_band": sample_c_withheld(N_TEST_WITHHELD, rng),
    }

    results = {}
    for pool_name, cs_pool in pools.items():
        print(f"\n=== Pool {pool_name} (N={len(cs_pool)}, "
              f"N_rep={N_REP}) ===")
        n_pool = len(cs_pool)
        # Per-probe accumulators (in c_tilde frame).
        biases = {name: np.zeros((n_pool, N_WC)) for name in probes}
        variances = {name: np.zeros((n_pool, N_WC)) for name in probes}
        mle_variances = np.zeros((n_pool, N_WC))

        for i, c_true in enumerate(cs_pool):
            c_tilde_true = V.T @ c_true
            # Per-probe reps in c_tilde.
            c_hat_reps = {name: np.zeros((N_REP, N_WC)) for name in probes}
            mle_reps = np.zeros((N_REP, N_WC))
            for r in range(N_REP):
                w, M_ctx, Y_ctx = implicit_w(
                    model, c_true, oracle, rng, SIGMA_Y, morphing)
                for name, p in probes.items():
                    c_hat = p.predict(w[None, :])[0]
                    c_hat_reps[name][r] = c_hat
                mu_SM_k, A_k, B_k = morphing.at(M_ctx)
                mle_reps[r] = mle_c_from_morphing(
                    Y_ctx, mu_SM_k, A_k, B_k, SIGMA_Y, c0=np.zeros(N_WC))
            mle_tilde = mle_reps @ V
            mle_variances[i] = mle_tilde.var(axis=0)
            for name in probes:
                c_hat_tilde = c_hat_reps[name] @ V
                biases[name][i] = c_hat_tilde.mean(axis=0) - c_tilde_true
                variances[name][i] = c_hat_tilde.var(axis=0)

        # Aggregate.
        mle_mean = mle_variances.mean(axis=0)
        print(f"  MLE variance per c_tilde dir: {mle_mean.round(5).tolist()}")
        pool_results = {"mle_variance_mean": mle_mean.tolist(), "probes": {}}
        for name in probes:
            mse = (biases[name] ** 2 + variances[name]).mean(axis=0)
            bias = biases[name].mean(axis=0)
            print(f"  {name:20s}  MSE per c_tilde = {mse.round(4).tolist()}  "
                  f"bias = {bias.round(4).tolist()}")
            pool_results["probes"][name] = {
                "mse_mean": mse.tolist(),
                "bias_mean": bias.tolist(),
            }
        results[pool_name] = pool_results

    out_path = OUT_DIR / "probe_capacity_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump({"sigma_y": SIGMA_Y, "n_train": N_TRAIN_PROBE,
                   "n_rep": N_REP, "pools": results}, f, indent=2)
    print(f"\nSaved diagnostic to {out_path}")

    # Verdict.
    print("\n=== Verdict ===")
    mle_inbox = np.array(results["inbox"]["mle_variance_mean"])
    best_per_dir = {}
    for name, summary in results["inbox"]["probes"].items():
        mse = np.array(summary["mse_mean"])
        ratio = mse / np.maximum(mle_inbox, 1e-30)
        print(f"  {name:20s}  MSE / MLE-variance per c_tilde = "
              f"{ratio.round(3).tolist()}")
        for a in range(N_WC):
            if a not in best_per_dir or ratio[a] < best_per_dir[a][1]:
                best_per_dir[a] = (name, ratio[a])
    for a, (name, ratio) in best_per_dir.items():
        print(f"  c_tilde_{a+1}: best probe = {name}  (MSE / MLE-var = "
              f"{ratio:.2f})")
    print("\nIf the best probe is within ~3x of MLE variance on the resolved "
          "directions (c_tilde_1, c_tilde_2), the bottleneck WAS the linear "
          "OLS probe; INV-2 should use that probe. Otherwise psi_theta is the "
          "bottleneck and INV-2's next step is to retrain psi_theta with a "
          "c-recoverability auxiliary loss.")


if __name__ == "__main__":
    main()
