r"""Capability cell: In-context / amortized PFN  ->  SMEFT parameter inference.

FM family favoured: in-context / amortized PFN (TabPFN / CausalFM / the ALETHIA
Intention closed-form ridge). The objective this cell rewards is *amortized
calibrated inference*: given an event set drawn at an unknown Wilson point c,
return a posterior over c in a single forward pass, with no per-scenario
gradient steps. The "learning algorithm" lives in the weights.

HEP task: infer the 4 Wilson coefficients (cHq3, cHq1, clq3, clq1) from one
event set of (log m_ll, cos θ*) — simulation-based inference on the SMEFT
substrate.

Metric (native to the objective): held-out interval *coverage* and a
*calibration error*, NOT point R². A model can have good point accuracy and be
badly calibrated; the PFN's selling point is calibration, so that is the score.

Ablation isolated: amortized one-pass inference vs per-scenario gradient
optimisation. The contrast is the exact-likelihood MLE (the statistically
optimal estimator) computed per scenario on a small subset — the amortized net
should approach it at a fraction of the per-scenario cost.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

import substrate as sub
import probes
from nets import DeepSetsEncoder, count_params, set_seed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_matrix"


class AmortizedPosterior(nn.Module):
    """Set encoder -> Gaussian posterior over c: mean (n_wc) and log-std (n_wc)."""

    def __init__(self, n_wc, d_out=48):
        super().__init__()
        self.enc = DeepSetsEncoder(d_out=d_out, d_emb=64, hidden=80)
        self.mean = nn.Linear(d_out, n_wc)
        self.logstd = nn.Linear(d_out, n_wc)

    def forward(self, X):
        z = self.enc(X)
        return self.mean(z), self.logstd(z).clamp(-6, 3)

    @torch.no_grad()
    def representation(self, X):
        return self.enc(X)


def gaussian_nll(mean, logstd, target):
    var = torch.exp(2 * logstd)
    return (0.5 * ((target - mean) ** 2 / var) + logstd).mean()


def train(model, Xtr, Ctr, *, steps=1200, bs=32, lr=2e-3, seed=0):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        idx = rng.choice(n, bs, replace=False)
        mean, logstd = model(Xtr[idx])
        loss = gaussian_nll(mean, logstd, Ctr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def _sigma_sm_grid(oracle, m_grid):
    """SM differential cross section σ_SM(m) on the m-grid (TeV), for the
    total-rate normalisation of the unbinned likelihood."""
    from modules.analytic_smeft import differential_xs
    res = differential_xs({}, m_grid * 1000.0, sqrt_s=oracle.sqrt_s,
                          lambda_scale=oracle.lam, order=oracle.order,
                          pdf=oracle.pdf)
    return np.asarray(res["sm_only"], float)


def _log_rate_ratio(oracle, c, m_grid, sigma_sm):
    """log[σ_tot(c)/σ_tot(SM)] via trapezoid of σ_SM(m)·μ(c,m) over the grid."""
    C = np.tile(c, (len(m_grid), 1))
    mu = oracle.truth(C, m_grid)
    num = np.trapezoid(sigma_sm * mu, m_grid)
    den = np.trapezoid(sigma_sm, m_grid)
    return float(np.log(max(num, 1e-30) / max(den, 1e-30)))


def exact_mle(oracle, X_np, n_wc, *, maxiter=40, bound=1.5, n_ev=120,
              m_grid=None, sigma_sm=None):
    """Per-scenario exact-likelihood MLE of the unbinned shape likelihood.

    The proper per-event normalised density ratio is
        log r_c(x) = log[dσ(c,x)/dσ(SM,x)]  -  log[σ_tot(c)/σ_tot(SM)],
    i.e. the cross-section-ratio log-weight MINUS the total-rate-ratio. The
    first term is ``event_log_likelihood_ratio``; the second normalises it so
    the likelihood is proper (without it the EFT tail makes mean log-weight
    increase without bound and the MLE runs to the boundary). The MLE
    maximises mean_x log r_c(x) — the optimal but non-amortized estimator.

    Bounded to the physical region [-bound, bound]^n_wc; events capped at n_ev.
    """
    from modules.surrogate.oracle_events import event_log_likelihood_ratio

    if m_grid is None:
        m_grid = np.linspace(0.3, 2.3, 40)
    if sigma_sm is None:
        sigma_sm = _sigma_sm_grid(oracle, m_grid)
    Xe = X_np[:n_ev]

    def neg_ll(c):
        shape = float(np.mean(event_log_likelihood_ratio(oracle, c, Xe)))
        norm = _log_rate_ratio(oracle, c, m_grid, sigma_sm)
        return -(shape - norm)

    res = minimize(neg_ll, np.zeros(n_wc), method="L-BFGS-B",
                   bounds=[(-bound, bound)] * n_wc,
                   options={"maxiter": maxiter, "eps": 1e-2})
    return res.x


def run(seed: int = 0, n_mle: int = 5) -> dict:
    t0 = time.time()
    data = sub.load_cache()
    cfg = sub.CONFIG
    Xtr = torch.tensor(data["train_X1"], dtype=torch.float32)
    Ctr = torch.tensor(data["train_c"], dtype=torch.float32)
    Xte = torch.tensor(data["test_X1"], dtype=torch.float32)
    Cte = data["test_c"]

    model = AmortizedPosterior(cfg.n_wc)
    train(model, Xtr, Ctr, seed=seed)

    model.eval()
    with torch.no_grad():
        mean_te, logstd_te = model(Xte)
    mean_te = mean_te.numpy()
    std_te = np.exp(logstd_te.numpy())

    # Native metric: coverage & calibration, pooled over the 4 Wilson dims.
    cov = probes.gaussian_coverage(Cte, mean_te, std_te)
    point_r2 = probes.r2_score(Cte, mean_te)

    # Capacity-controlled representation probe (for the matrix's common column).
    Ztr = model.representation(Xtr).numpy()
    Zte = model.representation(Xte).numpy()
    raw_tr = probes.raw_event_summary(data["train_X1"])
    raw_te = probes.raw_event_summary(data["test_X1"])
    probe = probes.probe_with_floor(Ztr, data["train_c"], Zte, Cte,
                                    raw_tr, raw_te, seed=seed)

    # Ablation contrast: exact-likelihood MLE (optimal, non-amortized) on a
    # small subset — the per-scenario non-vectorized oracle makes this slow, so
    # it is a statistical-floor *reference* (mean per-coordinate L2 error), not
    # a full-test metric. Compares the one-pass amortized net to the optimal
    # per-scenario estimator on matched scenarios.
    oracle = sub.make_oracle()
    mg = np.linspace(cfg.m_lo, cfg.m_hi, cfg.n_m_grid)
    from cell_pfn import _sigma_sm_grid
    ssm = _sigma_sm_grid(oracle, mg)
    mle_idx = np.arange(min(n_mle, len(Cte)))
    mle_pred = np.stack([
        exact_mle(oracle, data["test_X1"][i], cfg.n_wc, m_grid=mg,
                  sigma_sm=ssm, n_ev=64, maxiter=15) for i in mle_idx
    ])
    mle_err = float(np.mean(np.linalg.norm(mle_pred - Cte[mle_idx], axis=1)))
    amort_err_sub = float(np.mean(
        np.linalg.norm(mean_te[mle_idx] - Cte[mle_idx], axis=1)))

    result = {
        "cell": "pfn_inference",
        "fm_family": "In-context / amortized PFN",
        "hep_task": "SMEFT Wilson-coefficient inference from one event set",
        "metric_primary": {"name": "calibration_error (lower=better)",
                           "value": cov["calibration_error"]},
        "metrics": {
            "calibration_error": cov["calibration_error"],
            "coverage_levels": cov["levels"],
            "coverage_empirical": cov["coverage"],
            "point_r2_amortized": point_r2,
            "probe_r2_heldout": probe.r2,
            "probe_floor_r2": probe.floor_r2,
            "probe_margin": probe.margin,
            "ablation_mle_meanerr_subset": mle_err,
            "ablation_amortized_meanerr_subset": amort_err_sub,
            "ablation_n_subset": int(len(mle_idx)),
        },
        "ablation_isolated": ("amortized one-pass inference vs per-scenario "
                              "exact-MLE (optimal but non-amortized)"),
        "n_params": count_params(model),
        "wall_seconds": time.time() - t0,
        "config": {"d_z": probe.d_z, "n_train": int(Xtr.shape[0]),
                   "n_events": cfg.n_events},
    }
    OUT.mkdir(exist_ok=True)
    with open(OUT / "cell_pfn.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("cell", "metric_primary", "metrics",
                                        "n_params", "wall_seconds")}, indent=2))
