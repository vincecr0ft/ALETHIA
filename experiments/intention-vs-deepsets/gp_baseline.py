"""GP and kernel-ridge baselines on the analytic-SMEFT held-out pools.

Patch 8 of physics_patches.md: a Gaussian process with a sensible kernel
fit per-scenario on the same K=12 context is the "no representation
learning" control. If Intention beats the GP at the 95% bootstrap level,
the learned representation contributes beyond kernel choice; if they
tie, the contribution is the closed-form-update machinery (Sherman-
Morrison) rather than the representation.

Two baselines:
  GP (Matern 3/2): sklearn GaussianProcessRegressor with ML
    hyperparameter optimisation per scenario.
  KR (RBF kernel-ridge): sklearn KernelRidge with a fixed-width Gaussian
    kernel + ridge, alpha=1e-3.

Both consume the same (M_ctx, Y_ctx) and predict at M_query.

Outputs:
  experiments/intention-vs-deepsets/output_smeft/gp_baseline.json
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge

# Reuse the existing SMEFT held-out data.
NPZ = HERE / "output_smeft" / "results.npz"


def r2_per_scenario(y_pred, y_true):
    ss_res = np.sum((y_pred - y_true) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def gp_predict_one(M_ctx, Y_ctx, M_q, length_init=0.5):
    """Single-scenario GP fit on log(m/M_ref) with Matern-3/2 kernel."""
    X = np.log(M_ctx / 1.0).reshape(-1, 1)
    Xq = np.log(M_q / 1.0).reshape(-1, 1)
    k = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e3))
         * Matern(length_scale=length_init, length_scale_bounds=(1e-2, 1e2), nu=1.5)
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0)))
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=2, alpha=0.0)
    gp.fit(X, Y_ctx)
    return gp.predict(Xq)


def kr_predict_one(M_ctx, Y_ctx, M_q, gamma=1.0, alpha=1e-3):
    """Single-scenario kernel-ridge fit with RBF kernel on log(m/M_ref)."""
    X = np.log(M_ctx / 1.0).reshape(-1, 1)
    Xq = np.log(M_q / 1.0).reshape(-1, 1)
    kr = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
    kr.fit(X, Y_ctx)
    return kr.predict(Xq)


def main():
    d = np.load(NPZ)
    M_ctx_in  = d["test_in_M_ctx"];   Y_ctx_in  = d["test_in_Y_ctx"]
    M_q_in    = d["test_in_M_query"]; Y_q_in    = d["test_in_Y_query"]
    M_ctx_out = d["test_out_M_ctx"];  Y_ctx_out = d["test_out_Y_ctx"]
    M_q_out   = d["test_out_M_query"];Y_q_out   = d["test_out_Y_query"]

    S_in = M_ctx_in.shape[0]; S_out = M_ctx_out.shape[0]
    print(f"# scenarios: in={S_in}, out={S_out}")

    print("# GP (Matern 3/2)")
    t0 = time.time()
    yp_gp_in = np.array([gp_predict_one(M_ctx_in[s], Y_ctx_in[s], M_q_in[s]) for s in range(S_in)])
    yp_gp_out = np.array([gp_predict_one(M_ctx_out[s], Y_ctx_out[s], M_q_out[s]) for s in range(S_out)])
    print(f"  wall = {time.time() - t0:.1f}s")
    r2_gp_in = r2_per_scenario(yp_gp_in, Y_q_in)
    r2_gp_out = r2_per_scenario(yp_gp_out, Y_q_out)
    print(f"  in : median R^2 = {float(np.median(r2_gp_in)):+.4f}  p5 = {float(np.percentile(r2_gp_in, 5)):+.4f}")
    print(f"  out: median R^2 = {float(np.median(r2_gp_out)):+.4f}  p5 = {float(np.percentile(r2_gp_out, 5)):+.4f}")

    print("\n# Kernel ridge (RBF, gamma=1, alpha=1e-3)")
    t0 = time.time()
    yp_kr_in = np.array([kr_predict_one(M_ctx_in[s], Y_ctx_in[s], M_q_in[s]) for s in range(S_in)])
    yp_kr_out = np.array([kr_predict_one(M_ctx_out[s], Y_ctx_out[s], M_q_out[s]) for s in range(S_out)])
    print(f"  wall = {time.time() - t0:.1f}s")
    r2_kr_in = r2_per_scenario(yp_kr_in, Y_q_in)
    r2_kr_out = r2_per_scenario(yp_kr_out, Y_q_out)
    print(f"  in : median R^2 = {float(np.median(r2_kr_in)):+.4f}  p5 = {float(np.percentile(r2_kr_in, 5)):+.4f}")
    print(f"  out: median R^2 = {float(np.median(r2_kr_out)):+.4f}  p5 = {float(np.percentile(r2_kr_out, 5)):+.4f}")

    summary = {
        "gp_matern32": {
            "in":  {"median": float(np.median(r2_gp_in)),  "p5": float(np.percentile(r2_gp_in, 5))},
            "out": {"median": float(np.median(r2_gp_out)), "p5": float(np.percentile(r2_gp_out, 5))},
        },
        "kr_rbf": {
            "in":  {"median": float(np.median(r2_kr_in)),  "p5": float(np.percentile(r2_kr_in, 5))},
            "out": {"median": float(np.median(r2_kr_out)), "p5": float(np.percentile(r2_kr_out, 5))},
        },
    }
    out_path = HERE / "output_smeft" / "gp_baseline.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
