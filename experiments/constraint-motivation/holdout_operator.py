"""Operator-holdout test for the constraint-violating closed-form regressor.

Feature map: phi(c, m) = phi_c(c) tensored with psi_x(m), where
- phi_c is the canonical SMEFT polynomial basis through quadratic order:
    {1, c_1, ..., c_4, c_1^2, c_1 c_2, ..., c_4^2}     (1 + 4 + 10 = 15 dims)
- psi_x is a quartic polynomial in log(m / M_ref):
    {1, log(m/M_ref), ..., log^4(m/M_ref)}              (5 dims)
Total feature dim: 15 * 5 = 75. This is the morphing ansatz of
equation (2) in the paper.

For each operator i in {cHq3, cHq1, clq3, clq1}:
    1. Sample 200 training scenarios with c_i held at zero
       (other operators ~ U([-0.7, 0.7])).
    2. Sample K=12 m-values per scenario, get mu from the analytic
       SMEFT oracle. Flatten to (200 * 12 = 2400) (c, m) -> y triples.
    3. Bayesian ridge fit (closed form) of W in phi @ W = y.
    4. Test on 50 scenarios with c_i ~ U([-0.5, 0.5]) and other c
       at U([-0.5, 0.5]). Compute per-operator R^2.

This is the empirical hammer for the architectural-constraint motivation
(Patch 11 of physics_patches.md). The expected pattern from
empirical-results.md is per-operator R^2 < -20 on every held-out operator
when that operator is held at zero throughout training.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import json
import numpy as np

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle


WC_NAMES = ("cHq3", "cHq1", "clq3", "clq1")
N_WC = 4
M_REF = 1.0
M_RANGE = (0.3, 2.3)
N_TRAIN = 200
K_CTX = 12
N_TEST = 50
ALPHA_RIDGE = 1e-3
C_BOX = 0.7
SEED = 0

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def phi_c(c: np.ndarray) -> np.ndarray:
    """Polynomial basis in c through quadratic order. c: (n, 4) -> (n, 15)."""
    n = c.shape[0]
    cols = [np.ones(n)]
    cols.extend(c[:, i] for i in range(N_WC))
    for i in range(N_WC):
        for j in range(i, N_WC):
            cols.append(c[:, i] * c[:, j])
    return np.stack(cols, axis=1)


def psi_x(m: np.ndarray) -> np.ndarray:
    """Polynomial basis in log(m / M_ref) through quartic order. m: (n,) -> (n, 5)."""
    lm = np.log(m / M_REF)
    return np.stack([lm ** k for k in range(5)], axis=1)


def feature_map(c: np.ndarray, m: np.ndarray) -> np.ndarray:
    """phi(c, m) = phi_c(c) ⊗ psi_x(m). c: (n, 4), m: (n,) -> (n, 75)."""
    pc = phi_c(c)                     # (n, 15)
    px = psi_x(m)                     # (n,  5)
    return np.einsum("nc,nx->ncx", pc, px).reshape(c.shape[0], -1)


def make_training_pool(held_dim: int, n_scenarios: int, rng: np.random.Generator,
                       oracle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (c, m, y) triples with c_{held_dim} = 0 throughout."""
    cs = rng.uniform(-C_BOX, C_BOX, size=(n_scenarios, N_WC))
    cs[:, held_dim] = 0.0
    all_c, all_m, all_y = [], [], []
    for c in cs:
        ms = rng.uniform(*M_RANGE, size=K_CTX)
        cs_per = np.tile(c, (K_CTX, 1))
        ys = oracle.truth(cs_per, ms)
        all_c.append(cs_per); all_m.append(ms); all_y.append(ys)
    c_arr = np.concatenate(all_c, axis=0)
    m_arr = np.concatenate(all_m, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    return c_arr, m_arr, y_arr


def make_test_pool(held_dim: int, n_scenarios: int, rng: np.random.Generator,
                   oracle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test scenarios: c_{held_dim} ~ U([-0.5, 0.5]) non-zero, others same."""
    cs = rng.uniform(-0.5, 0.5, size=(n_scenarios, N_WC))
    # Ensure the held operator is meaningfully non-zero.
    signs = rng.choice([-1.0, 1.0], size=n_scenarios)
    cs[:, held_dim] = signs * rng.uniform(0.2, 0.5, size=n_scenarios)
    all_c, all_m, all_y = [], [], []
    for c in cs:
        ms = rng.uniform(*M_RANGE, size=K_CTX)
        cs_per = np.tile(c, (K_CTX, 1))
        ys = oracle.truth(cs_per, ms)
        all_c.append(cs_per); all_m.append(ms); all_y.append(ys)
    return (np.concatenate(all_c), np.concatenate(all_m),
            np.concatenate(all_y))


def fit_ridge(Phi: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    d = Phi.shape[1]
    A = Phi.T @ Phi + alpha * np.eye(d)
    return np.linalg.solve(A, Phi.T @ y)


def r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def main():
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    rng = np.random.default_rng(SEED)

    # ---- Reference: train on full Wilson box, evaluate on full Wilson box ----
    print("# Reference: train with all four operators active")
    c_tr, m_tr, y_tr = [], [], []
    cs = rng.uniform(-C_BOX, C_BOX, size=(N_TRAIN, N_WC))
    for c in cs:
        ms = rng.uniform(*M_RANGE, size=K_CTX)
        cs_per = np.tile(c, (K_CTX, 1))
        ys = oracle.truth(cs_per, ms)
        c_tr.append(cs_per); m_tr.append(ms); y_tr.append(ys)
    c_tr = np.concatenate(c_tr); m_tr = np.concatenate(m_tr); y_tr = np.concatenate(y_tr)
    Phi_tr = feature_map(c_tr, m_tr)
    W = fit_ridge(Phi_tr, y_tr, ALPHA_RIDGE)
    # Test on a fresh in-distribution pool.
    cs_te = rng.uniform(-C_BOX, C_BOX, size=(N_TEST, N_WC))
    c_te, m_te, y_te = [], [], []
    for c in cs_te:
        ms = rng.uniform(*M_RANGE, size=K_CTX)
        cs_per = np.tile(c, (K_CTX, 1))
        ys = oracle.truth(cs_per, ms)
        c_te.append(cs_per); m_te.append(ms); y_te.append(ys)
    c_te = np.concatenate(c_te); m_te = np.concatenate(m_te); y_te = np.concatenate(y_te)
    Phi_te = feature_map(c_te, m_te)
    y_pred = Phi_te @ W
    r2_ref = r2(y_pred, y_te)
    print(f"  in-box held-out R^2 = {r2_ref:+.4f}")

    # ---- Holdout: each operator held at zero in turn ----
    results = {"reference_R2": r2_ref, "holdouts": {}}
    print("\n# Operator-holdout test")
    print(f"  {'held':<8s} {'R2_full':>10s} {'per-op R2 vs test active':>40s}")
    for k, name in enumerate(WC_NAMES):
        rng_holdout = np.random.default_rng(SEED + 1 + k)
        c_h_tr, m_h_tr, y_h_tr = make_training_pool(k, N_TRAIN, rng_holdout, oracle)
        Phi_h_tr = feature_map(c_h_tr, m_h_tr)
        W_h = fit_ridge(Phi_h_tr, y_h_tr, ALPHA_RIDGE)

        c_h_te, m_h_te, y_h_te = make_test_pool(k, N_TEST, rng_holdout, oracle)
        Phi_h_te = feature_map(c_h_te, m_h_te)
        y_h_pred = Phi_h_te @ W_h
        r2_h_full = r2(y_h_pred, y_h_te)

        # Per-operator: project test pool onto each c-axis to read off
        # the held-out direction. We report R^2 against y_te restricted
        # to the held-out operator direction by integrating along all
        # other axes: for the morphing ansatz this is the SM-relative
        # rate at that operator.
        # Simpler and more directly comparable to empirical-results.md:
        # report the overall held-out R^2 along with the worst-case slice
        # over the test pool where the held operator dominates.
        held_dom_mask = np.abs(c_h_te[:, k]) > 0.3
        if held_dom_mask.any():
            r2_dom = r2(y_h_pred[held_dom_mask], y_h_te[held_dom_mask])
        else:
            r2_dom = np.nan

        print(f"  {name:<8s} {r2_h_full:+10.3f} {r2_dom:+40.3f}")
        results["holdouts"][name] = {
            "r2_full_test": float(r2_h_full),
            "r2_held_dominant": float(r2_dom),
        }

    out = OUT_DIR / "operator_holdout.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
