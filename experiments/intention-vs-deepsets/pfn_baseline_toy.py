"""PFN baseline on the polynomial-toy oracle (matches paper Table 1).

Identical to ``pfn_baseline.py`` but with ``data`` (DummyAnalyticOracle)
substituted for ``data_smeft`` (AnalyticSMEFTOracle), so the PFN result
sits in the same Wilson-coefficient-times-log-mass polynomial regime
the headline Intention / DeepSets / cheat-regressor numbers were
computed against.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from data import (
    make_oracle, sample_c, sample_c_shell,
    make_dataset, scenarios_to_tensors,
    N_WC, M_RANGE,
)
from pfn_baseline import (
    PFNHead, r2_per_scenario, mse_total, summarise, train,
    N_TRAIN_SCENARIOS, N_TEST_IN, N_TEST_OUT, K_CTX, Q_QUERY,
    C_MAX_TRAIN, C_OUTER, N_META_STEPS, LR, BATCH_S, SEED,
    D_MODEL, N_HEAD, DIM_FF, N_LAYERS,
)


OUTPUT_DIR = HERE / "output_pfn_toy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("# PFN baseline — polynomial-toy oracle (Table 1 comparison)")
    oracle = make_oracle(seed=0, noise_frac=0.0)
    train_scen = make_dataset(
        N_TRAIN_SCENARIOS,
        lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
        oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=1)
    test_in_scen = make_dataset(
        N_TEST_IN,
        lambda r: sample_c(r, 1, c_max=C_MAX_TRAIN)[0],
        oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=42)
    test_out_scen = make_dataset(
        N_TEST_OUT,
        lambda r: sample_c_shell(r, 1, c_inner=C_MAX_TRAIN,
                                 c_outer=C_OUTER)[0],
        oracle, K_ctx=K_CTX, Q_query=Q_QUERY, seed=43)
    train_t = scenarios_to_tensors(train_scen)
    test_in_t = scenarios_to_tensors(test_in_scen)
    test_out_t = scenarios_to_tensors(test_out_scen)

    model = PFNHead()
    n_params = model.n_params
    print(f"  PFN n_params = {n_params}")
    losses, hist, wall = train(
        model, train_t, test_in_t, N_META_STEPS, LR, BATCH_S, seed=SEED)
    print(f"  trained in {wall:.1f}s")
    model.eval()
    with torch.no_grad():
        yp_in = model(test_in_t["M_ctx"], test_in_t["Y_ctx"],
                      test_in_t["M_query"]).cpu().numpy()
        yp_out = model(test_out_t["M_ctx"], test_out_t["Y_ctx"],
                       test_out_t["M_query"]).cpu().numpy()

    res_in = summarise("PFN_toy", yp_in,
                       test_in_t["Y_query"].cpu().numpy())
    res_out = summarise("PFN_toy", yp_out,
                        test_out_t["Y_query"].cpu().numpy())

    print(f"  in : median R2 = {res_in['r2_median']:+.4f}  "
          f"p5 = {res_in['r2_p5']:+.4f}  MSE = {res_in['mse']:.4e}")
    print(f"  out: median R2 = {res_out['r2_median']:+.4f}  "
          f"p5 = {res_out['r2_p5']:+.4f}  MSE = {res_out['mse']:.4e}")

    summary = dict(
        oracle="polynomial_toy",
        config=dict(
            d_model=D_MODEL, n_head=N_HEAD, dim_feedforward=DIM_FF,
            n_layers=N_LAYERS, n_params=n_params,
            n_train=N_TRAIN_SCENARIOS, n_steps=N_META_STEPS,
            lr=LR, batch=BATCH_S, seed=SEED,
            K_ctx=K_CTX, Q_query=Q_QUERY,
            c_max_train=C_MAX_TRAIN, c_outer=C_OUTER,
        ),
        PFN=dict(
            in_=res_in, out=res_out, n_params=n_params, wall=wall,
        ),
    )
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
