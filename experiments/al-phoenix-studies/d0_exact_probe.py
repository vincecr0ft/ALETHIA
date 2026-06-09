"""Definitive: reproduce probe_efficiency's EXACT published W-fit, confirm it
matches the frozen npz on its own checkpoint, then apply the same authoritative
recipe to the s2027 sweep head and test contraction.

Removes the 'maybe my refit is just broken' doubt: if the exact recipe matches
frozen on output/ (cos ~1) but a same-recipe fit on s2027 shows no contraction
while frozen-on-s2027 does, the B.9 contraction is a checkpoint-basis artifact.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "full-chain-run"))

import probe_efficiency as pe
from modules.surrogate.intention import IntentionFM
from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from _common import (load_probe, angular_fisher_V, build_oracle, SIGMA_Y as SY_SWEEP)
from d0_positive_control import run_chain, target_c
import d0_positive_control as d0

CKPT_OUT = REPO / "experiments/full-chain-run/output/intention_fm.pt"
CKPT_S2027 = REPO / "experiments/full-chain-run/output_bimodal_random_mu_afb_stressed_s2027/intention_fm.pt"


def exact_W(ckpt, oracle_truth, morphing):
    """probe_efficiency.main's exact OLS probe fit on a given checkpoint."""
    model = IntentionFM(d_psi=16, hidden=64, alpha=1e-3)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(pe.SEED)
    c_train = pe.sample_c_inbox(pe.N_TRAIN_PROBE, rng)
    w_train = pe.batch_implicit_w(model, c_train, oracle_truth, rng, pe.SIGMA_Y, morphing)
    X = np.hstack([w_train, np.ones((pe.N_TRAIN_PROBE, 1))])
    Wb = np.linalg.inv(X.T @ X) @ X.T @ c_train
    return model, Wb[:16].T


def main():
    d0.N_CYCLES = 200
    # probe_efficiency builds the oracle with truth (mass-only) and morphing table
    oracle_pe = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    m_grid = np.linspace(pe.M_RANGE[0], pe.M_RANGE[1], 240)
    morphing = pe.MorphingTable(oracle_pe, m_grid)

    oracle_sweep = build_oracle(seed=2026)   # mu_FB sweep oracle (what stage_c uses)
    V, _ = angular_fisher_V(oracle_sweep)
    frozen = load_probe(oracle_sweep)["W"]

    print("## reproduce exact published probe on output/intention_fm.pt (frozen W's own head)")
    model_out, W_out = exact_W(CKPT_OUT, oracle_pe, morphing)
    # compare full W (4x16) frozen vs exact, row-wise cosine
    cos_rows = [abs(frozen[i] @ W_out[i]) / (np.linalg.norm(frozen[i]) * np.linalg.norm(W_out[i]) + 1e-30)
                for i in range(4)]
    print(f"    row-cos(exact_out, frozen) = {[round(c,3) for c in cos_rows]}  "
          f"(──> ~1 confirms I reproduced the published probe)")

    print("\n## reproduce exact published probe on the s2027 SWEEP head")
    model_s, W_s = exact_W(CKPT_S2027, oracle_pe, morphing)

    def contr(model, W, head_label):
        for acq in ("random", "param_epig_a"):
            rows = [run_chain(model, oracle_sweep, target_c(), V.T @ W, SY_SWEEP, V, acq, 2026 + s)
                    for s in range(3)]
            m = float(np.mean([r["contr_d1"] for r in rows]))
            print(f"    [{head_label}] {acq:14s} contr_d1={m:.4f}")

    print("\n## contraction with the EXACT authoritative probe, head-matched:")
    print("  output head + exact-output W:")
    contr(model_out, W_out, "out/exact-out")
    print("  s2027 head + exact-s2027 W:")
    contr(model_s, W_s, "s2027/exact-s2027")
    print("  s2027 head + frozen W (the PUBLISHED B.9 pairing):")
    contr(model_s, frozen, "s2027/frozen")


if __name__ == "__main__":
    main()
