"""Is the precursor B.9 contraction a checkpoint/basis-mismatch artifact?

The frozen probe_W_mass_only_v2 was fit on full-chain-run/output/intention_fm.pt
(probe_efficiency.MODEL_CHECKPOINT default). The B.9 sweep (_common.DEFAULT_CKPT)
loads output_bimodal_random_mu_afb_stressed_s2027/intention_fm.pt — a DIFFERENT
training (max weight diff 1.85). A W fit in one learned psi-basis is meaningless
in another.

Tests:
  (A) self-consistency: refit W on output/intention_fm.pt (the frozen W's OWN
      head), with the EXACT probe_efficiency recipe (mass-only mu, withhold
      |clq3| in [0.6,1.0], intercept, N=400, SEED 7771). Compare to frozen:
      cos(p1) should be ~1 and contraction ~equal if my procedure matches.
  (B) matched-basis contraction on output/intention_fm.pt: random vs param_epig_a
      with the self-consistent refit W. Does targeted beat random?
  (C) correct-probe-for-the-sweep-head: refit W on the s2027 head and run the
      same contraction. Does it contract at all?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import (
    N_WC, M_RANGE, K_CTX, SIGMA_Y, WITHHOLD_DIM, WITHHOLD_DIM_2,
    TARGET_C_LQ3, TARGET_C_HQ3, build_oracle, load_pretrained_model,
    load_probe, angular_fisher_V,
)
from d0_positive_control import run_chain, target_c

REPO = HERE.parent.parent
CKPT_PROBE = REPO / "experiments/full-chain-run/output/intention_fm.pt"          # frozen W's head
CKPT_SWEEP = REPO / "experiments/full-chain-run/output_bimodal_random_mu_afb_stressed_s2027/intention_fm.pt"
C_BOX, WB = 0.7, (0.6, 1.0)


def refit_W(head, oracle, observable, withhold, seed=7771, n_train=400, K=12):
    rng = np.random.default_rng(seed)
    cs = np.empty((n_train, N_WC)); i = 0
    while i < n_train:
        c = rng.uniform(-C_BOX, C_BOX, size=N_WC)
        if (not withhold) or abs(c[WITHHOLD_DIM]) < WB[0]:
            cs[i] = c; i += 1
    Ws = []
    for c in cs:
        M = rng.uniform(*M_RANGE, size=K)
        Y = (oracle.truth_mu_fb if observable == "mufb" else oracle.truth)(np.tile(c, (K, 1)), M)
        Y = Y + rng.normal(0.0, SIGMA_Y, size=K)
        _, w, _ = head.A_inv_and_w(M, Y)
        Ws.append(w)
    X = np.hstack([np.stack(Ws), np.ones((n_train, 1))])
    Wb, *_ = np.linalg.lstsq(X, cs, rcond=None)
    return Wb[:-1].T


def contr_pair(head, oracle, V, W, sigma_y, n_cyc=200, seeds=3):
    out = {}
    for acq in ("random", "param_epig_a"):
        rows = [run_chain(head, oracle, target_c(), V.T @ W, sigma_y, V, acq, 2026 + s)
                for s in range(seeds)]
        # run_chain uses module N_CYCLES; override via monkeypatch below
        out[acq] = float(np.mean([r["contr_d1"] for r in rows]))
    return out


def main():
    import d0_positive_control as d0
    d0.N_CYCLES = 200            # shorter for the diagnostic
    oracle = build_oracle(seed=2026)
    V, _ = angular_fisher_V(oracle)
    frozen = load_probe(oracle)["W"]

    print("## (A) self-consistency on the frozen W's OWN head (output/intention_fm.pt)")
    head_p = load_pretrained_model(CKPT_PROBE)
    W_match = refit_W(head_p, oracle, "mass", withhold=True)
    p1f = (V.T @ frozen)[1]; p1m = (V.T @ W_match)[1]
    cosm = abs(p1f @ p1m) / (np.linalg.norm(p1f) * np.linalg.norm(p1m) + 1e-30)
    print(f"    cos(refit_match p1, frozen p1) = {cosm:.4f}  (──> ~1 means my recipe matches the frozen fit)")
    cf = contr_pair(head_p, oracle, V, frozen, SIGMA_Y)
    cm = contr_pair(head_p, oracle, V, W_match, SIGMA_Y)
    print(f"    frozen W on its own head : random {cf['random']:.4f}  param_epig_a {cf['param_epig_a']:.4f}")
    print(f"    refit  W on its own head : random {cm['random']:.4f}  param_epig_a {cm['param_epig_a']:.4f}")

    print("\n## (C) correct probe refit on the SWEEP head (s2027) — what B.9 should have used")
    head_s = load_pretrained_model(CKPT_SWEEP)
    W_s = refit_W(head_s, oracle, "mass", withhold=True)
    cs = contr_pair(head_s, oracle, V, W_s, SIGMA_Y)
    cfs = contr_pair(head_s, oracle, V, frozen, SIGMA_Y)
    print(f"    refit-on-s2027 W : random {cs['random']:.4f}  param_epig_a {cs['param_epig_a']:.4f}")
    print(f"    frozen W on s2027: random {cfs['random']:.4f}  param_epig_a {cfs['param_epig_a']:.4f}  (the published pairing)")


if __name__ == "__main__":
    main()
