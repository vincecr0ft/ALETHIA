"""Localize WHY the frozen mass-only W contracts but refits don't.

Same precursor head, same A^-1. Vary only the probe W:
  frozen        : published probe_W_mass_only_v2 (fit on mass-only mu, withholding)
  refit_mufb    : OLS c~w on mu_FB labels, no withholding (d0 refit, does NOT contract)
  refit_massonly: OLS c~w on mass-only mu (oracle.truth), withholding (replicates probe_efficiency)

For each: P=V^T W, p1 = c-tilde_2 readout vector. On a fixed seed + leverage-grown
context, print contraction p1^T A_final^-1 p1 / p1^T A_seed^-1 p1, the cosine of p1
against frozen p1, and the decomposition of p1 onto the eigenvectors of A_seed
(how much of p1 sits on high- vs low-eigenvalue / high- vs low-leverage directions).
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
    load_probe, angular_fisher_V, build_pool_P0, ig_per_candidate,
)

C_BOX = 0.7
WITHHOLD_BAND = (0.6, 1.0)


def target_c():
    c = np.zeros(N_WC); c[WITHHOLD_DIM] = TARGET_C_LQ3; c[WITHHOLD_DIM_2] = TARGET_C_HQ3
    return c


def sample_c(n, rng, withhold):
    out = np.empty((n, N_WC)); i = 0
    while i < n:
        c = rng.uniform(-C_BOX, C_BOX, size=N_WC)
        if (not withhold) or abs(c[WITHHOLD_DIM]) < WITHHOLD_BAND[0]:
            out[i] = c; i += 1
    return out


def refit_W(head, oracle, rng, *, observable, withhold, n_train=400, K=12):
    cs = sample_c(n_train, rng, withhold)
    Ws = []
    for c in cs:
        M = rng.uniform(*M_RANGE, size=K)
        if observable == "mufb":
            Y = oracle.truth_mu_fb(np.tile(c, (K, 1)), M)
        else:
            Y = oracle.truth(np.tile(c, (K, 1)), M)
        Y = Y + rng.normal(0.0, SIGMA_Y, size=K)
        _, w, _ = head.A_inv_and_w(M, Y)
        Ws.append(w)
    Wm = np.stack(Ws); X = np.hstack([Wm, np.ones((n_train, 1))])
    Wb, *_ = np.linalg.lstsq(X, cs, rcond=None)
    return Wb[:-1].T


def grown_context(head, oracle, c_tgt, seed, n_cyc=200):
    rng = np.random.default_rng(seed)
    M = rng.uniform(0.5, 1.0, size=K_CTX)
    Y = oracle.truth_mu_fb(np.tile(c_tgt, (K_CTX, 1)), M)
    A_inv_seed, _, _ = head.A_inv_and_w(M, Y)
    pr = np.random.default_rng(rng.bit_generator.random_raw())
    for _ in range(n_cyc):
        pool = build_pool_P0(pr, size=30)
        ig = ig_per_candidate(head, M, Y, pool)
        pick = pool[[int(np.argmax(ig))]]
        Yp = oracle.truth_mu_fb(np.tile(c_tgt, (1, 1)), pick)
        M = np.concatenate([M, pick]); Y = np.concatenate([Y, Yp])
    A_inv_fin, _, _ = head.A_inv_and_w(M, Y)
    return A_inv_seed, A_inv_fin


def main():
    oracle = build_oracle(seed=2026)
    head = load_pretrained_model()
    c_tgt = target_c()
    V, _ = angular_fisher_V(oracle)
    rng = np.random.default_rng(7771)

    Ws = {
        "frozen": load_probe(oracle)["W"],
        "refit_mufb": refit_W(head, oracle, np.random.default_rng(7771), observable="mufb", withhold=False),
        "refit_massonly_wh": refit_W(head, oracle, np.random.default_rng(7771), observable="mass", withhold=True),
        "refit_mufb_wh": refit_W(head, oracle, np.random.default_rng(7771), observable="mufb", withhold=True),
    }

    A_inv_seed, A_inv_fin = grown_context(head, oracle, c_tgt, seed=2026)
    A_seed = np.linalg.inv(A_inv_seed)
    eval_, evec = np.linalg.eigh(A_seed)          # ascending
    order = np.argsort(eval_)[::-1]
    eval_, evec = eval_[order], evec[:, order]    # descending eigenvalue

    p1_frozen = (V.T @ Ws["frozen"])[1]
    print(f"{'W variant':20s} {'contr_d1':>9s} {'cos(p1,frozen)':>15s}  "
          f"weight on A-eig [top3 | bot3]")
    for name, W in Ws.items():
        P = V.T @ W
        p1 = P[1]
        contr = float(p1 @ A_inv_fin @ p1) / float(p1 @ A_inv_seed @ p1)
        cosf = float(abs(p1 @ p1_frozen) / (np.linalg.norm(p1) * np.linalg.norm(p1_frozen) + 1e-30))
        coeff = (evec.T @ p1)
        frac = coeff ** 2 / (coeff @ coeff)
        top3 = frac[:3].sum(); bot3 = frac[-3:].sum()
        print(f"{name:20s} {contr:9.4f} {cosf:15.4f}  "
              f"[{top3:.3f} | {bot3:.3f}]  eig(A) top={eval_[0]:.3g} bot={eval_[-1]:.3g}")
    print("\n(top3 = fraction of the c-tilde2 readout vector lying on the 3 "
          "highest-eigenvalue / highest-leverage directions of A_seed; bot3 on "
          "the 3 lowest. A vector on low-eigenvalue dirs of A sits where A^-1 is "
          "large and barely changes -> no contraction.)")


if __name__ == "__main__":
    main()
