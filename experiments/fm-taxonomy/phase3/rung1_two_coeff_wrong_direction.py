"""
Rung 1: two Wilson coefficients, and the "are we acquiring in the wrong
direction?" test.

n=2 SMEFT-DY morphing: a four-fermion coefficient c1 (stiff, tail-growing) and
a vertex coefficient c2 (flat, near-degenerate with the SM normalisation). The
morphing feature is v(c) = [1, c1, c2, c1^2, c2^2, c1 c2]; a working point c
queried on the oracle returns y = v(c) . T* + noise, with T* the ground-truth
templates. The Intention head is Bayesian linear regression over T.

Active learning chooses the next *working point* to query (this is the paper's
"working-point curvature acquisition"). We compare:
  random        - uniform pick from the pool
  leverage      - max info gain  1/2 log(1 + v^T A^-1 v / s2)   (BALD/D-optimal)
  param_a       - A-optimal on the TARGET functional (direction-aware)
  epig          - predictive info gain on a near-SM TARGET set

The quantity we are scored on is the prediction near the SM working point, where
real data lives. Hypothesis (the taxonomy's axis IV-A): leverage contracts the
posterior volume but does NOT beat random on the near-SM target, because the
high-leverage working points are extreme/high-curvature and mutually collinear
(cos_Ainv -> 1) thanks to the c1/c2 near-degeneracy; the direction-aware
param_a/epig recover the win.

numpy only. python3 rung1_two_coeff_wrong_direction.py
"""
import numpy as np

# ---- ground-truth morphing templates [T0, A1, A2, B11, B22, B12] ----------
# A1 (four-fermion interference): sizable.  A2 (vertex): small -> weak ID.
# Quadratics modest; B22 (vertex^2) tiny so c2 is hard to pin from any design.
T_TRUE = np.array([1.0, 0.90, 0.06, 0.35, 0.02, 0.10])
S2 = 0.04 ** 2          # observation variance per query
LAM = 1e-3              # ridge prior precision

def vfeat(c):
    c1, c2 = c
    return np.array([1.0, c1, c2, c1 * c1, c2 * c2, c1 * c2])

# ---- candidate pool of working points, and a near-SM target set -----------
g = np.linspace(-1.5, 1.5, 21)
POOL = np.array([(a, b) for a in g for b in g])          # 441 working points
V_POOL = np.array([vfeat(c) for c in POOL])              # 441 x 6
# target: predict the rate at near-SM working points (where data lives)
gt = np.linspace(-0.3, 0.3, 5)
TARGETS = np.array([(a, b) for a in gt for b in gt])
V_TGT = np.array([vfeat(c) for c in TARGETS])            # 25 x 6
P_BAR = V_TGT.mean(0)                                     # target direction

def cos_ainv_top_decile(Ainv, Vcand):
    """Mean |cos| between the top-decile-leverage candidates in the A^-1 metric."""
    lev = np.einsum('ij,jk,ik->i', Vcand, Ainv, Vcand)
    k = max(2, len(Vcand) // 10)
    idx = np.argsort(lev)[-k:]
    G = Vcand[idx] @ Ainv @ Vcand[idx].T
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = np.abs(G / np.outer(d, d))
    iu = np.triu_indices(len(idx), 1)
    return float(C[iu].mean())

def run(acq, seed, rounds=25, n_seedctx=6):
    rng = np.random.default_rng(seed)
    avail = list(range(len(POOL)))
    chosen = list(rng.choice(avail, n_seedctx, replace=False))
    for i in chosen:
        avail.remove(i)
    rec = []
    for r in range(rounds):
        V = V_POOL[chosen]                                   # context features
        y = V @ T_TRUE + rng.normal(0, np.sqrt(S2), len(V))  # noisy oracle queries
        A = V.T @ V / S2 + LAM * np.eye(6)                   # posterior precision
        Ainv = np.linalg.inv(A)
        w = Ainv @ (V.T @ y / S2)                            # posterior mean (MLE/MAP)
        # scored quantity: prediction error + predictive var on near-SM targets
        pred_err = np.sqrt(np.mean((V_TGT @ w - V_TGT @ T_TRUE) ** 2))
        pred_var = np.mean(np.einsum('ij,jk,ik->i', V_TGT, Ainv, V_TGT))
        cosA = cos_ainv_top_decile(Ainv, V_POOL[avail])
        rec.append((pred_err, pred_var, cosA))
        # --- acquisition: pick next working point from the pool ---
        Va = V_POOL[avail]
        AiV = Va @ Ainv                                      # (navail x 6)
        quad = np.einsum('ij,ij->i', AiV, Va)                # v^T A^-1 v
        denom = 1.0 + quad / S2
        if acq == 'random':
            j = rng.integers(len(avail))
        elif acq == 'leverage':
            j = int(np.argmax(0.5 * np.log(denom)))
        elif acq == 'param_a':                               # A-optimal on P_BAR
            cross = AiV @ P_BAR                              # (p^T A^-1 v)
            gain = (cross ** 2 / S2) / denom
            j = int(np.argmax(gain))
        elif acq == 'epig':                                  # pred-info on targets
            cross = AiV @ V_TGT.T                            # (navail x ntgt)
            gain = np.sum((cross ** 2 / S2) / denom[:, None], axis=1)
            j = int(np.argmax(gain))
        chosen.append(avail.pop(j))
    return np.array(rec)                                     # rounds x 3

ACQS = ['random', 'leverage', 'param_a', 'epig']
SEEDS = range(2000, 2040)
final = {a: [] for a in ACQS}
for s in SEEDS:
    for a in ACQS:
        final[a].append(run(a, s)[-1])                       # last round
final = {a: np.array(v) for a, v in final.items()}

ref = final['random']
print(f"n=2 EFT working-point acquisition: {len(list(SEEDS))} seeds, scored on "
      f"near-SM target prediction")
print(f"{'acq':>9} | {'pred_err':>10} {'pred_var':>10} {'cos_Ainv':>9} | "
      f"{'z_err':>6} {'z_var':>6}")
print("-" * 64)
for a in ACQS:
    F = final[a]
    err, var, cosA = F.mean(0)
    # z vs random on the paired difference (negative = better than random)
    derr = F[:, 0] - ref[:, 0]
    dvar = F[:, 1] - ref[:, 1]
    ze = derr.mean() / (derr.std(ddof=1) / np.sqrt(len(derr)) + 1e-30)
    zv = dvar.mean() / (dvar.std(ddof=1) / np.sqrt(len(dvar)) + 1e-30)
    ze = 0.0 if a == 'random' else ze
    zv = 0.0 if a == 'random' else zv
    print(f"{a:>9} | {err:10.4e} {var:10.4e} {cosA:9.3f} | {ze:6.1f} {zv:6.1f}")
print("\nz_err/z_var: paired z vs random; negative = beats random (lower err/var).")
print("Read: does leverage beat random on pred_err (z_err<0), or only on")
print("pred_var (z_var<0) while pred_err ties/loses? Do param_a/epig beat random")
print("on pred_err? cos_Ainv flags how collinear the high-leverage picks are.")
