"""
Rung B: a foundation-model representation makes active learning few-shot.

Build-up from rung A. Now the task is one member of a FAMILY of functions
f_c(x) = sum_k c_k g_k(x) spanned by K=4 smooth modes. A "foundation model" is
pretrained on the family the only way a label-free model can be: it sees many
family members and learns their shared low-dimensional structure (here via SVD
of the family's function values -- the learned representation V_FM).

For a NEW family member we run the same AL loop as rung A in three feature
spaces:
  FM         the K=4 learned family directions (the pretrained representation)
  generic    24 RBF features (rung A's representation; knows nothing of the family)
  and random acquisition as the baseline in each.

Claim: AL on the FM representation converges in ~K labels, because the new task
lives in the K-dim family span and uncertainty sampling pins those K directions.
Generic features need many more labels; random is slowest. AL + good
representation > AL + generic > random.

python3 rungB_representation.py
"""
import numpy as np

GRID = np.linspace(0.0, 1.0, 200)             # everything lives on this grid
SIGMA = 0.05
LAM = 1e-3
K = 4                                          # family latent dimension

# ---- the family: K smooth Fourier modes ----------------------------------
def modes(x):
    return np.stack([np.ones_like(x), np.sin(2*np.pi*x), np.cos(2*np.pi*x),
                     np.sin(4*np.pi*x)], axis=1)        # (n, 4)
G = modes(GRID)                                          # (200, 4) true basis

# ---- pretrain the "foundation model": learn the family span from many members
def pretrain_fm(seed, n_members=300):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n_members, K))
    F = C @ G.T + rng.normal(0, 0.02, size=(n_members, len(GRID)))   # family data
    _, _, Vt = np.linalg.svd(F - F.mean(0, keepdims=True), full_matrices=False)
    return Vt[:K].T                                       # (200, K) learned rep on grid

# ---- generic RBF representation (rung A) ----------------------------------
RBF_C = np.linspace(0, 1, 24)
PHI_RBF = np.exp(-40.0 * (GRID[:, None] - RBF_C[None, :]) ** 2)       # (200, 24)

def al_run(feat, strategy, c_star, seed, n_start=2, budget=14):
    """feat: (200, d) feature matrix on the grid. c_star: true family coeffs."""
    rng = np.random.default_rng(seed)
    y_true = G @ c_star                                   # target on grid
    d = feat.shape[1]
    avail = list(range(len(GRID)))
    chosen = list(rng.choice(avail, n_start, replace=False))
    for i in chosen:
        avail.remove(i)
    curve = []
    for _ in range(budget):
        P = feat[chosen]
        y = y_true[chosen] + rng.normal(0, SIGMA, len(chosen))
        A = P.T @ P / SIGMA**2 + LAM * np.eye(d)
        Ainv = np.linalg.inv(A)
        w = Ainv @ (P.T @ y / SIGMA**2)
        curve.append(np.sqrt(np.mean((feat @ w - y_true) ** 2)))
        Pa = feat[avail]
        if strategy == "random":
            j = int(rng.integers(len(avail)))
        else:
            var = np.einsum('ij,jk,ik->i', Pa, Ainv, Pa)
            j = int(np.argmax(var))
        chosen.append(avail.pop(j))
    return np.array(curve)

SEEDS = range(200, 240)
V_FM = pretrain_fm(seed=7)
configs = [("FM", V_FM, "uncertainty"), ("FM", V_FM, "random"),
           ("generic", PHI_RBF, "uncertainty"), ("generic", PHI_RBF, "random")]
results = {}
for name, feat, strat in configs:
    curves = []
    for s in SEEDS:
        c_star = np.random.default_rng(s).normal(size=K)
        curves.append(al_run(feat, strat, c_star, s))
    results[(name, strat)] = np.array(curves).mean(0)

budgets = [2, 4, 6, 9, 14]
print("RMSE vs #labels on a NEW family member (40 tasks). K=4 family dimension.")
print(f"{'#labels':>8} | {'FM+AL':>8} {'FM+rand':>8} {'gen+AL':>8} {'gen+rand':>9}")
print("-" * 50)
for b in budgets:
    row = [results[k][b-1] for k in [("FM","uncertainty"),("FM","random"),
                                     ("generic","uncertainty"),("generic","random")]]
    print(f"{b:>8} | {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:9.4f}")
print(f"\nAt {K} labels FM+AL should already be near zero (it only needs to pin K")
print("directions). Generic features and random need many more labels.")
print("Take-away: the foundation representation is what makes AL few-shot.")
