"""
Rung A: the simplest active-learning example that WORKS.

A clean, well-posed 1-D regression. We learn f(x) with a well-conditioned RBF
feature map and Bayesian linear regression. Active learning picks the next x
where the model is most uncertain (predictive variance = BALD/uncertainty
sampling in the Gaussian-linear case); the baseline picks x at random. We track
held-out RMSE vs the number of labels.

No SMEFT, no degeneracy, no nuisance. This is the positive control: in a
well-posed problem, information-driven acquisition beats random, cleanly and at
every budget. Everything later builds on this.

python3 rungA_simple_al.py
"""
import numpy as np

rng_global = np.random.default_rng(0)

# ---- the target function and a well-conditioned feature map ---------------
def f_true(x):
    return np.sin(6.0 * x) + 0.5 * np.sin(13.0 * x) + 0.3 * x

CENTERS = np.linspace(0.0, 1.0, 24)          # RBF centers
GAMMA = 40.0                                  # RBF width
SIGMA = 0.05                                  # label noise
LAM = 1e-3                                     # ridge prior precision

def phi(x):
    x = np.atleast_1d(x)[:, None]
    return np.exp(-GAMMA * (x - CENTERS[None, :]) ** 2)   # (n, 24)

POOL = np.linspace(0.0, 1.0, 400)             # candidate x to label
TEST = np.linspace(0.0, 1.0, 1000)            # held-out evaluation grid
PHI_TEST = phi(TEST)
Y_TEST = f_true(TEST)

def al_run(strategy, seed, n_start=3, budget=25):
    rng = np.random.default_rng(seed)
    avail = list(range(len(POOL)))
    chosen = list(rng.choice(avail, n_start, replace=False))
    for i in chosen:
        avail.remove(i)
    curve = []
    for _ in range(budget):
        X = POOL[chosen]
        PHI = phi(X)
        y = f_true(X) + rng.normal(0, SIGMA, len(X))
        A = PHI.T @ PHI / SIGMA**2 + LAM * np.eye(len(CENTERS))
        Ainv = np.linalg.inv(A)
        w = Ainv @ (PHI.T @ y / SIGMA**2)
        rmse = np.sqrt(np.mean((PHI_TEST @ w - Y_TEST) ** 2))
        curve.append(rmse)
        # acquisition over the remaining pool
        Pa = phi(POOL[avail])
        if strategy == "random":
            j = int(rng.integers(len(avail)))
        elif strategy == "uncertainty":             # max predictive variance (BALD)
            var = np.einsum('ij,jk,ik->i', Pa, Ainv, Pa)
            j = int(np.argmax(var))
        chosen.append(avail.pop(j))
    return np.array(curve)

SEEDS = range(100, 140)
res = {s: np.array([al_run(s, seed) for seed in SEEDS]) for s in ["random", "uncertainty"]}

budgets = [3, 5, 8, 12, 18, 25]
print("Simplest AL that works: held-out RMSE vs #labels (40 seeds, lower is better)")
print(f"{'#labels':>8} | {'random':>10} {'uncertainty':>12} | {'ratio rand/unc':>14}")
print("-" * 52)
for b in budgets:
    r = res["random"][:, b - 1].mean()
    u = res["uncertainty"][:, b - 1].mean()
    print(f"{b:>8} | {r:10.4f} {u:12.4f} | {r/u:14.2f}")
# significance at the final budget
d = res["random"][:, -1] - res["uncertainty"][:, -1]
z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print(f"\nAt {budgets[-1]} labels: uncertainty sampling beats random by {z:.1f} sigma "
      f"(paired over seeds).")
print("Mechanism: max-variance acquisition fills the domain; random clumps and")
print("leaves gaps. Well-posed problem => information-driven acquisition wins.")
