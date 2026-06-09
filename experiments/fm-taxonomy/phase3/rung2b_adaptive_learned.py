"""
Rung 2b: the ADAPTIVE learner. Does the AL win survive when the agent must
LEARN the sensitivities instead of being handed the oracle gradients?

Rung 2 used oracle-computed Jacobians (known-simulator optimal design). Here the
agent learns a morphing surrogate from a finite, NOISY set of oracle queries at
basis working points, and acquires using the surrogate's (imperfect) gradients
J_hat. The vertex A_FB signature is tiny (~0.002), so a noisy surrogate can lose
it entirely -- a cold start in which the agent does not yet know A_FB breaks the
degeneracy.

Three agents, all evaluated on the TRUE Fisher (decisions under the learned
model, outcome under reality):
  random            uniform allocation
  exploit           A-optimal on the learned J_hat (greedy)
  explore_exploit   A-optimal on J_hat, epsilon-greedy channel exploration

Question: with a poor surrogate, does greedy exploit fail to find A_FB (i.e.
reproduce "random wins") while explore_exploit recovers the rung-2 win?

PYTHONPATH=/home/vince/ALETHIA python3 rung2b_adaptive_learned.py
"""
import numpy as np
from modules.analytic_smeft import differential_xs, differential_afb

LAM = 1000.0
C_TRUE = {"c_lq^(3)": 0.5, "c_phi_q^(3)": 0.5}
C0 = np.array([0.5, 0.5])                       # truth in (c_lq3, c_phiq3)
m_mass = np.linspace(420.0, 2380.0, 40)
m_afb = np.linspace(600.0, 1800.0, 6)
BINW, LUMI, AFB_COST = 50.0, 3.0e6, 8.0

def xs(c1, c2, m):
    return differential_xs({"c_lq^(3)": c1, "c_phi_q^(3)": c2}, m, lambda_scale=LAM)["differential_xs"]
def afb(c1, c2, m):
    return differential_afb({"c_lq^(3)": c1, "c_phi_q^(3)": c2}, m, lambda_scale=LAM)["A_FB"]

# ---- truth: exact oracle gradients + Poisson variances (for EVALUATION) ----
H = 1e-3
def grad_true(fn, m):
    g1 = (fn(C0[0] + H, C0[1], m) - fn(C0[0] - H, C0[1], m)) / (2 * H)
    g2 = (fn(C0[0], C0[1] + H, m) - fn(C0[0], C0[1] - H, m)) / (2 * H)
    return np.array([g1, g2])
R0 = xs(C0[0], C0[1], m_mass)
A0 = afb(C0[0], C0[1], m_afb)
dR_true = grad_true(xs, m_mass)                 # (2, 40)
dA_true = grad_true(afb, m_afb)                 # (2, 6)
N_mass = LUMI * xs(0, 0, m_mass) * BINW
N_afb = LUMI * xs(0, 0, m_afb) * BINW / AFB_COST
var_mass = R0 ** 2 / N_mass                      # relative-rate Poisson var
var_afb = (1.0 - A0 ** 2) / N_afb

def true_cell_fisher(i, is_afb):
    """3x3 Fisher contribution (c1, c2, alpha) of one measured cell, true J."""
    if is_afb:
        J = np.array([dA_true[0, i], dA_true[1, i], 0.0]); v = var_afb[i]
    else:
        J = np.array([dR_true[0, i], dR_true[1, i], R0[i]]); v = var_mass[i]
    return J[:, None] * J[None, :] / v

# ---- learned surrogate: noisy morphing fit -> J_hat (for DECISIONS) --------
BASIS = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.2, 1.2],
                  [0.0, 0.6], [0.7, 0.0]])       # 6 working points (2D quadratic)
VB = np.array([[1, a, b, a * a, b * b, a * b] for a, b in BASIS])   # 6x6 morphing matrix
VBinv = np.linalg.inv(VB)

def learn_Jhat(fn, m, rel_noise, abs_noise, rng):
    """Fit morphing templates from noisy oracle queries; return d/dc at C0."""
    Y = np.array([fn(a, b, m) for a, b in BASIS])           # 6 x nbins (clean)
    Y = Y + rng.normal(0, rel_noise * np.abs(Y) + abs_noise, Y.shape)
    Theta = VBinv @ Y                                        # 6 x nbins templates
    a0, b0 = C0
    d1 = Theta[1] + 2 * Theta[3] * a0 + Theta[5] * b0        # d/dc1 at C0
    d2 = Theta[2] + 2 * Theta[4] * b0 + Theta[5] * a0        # d/dc2 at C0
    return np.array([d1, d2])

def cell_fisher_hat(i, is_afb, Jh_mass, Jh_afb):
    if is_afb:
        J = np.array([Jh_afb[0, i], Jh_afb[1, i], 0.0]); v = var_afb[i]
    else:
        J = np.array([Jh_mass[0, i], Jh_mass[1, i], R0[i]]); v = var_mass[i]
    return J[:, None] * J[None, :] / v

PRIOR = np.diag([1e-4, 1e-4, 1e-6])
N_MASS, N_AFB = len(m_mass), len(m_afb)
NCELL = N_MASS + N_AFB

def run(agent, seed, rel_noise, abs_noise, budget=24, n_seed=3, eps=0.3):
    rng = np.random.default_rng(seed)
    # the agent's learned (imperfect) sensitivities
    Jh_mass = learn_Jhat(xs, m_mass, rel_noise, abs_noise, rng)
    Jh_afb = learn_Jhat(afb, m_afb, rel_noise, abs_noise, rng)
    F_true = PRIOR.copy()                          # accumulated TRUE Fisher (outcome)
    F_hat = PRIOR.copy()                           # agent's believed Fisher (decisions)
    chosen = []
    for k in list(rng.choice(NCELL, n_seed, replace=False)):
        is_a = k >= N_MASS
        F_true = F_true + true_cell_fisher(k - N_MASS if is_a else k, is_a)
        F_hat = F_hat + cell_fisher_hat(k - N_MASS if is_a else k, is_a, Jh_mass, Jh_afb)
        chosen.append(k)
    for _ in range(budget):
        if agent == "random" or (agent == "explore_exploit" and rng.random() < eps):
            j = int(rng.integers(NCELL))
        else:                                       # A-optimal on the BELIEVED Fisher
            best, bj = -np.inf, 0
            for k in range(NCELL):
                is_a = k >= N_MASS
                Fn = F_hat + cell_fisher_hat(k - N_MASS if is_a else k, is_a, Jh_mass, Jh_afb)
                val = -np.trace(np.linalg.inv(Fn)[:2, :2])
                if val > best:
                    best, bj = val, k
            j = bj
        is_a = j >= N_MASS
        F_true = F_true + true_cell_fisher(j - N_MASS if is_a else j, is_a)
        F_hat = F_hat + cell_fisher_hat(j - N_MASS if is_a else j, is_a, Jh_mass, Jh_afb)
        chosen.append(j)
    sigma = np.sqrt(np.diag(np.linalg.inv(F_true)))[:2]      # TRUE marginal CR sigma
    n_afb = sum(1 for k in chosen if k >= N_MASS)
    return sigma, n_afb

SEEDS = range(5000, 5030)
REGIMES = [("good surrogate", 0.02, 1e-4), ("poor surrogate (cold)", 0.20, 3e-3)]
AGENTS = ["random", "exploit", "explore_exploit"]

print("Rung 2b: adaptive learner with LEARNED (noisy) sensitivities, real oracle.")
print("Evaluated on the TRUE Fisher. 30 seeds, budget 27 cells, A_FB = 6/46 of pool.\n")
for label, rel, ab in REGIMES:
    print(f"=== {label}  (rel_noise={rel}, abs_noise={ab}) ===")
    print(f"{'agent':>16} | {'sigma(c_lq3)':>13} {'sigma(c_phiq3)':>15} {'#AFB':>5}")
    print("-" * 58)
    for ag in AGENTS:
        S, NA = [], []
        for s in SEEDS:
            sg, na = run(ag, s, rel, ab)
            S.append(sg); NA.append(na)
        S = np.array(S)
        print(f"{ag:>16} | {S[:,0].mean():13.4e} {S[:,1].mean():15.4e} {np.mean(NA):5.1f}")
    print()
