"""
Rung 3: a four-fermion direction where AL wins ROBUSTLY (and for real).

The vertex direction (rung 2/2b) was the worst case: a ~0.002 A_FB signal below
any surrogate noise floor, so learned-sensitivity AL collapsed to random. Here we
take the chirality pair (c_lq^(1), c_lq^(3)). Both grow in the tail; A_FB
separates them with a LARGE signal (~0.03-0.10, including a sign flip for c_lq^(1)
at low mass). The question: does AL beat random by a useful margin, give a real
measurement (sigma << coefficient), AND survive LEARNED sensitivities?

Same machine as rung 2/2b: mass cells abundant, A_FB cells scarce+costly; agent
acquires A-optimal on sensitivities that are either exact or learned from a noisy
morphing surrogate. Evaluated on the TRUE Fisher.

PYTHONPATH=/home/vince/ALETHIA python3 rung3_fourfermion.py
"""
import numpy as np
from modules.analytic_smeft import differential_xs, differential_afb

LAM = 1000.0
PARAMS = ["c_lq^(1)", "c_lq^(3)"]
C0 = np.array([0.3, 0.3])
m_mass = np.linspace(420.0, 2380.0, 40)
m_afb = np.linspace(600.0, 1800.0, 6)
BINW, LUMI, AFB_COST = 50.0, 3.0e6, 8.0
H = 1e-3

def xs(c1, c2, m):
    return differential_xs({PARAMS[0]: c1, PARAMS[1]: c2}, m, lambda_scale=LAM)["differential_xs"]
def afb(c1, c2, m):
    return differential_afb({PARAMS[0]: c1, PARAMS[1]: c2}, m, lambda_scale=LAM)["A_FB"]

def grad(fn, m):
    g1 = (fn(C0[0] + H, C0[1], m) - fn(C0[0] - H, C0[1], m)) / (2 * H)
    g2 = (fn(C0[0], C0[1] + H, m) - fn(C0[0], C0[1] - H, m)) / (2 * H)
    return np.array([g1, g2])

R0, A0 = xs(C0[0], C0[1], m_mass), afb(C0[0], C0[1], m_afb)
dR_t, dA_t = grad(xs, m_mass), grad(afb, m_afb)
N_mass = LUMI * xs(0, 0, m_mass) * BINW
N_afb = LUMI * xs(0, 0, m_afb) * BINW / AFB_COST
var_mass, var_afb = R0 ** 2 / N_mass, (1 - A0 ** 2) / N_afb
N_MASS, N_AFB = len(m_mass), len(m_afb); NCELL = N_MASS + N_AFB
PRIOR = np.diag([1e-4, 1e-4, 1e-6])

BASIS = np.array([[-1, -1], [1, -1], [-1, 1], [1.2, 1.2], [0, 0.6], [0.7, 0]], float)
VB = np.array([[1, a, b, a * a, b * b, a * b] for a, b in BASIS]); VBinv = np.linalg.inv(VB)

def learn_J(fn, m, rel, ab, rng):
    Y = np.array([fn(a, b, m) for a, b in BASIS])
    Y = Y + rng.normal(0, rel * np.abs(Y) + ab, Y.shape)
    Th = VBinv @ Y
    a0, b0 = C0
    return np.array([Th[1] + 2 * Th[3] * a0 + Th[5] * b0, Th[2] + 2 * Th[4] * b0 + Th[5] * a0])

def tf(i, a):
    J = np.array([dA_t[0, i], dA_t[1, i], 0.]) if a else np.array([dR_t[0, i], dR_t[1, i], R0[i]])
    return np.outer(J, J) / (var_afb[i] if a else var_mass[i])
def hf(i, a, Jm, Ja):
    J = np.array([Ja[0, i], Ja[1, i], 0.]) if a else np.array([Jm[0, i], Jm[1, i], R0[i]])
    return np.outer(J, J) / (var_afb[i] if a else var_mass[i])

def run(agent, seed, rel, ab, budget=24, nsd=3):
    rng = np.random.default_rng(seed)
    Jm, Ja = learn_J(xs, m_mass, rel, ab, rng), learn_J(afb, m_afb, rel, ab, rng)
    Ft, Fh, ch = PRIOR.copy(), PRIOR.copy(), []
    for k in list(rng.choice(NCELL, nsd, replace=False)):
        a = k >= N_MASS; i = k - N_MASS if a else k
        Ft, Fh = Ft + tf(i, a), Fh + hf(i, a, Jm, Ja); ch.append(k)
    for _ in range(budget):
        if agent == "random":
            j = int(rng.integers(NCELL))
        else:
            best, j = -np.inf, 0
            for k in range(NCELL):
                a = k >= N_MASS; i = k - N_MASS if a else k
                val = -np.trace(np.linalg.inv(Fh + hf(i, a, Jm, Ja))[:2, :2])
                if val > best: best, j = val, k
        a = j >= N_MASS; i = j - N_MASS if a else j
        Ft, Fh = Ft + tf(i, a), Fh + hf(i, a, Jm, Ja); ch.append(j)
    return np.sqrt(np.diag(np.linalg.inv(Ft)))[:2], sum(1 for k in ch if k >= N_MASS)

SEEDS = range(6000, 6030)
print(f"Rung 3: four-fermion pair {PARAMS}, truth {C0.tolist()}. 30 seeds, budget 27, "
      f"A_FB=6/46 of pool.\nEvaluated on TRUE Fisher; 'exact' = oracle gradients, "
      f"else learned surrogate at the stated noise.\n")
print(f"{'agent':>10} {'surr_noise':>11} | {'sigma(c_lq1)':>12} {'sigma(c_lq3)':>12} {'#AFB':>5}")
print("-" * 60)
for label, rel, ab in [("exact", 0.0, 0.0), ("learned 2%", 0.02, 1e-3), ("learned 5%", 0.05, 3e-3)]:
    for agent in (["random"] if label == "exact" else []) + ["exploit"]:
        S, NA = [], []
        for s in SEEDS:
            sg, na = run(agent, s, rel, ab); S.append(sg); NA.append(na)
        S = np.array(S)
        tag = "random" if agent == "random" else f"exploit"
        print(f"{tag:>10} {label:>11} | {S[:,0].mean():12.4e} {S[:,1].mean():12.4e} {np.mean(NA):5.1f}")
