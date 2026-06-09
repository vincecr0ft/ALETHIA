"""
Rung 1b: the EFT space where active learning DOES beat random.

Two coefficients:
  c1  four-fermion  -- constrained by the high-mass spectrum
  c2  vertex        -- DEGENERATE on the mass spectrum, constrainable ONLY by
                       the forward-backward asymmetry A_FB (the angular channel)

Design space = (working point) x (observable in {mass, AFB}). The catch that
makes this an AL problem and not importance sampling: the angular designs are a
MINORITY of the pool (mass bins are numerous, A_FB measurements are few/costly),
so uniform sampling under-covers the only channel that constrains c2.

Target: estimate BOTH coefficients (joint posterior). The bottleneck is c2.
Law under test: c2 is constrainable but only by under-sampled designs, so AL
(D-/A-optimal/EPIG) should route budget to A_FB and beat random; random keeps
drawing the abundant mass designs and leaves c2 under-constrained.

numpy only. python3 rung1b_degeneracy_breaking.py
"""
import numpy as np

C_TRUE = np.array([0.30, 0.30])      # the two coefficients we want to recover
S2 = 0.05 ** 2                        # per-measurement variance
LAM = 1e-3                            # ridge prior precision

# ---- design pool: Jacobian (sensitivity) of each design to (c1, c2) -------
# MASS designs: strong on c1 (tail growth), ~blind to c2 (degenerate).
mwp = np.linspace(0.4, 2.4, 120)                      # 120 mass-bin designs (abundant)
J_mass = np.stack([0.3 * mwp ** 2, 0.02 * np.ones_like(mwp)], axis=1)
# AFB designs: the only channel sensitive to c2; weak on c1. RARE in the pool.
awp = np.linspace(0.6, 1.8, 6)                        # only 6 angular designs
J_afb = np.stack([0.08 * np.ones_like(awp), 0.55 * awp], axis=1)
J_POOL = np.vstack([J_mass, J_afb])                   # 36 x 2
KIND = np.array(['mass'] * len(mwp) + ['afb'] * len(awp))
AFB_FRAC = (KIND == 'afb').mean()

def run(acq, seed, rounds=12, n_seed=3):
    rng = np.random.default_rng(seed)
    avail = list(range(len(J_POOL)))
    chosen = list(rng.choice(avail, n_seed, replace=False))
    for i in chosen:
        avail.remove(i)
    for r in range(rounds):
        J = J_POOL[chosen]
        y = J @ C_TRUE + rng.normal(0, np.sqrt(S2), len(J))
        A = J.T @ J / S2 + LAM * np.eye(2)
        Ainv = np.linalg.inv(A)
        chat = Ainv @ (J.T @ y / S2)
        # --- acquisition over remaining pool ---
        Ja = J_POOL[avail]
        AiJ = Ja @ Ainv
        quad = np.einsum('ij,ij->i', AiJ, Ja)
        denom = 1.0 + quad / S2
        if acq == 'random':
            j = rng.integers(len(avail))
        elif acq == 'dopt':                                  # max logdet gain (leverage/BALD)
            j = int(np.argmax(np.log(denom)))
        elif acq == 'aopt':                                  # min total posterior variance (trace)
            # trace reduction = ||A^-1 v||^2 / (s2 + v^T A^-1 v)
            red = np.einsum('ij,ij->i', AiJ, AiJ) / (S2 + quad)
            j = int(np.argmax(red))
        chosen.append(avail.pop(j))
    # final posterior
    J = J_POOL[chosen]
    A = J.T @ J / S2 + LAM * np.eye(2)
    Ainv = np.linalg.inv(A)
    y = J @ C_TRUE + np.random.default_rng(seed + 1).normal(0, np.sqrt(S2), len(J))
    chat = Ainv @ (J.T @ y / S2)
    n_afb = int(np.sum(KIND[chosen] == 'afb'))
    return dict(err_c1=abs(chat[0] - C_TRUE[0]), err_c2=abs(chat[1] - C_TRUE[1]),
                var_c2=Ainv[1, 1], logdet=np.linalg.slogdet(Ainv)[1], n_afb=n_afb)

ACQS = ['random', 'dopt', 'aopt']
SEEDS = range(3000, 3080)
res = {a: [run(a, s) for s in SEEDS] for a in ACQS}

def col(a, k): return np.array([d[k] for d in res[a]])
ref_c2 = col('random', 'err_c2')

print(f"AFB designs are {100*AFB_FRAC:.0f}% of the pool ({(KIND=='afb').sum()}/{len(KIND)}); "
      f"c2 is constrainable ONLY through them. {len(list(SEEDS))} seeds, 12 rounds, "
      f"budget 15 of {len(J_POOL)}.")
print(f"{'acq':>8} | {'err_c1':>9} {'err_c2':>9} {'var_c2':>9} {'#AFB used':>9} | {'z(err_c2)':>9}")
print("-" * 72)
for a in ACQS:
    e1, e2, v2, na = col(a,'err_c1').mean(), col(a,'err_c2').mean(), col(a,'var_c2').mean(), col(a,'n_afb').mean()
    d = col(a,'err_c2') - ref_c2
    z = 0.0 if a == 'random' else d.mean() / (d.std(ddof=1)/np.sqrt(len(d)) + 1e-30)
    print(f"{a:>8} | {e1:9.4f} {e2:9.4f} {v2:9.2e} {na:9.1f} | {z:9.1f}")
print("\nz(err_c2): paired z vs random on the bottleneck coefficient; negative = AL beats random.")
print("If dopt/aopt draw more #AFB and get z(err_c2)<0, AL wins because it routes")
print("budget to the under-sampled channel that constrains the bottleneck direction.")
