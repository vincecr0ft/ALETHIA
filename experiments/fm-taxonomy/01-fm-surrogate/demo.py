"""Surrogate vs foundation-style representation: a numerical contrast.

This demo isolates taxonomy axis 1 (amortization-over-what) from writeup.md.

We work with a family of 1-D regression functions
    f_c(x) = sin(w*x + p) + a*x,   c = (w, p, a)
a sinusoid (frequency w, phase p) on a linear trend (slope a). The family
parameter c plays the role of the SMEFT Wilson coefficient in ALETHIA: it
indexes which member of the family we face.

Two modes are compared on the SAME unseen target function f_c*, given only a
handful (k) of labelled points from it:

  (A) Single-task surrogate. A model with no knowledge of the family is fit
      from scratch to the k points. It is amortized over INPUTS x only. With
      k small it has nothing to fall back on.

  (B) Foundation-style representation. A representation phi(x) is PRETRAINED on
      many functions drawn from the family (self-supervised-ish: it never sees
      the target, only sibling functions), so it learns a feature basis that
      spans the family. Adaptation to the new function is a linear readout on
      the k points (the "linear probe" reuse mechanism). It is amortized over
      the FAMILY.

The quantity the taxonomy predicts is the few-shot gap: with k small, (B)
beats (A) because the family structure is reusable; as k grows the surrogate
catches up, since the per-task signal eventually dominates.

Deps: numpy + scikit-learn only.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

RNG = np.random.default_rng(2026)
N_GRID = 400                     # dense grid for measuring true generalisation MSE
X = np.linspace(-3.0, 3.0, N_GRID).reshape(-1, 1)


def sample_params(rng):
    """Draw a family member c = (w, p, a)."""
    w = rng.uniform(0.5, 3.0)    # frequency
    p = rng.uniform(0.0, 2 * np.pi)  # phase
    a = rng.uniform(-0.5, 0.5)   # linear trend slope
    return w, p, a


def f(x, c):
    w, p, a = c
    return np.sin(w * x + p) + a * x


# ---------------------------------------------------------------------------
# (B) Pretrain a foundation-style representation on a FAMILY of functions.
# The shared trunk is an MLP trained to regress f_c(x) from the augmented
# input (x, c): forced to fit hundreds of family members with one body, its
# penultimate layer becomes a feature basis phi(x) that spans the family.
# We then DISCARD the head and freeze phi; adaptation is a linear probe.
# ---------------------------------------------------------------------------
def build_pretraining_set(n_funcs, pts_per_func, rng):
    xs, feats, ys = [], [], []
    for _ in range(n_funcs):
        c = sample_params(rng)
        xf = rng.uniform(-3.0, 3.0, size=(pts_per_func, 1))
        yf = f(xf, c)
        xs.append(xf)
        feats.append(np.repeat(np.array(c)[None, :], pts_per_func, axis=0))
        ys.append(yf)
    Xin = np.hstack([np.vstack(xs), np.vstack(feats)])  # (x, w, p, a)
    Y = np.vstack(ys).ravel()
    return Xin, Y


print("Pretraining foundation-style trunk on a family of functions ...")
Xpre, Ypre = build_pretraining_set(n_funcs=300, pts_per_func=40, rng=RNG)
trunk = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh",
                     max_iter=2000, random_state=0, alpha=1e-4)
trunk.fit(Xpre, Ypre)


def phi(xgrid, c_guess):
    """Frozen representation: hidden activations of the pretrained trunk.

    The trunk was trained on (x, c). At adaptation time the target's c is
    unknown, so we feed a neutral family-prior guess (the family means) as the
    c-slots. The features still carry the learned sinusoidal basis over x; the
    linear probe re-weights them to the specific target. This mimics reusing a
    pretrained encoder whose conditioning is supplied weakly at probe time.
    """
    c_fill = np.repeat(np.array(c_guess)[None, :], len(xgrid), axis=0)
    Z = np.hstack([xgrid, c_fill])
    # forward through hidden layers, stop at last hidden activation
    A = Z
    for W, b in zip(trunk.coefs_[:-1], trunk.intercepts_[:-1]):
        A = np.tanh(A @ W + b)
    return A


C_PRIOR = (1.75, np.pi, 0.0)     # family-mean guess for the c-slots


# ---------------------------------------------------------------------------
# Few-shot evaluation on UNSEEN target functions.
# ---------------------------------------------------------------------------
def eval_modes(k, n_targets=40, rng=None):
    rng = rng or np.random.default_rng(7)
    surr_mse, found_mse = [], []
    for _ in range(n_targets):
        c_star = sample_params(rng)
        xk = rng.uniform(-3.0, 3.0, size=(k, 1))
        yk = f(xk, c_star)
        ytrue = f(X, c_star).ravel()

        # (A) single-task surrogate: small MLP fit from scratch on k points
        surr = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh",
                            max_iter=2000, random_state=1, alpha=1e-3)
        surr.fit(xk, yk.ravel())
        surr_mse.append(np.mean((surr.predict(X) - ytrue) ** 2))

        # (B) foundation: linear probe on frozen pretrained features
        Phi_k = phi(xk, C_PRIOR)
        Phi_g = phi(X, C_PRIOR)
        probe = Ridge(alpha=1e-2)
        probe.fit(Phi_k, yk.ravel())
        found_mse.append(np.mean((probe.predict(Phi_g) - ytrue) ** 2))

    return np.median(surr_mse), np.median(found_mse)


print("\nFew-shot generalisation MSE on unseen family members")
print("(median over 40 target functions; lower is better)\n")
print(f"{'k (labelled pts)':>16} | {'surrogate (A)':>14} | {'foundation (B)':>15} | {'ratio A/B':>10}")
print("-" * 66)
for k in [3, 5, 10, 20, 50]:
    s, fnd = eval_modes(k, rng=np.random.default_rng(100 + k))
    print(f"{k:>16} | {s:>14.4f} | {fnd:>15.4f} | {s / fnd:>10.2f}")

print("\nInterpretation: ratio A/B > 1 means the pretrained representation wins.")
print("The gap is largest at small k (family structure is reusable, the lone")
print("surrogate has nothing to fall back on) and shrinks as k grows.")
