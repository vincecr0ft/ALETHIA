"""
Rung C: AL completes a manifold; physics (the residual) expands it.

The correction to the whole thread. Active learning works WITHIN a fixed span
(rung A/B). It cannot pick the DIRECTION of model expansion -- which new operator
to turn on. That direction comes from the structure: the dominant direction of
the fit residual names the next operator (the paper's psi-extension). Division of
labour: AL completes, physics expands.

True function uses 6 hierarchical modes. We compare:
  fixed (K=2) + AL    AL keeps acquiring inside a 2-mode span -> PLATEAUS at the
                      out-of-span floor; no amount of acquisition breaks it.
  staged + AL         AL completes the current span; when complete, the residual's
                      dominant remaining mode is switched on (psi-extension); AL
                      completes the larger span; repeat. -> steps down to the floor.
Plus a control at each expansion: switching on the residual-named mode vs a
RANDOM remaining mode -- to show the expansion direction must come from the
residual, not from chance (and not from AL).

python3 rungC_complete_then_expand.py
"""
import numpy as np

GRID = np.linspace(0.0, 1.0, 300)
SIGMA = 0.03
LAM = 1e-4

def modes(x):
    return np.stack([np.ones_like(x), np.sin(2*np.pi*x), np.cos(2*np.pi*x),
                     np.sin(4*np.pi*x), np.cos(4*np.pi*x), np.sin(6*np.pi*x)], axis=1)
G = modes(GRID)                                   # (300, 6)
C_TRUE = np.array([1.0, -0.8, 0.6, -0.4, 0.25, -0.15])   # hierarchical
Y = G @ C_TRUE                                     # target on grid

def fit_predict(span, X_idx, rng):
    """Fit the coefficients of the modes in `span` from labelled points; return
    grid prediction, posterior Ainv, and the design feature matrix."""
    Phi = G[:, span]
    Po = Phi[X_idx]
    y = Y[X_idx] + rng.normal(0, SIGMA, len(X_idx))
    A = Po.T @ Po / SIGMA**2 + LAM * np.eye(len(span))
    Ainv = np.linalg.inv(A)
    w = Ainv @ (Po.T @ y / SIGMA**2)
    return Phi @ w, Ainv, y - Po @ w           # grid pred, Ainv, labelled residual

def acquire(span, Ainv, avail):
    """Uncertainty sampling within the current span."""
    Pa = G[np.ix_(avail, span)]
    var = np.einsum('ij,jk,ik->i', Pa, Ainv, Pa)
    return avail[int(np.argmax(var))]

def residual_dominant_mode(span, X_idx, resid):
    """psi-extension: among modes NOT in span, the one most aligned with the
    labelled-point residual -- the direction the data says is missing."""
    cand = [k for k in range(G.shape[1]) if k not in span]
    scores = [abs(G[X_idx, k] @ resid) / (np.linalg.norm(G[X_idx, k]) + 1e-9) for k in cand]
    return cand[int(np.argmax(scores))], cand

def run(mode, seed, budget=30, stage_len=5):
    rng = np.random.default_rng(seed)
    span = [0, 1]                                  # start: simple 2-mode manifold
    avail = list(range(len(GRID)))
    chosen = list(rng.choice(avail, 2, replace=False))
    for i in chosen: avail.remove(i)
    curve, spansize = [], []
    for t in range(budget):
        pred, Ainv, resid = fit_predict(span, chosen, rng)
        curve.append(np.sqrt(np.mean((pred - Y) ** 2)))
        spansize.append(len(span))
        nxt = acquire(span, Ainv, avail)           # acquire with the matching fit
        chosen.append(nxt); avail.remove(nxt)
        # expansion step (staged modes only), when the current span is "complete"
        if mode != "fixed" and (t + 1) % stage_len == 0 and len(span) < G.shape[1]:
            dom, cand = residual_dominant_mode(span, chosen[:-1], resid)
            if mode == "staged":
                span = span + [dom]                # physics: residual-named mode
            elif mode == "staged_random":
                span = span + [int(rng.choice(cand))]   # control: random new mode
    return np.array(curve), np.array(spansize)

SEEDS = range(300, 330)
curves = {m: np.mean([run(m, s)[0] for s in SEEDS], axis=0) for m in
          ["fixed", "staged", "staged_random"]}
spansz = run("staged", 300)[1]

checkpts = [4, 9, 14, 19, 24, 29]
print("Total RMSE vs #labels (true function has 6 hierarchical modes, 30 seeds).")
print(f"{'#labels':>8} {'span K':>7} | {'fixed K=2+AL':>13} {'staged+AL':>10} {'staged(rand exp)':>16}")
print("-" * 62)
for c in checkpts:
    print(f"{c+1:>8} {spansz[c]:>7} | {curves['fixed'][c]:13.4f} {curves['staged'][c]:10.4f} "
          f"{curves['staged_random'][c]:16.4f}")
print("\nfixed K=2 + AL plateaus: acquisition cannot capture out-of-span structure.")
print("staged + AL steps down at each psi-extension (residual names the new mode).")
print("staged with RANDOM expansion is worse: the direction must come from the")
print("residual, not chance -- AL completes, physics expands.")
