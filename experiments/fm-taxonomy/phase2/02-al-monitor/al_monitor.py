"""Active-learning loop with a monitor at the centre — the mathematics of the learner.

Phase-2, axis IV (the learner) of the ALETHIA FM taxonomy. This is a deeper,
monitor-first build of the branch-04 demo
(experiments/fm-taxonomy/04-active-learning/demo.py): instead of reporting only
terminal z-scores, a *monitor* computes and logs the learner's full mathematical
state every round, for every acquisition function and every regime, and dumps it
to CSV. The point the run makes visible: posterior contraction and estimator
error are the *same* signal in a well-conditioned embedding and *diverge* on a
collinear ridge, and the monitor detects the regime live as the redundancy
diagnostic ``top-decile mean |cos_Ainv|`` crosses from spread (~0.5) to collinear
(~0.85+).

The learner is the closed-form Bayesian linear-Gaussian head ALETHIA's IntentionFM
exposes:

    y = psi(x)^T w + eps,   eps ~ N(0, sigma_y^2),   prior w ~ N(0, tau^2 I)
    A = sigma_y^-2 sum_i psi(x_i) psi(x_i)^T + tau^-2 I      (posterior precision)
    Cov(w) = sigma_y^2 A^{-1}                                (posterior covariance)

Every acquisition score and every monitored quantity is a closed-form rank-one
(Sherman-Morrison) update of A^{-1} — the O(D^2)/cycle machinery the frozen head
exposes, with no retraining and no MC sampling.

Acquisition functions mirrored faithfully from the real ALETHIA code:
  - leverage   = BALD / D-optimal     0.5 log(1 + psi^T A^{-1} psi)
                 (al-phoenix-studies/_common.py:ig_per_candidate)
  - param_a    = A-optimal on one c-direction
                 (modules/surrogate/intention/acquisition.py:param_epig_a_acquire)
  - param_d    = D-optimal on the resolved c-subspace
                 (modules/surrogate/intention/acquisition.py:param_epig_d_acquire)
  - epig       = predictive EPIG over a target set
                 (modules/surrogate/intention/acquisition.py:epig_acquire_m)
  - random     = uniform pool draw (baseline)

The contraction / MLE-error / cos_Ainv monitors mirror
  - al-phoenix-studies/_common.py:sigma_ctilde     (contraction)
  - al-phoenix-studies/04-active-learning/demo.py:cos_ainv_top_decile (redundancy)

Run:  python3 al_monitor.py
      python3 al_monitor.py --seeds 60 --rounds 40 --csv out.csv

Light deps: numpy required; matplotlib optional (guarded — CSV is the primary
artifact, plots are a bonus if matplotlib imports).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


# ===========================================================================
# The learner: closed-form Bayesian linear-Gaussian head (mirrors A_inv_and_w).
# ===========================================================================


class LinearGaussianHead:
    """Posterior over w for y = psi^T w + eps. Mirrors IntentionFM.A_inv_and_w."""

    def __init__(self, psi_fn, sigma_y, tau):
        self.psi = psi_fn                    # x (n,) -> Psi (n, D)
        self.sigma_y = float(sigma_y)
        self.tau = float(tau)

    def A_inv_and_w(self, X_ctx, Y_ctx):
        """Return (A_inv, w_map, A). A = sy^-2 Psi^T Psi + tau^-2 I."""
        Psi = self.psi(np.asarray(X_ctx, dtype=float))          # (K, D)
        D = Psi.shape[1]
        A = (Psi.T @ Psi) / self.sigma_y**2 + np.eye(D) / self.tau**2
        A_inv = np.linalg.inv(A)
        b = (Psi.T @ np.asarray(Y_ctx, dtype=float)) / self.sigma_y**2
        w_map = A_inv @ b
        return A_inv, w_map, A


# ===========================================================================
# Acquisition scores. All closed form; one rank-one update each.
# Mirrored from the cited ALETHIA modules (see module docstring).
# ===========================================================================


def score_leverage(head, A_inv, X_pool):
    """BALD / D-optimal: 0.5 log(1 + psi^T A^{-1} psi).

    Mirror of al-phoenix-studies/_common.py:ig_per_candidate.
    """
    Psi = head.psi(X_pool)
    lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
    return 0.5 * np.log1p(np.maximum(lev, 0.0))


def score_param_a(head, A_inv, X_pool, p_a):
    """A-optimal on direction p_a: -0.5 log(1 - sy^2 u_a^2 / ((1+lev) Sigma_aa)).

    Mirror of acquisition.py:param_epig_a_acquire's per-candidate score.
    """
    sy2 = head.sigma_y**2
    Psi = head.psi(X_pool)
    AP = Psi @ A_inv
    lev = np.einsum("pd,pd->p", AP, Psi)
    u_a = AP @ p_a
    Sigma_aa = sy2 * float(p_a @ A_inv @ p_a)
    arg = 1.0 - sy2 * (u_a**2) / ((1.0 + lev) * max(Sigma_aa, 1e-30))
    return -0.5 * np.log(np.maximum(arg, 1e-12))


def score_param_d(head, A_inv, X_pool, P_sub):
    """D-optimal on a resolved subspace P_sub (rows = directions).

    ΔH_D(p) = -0.5 log(1 - sy^2 (u^T Sigma^{-1} u)/(1+lev)),  u = P A^{-1} psi.
    Mirror of acquisition.py:param_epig_d_acquire's per-candidate score.
    """
    sy2 = head.sigma_y**2
    Psi = head.psi(X_pool)
    Sigma = sy2 * (P_sub @ A_inv @ P_sub.T)                     # (r, r)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(Sigma.shape[0]))
    AP = Psi @ A_inv                                            # (n_P, D)
    U = AP @ P_sub.T                                            # (n_P, r)
    lev = np.einsum("pd,pd->p", AP, Psi)
    # per-row u^T Sigma^{-1} u via a single triangular solve
    X = np.linalg.solve(L, U.T)                                 # (r, n_P)
    quad = np.einsum("ri,ri->i", X, X)
    arg = 1.0 - sy2 * quad / (1.0 + lev)
    return -0.5 * np.log(np.maximum(arg, 1e-12))


def score_epig(head, A_inv, X_pool, X_target):
    """Predictive EPIG over a target set.

    Mirror of acquisition.py:epig_acquire_m's per-candidate score.
    """
    Psi_P = head.psi(X_pool)
    Psi_T = head.psi(X_target)
    lev_T = np.einsum("id,de,ie->i", Psi_T, A_inv, Psi_T)
    lev_P = np.einsum("pd,de,pe->p", Psi_P, A_inv, Psi_P)
    K_TP = Psi_T @ A_inv @ Psi_P.T                              # (n_T, n_P)
    var_red = K_TP**2 / (lev_P[None, :] + 1.0)
    denom = np.maximum(lev_T[:, None] - var_red, 1e-12)
    log_ratio = np.log(np.maximum(lev_T[:, None], 1e-12) / denom)
    return 0.5 * log_ratio.mean(axis=0)


# ===========================================================================
# Monitored diagnostics.
# ===========================================================================


def cos_ainv_top_decile(head, A_inv, X_pool):
    """Mean |cos| of top-decile-leverage candidates in the A^{-1} metric.

    The Stage-B redundancy diagnostic (REPORT.md §Stage B): ~1.0 means the
    high-IG subspace is 1-D. Mirror of demo.py:cos_ainv_top_decile.
    """
    Psi = head.psi(X_pool)
    lev = np.einsum("pd,de,pe->p", Psi, A_inv, Psi)
    k = max(2, len(lev) // 10)
    top = np.argsort(lev)[::-1][:k]
    G = Psi[top] @ A_inv @ Psi[top].T                          # Gram in A^{-1} metric
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = np.abs(G / np.outer(d, d))
    iu = np.triu_indices_from(C, k=1)
    return float(C[iu].mean())


def cv_ig(head, A_inv, X_pool):
    """CV(IG) over the pool — the AL_separation published gate (REPORT.md)."""
    ig = score_leverage(head, A_inv, X_pool)
    m = ig.mean()
    return float(ig.std() / m) if m > 0 else 0.0


# ===========================================================================
# Feature maps.
# ===========================================================================


def make_rff(D, freqs, phases):
    """Random Fourier features: well-conditioned, target identifiable."""
    def psi(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.sqrt(2.0 / D) * np.cos(np.outer(x, freqs) + phases)
    return psi


def make_collinear(D, freqs, phases, leak):
    """Near-rank-deficient map: all but a tiny `leak` of the variance lives in a
    single dominant feature, so high-leverage pool points become collinear in
    the A^{-1} metric — the ALETHIA cos_Ainv -> 1 / weak-identifiability regime.
    """
    base = make_rff(D, freqs, phases)

    def psi(x):
        P = base(x)
        dom = P[:, :1]                      # one dominant direction
        rest = P[:, 1:] * leak              # everything else suppressed
        return np.concatenate([dom, rest], axis=1)
    return psi


# ===========================================================================
# Setup of one regime.
# ===========================================================================


def setup(rng, D, x_lo, x_hi, n_pool, n_target, sigma_y, tau, leak):
    """Build head, truth, seed context, pool, target set, and target direction.

    leak == 0 gives the well-conditioned RFF map; leak > 0 (small) gives the
    collinear ridge. The seed context is a thin draw in the low-x region (like
    ALETHIA's K=12 seed in [0.5, 1.0]); the target/eval band sits at high x.
    """
    freqs = rng.uniform(0.5, 6.0, size=D)
    phases = rng.uniform(0, 2 * np.pi, size=D)
    psi = make_rff(D, freqs, phases) if leak == 0.0 else make_collinear(
        D, freqs, phases, leak)
    head = LinearGaussianHead(psi, sigma_y, tau)
    w_true = rng.normal(0, tau, size=D)
    X_pool = rng.uniform(x_lo, x_hi, size=n_pool)
    X_target = rng.uniform(x_hi - 0.25 * (x_hi - x_lo), x_hi, size=n_target)
    eval_x = np.linspace(x_hi - 0.25 * (x_hi - x_lo), x_hi, 60)
    # targeted parameter direction: feature loading at the high-x eval edge
    p_a = psi(np.array([x_hi]))[0]
    p_a = p_a / np.linalg.norm(p_a)
    # resolved subspace for D-optimal: loadings at two high-x eval points
    P_sub = psi(np.array([x_hi, x_hi - 0.1 * (x_hi - x_lo)]))
    P_sub = P_sub / np.linalg.norm(P_sub, axis=1, keepdims=True)
    K = 6
    X_ctx = rng.uniform(x_lo, x_lo + 0.25 * (x_hi - x_lo), size=K)
    Y_ctx = psi(X_ctx) @ w_true + rng.normal(0, sigma_y, size=K)
    return dict(head=head, w_true=w_true, X_ctx=X_ctx, Y_ctx=Y_ctx,
                X_pool=X_pool, X_target=X_target, eval_x=eval_x, p_a=p_a,
                P_sub=P_sub, psi=psi)


# ===========================================================================
# One AL loop with the monitor firing every round.
# ===========================================================================


def run_loop(s, acq, rng, n_rounds, sigma_y):
    """Greedy round-by-round acquisition. acq in {random,leverage,param_a,param_d,epig}.

    Returns a list of per-round monitor dicts (the learner's mathematical state).
    """
    head, w_true = s["head"], s["w_true"]
    X_ctx, Y_ctx = list(s["X_ctx"]), list(s["Y_ctx"])
    X_pool, X_target = s["X_pool"], s["X_target"]
    eval_x, p_a, P_sub = s["eval_x"], s["p_a"], s["P_sub"]
    avail = np.ones(len(X_pool), dtype=bool)
    rows = []
    prev_logdet = None
    for r in range(n_rounds):
        A_inv, w_map, A = head.A_inv_and_w(np.array(X_ctx), np.array(Y_ctx))
        # --- the monitored mathematical state ---
        # 1. target contraction Var(c_a) = sy^2 p_a^T A^{-1} p_a   (sigma_ctilde)
        var_a = sigma_y**2 * float(p_a @ A_inv @ p_a)
        # 2. estimator (MAP/MLE) error on the target direction
        err_a = abs(float(p_a @ w_map) - float(p_a @ w_true))
        # 3. predictive RMSE on the high-x eval band
        Phi = head.psi(eval_x)
        pred_rmse = float(np.sqrt(np.mean((Phi @ w_map - Phi @ w_true) ** 2)))
        # 4. precision update: log det A (total information accrued)
        sign, logdet = np.linalg.slogdet(A)
        dlogdet = 0.0 if prev_logdet is None else float(logdet - prev_logdet)
        prev_logdet = float(logdet)
        # 5. redundancy diagnostic on the remaining pool
        pool_now = X_pool[avail] if avail.any() else X_pool
        cos_top = cos_ainv_top_decile(head, A_inv, pool_now)
        cvig = cv_ig(head, A_inv, pool_now)
        # 6. the acquisition score actually used (max over available pool)
        if acq == "random":
            score_used = 0.0
        else:
            s_all = _scores(acq, head, A_inv, X_pool, X_target, p_a, P_sub)
            score_used = float(np.max(np.where(avail, s_all, -np.inf)))

        rows.append(dict(
            regime="", acq=acq, seed=-1, round=r, context_size=len(X_ctx),
            var_a=var_a, err_a=err_a, pred_rmse=pred_rmse,
            logdet_A=float(logdet), dlogdet_A=dlogdet,
            cos_ainv_top=cos_top, cv_ig=cvig,
            gap=err_a - var_a, acq_score=score_used))

        if not avail.any():
            break
        # --- pick + fold (rank-one update happens implicitly via refit) ---
        if acq == "random":
            cand = int(rng.choice(np.flatnonzero(avail)))
        else:
            s_all = _scores(acq, head, A_inv, X_pool, X_target, p_a, P_sub)
            cand = int(np.argmax(np.where(avail, s_all, -np.inf)))
        x_new = X_pool[cand]
        y_new = float(head.psi(np.array([x_new]))[0] @ w_true
                      + rng.normal(0, sigma_y))
        X_ctx.append(x_new)
        Y_ctx.append(y_new)
        avail[cand] = False
    return rows


def _scores(acq, head, A_inv, X_pool, X_target, p_a, P_sub):
    if acq == "leverage":
        return score_leverage(head, A_inv, X_pool)
    if acq == "param_a":
        return score_param_a(head, A_inv, X_pool, p_a)
    if acq == "param_d":
        return score_param_d(head, A_inv, X_pool, P_sub)
    if acq == "epig":
        return score_epig(head, A_inv, X_pool, X_target)
    raise ValueError(acq)


# ===========================================================================
# Aggregation over seeds + reporting.
# ===========================================================================


def zscore(target, ref):
    """Signed z of mean(target)-mean(ref); negative = target smaller (better)."""
    d = target.mean() - ref.mean()
    sd = np.sqrt(target.var(ddof=1) / len(target) + ref.var(ddof=1) / len(ref))
    return d / sd if sd > 0 else 0.0


def run_regime(name, regime_kw, acqs, n_seeds, n_rounds, base_seed, all_rows):
    """Run every acquisition over n_seeds; append per-round rows to all_rows.

    Returns (terminal-metric arrays per acq, mean cos_Ainv at seed, mean final cos).
    """
    sigma_y = regime_kw["sigma_y"]
    term = {a: [] for a in acqs}            # terminal (var_a, err_a, pred_rmse)
    cos_seed, cos_final = [], []
    for si in range(n_seeds):
        rng_setup = np.random.default_rng(base_seed + si)
        s = setup(rng_setup, **regime_kw)
        A_inv0, _, _ = s["head"].A_inv_and_w(s["X_ctx"], s["Y_ctx"])
        cos_seed.append(cos_ainv_top_decile(s["head"], A_inv0, s["X_pool"]))
        for a in acqs:
            rng_a = np.random.default_rng(base_seed + si)   # same labels/noise
            rows = run_loop(s, a, rng_a, n_rounds, sigma_y)
            for row in rows:
                row["regime"] = name
                row["seed"] = base_seed + si
                all_rows.append(row)
            last = rows[-1]
            term[a].append((last["var_a"], last["err_a"], last["pred_rmse"]))
            if a == acqs[0]:
                cos_final.append(rows[-1]["cos_ainv_top"])
    term = {a: np.array(v) for a, v in term.items()}
    return term, float(np.mean(cos_seed)), float(np.mean(cos_final))


def report(name, term, cos_seed, acqs):
    print(f"\n=== {name} ===")
    flag = "collinear/redundant" if cos_seed > 0.7 else "spread"
    print(f"top-decile mean |cos_Ainv| at seed context: {cos_seed:.3f}  ({flag})")
    ref = term["random"]
    hdr = (f"{'acq':>10} | {'Var(c_a)':>10} {'err(c_a)':>10} {'pred_RMSE':>10}"
           f" | {'zVar':>6} {'zErr':>6} {'zRMSE':>6}")
    print(hdr)
    print("-" * len(hdr))
    for a in acqs:
        v = term[a]
        vm, em, rm = v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()
        if a == "random":
            zs = "   ref    ref    ref"
        else:
            zs = (f"{zscore(v[:, 0], ref[:, 0]):>6.1f} "
                  f"{zscore(v[:, 1], ref[:, 1]):>6.1f} "
                  f"{zscore(v[:, 2], ref[:, 2]):>6.1f}")
        print(f"{a:>10} | {vm:>10.4g} {em:>10.4g} {rm:>10.4g} | {zs}")


# ===========================================================================
# Optional plotting (guarded).
# ===========================================================================


def maybe_plot(all_rows, regimes, acqs, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                    # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc}); CSV is the artifact.")
        return
    # per-round trajectories averaged over seeds: cos_Ainv, gap, var_a
    def traj(regime, acq, key):
        vals = {}
        for row in all_rows:
            if row["regime"] == regime and row["acq"] == acq:
                vals.setdefault(row["round"], []).append(row[key])
        xs = sorted(vals)
        return np.array(xs), np.array([np.mean(vals[x]) for x in xs])

    nreg = len(regimes)
    fig, axes = plt.subplots(3, nreg, figsize=(5 * nreg, 11),
                             constrained_layout=True, squeeze=False)
    for ci, regime in enumerate(regimes):
        ax = axes[0][ci]
        for a in acqs:
            xs, ys = traj(regime, a, "cos_ainv_top")
            ax.plot(xs, ys, label=a)
        ax.axhline(0.7, ls="--", c="grey", alpha=0.6)
        ax.set_title(f"{regime}\ntop-decile mean |cos_Ainv|")
        ax.set_xlabel("round"); ax.set_ylabel("|cos_Ainv|")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1][ci]
        for a in acqs:
            xs, ys = traj(regime, a, "var_a")
            ax.semilogy(xs, ys, label=a)
        ax.set_title("contraction  Var(c_a) = sy^2 p_a^T A^-1 p_a")
        ax.set_xlabel("round"); ax.set_ylabel("Var(c_a)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[2][ci]
        for a in acqs:
            xs, ys = traj(regime, a, "err_a")
            ax.plot(xs, ys, label=a)
        ax.set_title("estimator error  |p_a^T (w_MAP - w_true)|")
        ax.set_xlabel("round"); ax.set_ylabel("err(c_a)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


# ===========================================================================
# Entry point.
# ===========================================================================


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--csv", default=None,
                   help="per-round monitor log (default: alongside this script)")
    p.add_argument("--png", default=None,
                   help="optional trajectory plot (default: alongside this script)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv or os.path.join(here, "monitor_log.csv")
    png_path = args.png or os.path.join(here, "monitor_trajectories.png")

    acqs = ["random", "leverage", "param_a", "param_d", "epig"]

    # Three regimes spanning the cos_Ainv axis. leak grows -> collinear ridge.
    # D, x-range, pool/target sizes, sigma_y, tau fixed across regimes so only
    # the embedding geometry (leak) changes — the monitor isolates geometry.
    common = dict(D=24, x_lo=0.0, x_hi=6.0, n_pool=200, n_target=40,
                  sigma_y=0.05, tau=1.0)
    regimes = [
        ("E1_wellposed", dict(**common, leak=0.0), 1000),
        ("E_intermediate", dict(**common, leak=0.12), 1500),
        ("E2_collinear", dict(**common, leak=0.02), 2000),
    ]

    all_rows = []
    summary = []
    for name, kw, seed in regimes:
        term, cos_seed, cos_final = run_regime(
            name, kw, acqs, args.seeds, args.rounds, seed, all_rows)
        report(name, term, cos_seed, acqs)
        summary.append((name, cos_seed, cos_final, term))

    # --- write the CSV (the primary artifact) ---
    fields = ["regime", "acq", "seed", "round", "context_size",
              "var_a", "err_a", "pred_rmse", "logdet_A", "dlogdet_A",
              "cos_ainv_top", "cv_ig", "gap", "acq_score"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fields})
    print(f"\n[csv] wrote {len(all_rows)} rows to {csv_path}")

    print("\nReading the monitor:")
    print(" - cos_Ainv at seed: E1 ~0.5 (spread) -> E2 ~0.85+ (collinear ridge).")
    print(" - E1: targeted methods show zVar<0 AND zErr<0 -> contraction and")
    print("   estimator error move together. Textbook AL win.")
    print(" - E2: leverage/epig show zVar near 0 or worse and zErr near 0 ->")
    print("   the contraction-vs-MLE gap. The monitor's `gap` column (err_a -")
    print("   var_a) and the cos_Ainv trajectory are the live detectors.")
    print(" - param_a/param_d (parameter-aware, directional) survive longest on")
    print("   the ridge — the Stage-B redundancy story, per-round.")

    if not args.no_plot:
        maybe_plot(all_rows, [r[0] for r in regimes], acqs, png_path)


if __name__ == "__main__":
    sys.exit(main())
