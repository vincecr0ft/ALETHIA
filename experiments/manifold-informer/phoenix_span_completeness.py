r"""Production span-completeness controller: AL completes, physics expands.

Generalises ``task4_span_completeness_loop.py`` from a one-shot, fixed-queue demo
into the end-to-end loop:

  * ACQUISITION drives working-point selection (curvature score over a live pool),
    with a ``random`` control arm.                                   [AL completes]
  * The residual-SVD fingerprint MONITORS span completeness and, on persistence,
    TRIGGERS a psi-extension (append the dominant residual direction).
                                                                     [the monitor + trigger]
  * Completing a manifold UNLOCKS the next SMEFT power-counting operator tier
    (four-fermion 𝒪(ŝ/Λ²) first, then vertex 𝒪(v²/Λ²)) -- "turn on more operators".
                                                                     [physics expands]
  * Every cycle, extension, and tier-unlock is an OpenTelemetry/Arize-Phoenix span.
                                                                     [observability]

Division of labour (the resolved frame): active learning completes a manifold;
the residual structure, ordered by SMEFT power counting, expands it. AL is never
asked to discover the expansion direction.

Usage:
    PYTHONPATH=/home/vince/ALETHIA python3 phoenix_span_completeness.py \
        [--strategy curvature|random|both] [--seed 7] [--max-cycles 40] \
        [--no-phoenix] [--out output_phoenix_loop]

Output: <out>/summary.json, <out>/spectra.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_events import sample_events, event_log_likelihood_ratio
from modules.surrogate.features import N_WC, WC_NAMES   # WC_NAMES = (cHq3, cHq1, clq3, clq1)


# ---------------------------------------------------------------------------
# Phoenix tracer (observability; no-op when PHOENIX_TRACING=0)
# ---------------------------------------------------------------------------
_PROJECT = "alethia-span-completeness"


class _NoopSpan:
    def set_attribute(self, *a, **k): pass
    def add_event(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _NoopTracer:
    def start_as_current_span(self, *a, **k): return _NoopSpan()


def tracer():
    """Arize-Phoenix/OTel tracer when available + enabled; a self-contained
    no-op otherwise (so the loop runs with no tracing dependency)."""
    if os.environ.get("PHOENIX_TRACING", "1").lower() in {"0", "off", "false"}:
        return _NoopTracer()
    try:
        if not hasattr(tracer, "_provider"):
            from phoenix.otel import register
            os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
            tracer._provider = register(project_name=_PROJECT, auto_instrument=False,
                                        protocol="http/protobuf", batch=False, verbose=False)
        return tracer._provider.get_tracer(f"{_PROJECT}.runtime")
    except Exception:                                # phoenix/otel/collector absent -> no-op
        return _NoopTracer()


# ---------------------------------------------------------------------------
# SMEFT power-counting tiers (the expansion schedule = the physics prior)
# WC_NAMES = (cHq3[0], cHq1[1], clq3[2], clq1[3])
# ---------------------------------------------------------------------------
TIERS = [
    {"name": "four_fermion_O(s/L2)", "dims": [2, 3]},   # clq3, clq1  -- leading
    {"name": "vertex_O(v2/L2)",      "dims": [0, 1]},   # cHq3, cHq1  -- subleading
]


@dataclass
class Config:
    n_probe: int = 4000
    sigma_floor: float = 1e-3
    ratio_threshold: float = 10.0
    persistence_n: int = 3
    completion_floor: float = 0.30   # residual sigma_1 below this => manifold complete
    silence_window: int = 3       # cycles with no extension that declare a tier complete
    max_cycles: int = 40
    max_psi_dim: int = 14         # cap on psi-extensions (safety)
    seed: int = 7
    strategy: str = "curvature"   # curvature | random
    out: str = "output_phoenix_loop"


# ---------------------------------------------------------------------------
# Encoder, residual, monitor, extension, acquisition  (core from task4)
# ---------------------------------------------------------------------------
def psi_mass_only(events: np.ndarray) -> np.ndarray:
    log_m = events[:, 0]
    return np.stack([np.ones_like(log_m), log_m, log_m**2, log_m**3, log_m**4], axis=1)


def projector_residual(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    A = psi.T @ psi + 1e-6 * np.eye(psi.shape[1])
    w = np.linalg.solve(A, psi.T @ target)
    return target - psi @ w


@dataclass
class SpanState:
    psi: np.ndarray
    history: list = field(default_factory=list)   # cached log_w of acquired working points
    sigma1: list = field(default_factory=list)    # post-extension residual sigma_1 per cycle


def residual_stats(state: SpanState, cfg: Config) -> dict:
    """SVD of the residual operator R = [resid(psi, log_w) for each acquired WP],
    recomputed against the CURRENT psi (so it is correct after any extension)."""
    if not state.history:
        return {"sigma1": 0.0, "ratio": float("inf"), "fired": False, "U": None}
    R = np.stack([projector_residual(state.psi, lw) for lw in state.history], axis=1)
    if R.shape[1] >= 2:
        U, S, _ = np.linalg.svd(R, full_matrices=False)
        s1, ratio = float(S[0]), float(S[0] / max(S[1], 1e-12))
    else:
        U, s1, ratio = None, float(np.linalg.norm(R[:, 0])), float("inf")
    fired = (s1 > cfg.sigma_floor) and (ratio > cfg.ratio_threshold)
    return {"sigma1": s1, "ratio": ratio, "fired": fired, "U": U}


def extend_psi(state: SpanState, U: np.ndarray):
    """Append the dominant residual direction. History is kept; the residual is
    recomputed against the enlarged psi on the next residual_stats call."""
    state.psi = np.concatenate([state.psi, U[:, 0].reshape(-1, 1)], axis=1)


def curvature_score(psi: np.ndarray, U_lead, log_w: np.ndarray) -> float:
    """Out-of-span residual energy a cached candidate would add, aligned with the
    current leading residual direction. No oracle call (log_w is cached)."""
    r = projector_residual(psi, log_w)
    norm = float(np.linalg.norm(r))
    if U_lead is None:
        return norm
    return norm * float(np.linalg.norm(U_lead.T @ r))


# ---------------------------------------------------------------------------
# Candidate working-point pool, per power-counting tier
# ---------------------------------------------------------------------------
def tier_pool(dims, rng, mags=(0.3, 0.5, 0.7)) -> list:
    """Working points exciting the given Wilson dims: singles + a few mixes."""
    pool = []
    for d in dims:
        for m in mags:
            c = np.zeros(N_WC); c[d] = m; pool.append(c)
    if len(dims) >= 2:
        for m in mags:
            c = np.zeros(N_WC); c[dims[0]] = m; c[dims[1]] = m * 0.7; pool.append(c)
    rng.shuffle(pool)
    return pool


# ---------------------------------------------------------------------------
# The production loop
# ---------------------------------------------------------------------------
def run_loop(cfg: Config, oracle, probe_events, strategy: str, lw_cache: dict | None = None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    tr = tracer()
    state = SpanState(psi=psi_mass_only(probe_events))
    lw_cache = {} if lw_cache is None else lw_cache   # shared across strategies

    def get_lw(c):                                        # cached oracle log-ratio (18s/call @4000)
        key = c.tobytes()
        if key not in lw_cache:
            lw_cache[key] = event_log_likelihood_ratio(oracle, c, probe_events)
        return lw_cache[key]

    pool = tier_pool(TIERS[0]["dims"], rng)
    unlocked = [TIERS[0]["name"]]
    locked = list(TIERS[1:])
    actions, extensions, unlocks = [], [], []
    cycles_since_extension = 0

    print(f"\n## strategy={strategy}  start D_psi={state.psi.shape[1]}  tier={TIERS[0]['name']}")
    with tr.start_as_current_span(f"span_completeness/{strategy}") as run_span:
        run_span.set_attribute("strategy", strategy)
        for k in range(cfg.max_cycles):
            if not pool:
                if not locked:
                    break
                tier = locked.pop(0)                      # exhausted -> turn on next operators
                pool = tier_pool(tier["dims"], rng)
                unlocked.append(tier["name"]); unlocks.append({"cycle": k, "tier": tier["name"]})
            # ----- acquisition: AL completes (pick the WP exposing most out-of-span energy) -----
            pre = residual_stats(state, cfg)
            if strategy == "curvature":
                U_lead = pre["U"][:, :1] if pre["U"] is not None else None
                scores = [curvature_score(state.psi, U_lead, get_lw(c)) for c in pool]
                j = int(np.argmax(scores))
            else:
                j = int(rng.integers(len(pool)))
            c = pool.pop(j)
            state.history.append(get_lw(c))

            with tr.start_as_current_span("cycle") as span:
                upd = residual_stats(state, cfg)
                sigma1_pre = upd["sigma1"]
                span.set_attribute("cycle", k); span.set_attribute("sigma1_pre", sigma1_pre)
                span.set_attribute("ratio", upd["ratio"] if np.isfinite(upd["ratio"]) else -1.0)
                span.set_attribute("fired", bool(upd["fired"]))
                # ----- physics expands: extend on the dominant residual until completed -----
                n_ext = 0
                while (upd["sigma1"] > cfg.completion_floor and upd["U"] is not None
                       and state.psi.shape[1] < cfg.max_psi_dim):
                    extend_psi(state, upd["U"])
                    extensions.append({"cycle": k, "new_D_psi": int(state.psi.shape[1])})
                    span.add_event("psi_extend", {"new_D_psi": int(state.psi.shape[1])})
                    n_ext += 1
                    upd = residual_stats(state, cfg)
                decision = f"extend x{n_ext}" if n_ext else ("watch" if not upd["fired"] else "fired")
                span.set_attribute("D_psi", int(state.psi.shape[1]))
                span.set_attribute("sigma1_post", upd["sigma1"])
                span.set_attribute("decision", decision)

            cycles_since_extension = 0 if n_ext else cycles_since_extension + 1
            # tier complete: acquisition no longer reveals structure -> turn on next operators
            if cycles_since_extension >= cfg.silence_window and locked:
                tier = locked.pop(0)
                pool += tier_pool(tier["dims"], rng)
                unlocked.append(tier["name"]); unlocks.append({"cycle": k, "tier": tier["name"]})
                decision += "+unlock:" + tier["name"]; cycles_since_extension = 0

            state.sigma1.append(upd["sigma1"])
            actions.append({"cycle": k, "c": c.tolist(), "sigma1_pre": sigma1_pre,
                            "sigma1_post": upd["sigma1"], "n_extensions": n_ext,
                            "D_psi": int(state.psi.shape[1]), "decision": decision})
            print(f"  cyc {k:2d}: σ1 {sigma1_pre:.3f}->{upd['sigma1']:.3f} "
                  f"D_psi={state.psi.shape[1]} -> {decision}")
            # terminate once everything is on and the residual stays completed
            if (not locked and upd["sigma1"] <= cfg.completion_floor
                    and cycles_since_extension >= cfg.silence_window):
                break

    return {"strategy": strategy, "actions": actions, "extensions": extensions,
            "unlocks": unlocks, "final_D_psi": int(state.psi.shape[1]),
            "tiers_unlocked": unlocked, "sigma1_trace": state.sigma1,
            "final_sigma1": state.sigma1[-1] if state.sigma1 else None,
            "n_cycles": len(actions)}


def compute_gates(res: dict, cfg: Config, ctrl: dict | None) -> dict:
    g = {}
    g["at_least_one_extension"] = len(res["extensions"]) > 0
    g["all_tiers_unlocked"] = len(res["tiers_unlocked"]) == len(TIERS)
    g["manifold_completed"] = (res["final_sigma1"] is not None
                               and res["final_sigma1"] <= cfg.completion_floor)
    def first_complete(r):                            # first cycle with sigma1 <= floor
        for i, s in enumerate(r["sigma1_trace"]):
            if s <= cfg.completion_floor:
                return i
        return len(r["sigma1_trace"])
    if ctrl is not None:                              # AL completes at least as fast as random
        g["curvature_completes_no_later_than_random"] = first_complete(res) <= first_complete(ctrl)
    return g


def make_plot(results: list, cfg: Config, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = {"curvature": "C3", "random": "C0"}
    for res in results:
        s1 = np.maximum(res["sigma1_trace"], 1e-6)
        ax.semilogy(range(len(s1)), s1, "-o", c=colors.get(res["strategy"], "C2"),
                    ms=5, label=f"{res['strategy']} (D_psi {res['final_D_psi']})")
        for e in res["extensions"]:
            ax.axvline(e["cycle"] + 0.5, ls=":", c=colors.get(res["strategy"], "C2"), alpha=0.5)
        for u in res["unlocks"]:
            ax.axvline(u["cycle"] + 0.5, ls="--", c="grey", alpha=0.4)
    ax.axhline(cfg.completion_floor, ls="--", c="k", alpha=0.4, label="completion floor τ")
    ax.set_xlabel("cycle"); ax.set_ylabel(r"$\sigma_1$ of residual operator")
    ax.set_title("Production span-completeness loop\n"
                 "AL completes (acquisition) · residual triggers ψ-extension (:) · "
                 "tier unlock (--)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(out / "spectra.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="both", choices=["curvature", "random", "both"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-probe", type=int, default=4000)
    ap.add_argument("--max-cycles", type=int, default=40)
    ap.add_argument("--no-phoenix", action="store_true")
    ap.add_argument("--out", default="output_phoenix_loop")
    args = ap.parse_args()
    if args.no_phoenix:
        os.environ["PHOENIX_TRACING"] = "0"

    cfg = Config(seed=args.seed, n_probe=args.n_probe, max_cycles=args.max_cycles, out=args.out)
    out = HERE / cfg.out; out.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    print("# Production span-completeness controller")
    oracle = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    probe_events = sample_events(oracle, np.zeros(N_WC), cfg.n_probe, seed=cfg.seed)

    strategies = ["curvature", "random"] if args.strategy == "both" else [args.strategy]
    shared_lw: dict = {}            # one oracle call per unique working point, reused across arms
    results = [run_loop(cfg, oracle, probe_events, s, shared_lw) for s in strategies]
    by = {r["strategy"]: r for r in results}
    ctrl = by.get("random")
    gates = compute_gates(by.get("curvature", results[0]), cfg, ctrl)

    print("\n## Gates")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    make_plot(results, cfg, out)
    summary = {"config": asdict(cfg), "tiers": TIERS, "results": results,
               "gates": gates, "all_pass": all(gates.values()),
               "wall_seconds": time.perf_counter() - t0}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n# wrote {out}/{{summary.json, spectra.png}}")
    print(f"# Overall: {'PASS' if summary['all_pass'] else 'FAIL'} "
          f"(wall={summary['wall_seconds']:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
