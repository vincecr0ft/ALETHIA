"""MadGraph fidelity cross-check on the withheld-band target scenario.

Picks K m-values inside the engineered drift band, queries both the
analytic SMEFT oracle (T1) and MadGraph (T2 at LO with nevents=500), and
records per-point disagreement. Emits Phoenix spans under the
`aletheia.drift.fidelity` schema from docs/research/03-drift/eval.md
section 3.

This is the closest practical analogue of the T2 escalation tier from
agent (a)'s tier ladder in docs/research/01-oracle/summary.md section 2.3.
Each MG call is ~30 seconds; the script defaults to 50 cross-check points
(~25 minutes wall time).

Output:
- experiments/full-chain-run/output/mg_crosscheck.npz
- experiments/full-chain-run/output/mg_crosscheck_summary.json
- docs/research/plots/full-chain/mg_crosscheck.png
"""
from __future__ import annotations

import sys, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from phoenix.otel import register

from modules.surrogate.oracle_smeft import AnalyticSMEFTOracle
from modules.surrogate.oracle_madgraph import MadGraphSMEFTOracle

TRACER_PROVIDER = register(project_name="alethia", auto_instrument=False,
                           protocol="http/protobuf")
tracer = TRACER_PROVIDER.get_tracer("alethia.mg_crosscheck")

OUT = HERE / "output"
PLOT_DIR = HERE.parent.parent / "docs" / "research" / "plots" / "full-chain"

# Target scenario in the withheld band (same as run.py).
TARGET_C = np.array([0.0, 0.0, 0.8, 0.0])  # c_lq^(3) = 0.8
M_GRID = np.linspace(0.4, 2.2, 50)         # 50 m-values across the band

# MG configuration: nevents=500 keeps each call near 30s with ~5% MC noise
# on mu in tails.
MG_NEVENTS = 500


def main():
    print(f"MG cross-check: target c = {TARGET_C.tolist()}")
    print(f"               {len(M_GRID)} m-values in [{M_GRID[0]}, {M_GRID[-1]}] TeV")
    print(f"               MG nevents = {MG_NEVENTS} (~30s/call)")
    print(f"               estimated wall time: {len(M_GRID) * 30 / 60:.0f} min")

    analytic = AnalyticSMEFTOracle(pdf="analytic", noise_frac=0.0)
    mg = MadGraphSMEFTOracle(nevents=MG_NEVENTS)

    mu_t1 = np.zeros(len(M_GRID))
    mu_t2 = np.zeros(len(M_GRID))
    cost_t1 = np.zeros(len(M_GRID))
    cost_t2 = np.zeros(len(M_GRID))

    with tracer.start_as_current_span("chain.mg_crosscheck") as root:
        root.set_attribute("aletheia.fidelity.target_c",
                           TARGET_C.tolist())
        root.set_attribute("aletheia.fidelity.n_points", len(M_GRID))
        root.set_attribute("aletheia.fidelity.mg_nevents", MG_NEVENTS)

        for i, m in enumerate(M_GRID):
            # T1: analytic.
            with tracer.start_as_current_span("tool.oracle.query") as sp:
                t0 = time.time()
                mu_t1[i] = analytic.truth(TARGET_C[None, :],
                                          np.array([m]))[0]
                cost_t1[i] = time.time() - t0
                sp.set_attribute("aletheia.oracle.fidelity_tier", "T1")
                sp.set_attribute("aletheia.oracle.n_points", 1)
                sp.set_attribute("aletheia.oracle.cost_seconds",
                                 float(cost_t1[i]))
                sp.set_attribute("aletheia.oracle.m_tev", float(m))
                sp.set_attribute("aletheia.oracle.mu", float(mu_t1[i]))

            # T2: MadGraph.
            with tracer.start_as_current_span("tool.oracle.query") as sp:
                t0 = time.time()
                mu_t2[i] = mg.truth(TARGET_C[None, :], np.array([m]))[0]
                cost_t2[i] = time.time() - t0
                sp.set_attribute("aletheia.oracle.fidelity_tier", "T2")
                sp.set_attribute("aletheia.oracle.n_points", 1)
                sp.set_attribute("aletheia.oracle.cost_seconds",
                                 float(cost_t2[i]))
                sp.set_attribute("aletheia.oracle.m_tev", float(m))
                sp.set_attribute("aletheia.oracle.mu", float(mu_t2[i]))
                sp.set_attribute("aletheia.oracle.mc_nevents",
                                 MG_NEVENTS)

            # Compare.
            rel_err = (mu_t2[i] - mu_t1[i]) / max(abs(mu_t1[i]), 1e-12)
            # MG MC noise floor at nevents=500 is ~5%; we flag |z| > 3.
            mc_sigma_t2 = 0.05 * abs(mu_t2[i])
            z_score = (mu_t2[i] - mu_t1[i]) / max(mc_sigma_t2, 1e-12)
            with tracer.start_as_current_span(
                    "tool.drift.fidelity") as df:
                df.set_attribute("aletheia.drift.fid.t_low", "T1")
                df.set_attribute("aletheia.drift.fid.t_high", "T2")
                df.set_attribute("aletheia.drift.fid.m_tev", float(m))
                df.set_attribute("aletheia.drift.fid.mu_t1",
                                 float(mu_t1[i]))
                df.set_attribute("aletheia.drift.fid.mu_t2",
                                 float(mu_t2[i]))
                df.set_attribute("aletheia.drift.fid.delta_mu",
                                 float(mu_t2[i] - mu_t1[i]))
                df.set_attribute("aletheia.drift.fid.rel_err",
                                 float(rel_err))
                df.set_attribute("aletheia.drift.fid.z_score",
                                 float(z_score))
                df.set_attribute("aletheia.drift.fid.fired",
                                 bool(abs(z_score) > 3))

            print(f"  [{i+1:2d}/{len(M_GRID)}]  m={m:5.2f} TeV  "
                  f"T1 mu={mu_t1[i]:7.2f}  T2 mu={mu_t2[i]:7.2f}  "
                  f"rel={rel_err*100:+6.2f}%  z={z_score:+5.2f}  "
                  f"t_T2={cost_t2[i]:5.1f}s")

    # Summary stats.
    rel_errs = (mu_t2 - mu_t1) / np.maximum(np.abs(mu_t1), 1e-12)
    z_scores = (mu_t2 - mu_t1) / np.maximum(0.05 * np.abs(mu_t2), 1e-12)
    n_fired = int((np.abs(z_scores) > 3).sum())

    np.savez(OUT / "mg_crosscheck.npz",
             M=M_GRID, mu_t1=mu_t1, mu_t2=mu_t2,
             cost_t1=cost_t1, cost_t2=cost_t2,
             rel_errs=rel_errs, z_scores=z_scores)

    summary = dict(
        target_c=TARGET_C.tolist(),
        n_points=len(M_GRID),
        m_range_tev=[float(M_GRID[0]), float(M_GRID[-1])],
        mg_nevents=MG_NEVENTS,
        total_t1_seconds=float(cost_t1.sum()),
        total_t2_seconds=float(cost_t2.sum()),
        mean_t2_per_call=float(cost_t2.mean()),
        median_rel_err=float(np.median(np.abs(rel_errs))),
        p95_rel_err=float(np.percentile(np.abs(rel_errs), 95)),
        max_rel_err=float(np.max(np.abs(rel_errs))),
        n_fidelity_fired=n_fired,
        mu_t1_range=[float(mu_t1.min()), float(mu_t1.max())],
        mu_t2_range=[float(mu_t2.min()), float(mu_t2.max())],
    )
    with open(OUT / "mg_crosscheck_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))

    # Plot.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True,
                             sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(M_GRID, mu_t1, "C0-o", lw=1.4, ms=5,
            label=f"T1 analytic ({cost_t1.sum():.1f} s total)")
    ax.errorbar(M_GRID, mu_t2, yerr=0.05 * np.abs(mu_t2), fmt="C3-s",
                lw=1.4, ms=5, capsize=2,
                label=f"T2 MadGraph nevents={MG_NEVENTS} "
                      f"({cost_t2.sum()/60:.1f} min total)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mu = \sigma_{\mathrm{total}} / \sigma_{\mathrm{SM}}$  (log)")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title(
        f"Fidelity ladder T1 vs T2 at target $c_{{lq}}^{{(3)}}=0.8$\n"
        f"{len(M_GRID)} m-points across the band; per-MG-call cost "
        f"{cost_t2.mean():.1f} s")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.axhline(0, color="black", lw=0.6)
    ax.fill_between(M_GRID, -5, 5, color="gray", alpha=0.15,
                    label=r"$\pm 5\%$ (MG MC at $n_{ev}=500$)")
    ax.plot(M_GRID, rel_errs * 100, "C2-o", lw=1.2, ms=4,
            label="(T2 - T1) / T1  [%]")
    ax.set_xlabel(r"$m_{\ell\ell}$  [TeV]")
    ax.set_ylabel("relative\ndisagreement [%]")
    ax.set_ylim(-15, 15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("MadGraph cross-check on the engineered-drift target",
                 fontsize=12)
    fig.savefig(PLOT_DIR / "mg_crosscheck.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {PLOT_DIR / 'mg_crosscheck.png'}")


if __name__ == "__main__":
    main()
