"""Gate 2 of INV-1: SM forward-backward asymmetry sanity plot.

Computes ``A_FB(m_ll) = (3/4) D_had(m_ll) / S_had(m_ll)`` at ``c = 0`` on a
fine m-grid and compares to the textbook asymptotic value

    A_FB^infty = (3/4) (g_L^2 - g_R^2)_q (g_L^2 - g_R^2)_l
                 / [ (g_L^2 + g_R^2)_q (g_L^2 + g_R^2)_l ],

flavour-averaged over the up-type and down-type contributions weighted by the
parton-luminosity ratio at the highest m on the grid. The plot saves to
``docs/research/plots/afb_sm_sanity.png``.

Convention check: the asymptotic A_FB^infty for pp -> ll at high mass is
positive (left-handed coupling dominates and up valence > down valence). A
negative computed curve at high m_ll indicates the cos-theta*_CS sign / quark-
direction assignment in ``_luminosity_asym`` is wrong; gate 2 of INV-1
flags it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from modules.analytic_smeft.constants import (
    LEPTON_CHARGE,
    LEPTON_T3,
    QUARK_CHARGE,
    QUARK_T3,
    SIN2_THETA_W,
)
from modules.analytic_smeft.pdfs import get_pdf
from modules.analytic_smeft.smeft import _dsigma_dm_S_D_hadronic, _normalise_wc

PLOT_PATH = Path("/home/vince/ALETHIA/docs/research/plots/afb_sm_sanity.png")


def _sm_g_lr(t3: float, q: float) -> tuple[float, float]:
    g_l = t3 - q * SIN2_THETA_W
    g_r = -q * SIN2_THETA_W
    return g_l, g_r


def _afb_asymptotic_quark(pid: int) -> float:
    """Z-exchange-only A_FB at infinite m_ll for partonic q qbar -> l+ l-."""
    g_lq, g_rq = _sm_g_lr(QUARK_T3[abs(pid)], QUARK_CHARGE[abs(pid)])
    g_ll, g_rl = _sm_g_lr(LEPTON_T3, LEPTON_CHARGE)
    diff_q = g_lq**2 - g_rq**2
    diff_l = g_ll**2 - g_rl**2
    sum_q = g_lq**2 + g_rq**2
    sum_l = g_ll**2 + g_rl**2
    return 0.75 * diff_q * diff_l / (sum_q * sum_l)


def _luminosity_weighted_asymptote(pdf, m_max_gev: float, sqrt_s: float) -> float:
    """Combine quark-flavour asymptotic A_FB by luminosity weights at m_max.

    Uses the symmetric S luminosity for the denominator weighting and the
    asymmetric D luminosity for the numerator weighting, matching the
    hadronic A_FB definition.
    """
    from modules.analytic_smeft.smeft import _luminosity, _luminosity_asym
    s = sqrt_s**2
    tau = (m_max_gev / sqrt_s) ** 2
    num = 0.0
    den = 0.0
    for pid in (1, 2, 3, 4, 5):
        phi_sym = _luminosity(pdf, pid, tau, m_max_gev)
        phi_asym = _luminosity_asym(pdf, pid, tau, m_max_gev)
        a_inf = _afb_asymptotic_quark(pid)
        # In the high-m limit S_parton and D_parton are dominated by Z
        # exchange; A_FB_parton ~ a_inf. Then
        #   D_had / S_had ~ sum_q phi_asym(q) sum_q^pdg D_parton(q)
        # ~ sum_q phi_asym(q) (g_L^2 - g_R^2)_q (g_L^2 - g_R^2)_l
        # / [ sum_q phi_sym(q) (g_L^2 + g_R^2)_q (g_L^2 + g_R^2)_l ]
        # times 3/4. Easier: weight quark-by-quark contributions to S and D
        # in their high-m partonic limits (a_inf bakes in the 3/4 already).
        sum_q = (
            (QUARK_T3[abs(pid)] - QUARK_CHARGE[abs(pid)] * SIN2_THETA_W) ** 2
            + (QUARK_CHARGE[abs(pid)] * SIN2_THETA_W) ** 2
        )
        sum_l = (
            (LEPTON_T3 - LEPTON_CHARGE * SIN2_THETA_W) ** 2
            + (LEPTON_CHARGE * SIN2_THETA_W) ** 2
        )
        den += phi_sym * sum_q * sum_l
        num += phi_asym * (4.0 / 3.0) * a_inf * sum_q * sum_l
    return num / den if den > 0 else float("nan")


def main() -> None:
    # SM run: c = 0.
    wc = _normalise_wc({})
    lam = 2000.0
    sqrt_s = 13000.0
    s = sqrt_s**2

    # Use the analytic toy PDF: the ratio A_FB is PDF-dependent through the
    # valence-sea split, but the toy keeps the qualitative sign and rough
    # asymptote, and the test does not rely on LHAPDF. The shape is what
    # gate 2 sanity-checks (rising magnitude, correct sign at high m_ll).
    try:
        pdf = get_pdf("auto")
        pdf_name = pdf.name
    except Exception:
        pdf = get_pdf("analytic")
        pdf_name = "analytic"
    print(f"PDF backend: {pdf_name}")

    m_grid_tev = np.linspace(0.3, 2.5, 50)
    m_grid_gev = m_grid_tev * 1000.0

    S_sm, _, _, D_sm, _, _ = _dsigma_dm_S_D_hadronic(m_grid_gev, wc, lam, pdf, s)
    # A_FB(m_ll) = (3/4) D_had / S_had at c = 0.
    afb = 0.75 * D_sm / S_sm

    # Asymptotic value at m_ll = m_max with luminosity weighting.
    a_inf = _luminosity_weighted_asymptote(pdf, m_grid_gev[-1], sqrt_s)
    print(f"computed A_FB(m_ll = 2.5 TeV) = {afb[-1]:.4f}")
    print(f"asymptotic A_FB^infty (luminosity-weighted at 2.5 TeV) = {a_inf:.4f}")
    print(f"sign agreement: {np.sign(afb[-1]) == np.sign(a_inf)}")

    # Parton-level reference, no PDF folding (Z-only): A_FB^infty per flavour.
    a_inf_up = _afb_asymptotic_quark(2)
    a_inf_down = _afb_asymptotic_quark(1)
    print(f"asymptotic A_FB^infty (parton-level, up   q): {a_inf_up:.4f}")
    print(f"asymptotic A_FB^infty (parton-level, down q): {a_inf_down:.4f}")

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(m_grid_tev, afb, lw=2.0, color="C0",
            label=r"computed $A_{FB}(m_{ll}) = \frac{3}{4} D_{had}/S_{had}$ at $c = 0$")
    ax.axhline(a_inf, ls="--", color="C3",
               label=fr"PDF-weighted asymptote at 2.5 TeV: {a_inf:.3f}")
    ax.axhline(a_inf_up, ls=":", color="C2", alpha=0.7,
               label=fr"Z-only partonic $A_{{FB}}^\infty$ (up): {a_inf_up:.3f}")
    ax.axhline(a_inf_down, ls=":", color="C4", alpha=0.7,
               label=fr"Z-only partonic $A_{{FB}}^\infty$ (down): {a_inf_down:.3f}")
    ax.axhline(0.0, color="0.5", lw=0.5)
    ax.set_xlabel(r"$m_{\ell\ell}$ [TeV]")
    ax.set_ylabel(r"$A_{FB}(m_{\ell\ell})$")
    ax.set_title(
        f"INV-1 gate 2: SM $A_{{FB}}$ sanity ({pdf_name} PDF, "
        r"$\sqrt{s} = 13$ TeV, $\Lambda = 2$ TeV)"
    )
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130)
    print(f"saved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
