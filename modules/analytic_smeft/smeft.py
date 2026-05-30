"""Analytic leading-order SMEFT Drell-Yan cross section.

Closed-form neutral-current Drell-Yan, ``p p -> l+ l-``, at parton level via
``gamma*/Z`` exchange plus dimension-6 SMEFT operators (Warsaw basis). The
energy-growing four-fermion contact structure follows Greljo & Marzocca,
"High-pT dilepton tails and Wilson coefficients" (arXiv:1704.09015).

Physics summary
---------------
The partonic amplitude for ``q qbar -> l+ l-`` is written, per quark flavour
``q`` and per chirality channel ``(X, Y)`` (X = quark chirality, Y = lepton
chirality), as a single current-times-current coefficient

    A_XY(s) = A_SM_XY(s) + A_BSM_XY(s)

with the SM part the photon + Z exchange and the dim-6 part

    A_BSM_XY(s) = c4f_XY / Lambda^2                         (four-fermion contact)
                + (Z propagator) * (vertex-coupling shifts)  (~ v^2 / Lambda^2)

The partonic cross section, after the trivial angular integral, is

    sigma_q(s) = s / (48 pi) * sum_XY |A_XY(s)|^2

which reproduces the textbook ``sigma = 4 pi alpha^2 / (3 s) * Q_q^2`` in the
pure-QED limit. Squaring and grouping by power of 1/Lambda^2 gives the
``sm_only`` / ``interference`` / ``bsm_squared`` decomposition. The hadronic
differential cross section convolves each partonic piece with the *same*
quark-antiquark parton luminosity, so the SMEFT ratios are PDF-independent.

The four-fermion contact is constant in ``s`` while the SM amplitude falls as
``1/s``; hence the interference / SM ratio grows as ``(M_ll/Lambda)^2`` and the
BSM^2 / SM ratio as ``(M_ll/Lambda)^4`` -- the EFT energy growth.

Operator set (14, Warsaw basis, dimension-6)
--------------------------------------------
Four-fermion: ``c_lq^(1) c_lq^(3) c_eu c_ed c_lu c_ld c_qe``.
Vertex:       ``c_phi_l^(1) c_phi_l^(3) c_phi_e c_phi_q^(1) c_phi_q^(3)
               c_phi_u c_phi_d``.
Any coefficient not supplied defaults to 0.

No Pythia, Delphes, MadGraph, showering or detector effects -- numpy + scipy
only.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from .constants import (
    COS2_THETA_W,
    E_CHARGE_SQ,
    GAMMA_Z,
    GEV2_TO_PB,
    LEPTON_CHARGE,
    LEPTON_T3,
    M_Z,
    QUARK_CHARGE,
    QUARK_T3,
    SIN2_THETA_W,
    UP_TYPE_QUARKS,
    V_HIGGS,
)
from .pdfs import PDFSet, get_pdf

# --- The 14 supported Warsaw-basis dimension-6 operators ---
FOUR_FERMION_OPERATORS = (
    "c_lq^(1)", "c_lq^(3)", "c_eu", "c_ed", "c_lu", "c_ld", "c_qe",
)
VERTEX_OPERATORS = (
    "c_phi_l^(1)", "c_phi_l^(3)", "c_phi_e",
    "c_phi_q^(1)", "c_phi_q^(3)", "c_phi_u", "c_phi_d",
)
OPERATORS = FOUR_FERMION_OPERATORS + VERTEX_OPERATORS

# Chirality channels: (quark chirality, lepton chirality).
CHANNELS = (("L", "L"), ("L", "R"), ("R", "L"), ("R", "R"))

# Light quark flavours contributing to Drell-Yan (PDG ids).
_QUARK_FLAVOURS = (1, 2, 3, 4, 5)

# Gauss-Legendre nodes for the parton-luminosity, bin, and pT-kernel integrals.
_LUM_NODES, _LUM_WEIGHTS = np.polynomial.legendre.leggauss(64)
_BIN_NODES, _BIN_WEIGHTS = np.polynomial.legendre.leggauss(8)
_PT_NODES, _PT_WEIGHTS = np.polynomial.legendre.leggauss(32)


# --------------------------------------------------------------------------
# Wilson coefficient handling
# --------------------------------------------------------------------------
def _normalise_wc(wilson_coefficients: Optional[dict]) -> dict:
    """Return a full coefficient dict (all 14 keys, missing ones = 0.0)."""
    wc = {key: 0.0 for key in OPERATORS}
    if wilson_coefficients:
        unknown = set(wilson_coefficients) - set(OPERATORS)
        if unknown:
            raise ValueError(
                f"Unknown Wilson coefficient(s): {sorted(unknown)}. "
                f"Supported operators: {list(OPERATORS)}"
            )
        for key, value in wilson_coefficients.items():
            wc[key] = float(value)
    return wc


# --------------------------------------------------------------------------
# Couplings
# --------------------------------------------------------------------------
def _g_quark_sm(chirality: str, pid: int) -> float:
    """SM neutral-current chiral coupling of quark ``pid``."""
    charge = QUARK_CHARGE[abs(pid)]
    t3 = QUARK_T3[abs(pid)]
    return (t3 - charge * SIN2_THETA_W) if chirality == "L" else (-charge * SIN2_THETA_W)


def _g_lepton_sm(chirality: str) -> float:
    """SM neutral-current chiral coupling of the charged lepton."""
    if chirality == "L":
        return LEPTON_T3 - LEPTON_CHARGE * SIN2_THETA_W
    return -LEPTON_CHARGE * SIN2_THETA_W


def _kappa_lepton(chirality: str, wc: dict) -> float:
    """Vertex-operator shift kappa for the lepton (delta g = v^2/Lambda^2 * kappa)."""
    if chirality == "L":
        return -0.5 * (wc["c_phi_l^(1)"] + wc["c_phi_l^(3)"])
    return -0.5 * wc["c_phi_e"]


def _kappa_quark(chirality: str, is_up: bool, wc: dict) -> float:
    """Vertex-operator shift kappa for a quark.

    The isotriplet ``c_phi_q^(3)`` enters left-handed up and down quarks with
    opposite sign (the tau^3 projection).
    """
    if chirality == "L":
        if is_up:
            return -0.5 * (wc["c_phi_q^(1)"] - wc["c_phi_q^(3)"])
        return -0.5 * (wc["c_phi_q^(1)"] + wc["c_phi_q^(3)"])
    return -0.5 * (wc["c_phi_u"] if is_up else wc["c_phi_d"])


def _contact(channel: tuple, is_up: bool, wc: dict) -> float:
    """Four-fermion Wilson coefficient feeding chirality ``channel``.

    Like ``c_phi_q^(3)``, the isotriplet ``c_lq^(3)`` splits up vs down quarks.
    """
    qx, ly = channel
    if (qx, ly) == ("L", "L"):
        return (wc["c_lq^(1)"] - wc["c_lq^(3)"]) if is_up else (wc["c_lq^(1)"] + wc["c_lq^(3)"])
    if (qx, ly) == ("L", "R"):
        return wc["c_qe"]
    if (qx, ly) == ("R", "L"):
        return wc["c_lu"] if is_up else wc["c_ld"]
    # ("R", "R")
    return wc["c_eu"] if is_up else wc["c_ed"]


# --------------------------------------------------------------------------
# Amplitudes and partonic cross section
# --------------------------------------------------------------------------
def _amplitudes(pid: int, s_hat: np.ndarray, wc: dict, lam: float):
    """SM and dim-6 amplitudes for flavour ``pid``, shape ``(4, len(s_hat))``."""
    charge_q = QUARK_CHARGE[abs(pid)]
    is_up = abs(pid) in UP_TYPE_QUARKS
    z_factor = E_CHARGE_SQ / (SIN2_THETA_W * COS2_THETA_W)
    prop_gamma = 1.0 / s_hat
    prop_z = 1.0 / (s_hat - M_Z**2 + 1j * M_Z * GAMMA_Z)
    v2_over_lam2 = V_HIGGS**2 / lam**2

    a_sm, a_bsm = [], []
    for channel in CHANNELS:
        qx, ly = channel
        g_q = _g_quark_sm(qx, pid)
        g_l = _g_lepton_sm(ly)

        # SM: photon (vector, chirality-blind) + Z with SM couplings.
        photon = E_CHARGE_SQ * charge_q * LEPTON_CHARGE * prop_gamma
        z_sm = z_factor * g_q * g_l * prop_z
        a_sm.append(photon + z_sm)

        # Dimension-6: four-fermion contact + linearised Z-vertex shift.
        contact = _contact(channel, is_up, wc) / lam**2
        dg_q = v2_over_lam2 * _kappa_quark(qx, is_up, wc)
        dg_l = v2_over_lam2 * _kappa_lepton(ly, wc)
        z_shift = z_factor * (dg_q * g_l + g_q * dg_l) * prop_z
        a_bsm.append(contact + z_shift)

    return np.array(a_sm), np.array(a_bsm)


def _partonic_xs(pid: int, s_hat: np.ndarray, wc: dict, lam: float):
    """Partonic cross sections (GeV^-2): (sm_only, interference, bsm_squared)."""
    a_sm, a_bsm = _amplitudes(pid, s_hat, wc, lam)
    prefactor = s_hat / (48.0 * math.pi)
    sm = prefactor * np.sum(np.abs(a_sm) ** 2, axis=0)
    interference = prefactor * np.sum(2.0 * np.real(np.conj(a_sm) * a_bsm), axis=0)
    bsm_squared = prefactor * np.sum(np.abs(a_bsm) ** 2, axis=0)
    return sm, interference, bsm_squared


# Forward-backward channel signs: aligned chiralities (LL, RR) contribute
# (1+cos theta*)^2 and integrate to +3/4 * |a|^2 above cos theta* = 0;
# anti-aligned (LR, RL) contribute (1-cos theta*)^2 and integrate to -3/4 * |a|^2.
# The factor 3/4 is the angular average of cos theta* over the (1+cos theta*)^2
# distribution; see e.g. Halzen & Martin section 12.6.
_AFB_SIGNS = np.array([
    +1.0 if channel in (("L", "L"), ("R", "R")) else -1.0 for channel in CHANNELS
])


def _partonic_xs_afb(pid: int, s_hat: np.ndarray, wc: dict, lam: float):
    """Partonic forward-minus-backward numerator (GeV^-2).

    Returns ``(sm_only_afb, interference_afb, bsm_squared_afb)``. The numerator
    of the partonic ``A_FB = (sigma_F - sigma_B) / (sigma_F + sigma_B)`` is the
    LL + RR minus LR + RL combination of channel-resolved squared amplitudes,
    weighted by the 3/4 angular-averaging factor.
    """
    a_sm, a_bsm = _amplitudes(pid, s_hat, wc, lam)
    prefactor = s_hat / (48.0 * math.pi) * 0.75
    signs = _AFB_SIGNS[:, None] if a_sm.ndim == 2 else _AFB_SIGNS
    sm = prefactor * np.sum(signs * np.abs(a_sm) ** 2, axis=0)
    interference = prefactor * np.sum(
        signs * 2.0 * np.real(np.conj(a_sm) * a_bsm), axis=0)
    bsm_squared = prefactor * np.sum(signs * np.abs(a_bsm) ** 2, axis=0)
    return sm, interference, bsm_squared


def _partonic_S_D(pid: int, s_hat: np.ndarray, wc: dict, lam: float):
    """Partonic S and D pieces (GeV^-2) for the cos-theta* differential.

    The partonic dilepton angular distribution factorises as

        d sigma_hat / d cos theta*  =  (3/8) [ (1 + cos^2 theta*) S
                                              + 2 cos theta* D ],

    with the partonic prefactor ``s_hat / (48 pi)`` already absorbed into
    ``S`` and ``D`` so that integrating over ``cos theta* in [-1, 1]``
    reproduces :func:`_partonic_xs` (gate 1).

    Returns ``(S_sm, S_int, S_bsm, D_sm, D_int, D_bsm)``: the S triplet is
    identical to :func:`_partonic_xs`, the D triplet is
    :func:`_partonic_xs_afb` with the 3/4 angular-averaging factor stripped
    so that ``D_parton`` is purely the LL+RR vs LR+RL chiral combination.
    """
    a_sm, a_bsm = _amplitudes(pid, s_hat, wc, lam)
    prefactor = s_hat / (48.0 * math.pi)
    signs = _AFB_SIGNS[:, None] if a_sm.ndim == 2 else _AFB_SIGNS
    S_sm = prefactor * np.sum(np.abs(a_sm) ** 2, axis=0)
    S_int = prefactor * np.sum(2.0 * np.real(np.conj(a_sm) * a_bsm), axis=0)
    S_bsm = prefactor * np.sum(np.abs(a_bsm) ** 2, axis=0)
    D_sm = prefactor * np.sum(signs * np.abs(a_sm) ** 2, axis=0)
    D_int = prefactor * np.sum(
        signs * 2.0 * np.real(np.conj(a_sm) * a_bsm), axis=0)
    D_bsm = prefactor * np.sum(signs * np.abs(a_bsm) ** 2, axis=0)
    return S_sm, S_int, S_bsm, D_sm, D_int, D_bsm


# --------------------------------------------------------------------------
# Parton luminosity and hadronic convolution
# --------------------------------------------------------------------------
def _luminosity(pdf: PDFSet, pid: int, tau: float, scale: float) -> float:
    """Symmetrised q-qbar parton luminosity ``Phi_qqbar(tau)`` for flavour pid.

    Phi(tau) = (1/tau) * integral_tau^1 dx/x [xf_q(x) xf_qbar(tau/x) + (q<->qbar)].
    """
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    log_tau = math.log(tau)
    # Map Gauss-Legendre nodes from [-1, 1] to u = ln(x) in [ln tau, 0].
    u = 0.5 * (0.0 - log_tau) * _LUM_NODES + 0.5 * log_tau
    jac = 0.5 * (0.0 - log_tau)
    x = np.exp(u)
    x2 = tau / x
    integrand = (
        pdf.xf(pid, x, scale) * pdf.xf(-pid, x2, scale)
        + pdf.xf(-pid, x, scale) * pdf.xf(pid, x2, scale)
    )
    return (1.0 / tau) * jac * float(np.sum(_LUM_WEIGHTS * integrand))


def _luminosity_asym(pdf: PDFSet, pid: int, tau: float, scale: float) -> float:
    """Asymmetric (valence-minus-sea) q-qbar parton luminosity for flavour pid.

    The hadronic forward-backward dilution: the antisymmetric piece D of the
    cos theta* distribution survives the pp -> ll convolution only because
    valence-quark PDFs exceed sea-antiquark PDFs at large x, so on average the
    quark moves along the dilepton boost direction. In the Collins-Soper frame
    with z-axis along ``sign(p_z^{ll})`` this is

        Phi^asym(tau)  =  sum over y of sign(y)
                        * [f_q(x1) f_qbar(x2) - f_qbar(x1) f_q(x2)],

    with ``x1,2 = sqrt(tau) e^{+/- y}``. Folding to ``y > 0`` and changing
    variables to ``u = ln x1`` over ``[ln sqrt(tau), 0]``:

        Phi^asym(tau)  =  2 * integral_{sqrt tau}^{1} dx/x
                          * [ f_q(x) f_qbar(tau/x) - f_qbar(x) f_q(tau/x) ].

    Same 64-node Gauss-Legendre quadrature as :func:`_luminosity`. Returns 0
    outside ``0 < tau < 1``.

    The convention ``sign(y) = sign(cos theta*_CS)`` is the standard CS-frame
    assignment for which the SM A_FB at high m_ll is positive (left-handed
    Z-coupling to up-type quarks dominates). This sign is what gate 2 of the
    INV-1 spec verifies against the asymptotic formula.

    If ``pdf.xf`` cannot be queried per-flavour (the LHAPDF backend can; the
    :class:`AnalyticPDF` toy also can — both expose ``xf(pid, x, Q)``), the
    quadrature integrand silently mis-dilutes, so we sanity-check the backend
    here and raise rather than fall through.
    """
    if not hasattr(pdf, "xf"):
        raise RuntimeError(
            f"PDF backend {type(pdf).__name__} does not expose per-flavour "
            f"xf(pid, x, Q); the asymmetric luminosity cannot be computed."
        )
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    # Fold y > 0: integrate u = ln(x) in [ln sqrt(tau), 0] = [0.5 ln tau, 0].
    half_log_tau = 0.5 * math.log(tau)
    u = 0.5 * (0.0 - half_log_tau) * _LUM_NODES + 0.5 * half_log_tau
    jac = 0.5 * (0.0 - half_log_tau)
    x = np.exp(u)
    x2 = tau / x
    # x f_q(x) * x f_qbar(tau/x) = tau * f_q(x) f_qbar(tau/x).
    integrand = (
        pdf.xf(pid, x, scale) * pdf.xf(-pid, x2, scale)
        - pdf.xf(-pid, x, scale) * pdf.xf(pid, x2, scale)
    )
    # Factor of 2 from folding y > 0 only, then (1/tau) to strip xf -> f f.
    return 2.0 * (1.0 / tau) * jac * float(np.sum(_LUM_WEIGHTS * integrand))


def _dsigma_dm(m: np.ndarray, wc: dict, lam: float, pdf: PDFSet, s: float):
    """Hadronic differential cross section dsigma/dm_ll (pb/GeV) at masses ``m``.

    Returns ``(sm_only, interference, bsm_squared)``.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    s_hat = m**2
    tau = s_hat / s
    sm = np.zeros_like(m)
    interference = np.zeros_like(m)
    bsm_squared = np.zeros_like(m)

    for pid in _QUARK_FLAVOURS:
        p_sm, p_int, p_bsm = _partonic_xs(pid, s_hat, wc, lam)
        lumi = np.array(
            [_luminosity(pdf, pid, t, scale) for t, scale in zip(tau, m)]
        )
        sm += p_sm * lumi
        interference += p_int * lumi
        bsm_squared += p_bsm * lumi

    # dsigma/dm = (2 m / s) * GeV^2->pb * sum_q sigma_q(m^2) * Phi_q(m^2/s).
    flux = 2.0 * m / s * GEV2_TO_PB
    return flux * sm, flux * interference, flux * bsm_squared


def _dsigma_dm_afb_numerator(m: np.ndarray, wc: dict, lam: float, pdf: PDFSet, s: float):
    """FB-asymmetric piece of dsigma/dm_ll (pb/GeV) at masses ``m``.

    Numerator of the differential ``A_FB(m_ll) = (dsigma_F - dsigma_B) /
    (dsigma_F + dsigma_B)``. Returns ``(sm_only_afb, interference_afb,
    bsm_squared_afb)`` with the same shape conventions as :func:`_dsigma_dm`.

    The partonic D piece (chiral asymmetry, signed by ``sign(cos theta*)``)
    survives the hadronic pp -> ll convolution only through the
    valence-minus-sea PDF luminosity (:func:`_luminosity_asym`). Using the
    symmetric luminosity here was the source of the ~30% gate-3 mismatch
    observed in the INV-1 MG cross-check (the symmetric Phi^sym was over-
    weighting the same chiral D-parton at all rapidities, producing an
    undiluted A_FB).
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    s_hat = m**2
    tau = s_hat / s
    sm = np.zeros_like(m)
    interference = np.zeros_like(m)
    bsm_squared = np.zeros_like(m)
    for pid in _QUARK_FLAVOURS:
        p_sm, p_int, p_bsm = _partonic_xs_afb(pid, s_hat, wc, lam)
        lumi = np.array(
            [_luminosity_asym(pdf, pid, t, scale) for t, scale in zip(tau, m)]
        )
        sm += p_sm * lumi
        interference += p_int * lumi
        bsm_squared += p_bsm * lumi
    flux = 2.0 * m / s * GEV2_TO_PB
    return flux * sm, flux * interference, flux * bsm_squared


def _dsigma_dm_S_D_hadronic(m: np.ndarray, wc: dict, lam: float, pdf: PDFSet, s: float):
    """Hadronic ``S_had`` and ``D_had`` pieces of ``d sigma / d m_ll``.

    ``S_had`` uses the symmetrised q-qbar luminosity (:func:`_luminosity`) and
    is the conventional rate convolution. ``D_had`` uses the asymmetric
    (valence-minus-sea) luminosity (:func:`_luminosity_asym`), which is the
    forward-backward dilution that survives the unknown quark direction in
    pp. Both are returned per morphing channel
    ``(sm_only, interference, bsm_squared)`` in pb/GeV (the ``flux`` factor
    ``2 m / s * GeV^2 -> pb`` is already applied).

    The cos-theta* bin integral assembles these as

        dsigma_bin / d m_ll  =  (3/8) [ w_S * S_had  +  w_D * D_had ]

    with ``w_S = integral (1 + u^2) du`` and ``w_D = integral 2 u du`` over
    the bin ``[u_lo, u_hi]`` (see :func:`differential_xs_costheta_bin`).
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    s_hat = m**2
    tau = s_hat / s
    S_sm = np.zeros_like(m)
    S_int = np.zeros_like(m)
    S_bsm = np.zeros_like(m)
    D_sm = np.zeros_like(m)
    D_int = np.zeros_like(m)
    D_bsm = np.zeros_like(m)
    for pid in _QUARK_FLAVOURS:
        Ssm, Sint, Sbsm, Dsm, Dint, Dbsm = _partonic_S_D(pid, s_hat, wc, lam)
        lumi_sym = np.array(
            [_luminosity(pdf, pid, t, scale) for t, scale in zip(tau, m)]
        )
        lumi_asym = np.array(
            [_luminosity_asym(pdf, pid, t, scale) for t, scale in zip(tau, m)]
        )
        S_sm += Ssm * lumi_sym
        S_int += Sint * lumi_sym
        S_bsm += Sbsm * lumi_sym
        D_sm += Dsm * lumi_asym
        D_int += Dint * lumi_asym
        D_bsm += Dbsm * lumi_asym
    flux = 2.0 * m / s * GEV2_TO_PB
    return (
        flux * S_sm, flux * S_int, flux * S_bsm,
        flux * D_sm, flux * D_int, flux * D_bsm,
    )


def _dsigma_dpt(pt: np.ndarray, wc: dict, lam: float, pdf: PDFSet, s: float):
    """Hadronic differential cross section dsigma/dpT_l (pb/GeV) at momenta ``pt``.

    At leading order the dilepton system has no transverse recoil, so a single
    charged lepton carries pT = (m_ll / 2) sin(theta*). The forward-backward
    asymmetric part of the angular distribution cancels between the two
    theta* -> pi - theta* solutions, leaving a *universal* (coefficient- and
    piece-independent) pT kernel. Marginalising over the dilepton mass,

        dsigma/dpT = (3/2) * integral_0^{pi/2} dpsi (1 + cos^2 psi)
                     * (dsigma/dm)|_{m = 2 pT / sin psi}.

    The substitution ``m = 2 pT / sin psi`` removes the integrable Jacobian-peak
    singularity at the kinematic edge m = 2 pT. Returns ``(sm, interference,
    bsm_squared)``.
    """
    pt = np.atleast_1d(np.asarray(pt, dtype=float))
    # psi nodes on (0, pi/2].
    half_range = 0.25 * math.pi
    psi = half_range * (_PT_NODES + 1.0)
    weight = _PT_WEIGHTS * (1.0 + np.cos(psi) ** 2)
    sin_psi = np.sin(psi)

    sm = np.empty_like(pt)
    interference = np.empty_like(pt)
    bsm_squared = np.empty_like(pt)
    for i, p in enumerate(pt):
        m = 2.0 * p / sin_psi
        d_sm, d_int, d_bsm = _dsigma_dm(m, wc, lam, pdf, s)
        sm[i] = 1.5 * half_range * float(np.sum(weight * d_sm))
        interference[i] = 1.5 * half_range * float(np.sum(weight * d_int))
        bsm_squared[i] = 1.5 * half_range * float(np.sum(weight * d_bsm))
    return sm, interference, bsm_squared


def _bin_average(lo: float, hi: float, kernel):
    """Bin-average a differential cross section over ``[lo, hi]`` (8-node GL).

    ``kernel(nodes)`` returns ``(sm, interference, bsm_squared)`` at ``nodes``.
    """
    half = 0.5 * (hi - lo)
    centre = 0.5 * (hi + lo)
    sm, interference, bsm_squared = kernel(half * _BIN_NODES + centre)
    # average = (1/(hi-lo)) * integral = (half/(hi-lo)) * sum(w * f).
    norm = half / (hi - lo)
    return (
        norm * float(np.sum(_BIN_WEIGHTS * sm)),
        norm * float(np.sum(_BIN_WEIGHTS * interference)),
        norm * float(np.sum(_BIN_WEIGHTS * bsm_squared)),
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def differential_afb(
    wilson_coefficients: Optional[dict],
    m_ll: Union[float, np.ndarray],
    *,
    sqrt_s: float = 13000.0,
    lambda_scale: float = 1000.0,
    order: str = "quadratic",
    pdf: Union[str, PDFSet] = "auto",
) -> dict:
    """Pointwise forward-backward asymmetry ``A_FB(m_ll)``.

    Companion to :func:`differential_xs` exposing the differential FB
    asymmetry as a function of dilepton invariant mass. The chirality
    structure of the cross section, which is integrated out in
    ``d sigma / d m_ll`` and in ``d sigma / d pT_l``, drives ``A_FB`` and
    distinguishes the left-handed and right-handed vertex operators
    (``c_phi_q^(3)`` and ``c_phi_q^(1)``) and the four-fermion operators
    of different chiralities. This is the discriminating observable for
    the residual non-disclosure pattern reported in
    Section "Identifiability" of the paper.

    Returns a dict with ``m_ll``, ``A_FB``, ``A_FB_SM``, ``A_FB_BSM``,
    ``sm_xs``, ``afb_numerator``.
    """
    if order not in ("linear", "quadratic"):
        raise ValueError(f"order must be 'linear' or 'quadratic', got {order!r}.")
    wc = _normalise_wc(wilson_coefficients)
    pdf_set = get_pdf(pdf)
    s = float(sqrt_s) ** 2
    lam = float(lambda_scale)
    m_arr = np.atleast_1d(np.asarray(m_ll, dtype=float))

    sm_xs, int_xs, bsm_xs = _dsigma_dm(m_arr, wc, lam, pdf_set, s)
    sm_afb, int_afb, bsm_afb = _dsigma_dm_afb_numerator(m_arr, wc, lam, pdf_set, s)

    if order == "linear":
        denom = sm_xs + int_xs
        numer = sm_afb + int_afb
    else:
        denom = sm_xs + int_xs + bsm_xs
        numer = sm_afb + int_afb + bsm_afb

    denom_safe = np.where(np.abs(denom) > 1e-30, denom, 1.0)
    a_fb_total = numer / denom_safe
    a_fb_sm = sm_afb / np.where(np.abs(sm_xs) > 1e-30, sm_xs, 1.0)
    return {
        "m_ll": m_arr,
        "A_FB": a_fb_total,
        "A_FB_SM": a_fb_sm,
        "sm_xs": sm_xs,
        "afb_numerator": numer,
        "wilson_coefficients": {key: wc[key] for key in OPERATORS},
        "process": "pp_to_ll",
    }


def differential_xs_costheta_bin(
    wilson_coefficients: Optional[dict],
    m_ll: Union[float, np.ndarray],
    costheta_bin: tuple,
    *,
    sqrt_s: float = 13000.0,
    lambda_scale: float = 1000.0,
    order: str = "quadratic",
    pdf: Union[str, PDFSet] = "auto",
) -> dict:
    """Pointwise differential cross section ``d sigma / d m_ll`` (pb/GeV)
    integrated over a single ``cos theta*_CS`` bin.

    Splits the partonic angular distribution into a symmetric and an
    antisymmetric piece,

        d sigma_hat / d cos theta*  =  (3/8) [ (1 + cos^2 theta*) S(c, s_hat)
                                              + 2 cos theta* D(c, s_hat) ],

    with ``S(c, s_hat) = sum_channels |M_ij|^2`` (rate piece) and
    ``D(c, s_hat) = sum_channels SIGN(ij) |M_ij|^2`` (forward-backward
    piece, ``SIGN(LL) = SIGN(RR) = +1``, ``SIGN(LR) = SIGN(RL) = -1``).
    The bin observable for ``cos theta*_CS in [u_lo, u_hi]`` is

        d sigma_bin / d m_ll  =  (3/8) [ w_S(bin) * S_had(m_ll)
                                       + w_D(bin) * D_had(m_ll) ],
        w_S(bin) = (u + u^3 / 3)|_{u_lo}^{u_hi},
        w_D(bin) = u_hi^2 - u_lo^2.

    ``S_had`` uses the existing symmetrised q-qbar luminosity. ``D_had``
    uses an asymmetric (valence-minus-sea) luminosity that implements the
    hadronic FB dilution in the Collins-Soper frame
    (see :func:`_luminosity_asym`).

    Args:
      wilson_coefficients: Warsaw-basis dim-6 coefficients (see ``OPERATORS``).
        ``None`` or ``{}`` means SM only.
      m_ll: dilepton invariant mass values in GeV. Scalar or 1-D array.
      costheta_bin: ``(u_lo, u_hi)``, with ``-1 <= u_lo < u_hi <= 1``.
      sqrt_s: hadronic centre-of-mass energy in GeV.
      lambda_scale: EFT scale Lambda in GeV.
      order: ``"linear"`` keeps O(Lambda^-2); ``"quadratic"`` adds O(Lambda^-4).
      pdf: PDF spec; see :func:`modules.analytic_smeft.pdfs.get_pdf`.

    Returns:
      dict with ``m_ll`` (GeV), ``costheta_bin`` (``(u_lo, u_hi)``),
      ``differential_xs`` (pb/GeV), ``sm_only``, ``interference``,
      ``bsm_squared`` (``None`` for ``order="linear"``), ``S_had`` and
      ``D_had`` (per-channel triples of the hadronic S and D pieces, both in
      pb/GeV at ``order="quadratic"``, summed only over ``sm_only +
      interference`` at ``order="linear"``), and the angular weights
      ``w_S`` and ``w_D``. Summing the bin observable over a partition of
      ``[-1, 1]`` recovers :func:`differential_xs` (gate 1 of INV-1).
    """
    if order not in ("linear", "quadratic"):
        raise ValueError(f"order must be 'linear' or 'quadratic', got {order!r}.")
    u_lo, u_hi = float(costheta_bin[0]), float(costheta_bin[1])
    if not (-1.0 - 1e-12 <= u_lo < u_hi <= 1.0 + 1e-12):
        raise ValueError(
            f"costheta_bin must satisfy -1 <= u_lo < u_hi <= 1, "
            f"got ({u_lo}, {u_hi})."
        )
    # Closed-form angular weights -- exact, no quadrature.
    w_S = (u_hi + u_hi**3 / 3.0) - (u_lo + u_lo**3 / 3.0)
    w_D = u_hi**2 - u_lo**2

    wc = _normalise_wc(wilson_coefficients)
    pdf_set = get_pdf(pdf)
    s = float(sqrt_s) ** 2
    lam = float(lambda_scale)
    m_arr = np.atleast_1d(np.asarray(m_ll, dtype=float))

    S_sm, S_int, S_bsm, D_sm, D_int, D_bsm = _dsigma_dm_S_D_hadronic(
        m_arr, wc, lam, pdf_set, s
    )
    # (3/8) [(1+u^2) S + 2u D]  ->  bin: (3/8) [w_S * S + w_D * D].
    coeff = 0.375  # 3/8.
    sm = coeff * (w_S * S_sm + w_D * D_sm)
    interference = coeff * (w_S * S_int + w_D * D_int)
    bsm_squared = coeff * (w_S * S_bsm + w_D * D_bsm)

    if order == "linear":
        total = sm + interference
        bsm_out: Optional[np.ndarray] = None
    else:
        total = sm + interference + bsm_squared
        bsm_out = bsm_squared

    return {
        "m_ll": m_arr,
        "costheta_bin": (u_lo, u_hi),
        "differential_xs": total,
        "sm_only": sm,
        "interference": interference,
        "bsm_squared": bsm_out,
        "S_had": (S_sm, S_int, S_bsm),
        "D_had": (D_sm, D_int, D_bsm),
        "w_S": w_S,
        "w_D": w_D,
        "wilson_coefficients": {key: wc[key] for key in OPERATORS},
        "process": "pp_to_ll",
    }


def differential_xs_pt(
    wilson_coefficients: Optional[dict],
    pt_l: Union[float, np.ndarray],
    *,
    sqrt_s: float = 13000.0,
    lambda_scale: float = 1000.0,
    order: str = "quadratic",
    pdf: Union[str, PDFSet] = "auto",
) -> dict:
    """Pointwise differential cross section ``d sigma / d pT_l`` (pb/GeV).

    Companion to :func:`differential_xs` that exposes the lepton transverse
    momentum spectrum at parton level. At LO the dilepton system has no
    transverse recoil, so the kernel is the angular integral of
    ``d sigma / d m_ll`` over ``m_ll = 2 pT / sin(theta*)`` weighted by
    ``(1 + cos^2 theta*)``. The resulting projection has different SMEFT
    operator weighting than ``d sigma / d m_ll`` because the angular
    distribution depends on the chirality structure of each operator;
    this asymmetry is the basis of multi-observable identifiability.

    Returns the same dict shape as :func:`differential_xs` with the
    ``m_ll`` key replaced by ``pt_l``.
    """
    if order not in ("linear", "quadratic"):
        raise ValueError(f"order must be 'linear' or 'quadratic', got {order!r}.")
    wc = _normalise_wc(wilson_coefficients)
    pdf_set = get_pdf(pdf)
    s = float(sqrt_s) ** 2
    lam = float(lambda_scale)
    pt_arr = np.atleast_1d(np.asarray(pt_l, dtype=float))
    sm, interference, bsm_squared = _dsigma_dpt(pt_arr, wc, lam, pdf_set, s)
    if order == "linear":
        total = sm + interference
        bsm_out: Optional[np.ndarray] = None
    else:
        total = sm + interference + bsm_squared
        bsm_out = bsm_squared
    return {
        "pt_l": pt_arr,
        "differential_xs": total,
        "sm_only": sm,
        "interference": interference,
        "bsm_squared": bsm_out,
        "wilson_coefficients": {key: wc[key] for key in OPERATORS},
        "process": "pp_to_ll",
    }


def differential_xs(
    wilson_coefficients: Optional[dict],
    m_ll: Union[float, np.ndarray],
    *,
    process: str = "pp_to_ll",
    sqrt_s: float = 13000.0,
    lambda_scale: float = 1000.0,
    order: str = "quadratic",
    pdf: Union[str, PDFSet] = "auto",
) -> dict:
    """Pointwise differential cross section ``d sigma / d m_ll`` (pb/GeV).

    Companion to :func:`simulate_analytic_smeft`: same physics, no binning.
    Surrogates and downstream integrators consume the differential directly.

    Args:
      wilson_coefficients: Warsaw-basis dim-6 coefficients (see ``OPERATORS``).
        ``None`` or ``{}`` means SM only.
      m_ll: dilepton invariant mass values in GeV. Scalar or 1-D array.
      process: only ``"pp_to_ll"`` is supported.
      sqrt_s: hadronic centre-of-mass energy in GeV.
      lambda_scale: EFT scale Lambda in GeV.
      order: ``"linear"`` keeps O(Lambda^-2); ``"quadratic"`` adds O(Lambda^-4).
      pdf: PDF spec; see :func:`modules.analytic_smeft.pdfs.get_pdf`.

    Returns:
      dict with ``m_ll``, ``differential_xs``, ``sm_only``, ``interference``,
      ``bsm_squared`` (``None`` for ``order="linear"``), ``wilson_coefficients``,
      and ``process``. All cross-section arrays are pb/GeV, shape ``m_ll.shape``.
    """
    if process != "pp_to_ll":
        raise ValueError(f"Unsupported process {process!r}; only 'pp_to_ll' is implemented.")
    if order not in ("linear", "quadratic"):
        raise ValueError(f"order must be 'linear' or 'quadratic', got {order!r}.")

    wc = _normalise_wc(wilson_coefficients)
    pdf_set = get_pdf(pdf)
    s = float(sqrt_s) ** 2
    lam = float(lambda_scale)
    m_arr = np.atleast_1d(np.asarray(m_ll, dtype=float))

    sm, interference, bsm_squared = _dsigma_dm(m_arr, wc, lam, pdf_set, s)
    if order == "linear":
        total = sm + interference
        bsm_out: Optional[np.ndarray] = None
    else:
        total = sm + interference + bsm_squared
        bsm_out = bsm_squared

    return {
        "m_ll": m_arr,
        "differential_xs": total,
        "sm_only": sm,
        "interference": interference,
        "bsm_squared": bsm_out,
        "wilson_coefficients": {key: wc[key] for key in OPERATORS},
        "process": process,
    }


def simulate_analytic_smeft(
    wilson_coefficients: dict,
    process: str = "pp_to_ll",
    sqrt_s: float = 13000.0,
    observable: str = "m_ll",
    bins: Optional[np.ndarray] = None,
    lambda_scale: float = 1000.0,
    order: str = "linear",
    pdf: Union[str, PDFSet] = "auto",
) -> dict:
    """Analytic leading-order SMEFT Drell-Yan cross section.

    Args:
      wilson_coefficients: Warsaw-basis dimension-6 coefficients (see
        ``OPERATORS``). Missing keys default to 0.0; unknown keys raise.
      process: only ``"pp_to_ll"`` (neutral-current Drell-Yan) is supported.
      sqrt_s: hadronic centre-of-mass energy in GeV.
      observable: ``"m_ll"`` (dilepton invariant mass) or ``"pT_l"`` (single
        charged-lepton transverse momentum). Sets what ``bins`` ranges over.
      bins: 1-D array of bin edges for ``observable``; a default high-mass
        binning is used when ``None``.
      lambda_scale: EFT scale Lambda in GeV.
      order: ``"linear"`` keeps O(Lambda^-2); ``"quadratic"`` adds O(Lambda^-4).
      pdf: PDF spec -- ``"auto"`` (lhapdf CT18NNLO if installed, else the
        analytic toy), ``"analytic"``, an lhapdf set name, or a ``PDFSet``.

    Returns:
      dict with ``bin_centers``, ``bin_edges``, ``differential_xs`` (pb/GeV),
      ``sm_only``, ``interference``, ``bsm_squared`` (``None`` when
      ``order="linear"``), ``wilson_coefficients`` and ``process``.
    """
    if process != "pp_to_ll":
        raise ValueError(f"Unsupported process {process!r}; only 'pp_to_ll' is implemented.")
    if order not in ("linear", "quadratic"):
        raise ValueError(f"order must be 'linear' or 'quadratic', got {order!r}.")
    if observable not in ("m_ll", "pT_l"):
        raise ValueError(f"Unsupported observable {observable!r}; use 'm_ll' or 'pT_l'.")

    wc = _normalise_wc(wilson_coefficients)
    pdf_set = get_pdf(pdf)
    s = float(sqrt_s) ** 2
    lam = float(lambda_scale)

    if observable == "m_ll":
        default_bins = np.array([200.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0, 3000.0])

        def kernel(nodes):
            return _dsigma_dm(nodes, wc, lam, pdf_set, s)
    else:  # pT_l
        default_bins = np.array([50.0, 100.0, 200.0, 350.0, 600.0, 1000.0])

        def kernel(nodes):
            return _dsigma_dpt(nodes, wc, lam, pdf_set, s)

    edges = np.asarray(default_bins if bins is None else bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bins must be a 1-D array of at least 2 bin edges.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("bins must be strictly increasing.")
    centers = 0.5 * (edges[:-1] + edges[1:])

    sm = np.empty(centers.size)
    interference = np.empty(centers.size)
    bsm_squared = np.empty(centers.size)
    for i in range(centers.size):
        sm[i], interference[i], bsm_squared[i] = _bin_average(edges[i], edges[i + 1], kernel)

    if order == "linear":
        differential_xs = sm + interference
        bsm_out: Optional[np.ndarray] = None
    else:
        differential_xs = sm + interference + bsm_squared
        bsm_out = bsm_squared

    return {
        "bin_centers": centers,
        "bin_edges": edges,
        "differential_xs": differential_xs,
        "sm_only": sm,
        "interference": interference,
        "bsm_squared": bsm_out,
        "wilson_coefficients": {key: wc[key] for key in OPERATORS},
        "process": process,
    }
