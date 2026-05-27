"""Standard Model electroweak constants for the analytic SMEFT calculator.

Central values follow PDG 2024. Natural units: energies in GeV, cross
sections converted to pb via ``GEV2_TO_PB``.
"""

from __future__ import annotations

import math

# --- Boson masses and widths (GeV) ---
M_Z = 91.1876
GAMMA_Z = 2.4952
M_W = 80.377

# --- Electroweak mixing and couplings ---
SIN2_THETA_W = 0.23122          # on-shell sin^2(theta_W)
COS2_THETA_W = 1.0 - SIN2_THETA_W
ALPHA_EM = 1.0 / 127.951        # running alpha_em near the Z scale
E_CHARGE_SQ = 4.0 * math.pi * ALPHA_EM   # e^2

# --- Higgs vacuum expectation value (GeV) ---
V_HIGGS = 246.22

# --- QCD / colour ---
N_COLORS = 3

# --- Unit conversion: (hbar c)^2 in GeV^2 * pb ---
GEV2_TO_PB = 3.8937966e8

# --- Quark electric charges (units of e), keyed by PDG id ---
QUARK_CHARGE = {1: -1.0 / 3.0, 2: 2.0 / 3.0, 3: -1.0 / 3.0, 4: 2.0 / 3.0, 5: -1.0 / 3.0}

# --- Weak isospin T^3 of the left-handed quark, keyed by PDG id ---
QUARK_T3 = {1: -0.5, 2: 0.5, 3: -0.5, 4: 0.5, 5: -0.5}

# --- Charged-lepton quantum numbers ---
LEPTON_CHARGE = -1.0
LEPTON_T3 = -0.5

# Up-type PDG ids carry T^3 = +1/2; down-type carry -1/2.
UP_TYPE_QUARKS = (2, 4)
DOWN_TYPE_QUARKS = (1, 3, 5)
