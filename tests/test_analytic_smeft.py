"""Tests for :func:`modules.analytic_smeft.simulate_analytic_smeft`.

The SM-limit tests use tabulated reference values. The structural tests
(linear interference, sign flip, quadratic energy growth) are exact physics
statements independent of the PDF, so they pin ``pdf="analytic"`` for speed
and determinism. ``test_sm_limit_ct18nnlo`` exercises the real LHAPDF path and
is skipped automatically when ``lhapdf`` is not importable.
"""

import numpy as np
import pytest

from modules.analytic_smeft import simulate_analytic_smeft
from modules.analytic_smeft.smeft import OPERATORS

# --- SM-limit references ----------------------------------------------------
# Leading-order analytic neutral-current Drell-Yan dsigma/dm_ll (pb/GeV), all
# Wilson coefficients zero, sqrt(s) = 13 TeV, m_ll bin edges below.
SM_BINS = np.array([200.0, 400.0, 600.0, 1000.0, 2000.0])

# With the built-in analytic toy PDF (no DGLAP evolution): a self-contained
# regression baseline. The toy undershoots a real PDF by a few-fold; the SMEFT
# *ratios* are PDF-independent and unaffected.
SM_REFERENCE_TOY = np.array([4.815723e-03, 5.273428e-04, 8.817434e-05, 7.176694e-06])

# With CT18NNLO via LHAPDF: PDF-quality leading-order reference.
SM_REFERENCE_CT18 = np.array([3.579295e-02, 2.586637e-03, 3.132997e-04, 1.634202e-05])


def _require_ct18nnlo():
    """Skip the calling test unless LHAPDF + CT18NNLO is reachable.

    Checks via the module's own PDF resolver, which self-discovers the
    project-vendored LHAPDF build (`make lhapdf`) -- a bare `import lhapdf`
    would not, since that build is not on the default path.
    """
    from modules.analytic_smeft.pdfs import LHAPDFSet

    try:
        LHAPDFSet("CT18NNLO")
    except Exception as exc:  # ImportError, or the PDF set is not installed
        pytest.skip(f"LHAPDF/CT18NNLO not available: {exc}")


def test_sm_limit():
    """All Wilson coefficients zero -> tabulated SM cross section within 5%."""
    result = simulate_analytic_smeft({}, bins=SM_BINS, pdf="analytic")
    sm = result["sm_only"]

    # No operators on => interference vanishes identically.
    assert np.allclose(result["interference"], 0.0)
    # differential_xs reduces to the SM piece.
    assert np.allclose(result["differential_xs"], sm)

    # Matches the tabulated SM Drell-Yan cross section to within 5%.
    rel_dev = np.abs(sm - SM_REFERENCE_TOY) / SM_REFERENCE_TOY
    assert np.all(rel_dev < 0.05), f"relative deviation from SM table: {rel_dev}"

    # Independent physical sanity (not circular): positive, strictly falling,
    # and within a plausible magnitude window for high-mass DY.
    assert np.all(sm > 0.0)
    assert np.all(np.diff(sm) < 0.0)
    assert np.all((sm > 1e-8) & (sm < 1e-1))


def test_sm_limit_ct18nnlo():
    """SM limit with CT18NNLO PDFs (LHAPDF) -> PDF-quality reference within 5%."""
    _require_ct18nnlo()
    result = simulate_analytic_smeft({}, bins=SM_BINS, pdf="CT18NNLO")
    sm = result["sm_only"]

    assert np.allclose(result["interference"], 0.0)
    rel_dev = np.abs(sm - SM_REFERENCE_CT18) / SM_REFERENCE_CT18
    assert np.all(rel_dev < 0.05), f"relative deviation from CT18NNLO table: {rel_dev}"
    assert np.all(sm > 0.0)
    assert np.all(np.diff(sm) < 0.0)


def test_linear_interference():
    """Interference scales linearly in a single four-fermion coefficient."""
    bins = np.array([200.0, 500.0, 1000.0, 2000.0])
    coeffs = [0.01, 0.1, 1.0]
    interference = {
        c: simulate_analytic_smeft(
            {"c_lq^(1)": c}, bins=bins, pdf="analytic"
        )["interference"]
        for c in coeffs
    }

    # interference / c must be independent of c (pure linear scaling).
    base = interference[0.1] / 0.1
    for c in coeffs:
        assert np.allclose(interference[c] / c, base, rtol=1e-8), f"nonlinear at c={c}"

    # The interference must be genuinely non-zero in every bin.
    assert np.all(np.abs(base) > 0.0)


def test_sign_flip():
    """c_lq^(1) -> -c_lq^(1) flips the interference sign in every bin."""
    bins = np.array([200.0, 500.0, 1000.0, 2000.0])
    pos = simulate_analytic_smeft({"c_lq^(1)": 0.3}, bins=bins, pdf="analytic")["interference"]
    neg = simulate_analytic_smeft({"c_lq^(1)": -0.3}, bins=bins, pdf="analytic")["interference"]

    assert np.allclose(pos, -neg, rtol=1e-8)
    # Genuine sign flip in every bin (no accidental zeros).
    assert np.all(pos * neg < 0.0)


def test_quadratic_growth():
    """The BSM^2 piece carries the EFT energy growth.

    The four-fermion contact amplitude grows as (M_ll/Lambda)^2 relative to the
    SM, so |BSM|^2 / SM grows as (M_ll/Lambda)^4 with a constant prefactor.
    """
    lam = 1000.0
    masses = np.array([500.0, 1000.0, 1500.0, 2000.0])
    bsm_sq = np.empty(masses.size)
    sm = np.empty(masses.size)
    for i, m in enumerate(masses):
        # Narrow bins so the bin centre is the effective mass.
        result = simulate_analytic_smeft(
            {"c_lq^(1)": 0.2},
            bins=np.array([m - 10.0, m + 10.0]),
            order="quadratic",
            lambda_scale=lam,
            pdf="analytic",
        )
        assert result["bsm_squared"] is not None
        bsm_sq[i] = result["bsm_squared"][0]
        sm[i] = result["sm_only"][0]

    # linear order must not return a BSM^2 array.
    linear = simulate_analytic_smeft(
        {"c_lq^(1)": 0.2}, bins=np.array([490.0, 510.0]), order="linear", pdf="analytic"
    )
    assert linear["bsm_squared"] is None

    # BSM^2 / SM follows (M_ll/Lambda)^4: the prefactor is constant in the tail.
    growth = (masses / lam) ** 4
    prefactor = (bsm_sq / sm) / growth
    assert np.all(np.abs(prefactor / prefactor[0] - 1.0) < 0.15)
    # ... and the ratio rises monotonically toward the tail.
    assert np.all(np.diff(bsm_sq / sm) > 0.0)


def test_zero_lambda_safe():
    """Lambda = 1 TeV gives no NaN/inf for any single |c| <= 1."""
    bins = np.array([200.0, 400.0, 700.0, 1200.0, 2000.0])
    for operator in OPERATORS:
        for value in (1.0, -1.0, 0.5):
            result = simulate_analytic_smeft(
                {operator: value},
                bins=bins,
                order="quadratic",
                lambda_scale=1000.0,
                pdf="analytic",
            )
            for key in ("differential_xs", "sm_only", "interference", "bsm_squared"):
                array = np.asarray(result[key], dtype=float)
                assert np.all(np.isfinite(array)), f"{key} non-finite for {operator}={value}"


def test_unknown_coefficient_rejected():
    """A misspelt / unsupported coefficient name raises rather than silently 0."""
    with pytest.raises(ValueError, match="Unknown Wilson coefficient"):
        simulate_analytic_smeft({"c_not_an_operator": 1.0})


def test_ptl_observable():
    """The pT_l observable: positive falling spectrum, SMEFT structure intact."""
    bins = np.array([100.0, 200.0, 400.0, 800.0])

    # SM pT spectrum: positive everywhere, strictly falling toward the tail.
    sm = simulate_analytic_smeft({}, observable="pT_l", bins=bins, pdf="analytic")
    assert np.all(sm["sm_only"] > 0.0)
    assert np.all(np.diff(sm["sm_only"]) < 0.0)
    assert np.allclose(sm["interference"], 0.0)

    # Interference stays linear in a four-fermion coefficient and flips sign --
    # the universal pT kernel preserves the SMEFT decomposition.
    pos = simulate_analytic_smeft(
        {"c_lq^(1)": 0.2}, observable="pT_l", bins=bins, pdf="analytic"
    )
    neg = simulate_analytic_smeft(
        {"c_lq^(1)": -0.2}, observable="pT_l", bins=bins, pdf="analytic"
    )
    small = simulate_analytic_smeft(
        {"c_lq^(1)": 0.02}, observable="pT_l", bins=bins, pdf="analytic"
    )
    assert np.allclose(pos["interference"], -neg["interference"], rtol=1e-8)
    assert np.allclose(pos["interference"] / 0.2, small["interference"] / 0.02, rtol=1e-8)
    assert np.all(pos["interference"] * neg["interference"] < 0.0)

    # order plumbing carries over to pT_l.
    linear = simulate_analytic_smeft({}, observable="pT_l", order="linear", pdf="analytic")
    assert linear["bsm_squared"] is None
    quad = simulate_analytic_smeft(
        {"c_lq^(1)": 0.5}, observable="pT_l", order="quadratic", pdf="analytic"
    )
    assert quad["bsm_squared"] is not None and np.all(quad["bsm_squared"] > 0.0)

    # Kernel normalisation cross-check: pT > pT_min requires m_ll > 2 pT_min, so
    # the pT-integrated rate recovers an O(1) (threshold-limited, < 1) fraction
    # of the m_ll > 2 pT_min total. A wrong kernel norm would miss this badly.
    pt_min = 150.0
    m_edges = np.geomspace(2.0 * pt_min, 6000.0, 24)
    pt_edges = np.geomspace(pt_min, 3000.0, 24)
    sig_m = float(np.sum(
        simulate_analytic_smeft(
            {}, observable="m_ll", bins=m_edges, pdf="analytic"
        )["differential_xs"] * np.diff(m_edges)
    ))
    sig_pt = float(np.sum(
        simulate_analytic_smeft(
            {}, observable="pT_l", bins=pt_edges, pdf="analytic"
        )["differential_xs"] * np.diff(pt_edges)
    ))
    assert 0.5 < sig_pt / sig_m < 0.85, f"pT/m closure ratio {sig_pt / sig_m}"


def test_ct18nnlo_smeft_structure():
    """SMEFT decomposition holds with real CT18NNLO PDFs (internal LHAPDF build).

    The structural properties are PDF-independent by construction; this checks
    they survive the full LHAPDF path, and that the pieces recompose exactly.
    """
    _require_ct18nnlo()
    bins = np.array([200.0, 500.0, 1000.0, 2000.0])

    # Interference is exactly linear in the coefficient and flips sign, even
    # with real PDFs.
    pos = simulate_analytic_smeft({"c_lq^(1)": 0.4}, bins=bins, pdf="CT18NNLO")
    neg = simulate_analytic_smeft({"c_lq^(1)": -0.4}, bins=bins, pdf="CT18NNLO")
    small = simulate_analytic_smeft({"c_lq^(1)": 0.04}, bins=bins, pdf="CT18NNLO")
    assert np.allclose(pos["interference"], -neg["interference"], rtol=1e-8)
    assert np.allclose(pos["interference"] / 0.4, small["interference"] / 0.04, rtol=1e-8)

    # differential_xs recomposes exactly from the three pieces at quadratic order.
    quad = simulate_analytic_smeft(
        {"c_lq^(1)": 0.4}, bins=bins, pdf="CT18NNLO", order="quadratic"
    )
    recomposed = quad["sm_only"] + quad["interference"] + quad["bsm_squared"]
    assert np.allclose(quad["differential_xs"], recomposed, rtol=1e-12)
    assert np.all(quad["sm_only"] > 0.0)
