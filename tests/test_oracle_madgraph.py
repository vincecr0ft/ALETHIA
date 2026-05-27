"""Smoke tests for the MadGraph oracle adapter.

These are gated on MG5_aMC being installed at vendor/MG5_aMC and are slow
(~20 s/query on a laptop). They're marked ``slow`` so the normal test
suite skips them; opt in with ``pytest -m slow``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modules.surrogate import N_WC, WC_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
MG_PROCESS_DIR = REPO_ROOT / "vendor" / "MG5_aMC" / "processes" / "dy_smeft"

slow = pytest.mark.slow

_mg_available = (MG_PROCESS_DIR / "bin" / "generate_events").exists()
_skip_no_mg = pytest.mark.skipif(
    not _mg_available, reason="MG process dir missing; run scripts/install_madgraph.sh"
)


@slow
@_skip_no_mg
def test_card_editors_zero_smeft_block(tmp_path: Path) -> None:
    """``_zero_smeft_block`` clears every Wilson coefficient line and
    ``_set_smeft_value`` updates exactly the targeted lhacode."""
    from modules.surrogate.oracle_madgraph import _set_smeft_value, _zero_smeft_block

    sample = (
        "Block smeft \n"
        "    1 1.000000e-01 # cG \n"
        "   24 9.000000e-03 # cHq1 \n"
        "   25 2.000000e-04 # cHq3 \n"
        "   35 4.000000e-05 # clq1 \n"
        "   36 5.000000e-05 # clq3 \n"
        "###################################\n"
        "Block smeftcutoff \n"
        "    1 1.000000e+03 # LambdaSMEFT \n"
    )
    zeroed = _zero_smeft_block(sample)
    assert "1.000000e-01" not in zeroed   # cG cleared
    assert "9.000000e-03" not in zeroed   # cHq1 cleared
    assert "0.000000e+00" in zeroed
    # Lambda block should be untouched.
    assert "1.000000e+03 # LambdaSMEFT" in zeroed
    # Now set clq1 = 0.42.
    updated = _set_smeft_value(zeroed, 35, 0.42)
    assert "4.200000e-01" in updated
    # Only lhacode 35 changed; lhacode 36 still zero.
    for line in updated.splitlines():
        if line.lstrip().startswith("36 "):
            assert "0.000000e+00" in line


@slow
@_skip_no_mg
def test_xs_parser() -> None:
    """`Integrated weight (pb)` line parses to a float matching the banner."""
    from modules.surrogate.oracle_madgraph import _parse_xs_from_banner
    text = (
        "blah blah\n"
        "#  Integrated weight (pb)  :       0.00271676\n"
        "other\n"
    )
    p = REPO_ROOT / "tests" / "_tmp_banner.txt"
    p.write_text(text)
    try:
        assert _parse_xs_from_banner(p) == pytest.approx(0.00271676, rel=1e-6)
    finally:
        p.unlink()


@slow
@_skip_no_mg
def test_mg_sm_limit() -> None:
    """Real MG run at all WCs = 0 returns mu ~ 1 to within MC noise (5%)."""
    from modules.surrogate import MadGraphSMEFTOracle

    oracle = MadGraphSMEFTOracle(nevents=1000, m_window_tev=0.20)
    c = np.zeros((1, N_WC))
    mu = oracle.truth(c, np.array([1.0]))
    assert mu.shape == (1,)
    # σ_SM/σ_SM should give exactly 1 minus a small MC fluctuation when the
    # numerator is recomputed (it's a fresh MG run, not the cached value).
    assert 0.93 < mu[0] < 1.07


@slow
@_skip_no_mg
def test_mg_eft_growth() -> None:
    """Activating clq1 drives mu well above 1 at high mass.

    With Lambda = 1 TeV, clq1 = +0.3, m_ll = 1.1 TeV: hand-checked smoke
    run gave mu ≈ 6.4. Allow ±15% for MC noise across runs.
    """
    from modules.surrogate import MadGraphSMEFTOracle

    oracle = MadGraphSMEFTOracle(
        nevents=1000, m_window_tev=0.20, lambda_gev=1000.0
    )
    c = np.zeros((1, N_WC))
    c[0, WC_NAMES.index("clq1")] = 0.3
    mu = oracle.truth(c, np.array([1.1]))
    assert 5.4 < mu[0] < 7.4, f"got mu = {mu[0]:.3f}, expected ~6.4"


@slow
@_skip_no_mg
def test_mg_precompute_sm_cache_hits() -> None:
    """``precompute_sm`` fills the cache so subsequent queries skip the SM run."""
    from modules.surrogate import MadGraphSMEFTOracle

    oracle = MadGraphSMEFTOracle(nevents=1000, m_window_tev=0.20)
    grid = np.array([0.5, 1.0, 1.5])
    oracle.precompute_sm(grid)
    assert set(round(float(m), 6) for m in grid) <= set(oracle._sigma_sm_cache)
    # A query at an EXACT grid point should not trigger a fresh SM run.
    before = dict(oracle._sigma_sm_cache)
    c = np.zeros((1, N_WC))
    c[0, WC_NAMES.index("clq3")] = 0.05
    oracle.truth(c, np.array([1.0]))
    # Only the BSM run was new; SM at m=1.0 was already cached.
    assert oracle._sigma_sm_cache == before
