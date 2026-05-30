"""Offline tests for the MCP-rubric selector in eigen_query.py.

The HTTP-side (fetch_eigen_spans) is not unit-tested because it
requires a live Phoenix; the parsing/selection logic is what matters
and can be exercised against synthesised spans.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "full-chain-run"))

from eigen_query import select_lis_gap


def _make_span(cycle, lam, U_low, d, k):
    return {
        "span_id": f"s-{cycle}",
        "start_time": cycle,
        "attrs": {
            "aletheia.cycle.index": cycle,
            "aletheia.eigen.lambda": list(map(float, lam)),
            "aletheia.eigen.U_low_flat":
                np.asarray(U_low).flatten(order="F").tolist(),
            "aletheia.eigen.d": int(d),
            "aletheia.eigen.k": int(k),
        },
    }


def test_select_lis_gap_empty_when_no_firings():
    """No cycle has a trailing eigenvalue below theta * lam_max => no firings."""
    d, k = 6, 3
    spans = []
    for c in range(3):
        # well-conditioned: spectrum ~ [1, 2, 3, 4, 5, 6]
        spans.append(_make_span(c, [1, 2, 3, 4, 5, 6], np.eye(d)[:, :k], d, k))
    out = select_lis_gap(spans, theta=0.01, since_cycle=0, top_k=k)
    assert out["fired_cycles"] == []
    assert out["U_low_union"] == []


def test_select_lis_gap_fires_when_low_eigenvalue():
    d, k = 6, 3
    spans = [
        _make_span(0, [1, 2, 3, 4, 5, 6], np.eye(d)[:, :k], d, k),
        # cycle 1 has a tiny smallest eigenvalue
        _make_span(1, [1e-6, 2, 3, 4, 5, 6], np.eye(d)[:, :k], d, k),
    ]
    out = select_lis_gap(spans, theta=1e-3, since_cycle=0, top_k=k)
    assert out["fired_cycles"] == [1]
    U = np.asarray(out["U_low_union"])
    # union of one block of rank 3 is rank 3
    assert U.shape == (d, 3)


def test_select_lis_gap_respects_since_cycle():
    d, k = 4, 2
    spans = [
        _make_span(0, [1e-9, 1, 2, 3], np.eye(d)[:, :k], d, k),
        _make_span(5, [1e-9, 1, 2, 3], np.eye(d)[:, :k], d, k),
    ]
    out = select_lis_gap(spans, theta=1e-3, since_cycle=3, top_k=k)
    assert out["fired_cycles"] == [5]


def test_select_lis_gap_union_orthonormal():
    """Two firings with overlapping low subspaces should orthonormalise."""
    d, k = 6, 2
    U_a = np.eye(d)[:, :k]                   # columns 0, 1
    U_b = np.eye(d)[:, 1:3]                  # columns 1, 2 (overlaps)
    spans = [
        _make_span(0, [1e-9, 1e-8, 1, 2, 3, 4], U_a, d, k),
        _make_span(1, [1e-9, 1e-8, 1, 2, 3, 4], U_b, d, k),
    ]
    out = select_lis_gap(spans, theta=1e-3, since_cycle=0, top_k=k)
    U = np.asarray(out["U_low_union"])
    # rank should be 3 (span of e_0, e_1, e_2)
    assert U.shape == (d, 3)
    # columns orthonormal
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-9)


def test_select_lis_gap_handles_inconsistent_d_silently():
    """Spans with mismatched d_psi should be skipped, not crash."""
    spans = [
        _make_span(0, [1e-9, 1, 2, 3], np.eye(4)[:, :2], 4, 2),
        _make_span(1, [1e-9, 1, 2, 3, 4, 5], np.eye(6)[:, :2], 6, 2),
    ]
    # First span sets d=4; second has d=6 and is skipped.
    out = select_lis_gap(spans, theta=1e-3, since_cycle=0, top_k=2)
    assert out["fired_cycles"] == [0]


def test_select_lis_gap_n_spans_seen():
    d, k = 4, 2
    spans = [_make_span(c, [1, 2, 3, 4], np.eye(d)[:, :k], d, k) for c in range(7)]
    out = select_lis_gap(spans, theta=0.01, since_cycle=0, top_k=k)
    assert out["n_spans_seen"] == 7
