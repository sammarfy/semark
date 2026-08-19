"""spec §5.3 (branch candidate selection, pure part of test_branch_rollout):
select smallest ordinary-candidate set covering the target mass; never silently truncate."""
import numpy as np

from src.prewm.candidates import select_candidates


def test_covers_target_mass():
    V = 3072
    probs = np.zeros(V)
    probs[10] = 0.8; probs[11] = 0.15; probs[12] = 0.04; probs[13] = 0.01
    ordinary = np.zeros(V, bool); ordinary[:2048] = True
    cs = select_candidates(probs, ordinary, target_mass=0.995, k_max=8)
    assert set(cs.ids.tolist()) >= {10, 11, 12}       # covers >=0.99
    assert cs.covered_mass >= 0.99
    assert not cs.low_coverage


def test_specials_excluded():
    V = 3072
    probs = np.zeros(V)
    probs[2100] = 0.5   # special (>=2048) - must be excluded
    probs[5] = 0.5
    ordinary = np.zeros(V, bool); ordinary[:2048] = True
    cs = select_candidates(probs, ordinary, target_mass=0.995, k_max=8)
    assert 2100 not in cs.ids.tolist()
    assert 5 in cs.ids.tolist()
    assert np.isclose(cs.ordinary_mass, 0.5)          # only the ordinary half


def test_low_coverage_flagged_not_truncated_silently():
    V = 3072
    # flat over 100 ordinary tokens -> 8 candidates cover only ~0.08 of ordinary mass
    probs = np.zeros(V); probs[:100] = 1.0 / 100
    ordinary = np.zeros(V, bool); ordinary[:2048] = True
    cs = select_candidates(probs, ordinary, target_mass=0.995, k_max=8, min_required=0.99, k_expand=12)
    assert cs.capped
    assert cs.low_coverage        # explicitly flagged, not hidden
    assert cs.covered_mass < 0.99
