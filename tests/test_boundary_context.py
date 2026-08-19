"""spec §6/§12: boundary curve is indexed by distance from the new boundary; no DTW;
physical-time alignment is the caller's responsibility (we operate on aligned frames)."""
import numpy as np

from src.prewm.boundary import error_vs_distance, aggregate_by_distance, context_length


def test_error_curve_indexing():
    d = 16
    z_full = np.random.default_rng(0).standard_normal((40, d))
    z_suffix = z_full.copy()
    z_suffix[:5] += 1.0                    # corrupt first 5 frames (near boundary)
    cur = error_vs_distance(z_full, z_suffix, first_frame_distance=0)
    assert cur["distance"][0] == 0
    assert cur["cosine"][10] > 0.999       # far frames match
    assert cur["cosine"][0] < 0.999        # near-boundary frame degraded


def test_context_length_predeclared_criterion():
    # build an aggregate where cosine recovers >=0.999 from distance 6 onward
    dists = list(range(0, 20))
    curves = []
    for _ in range(4):
        cos = np.array([0.9 + 0.02 * min(x, 5) for x in dists])  # ramps to ~1.0 by 5
        cos = np.clip(cos, 0, 1.0)
        cos[6:] = 0.9999
        l2 = np.where(np.array(dists) >= 6, 0.001, 0.05)
        curves.append({"distance": np.array(dists), "cosine": cos, "norm_l2": l2})
    agg = aggregate_by_distance(curves)
    out = context_length(agg, cos_thresh=0.999, l2_thresh=0.01)
    assert out["b_context"] == 6


def test_context_length_none_when_never_stable():
    dists = np.arange(10)
    curves = [{"distance": dists, "cosine": np.full(10, 0.95), "norm_l2": np.full(10, 0.1)}]
    agg = aggregate_by_distance(curves)
    assert context_length(agg)["b_context"] is None
