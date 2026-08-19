"""Boundary/context diagnostic curve (spec §6 / E0-e, Q10).

The ~29-frame chunk-seam result only proves sensitivity to an ARTIFICIAL split. E0-e asks
the real question: how many frames of left context does a NORMAL encode need before the
latent stabilizes? We compare suffix re-encodes (reduced left context) against the full
encode, indexed by distance from the new boundary, and read a context length AFTER seeing
the curve using a PREDECLARED numerical-stability criterion.

Pure numpy; the encode/re-encode itself runs on the GPU host and feeds arrays here.
"""
from __future__ import annotations

import numpy as np

from ..metrics import rowwise_cosine, rowwise_l2


def error_vs_distance(z_full: np.ndarray, z_suffix: np.ndarray, first_frame_distance: int = 0):
    """Per-frame error between a suffix re-encode and the full encode, aligned by physical
    time, indexed by distance from the suffix's new left boundary.

    z_full, z_suffix: [n, d] aligned frames (same physical times).
    first_frame_distance: distance label of z_suffix[0] from the boundary (usually 0).
    Returns dict of arrays: distance, cosine, norm_l2.
    """
    a = np.asarray(z_full, dtype=np.float64)
    b = np.asarray(z_suffix, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    cos = rowwise_cosine(a, b)
    denom = np.linalg.norm(a, axis=-1)
    nl2 = np.where(denom > 0, rowwise_l2(a, b) / denom, np.nan)
    dist = np.arange(first_frame_distance, first_frame_distance + n)
    return {"distance": dist, "cosine": cos, "norm_l2": nl2}


def aggregate_by_distance(curves: list[dict]) -> dict:
    """Aggregate multiple per-suffix curves into median cosine / L2 per distance."""
    by_d = {}
    for c in curves:
        for dist, cos, l2 in zip(c["distance"], c["cosine"], c["norm_l2"]):
            by_d.setdefault(int(dist), {"cos": [], "l2": []})
            by_d[int(dist)]["cos"].append(float(cos))
            by_d[int(dist)]["l2"].append(float(l2))
    dists = sorted(by_d)
    return {
        "distance": np.array(dists),
        "median_cosine": np.array([np.median(by_d[d]["cos"]) for d in dists]),
        "median_norm_l2": np.array([np.nanmedian(by_d[d]["l2"]) for d in dists]),
        "n": np.array([len(by_d[d]["cos"]) for d in dists]),
    }


def descriptive_context_length(agg: dict, min_n: int = 20, cos_margin: float = 0.003,
                               l2_factor: float = 1.3, window: int = 6) -> dict:
    """Relative-to-floor context length for when the absolute thresholds are numerically
    inappropriate (spec §6: report the full curve, pick a descriptive length after seeing it).

    Estimates the cosine plateau and L2 floor from well-sampled far distances, then returns the
    smallest well-sampled distance whose own value AND the next `window` well-sampled values are
    within `cos_margin` of the plateau and `l2_factor`x the floor. Robust to noisy small-n tails.
    """
    d = np.asarray(agg["distance"]); cos = np.asarray(agg["median_cosine"])
    l2 = np.asarray(agg["median_norm_l2"]); n = np.asarray(agg["n"])
    rel = n >= min_n
    far = rel & (d >= max(10, int(0.3 * d.max())))
    if far.sum() < 3:
        return {"b_context": None, "reason": "not enough well-sampled far distances"}
    plateau = float(np.median(cos[far])); floor = float(np.median(l2[far]))
    rel_idx = [i for i in range(len(d)) if rel[i]]
    b = None
    for pos, i in enumerate(rel_idx):
        seg = rel_idx[pos:pos + 1 + window]
        if all(cos[j] >= plateau - cos_margin and l2[j] <= l2_factor * floor for j in seg):
            b = int(d[i]); break
    return {"b_context": b, "plateau_cosine": plateau, "l2_floor": floor,
            "criterion": f"cos >= plateau-{cos_margin} and L2 <= {l2_factor}x floor over a "
                         f"{window}-frame window at well-sampled distances (n>={min_n})"}


def context_length(agg: dict, cos_thresh: float = 0.999, l2_thresh: float = 0.01) -> dict:
    """Predeclared criterion: smallest distance from which median cosine >= cos_thresh AND
    median norm-L2 <= l2_thresh for all larger distances. Returns b_context or None + curve
    so the caller can report the full curve if thresholds are numerically inappropriate.
    """
    d = agg["distance"]
    cos = agg["median_cosine"]
    l2 = agg["median_norm_l2"]
    ok = (cos >= cos_thresh) & (l2 <= l2_thresh)
    b = None
    for i in range(len(d)):
        if np.all(ok[i:]):
            b = int(d[i])
            break
    return {"b_context": b, "cos_thresh": cos_thresh, "l2_thresh": l2_thresh,
            "criterion": "median cosine >= %.4f AND median norm_l2 <= %.4f from b onward"
                         % (cos_thresh, l2_thresh)}
