"""Exact local same-context stochastic freedom (spec §3.1, Q3/Q4).

    collision_t   = sum_v p_t(v)^2
    local_disagree_t = 1 - collision_t          (prob two i.i.d. draws from p_t differ)
    D_local = mean_t(local_disagree_t)

D_local measures the freedom the sampler has AT A FIXED CONTEXT. It is NOT the ~87%
full-trajectory seed disagreement, which also includes the autoregressive cascade
(different contexts compound). Pure numpy.
"""
from __future__ import annotations

import numpy as np


def collision(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """sum_v p(v)^2 over the given axis. p may be dense or over a support (renormalized)."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pn = np.where(s > 0, p / s, 0.0)
    return (pn * pn).sum(axis=axis)


def local_disagreement(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """1 - collision(p)."""
    return 1.0 - collision(p, axis=axis)


def d_local(frame_probs: list[np.ndarray]) -> dict:
    """Aggregate local disagreement across frames.

    frame_probs: list of 1-D probability vectors (one per frame; ragged support allowed).
    Returns mean/median/p10/p90 of local_disagree_t.
    """
    vals = np.array([float(local_disagreement(p)) for p in frame_probs], dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "n": 0}
    return {"mean": float(vals.mean()), "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90)),
            "n": int(vals.size)}
