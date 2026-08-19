"""Branch-rollout candidate selection (spec §5.3).

p_t is peaked, so i.i.d. sampling would repeat the top token. Instead pick the smallest
ordinary-candidate set whose cumulative ordinary probability mass >= target, capped at
K_max, and RECORD the covered mass. Never silently renormalize a low-coverage subset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateSet:
    ids: np.ndarray          # selected ordinary candidate ids (talker vocab)
    probs: np.ndarray        # their p_t weights (unnormalized over full vocab)
    covered_mass: float      # cumulative ordinary mass covered
    ordinary_mass: float     # total ordinary (non-special) mass at this frame
    capped: bool             # True if K_max hit before reaching target
    low_coverage: bool       # True if covered_mass < min_required


def select_candidates(probs: np.ndarray, ordinary_mask: np.ndarray,
                      target_mass: float = 0.995, k_max: int = 8,
                      min_required: float = 0.99, k_expand: int = 12) -> CandidateSet:
    """Select ordinary candidates by descending p_t up to cumulative `target_mass`.

    probs: [V] full talker distribution at this frame.
    ordinary_mask: [V] bool, True for ordinary codec candidates (specials excluded).
    Expands to k_expand if k_max fails to cover min_required; flags low_coverage otherwise.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    mask = np.asarray(ordinary_mask, dtype=bool).ravel()
    ord_ids = np.where(mask)[0]
    ord_p = probs[ord_ids]
    ordinary_mass = float(ord_p.sum())
    order = ord_ids[np.argsort(ord_p)[::-1]]
    order_p = probs[order]
    cum = np.cumsum(order_p)
    # smallest k reaching target_mass * ordinary_mass
    need = target_mass * ordinary_mass
    k = int(np.searchsorted(cum, need) + 1) if ordinary_mass > 0 else 0
    capped = False
    if k > k_max:
        k, capped = k_max, True
        if cum[k_max - 1] < min_required * ordinary_mass and k_expand > k_max:
            k = min(k_expand, len(order))
            capped = k >= k_max
    k = max(1, min(k, len(order)))
    sel = order[:k]
    covered = float(probs[sel].sum())
    low = covered < min_required * ordinary_mass if ordinary_mass > 0 else True
    return CandidateSet(ids=sel, probs=probs[sel], covered_mass=covered,
                        ordinary_mass=ordinary_mass, capped=capped, low_coverage=low)
