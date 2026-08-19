"""Text-level bootstrap (spec §8, §12). Resamples TEXT ids, never individual frames.

Frames within a text are autocorrelated; bootstrapping frames independently would
understate uncertainty. Every resample here draws whole texts with replacement.
"""
from __future__ import annotations

import numpy as np


def bootstrap_text_indices(text_ids: list[str], n_replicates: int, seed: int = 7) -> list[list[str]]:
    """Return n_replicates resamples, each a list of text ids drawn WITH REPLACEMENT."""
    ids = list(text_ids)
    rng = np.random.default_rng(seed)
    n = len(ids)
    return [[ids[i] for i in rng.integers(0, n, size=n)] for _ in range(n_replicates)]


def bootstrap_statistic(text_ids: list[str], stat_fn, n_replicates: int = 200,
                        seed: int = 7, ci: float = 0.95) -> dict:
    """Bootstrap a scalar statistic over text resamples.

    stat_fn(list_of_text_ids) -> float. Returns point (on full set), mean, and CI.
    """
    point = float(stat_fn(list(text_ids)))
    reps = bootstrap_text_indices(text_ids, n_replicates, seed)
    vals = np.array([stat_fn(r) for r in reps], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    lo = float(np.percentile(vals, 100 * (1 - ci) / 2)) if vals.size else float("nan")
    hi = float(np.percentile(vals, 100 * (1 + ci) / 2)) if vals.size else float("nan")
    return {"point": point, "mean": float(vals.mean()) if vals.size else float("nan"),
            "ci_low": lo, "ci_high": hi, "n_replicates": int(vals.size)}
