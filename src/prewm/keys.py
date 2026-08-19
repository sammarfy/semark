"""Fixed diagnostic key directions (spec §3.2, §4, §10.3).

A key is a unit direction u in the d-dim whitened detector space. These are DIAGNOSTIC
directions only — NOT watermark embedding keys (no tilt is applied). The same fixed key
list is reused for sigma_matched, V, and C so the normalization stays consistent.
"""
from __future__ import annotations

import numpy as np


def diagnostic_keys(n_keys: int, dim: int, base_seed: int = 20260819) -> np.ndarray:
    """Return [n_keys, dim] unit-norm directions, one per fixed key seed.

    Each key i is drawn from its own seed (base_seed + i) so the bank is reproducible and
    a single key can be regenerated independently.
    """
    U = np.empty((n_keys, dim), dtype=np.float64)
    for i in range(n_keys):
        rng = np.random.default_rng(base_seed + i)
        v = rng.standard_normal(dim)
        U[i] = v / np.linalg.norm(v)
    return U
