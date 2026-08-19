"""Text-level fit/dev/test splits with hard leakage guarantees (spec §7.1, §12)."""
from __future__ import annotations

import hashlib

import numpy as np


def text_splits(text_ids: list[str], n_fit: int, n_dev: int, n_test: int,
                seed: int = 20260819) -> dict:
    """Deterministic disjoint text-level split. No text appears in two splits."""
    ids = sorted(set(text_ids))
    if len(ids) < n_fit + n_dev + n_test:
        raise ValueError(f"need {n_fit+n_dev+n_test} texts, have {len(ids)}")
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(ids)))
    order = [ids[i] for i in perm]
    fit = sorted(order[:n_fit])
    dev = sorted(order[n_fit:n_fit + n_dev])
    test = sorted(order[n_fit + n_dev:n_fit + n_dev + n_test])
    assert not (set(fit) & set(dev)) and not (set(fit) & set(test)) and not (set(dev) & set(test))
    return {"fit": fit, "dev": dev, "test": test,
            "hash": hashlib.sha256(("|".join(fit + ["/"] + dev + ["/"] + test)).encode()).hexdigest()[:16]}


def assert_disjoint(splits: dict) -> None:
    f, d, t = set(splits["fit"]), set(splits["dev"]), set(splits["test"])
    leaks = (f & d) | (f & t) | (d & t)
    if leaks:
        raise AssertionError(f"split leakage: {sorted(leaks)}")
