"""spec §8/§12: bootstrap resamples TEXT ids (with replacement), never individual frames."""
import numpy as np

from src.prewm.bootstrap import bootstrap_text_indices, bootstrap_statistic


def test_resamples_are_text_ids_with_replacement():
    ids = [f"t{i}" for i in range(20)]
    reps = bootstrap_text_indices(ids, n_replicates=50, seed=3)
    assert len(reps) == 50
    for r in reps:
        assert len(r) == len(ids)                 # same size as original
        assert set(r).issubset(set(ids))          # only original text ids
    # with replacement -> at least one replicate has a duplicate
    assert any(len(set(r)) < len(ids) for r in reps)


def test_bootstrap_statistic_ci_brackets_point():
    ids = [f"t{i}" for i in range(30)]
    vals = {t: float(i) for i, t in enumerate(ids)}
    stat = lambda subset: np.mean([vals[t] for t in subset])
    out = bootstrap_statistic(ids, stat, n_replicates=200, seed=1)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["n_replicates"] == 200


def test_frame_level_resampling_absent():
    # sanity: the API only exposes text-id resampling; no frame index leaks in
    ids = ["a", "b", "c"]
    reps = bootstrap_text_indices(ids, 5, seed=0)
    assert all(all(x in ids for x in r) for r in reps)
