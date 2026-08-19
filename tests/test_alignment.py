"""§16.E: deterministic delay compensation, cross-corr verification, no DTW."""
import numpy as np
import pytest

from src import alignment
from src.alignment import (
    crosscorr_peak_lag,
    align_after_offset,
    norm_sequence,
    USES_DTW,
)


def test_no_dtw_used_in_e0():
    assert USES_DTW is False


def test_crosscorr_recovers_injected_lag():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(120)
    offset = 3
    shifted = np.concatenate([np.zeros(offset), base])[: len(base)]
    # b[t] ~ a[t-offset]; peak lag should be +offset
    lag = crosscorr_peak_lag(base, shifted, max_lag=8)
    assert lag == offset


def test_align_after_offset_one_to_one_and_verified():
    rng = np.random.default_rng(1)
    d = 12
    T = 60
    original = rng.standard_normal((T, d))
    offset = 2
    # reencoded = original delayed by `offset` frames, plus tiny noise
    reenc = np.concatenate([rng.standard_normal((offset, d)) * 0.01, original], axis=0)
    reenc = reenc + rng.standard_normal(reenc.shape) * 1e-3

    o_al, r_al, align = align_after_offset(original, reenc, frame_offset=offset, max_lag=6)
    assert align.offset_verified
    assert align.frame_offset == offset == align.frame_offset_crosscorr_peak
    assert o_al.shape == r_al.shape
    assert align.aligned_length == o_al.shape[0]
    # after removing the deterministic delay the frames correspond one-to-one
    assert np.allclose(o_al, r_al, atol=0.05)


def test_wrong_offset_is_hard_stop():
    rng = np.random.default_rng(2)
    d = 8
    original = rng.standard_normal((40, d))
    reenc = np.concatenate([np.zeros((4, d)), original], axis=0)  # true delay 4
    with pytest.raises(AssertionError):
        # claim offset 0 -> cross-correlation peak (4) disagrees -> §20.10 stop
        align_after_offset(original, reenc, frame_offset=0, max_lag=6)


def test_norm_sequence_shape():
    x = np.ones((10, 5))
    ns = norm_sequence(x)
    assert ns.shape == (10,)
    assert np.allclose(ns, np.sqrt(5))
