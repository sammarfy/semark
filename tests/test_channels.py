"""spec §7.3: derivative channel transforms preserve length/timing (synchronous families)
and are separated from timing-changing families."""
import numpy as np

from src.prewm.channels import (
    apply_synchronous, awgn, resample_roundtrip, one_pole_lowpass,
    SYNCHRONOUS_FAMILIES, TIMING_FAMILIES,
)


def test_synchronous_families_preserve_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(24000).astype(np.float32)
    for fam in SYNCHRONOUS_FAMILIES:
        y = apply_synchronous(fam, x, 24000, rng)
        assert len(y) == len(x), (fam, len(y))


def test_clean_is_identity():
    x = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
    assert np.allclose(apply_synchronous("clean", x, 24000, np.random.default_rng(0)), x)


def test_awgn_reduces_snr_monotonically():
    rng = np.random.default_rng(2)
    x = np.sin(np.linspace(0, 100, 24000)).astype(np.float32)
    e_hi = np.mean((awgn(x, 30, rng) - x) ** 2)
    e_lo = np.mean((awgn(x, 10, rng) - x) ** 2)
    assert e_lo > e_hi                       # lower SNR -> more noise energy


def test_lowpass_reduces_high_freq_energy():
    t = np.linspace(0, 1, 24000, endpoint=False)
    hi = np.sin(2 * np.pi * 8000 * t).astype(np.float32)
    y = one_pole_lowpass(hi)
    assert np.mean(y ** 2) < np.mean(hi ** 2)


def test_families_are_disjoint():
    assert not (set(SYNCHRONOUS_FAMILIES) & set(TIMING_FAMILIES))
