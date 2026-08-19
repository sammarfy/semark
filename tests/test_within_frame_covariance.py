"""spec §1B/§12: the estimator computes mean_t(within-frame cov), NOT pooled covariance.

This is the single most important invariant of the milestone: pooling frames with
different means manufactures a false positive channel covariance.
"""
import numpy as np

from src.prewm.covariance import (
    frame_moments, aggregate_within_frame, pooled_covariance,
)


def _two_frames_with_no_within_but_large_pooled(seed=0):
    """Two frames. Within each frame phi and psi are UNCORRELATED, but the two frames sit
    at very different (phi, psi) means, so pooling sees a strong (spurious) correlation."""
    rng = np.random.default_rng(seed)
    K = 200
    # frame A centered near (0,0); independent phi/psi -> within-cov ~ 0
    phiA = rng.standard_normal(K)
    psiA = rng.standard_normal(K)
    pA = np.ones(K) / K
    # frame B centered near (10,10); independent again -> within-cov ~ 0
    phiB = 10 + rng.standard_normal(K)
    psiB = 10 + rng.standard_normal(K)
    pB = np.ones(K) / K
    return (phiA, psiA, pA), (phiB, psiB, pB)


def test_within_frame_is_not_pooled():
    (phiA, psiA, pA), (phiB, psiB, pB) = _two_frames_with_no_within_but_large_pooled()
    mA = frame_moments(phiA, psiA, pA)
    mB = frame_moments(phiB, psiB, pB)
    agg = aggregate_within_frame([mA, mB])

    pooled = pooled_covariance(np.concatenate([phiA, phiB]), np.concatenate([psiA, psiB]))

    # within-frame channel covariance is ~0 (each frame uncorrelated)
    assert abs(agg.C_raw) < 0.2, agg.C_raw
    # pooled is large and positive (spurious between-frame structure) -> ~25 (=Var of the 0/10 shift)
    assert pooled > 5.0, pooled
    # they must NOT agree — the whole point of the invariant
    assert abs(agg.C_raw - pooled) > 5.0


def test_production_path_matches_mean_within():
    """aggregate_within_frame == arithmetic mean of per-frame C_t (equal weights)."""
    (a, b) = _two_frames_with_no_within_but_large_pooled(1)
    mA, mB = frame_moments(*a), frame_moments(*b)
    agg = aggregate_within_frame([mA, mB])
    assert np.isclose(agg.C_raw, 0.5 * (mA.C + mB.C))
    assert np.isclose(agg.V, 0.5 * (mA.V + mB.V))


def test_frame_moments_weighted():
    # analytic: p-weighted cov of a tiny explicit frame
    phi = np.array([1.0, -1.0])
    psi = np.array([2.0, -2.0])
    p = np.array([0.5, 0.5])
    m = frame_moments(phi, psi, p)
    assert np.isclose(m.V, 1.0)      # Var[phi] = 1
    assert np.isclose(m.S, 4.0)      # Var[psi] = 4
    assert np.isclose(m.C, 2.0)      # Cov = 2
    assert np.isclose(m.j_t(), 1.0)  # perfectly correlated
