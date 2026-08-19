"""spec §5/§12 (pure parts): branch aggregation, CRN determinism, natural-reproduction check.
The runtime forcing (forced natural token reproduces base) is validated on the GPU host."""
import numpy as np

from src.prewm.branch import (
    continuation_seeds, BranchFrame, aggregate_frames, natural_reproduction_ok,
)


def test_common_random_numbers_shared_and_deterministic():
    a = continuation_seeds(3, 2)
    b = continuation_seeds(3, 2)
    assert a == b                         # deterministic
    assert continuation_seeds(4, 2) != a  # differ by anchor
    assert len(a) == 2


def test_identity_channel_branch_gives_J_one():
    """psi_bar == phi at every anchor -> aggregate J == 1 (mandatory identity check)."""
    rng = np.random.default_rng(0)
    frames = []
    for aid in range(6):
        K = rng.integers(3, 8)
        phi = rng.standard_normal(K)
        p = rng.random(K); p /= p.sum()
        frames.append(BranchFrame(aid, f"t{aid%3}", 10 + aid,
                                  np.arange(K), p, phi, phi.copy(), covered_mass=1.0))
    agg = aggregate_frames(frames, equal="state")
    assert np.isclose(agg.J_clean, 1.0, atol=1e-9)
    assert np.isclose(agg.C_raw, agg.V, atol=1e-10)


def test_aggregate_is_mean_within_frame_not_pooled():
    # two anchors with no within correlation but different means -> aggregate C ~ 0
    f1 = BranchFrame(0, "t0", 5, np.arange(50),
                     np.ones(50) / 50, np.random.default_rng(1).standard_normal(50),
                     np.random.default_rng(2).standard_normal(50), covered_mass=1.0)
    f2 = BranchFrame(1, "t1", 6, np.arange(50),
                     np.ones(50) / 50, 10 + np.random.default_rng(3).standard_normal(50),
                     10 + np.random.default_rng(4).standard_normal(50), covered_mass=1.0)
    agg = aggregate_frames([f1, f2])
    assert abs(agg.C_raw) < 0.3          # within-frame, not the ~25 a pooled estimate would give


def test_natural_reproduction_pass_and_fail():
    base = np.array([1, 2, 3, 4, 5])
    assert natural_reproduction_ok(base, base.copy())["ok"]
    bad = np.array([1, 2, 9, 4, 5])
    assert not natural_reproduction_ok(base, bad)["ok"]
    # latent tolerance
    z = np.random.default_rng(0).standard_normal((10, 4))
    assert natural_reproduction_ok(base, base.copy(), z, z + 1e-6)["ok"]
    assert not natural_reproduction_ok(base, base.copy(), z, z + 1.0)["ok"]
