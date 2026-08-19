"""spec §3.1 / §12: D_local = 1 - sum p^2, verified against analytic cases."""
import numpy as np

from src.prewm.stochasticity import collision, local_disagreement, d_local


def test_uniform_k():
    for K in (2, 5, 100):
        p = np.ones(K) / K
        assert np.isclose(collision(p), 1.0 / K)
        assert np.isclose(local_disagreement(p), 1.0 - 1.0 / K)


def test_deterministic_distribution():
    p = np.array([1.0, 0.0, 0.0])
    assert np.isclose(collision(p), 1.0)
    assert np.isclose(local_disagreement(p), 0.0)


def test_two_point():
    p = np.array([0.9, 0.1])
    assert np.isclose(collision(p), 0.81 + 0.01)
    assert np.isclose(local_disagreement(p), 1 - 0.82)


def test_renormalizes_support():
    p = np.array([2.0, 2.0])  # unnormalized -> uniform over 2
    assert np.isclose(collision(p), 0.5)


def test_d_local_aggregate():
    frames = [np.array([1.0, 0.0]), np.ones(4) / 4]  # disagree 0.0 and 0.75
    out = d_local(frames)
    assert np.isclose(out["mean"], (0.0 + 0.75) / 2)
    assert out["n"] == 2
