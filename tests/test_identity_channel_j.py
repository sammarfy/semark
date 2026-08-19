"""spec §5.6: mandatory identity-channel test. If psi == phi then C = V = sigma^2 and J = 1."""
import numpy as np

from src.prewm.covariance import frame_moments, aggregate_within_frame


def test_identity_channel_single_frame():
    rng = np.random.default_rng(0)
    K = 50
    phi = rng.standard_normal(K)
    psi = phi.copy()                       # identity channel
    p = rng.random(K); p /= p.sum()
    m = frame_moments(phi, psi, p)
    assert np.isclose(m.C, m.V, atol=1e-12)
    assert np.isclose(m.S, m.V, atol=1e-12)
    assert np.isclose(m.j_t(), 1.0, atol=1e-9)


def test_identity_channel_aggregate_j_is_one():
    rng = np.random.default_rng(1)
    moments = []
    for _ in range(20):
        K = rng.integers(5, 60)
        phi = rng.standard_normal(K)
        p = rng.random(K); p /= p.sum()
        moments.append(frame_moments(phi, phi.copy(), p))
    agg = aggregate_within_frame(moments)
    assert np.isclose(agg.C_raw, agg.V, atol=1e-10)
    assert np.isclose(agg.sigma_matched ** 2, agg.V, atol=1e-10)
    assert np.isclose(agg.J_clean, 1.0, atol=1e-9)


def test_cauchy_schwarz_per_state():
    """|J_t| <= 1 + tol for any channel (Cauchy-Schwarz)."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        K = rng.integers(3, 40)
        phi = rng.standard_normal(K)
        psi = 0.4 * phi + rng.standard_normal(K)   # arbitrary partially-correlated channel
        p = rng.random(K); p /= p.sum()
        j = frame_moments(phi, psi, p).j_t()
        assert abs(j) <= 1.0 + 1e-9, j
