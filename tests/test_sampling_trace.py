"""§16.D: sampling-trace reconstruction and the raw>=final entropy invariant."""
import numpy as np

from src import metrics
from src.sampling_trace import (
    softmax,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    reconstruct_raw_and_final,
    entropy_pair_ok,
)


def test_softmax_sums_to_one():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 4096))
    p = softmax(logits)
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_reconstruct_from_support_matches_dense():
    """softmax over the retained support == dense final probs on that support."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(4096)
    _, p_final = reconstruct_raw_and_final(logits, temperature=1.0, top_k=50, top_p=1.0)
    support = np.where(p_final > 0)[0]
    # Reconstruct from the stored support logits only.
    support_logits = logits[support]
    recon = softmax(support_logits)
    assert np.allclose(recon, p_final[support])
    assert np.isclose(p_final.sum(), 1.0)
    assert support.size == 50


def test_sampled_id_in_support():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal(4096)
    _, p_final = reconstruct_raw_and_final(logits, temperature=0.9, top_k=40, top_p=0.95)
    support = set(np.where(p_final > 0)[0].tolist())
    # draw a few samples; each must lie in the retained support
    draws = np.random.default_rng(3).choice(4096, size=200, p=p_final)
    assert set(draws.tolist()).issubset(support)


def test_raw_ge_final_entropy_invariant():
    """§16.D: H(p_raw) >= H(p_final) for every step, since truncation removes mass."""
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((64, 4096))
    p_raw, p_final = reconstruct_raw_and_final(logits, temperature=1.0, top_k=64, top_p=0.9)
    hr = metrics.entropy(p_raw, axis=-1)
    hf = metrics.entropy(p_final, axis=-1)
    assert np.all(hr + 1e-9 >= hf)
    assert entropy_pair_ok(p_raw, p_final)


def test_temperature_monotone_entropy():
    """Higher temperature -> higher entropy of p_raw (no truncation)."""
    rng = np.random.default_rng(5)
    logits = rng.standard_normal(4096)
    h_lo = metrics.entropy(softmax(apply_temperature(logits, 0.5)))
    h_hi = metrics.entropy(softmax(apply_temperature(logits, 2.0)))
    assert h_hi > h_lo


def test_top_k_support_size():
    logits = np.linspace(0, 10, 4096)
    filtered = apply_top_k(logits, 17)
    assert np.isfinite(filtered).sum() == 17


def test_top_p_keeps_at_least_one():
    logits = np.array([10.0, 0.0, 0.0, 0.0])
    filtered = apply_top_p(logits, 0.0001)
    assert np.isfinite(filtered).sum() >= 1


def test_top_p_nucleus_threshold():
    # probs ~ [0.6439, 0.2369, 0.0871, 0.0321]; top_p=0.8 keeps the first two.
    logits = np.log(np.array([0.64, 0.24, 0.09, 0.03]))
    filtered = apply_top_p(logits, 0.8)
    assert np.isfinite(filtered[0]) and np.isfinite(filtered[1])
    assert not np.isfinite(filtered[2]) and not np.isfinite(filtered[3])
