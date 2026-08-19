"""spec §12: the saved final p_t equals the distribution passed to the sampler
(temperature/top-k/top-p already applied), and it is a valid probability vector."""
import numpy as np

from src.sampling_trace import reconstruct_raw_and_final, softmax
from src.prewm.keys import diagnostic_keys


def test_final_probs_sum_to_one_over_support():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(3072)
    _, p_final = reconstruct_raw_and_final(logits, temperature=0.9, top_k=50, top_p=1.0)
    assert np.isclose(p_final.sum(), 1.0)
    assert (p_final >= 0).all()
    assert (p_final > 0).sum() == 50            # top-k applied


def test_support_reconstruction_matches_dense():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(3072)
    temp = 0.9
    _, p_final = reconstruct_raw_and_final(logits, temperature=temp, top_k=50, top_p=1.0)
    sup = np.where(p_final > 0)[0]
    # reconstruct from stored RAW support logits with the SAME temperature (the E0-d1 path)
    recon = softmax(logits[sup] / temp)
    assert np.allclose(recon, p_final[sup])


def test_diagnostic_keys_deterministic_and_unit():
    a = diagnostic_keys(64, 256)
    b = diagnostic_keys(64, 256)
    assert np.array_equal(a, b)                 # same fixed key list reused
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert a.shape == (64, 256)
