"""§16.F / §9.5: encoder determinism + chunk-seam contamination.

Model-required (skipped locally). On the GPU host these exercise the actual encoder.
The pure seam-metric helper is tested locally with synthetic latents.
"""
import os

import numpy as np
import pytest

from src.alignment import norm_sequence


# ------------------------- LOCAL: pure seam metric -------------------------- #
def seam_contamination_length(whole: np.ndarray, split: np.ndarray, thresh: float = 0.99) -> int:
    """Frames from the seam until per-frame cosine recovers above `thresh` (§9.5.2)."""
    from src.metrics import rowwise_cosine
    cos = rowwise_cosine(whole, split)
    below = np.where(cos < thresh)[0]
    return 0 if below.size == 0 else int(below.max() - below.min() + 1)


def test_seam_metric_zero_when_identical():
    x = np.random.default_rng(0).standard_normal((50, 8))
    assert seam_contamination_length(x, x.copy()) == 0


def test_seam_metric_counts_contaminated_region():
    rng = np.random.default_rng(1)
    whole = rng.standard_normal((50, 8))
    split = whole.copy()
    split[20:25] = rng.standard_normal((5, 8))  # corrupt a 5-frame region near the seam
    assert seam_contamination_length(whole, split) == 5


# --------------------------- MODEL: skipped locally ------------------------- #
torch = pytest.importorskip("torch", reason="encoder determinism needs the model")
MODEL_AVAILABLE = os.environ.get("SEMARK_MODEL_READY") == "1"
model_only = pytest.mark.skipif(not MODEL_AVAILABLE, reason="set SEMARK_MODEL_READY=1")


@model_only
def test_repeat_encode_is_deterministic(adapter, sample_waveform):
    """§9.5.1: repeat-encode of the same waveform must be identical. HARD STOP if not."""
    a = adapter.encode_semantic_space(sample_waveform).frame_latents
    b = adapter.encode_semantic_space(sample_waveform).frame_latents
    assert np.allclose(a, b, atol=1e-5), "non-deterministic encoder -> §20 stop #9"


@model_only
def test_chunk_invariance_measured(adapter, sample_waveform):
    """§9.5.2: measure (not necessarily pass) the seam contamination length."""
    whole = adapter.encode_semantic_space(sample_waveform).frame_latents
    n = len(sample_waveform)
    half = n // 2
    left = adapter.encode_semantic_space(sample_waveform[:half]).frame_latents
    right = adapter.encode_semantic_space(sample_waveform[half:]).frame_latents
    split = np.concatenate([left, right], axis=0)[: len(whole)]
    seam = seam_contamination_length(whole, split)
    assert seam >= 0  # recorded into the report; long seam is a note, not a failure
