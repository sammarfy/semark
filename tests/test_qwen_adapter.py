"""§16.A: adapter/model tests. Skipped unless the model is loaded on a GPU host.

These run on Colab after `SEMARK_MODEL_READY=1` is set and fixtures provide a loaded
adapter (see experiments/run_e0_generate.py for the loading recipe).
"""
import os

import pytest

torch = pytest.importorskip("torch", reason="adapter tests need torch + the model")

MODEL_AVAILABLE = os.environ.get("SEMARK_MODEL_READY") == "1"
model_only = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="set SEMARK_MODEL_READY=1 on the GPU host after loading"
)


@model_only
def test_model_and_tokenizer_load(adapter):
    assert adapter.model is not None
    assert adapter.tokenizer is not None


@model_only
def test_semantic_codebook_exists_and_matches_config(adapter):
    cb = adapter.get_semantic_codebook()
    assert cb.ndim == 2
    V, d = cb.shape
    # Sizes are READ at runtime, not hardcoded (D1: codec semantic codebook is 2048x256).
    assert V == adapter.semantic_vocab_size > 0, (V, adapter.semantic_vocab_size)
    assert d == adapter.semantic_dim > 0
    # D4: talker semantic head may be wider than the codec codebook.
    assert adapter.talker_vocab_size >= V


@model_only
def test_frame_rate_recorded(adapter):
    md = adapter.get_model_metadata()
    assert md["frame_rate_hz"] == 12.5
    assert md["sample_rate_hz"] == 24000


@model_only
def test_generation_reproducible_under_fixed_seed(adapter, sample_text, voice_conditioning):
    a = adapter.generate_trace(sample_text, seed=101, sample_id="t_a",
                               voice_id="repo_default", voice_clone_prompt=voice_conditioning)
    b = adapter.generate_trace(sample_text, seed=101, sample_id="t_b",
                               voice_id="repo_default", voice_clone_prompt=voice_conditioning)
    import numpy as np
    assert np.array_equal(np.asarray(a.semantic_ids), np.asarray(b.semantic_ids))


@model_only
def test_sampling_config_recorded(adapter):
    cfg = adapter.get_sampling_config()
    for k in ("temperature", "top_k", "top_p", "repetition_penalty", "do_sample"):
        assert k in cfg
