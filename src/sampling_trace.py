"""Generation-distribution tracing (Notion §7, §7.1).

Two distributions are captured per semantic generation step:

    p_t^raw   = softmax(logits after penalties & temperature, BEFORE top-k/top-p)
    p_t^final = softmax(logits after top-k/top-p)          (the one actually sampled)

Rationale (§7.1): with a fixed top-k the support size is constant and carries no
information; the actionable quantity is the PAIR of entropies. If H(raw) is healthy
but H(final) is small, truncation is destroying watermark capacity.

This module holds the PURE reconstruction math (numpy, no torch) plus the data
structures. The actual per-step logit capture during Qwen generation lives in
adapters/qwen3_tts.py, which reuses `apply_warpers` below so the reconstruction that
unit tests verify is byte-for-byte the reconstruction used at generation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Data structures (§4 required schema). Fields are typed Any so this module     #
# does not import torch; tensors may be torch or numpy at runtime.             #
# --------------------------------------------------------------------------- #
@dataclass
class GenerationTrace:
    sample_id: str
    text: str
    voice_id: str
    seed: int
    waveform: Any
    sample_rate: int
    semantic_ids: Any                 # [T]
    all_codec_ids: Any = None         # [K, T] if available
    final_semantic_logits: Any = None  # dense [T, V] OR reconstructable support form
    final_semantic_probs: Any = None   # optional
    raw_semantic_logits: Any = None    # post-temperature, PRE-truncation (§7.1)
    raw_semantic_probs: Any = None     # dense [T, V] fp16 (cheap at V=4096)
    support_ids: Any = None            # [T][variable] truncated support per step
    support_logits: Any = None         # matching final logits over support
    valid_frame_mask: Any = None       # [T]
    sampling_metadata: dict = field(default_factory=dict)
    model_metadata: dict = field(default_factory=dict)


@dataclass
class SemanticSpace:
    codebook: Any                      # [V, d]
    frame_latents: Any                 # [T, d]
    semantic_ids: Any                  # [T]
    valid_frame_mask: Any              # [T]
    coordinate_system_name: str
    hook_metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure reconstruction math (numpy). Used by BOTH generation and unit tests.     #
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: Optional[float]) -> np.ndarray:
    if temperature is None or temperature == 1.0:
        return np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return np.asarray(logits, dtype=np.float64) / float(temperature)


def apply_top_k(logits: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    """Mask all but the top_k logits to -inf (per row). None/<=0 disables."""
    x = np.array(logits, dtype=np.float64)
    if not top_k or top_k <= 0 or top_k >= x.shape[-1]:
        return x
    # threshold = k-th largest value per row
    kth = np.sort(x, axis=-1)[..., -top_k][..., None]
    x[x < kth] = -np.inf
    return x


def apply_top_p(logits: np.ndarray, top_p: Optional[float]) -> np.ndarray:
    """Nucleus filtering: keep the smallest set whose cumulative prob >= top_p.

    Matches the standard HF convention: the first token crossing the threshold is
    kept (so at least one token always survives).
    """
    if top_p is None or top_p >= 1.0 or top_p <= 0.0:
        return np.asarray(logits, dtype=np.float64)
    x = np.asarray(logits, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1]).copy()
    for i in range(flat.shape[0]):
        row = flat[i]
        order = np.argsort(row)[::-1]
        probs = softmax(row[order])
        cum = np.cumsum(probs)
        # HF convention: remove where cumulative prob exceeds top_p, but keep the first
        # token that crosses the threshold (shift the mask right by one, never drop #0).
        remove_sorted = cum > top_p
        remove_sorted[1:] = remove_sorted[:-1].copy()
        remove_sorted[0] = False
        row[order[remove_sorted]] = -np.inf
        flat[i] = row
    return flat.reshape(x.shape)


def reconstruct_raw_and_final(
    logits: np.ndarray,
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """From PRE-warp model logits, reconstruct (p_raw, p_final) per §7.1.

    p_raw  = softmax(temperature(logits))                 [BEFORE top-k/top-p]
    p_final= softmax(top_p(top_k(temperature(logits))))   [what is sampled]

    Note: repetition penalty (if any) is context-dependent and applied by the model
    before this point; callers must record whether penalties != identity and, if so,
    capture logits AFTER penalties so this reconstruction stays exact.
    """
    tempered = apply_temperature(logits, temperature)
    p_raw = softmax(tempered)
    truncated = apply_top_p(apply_top_k(tempered, top_k), top_p)
    p_final = softmax(truncated)
    return p_raw, p_final


def entropy_pair_ok(p_raw: np.ndarray, p_final: np.ndarray, tol: float = 1e-9) -> bool:
    """Sanity invariant (§16.D): H(p_raw) >= H(p_final) for every step.

    Truncation only removes mass, so the truncated-and-renormalized distribution
    cannot have higher entropy than the pre-truncation one.
    """
    from .metrics import entropy  # local import to avoid cycle at module load

    hr = entropy(p_raw, axis=-1)
    hf = entropy(p_final, axis=-1)
    return bool(np.all(hr + tol >= hf))
