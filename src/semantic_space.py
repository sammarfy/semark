"""Canonical semantic-space extraction (Notion §5, §11).

Confirmed at runtime (Qwen3-TTS-12Hz-0.6B, via dump probes):

  tokenizer wrapper (Qwen3TTSTokenizer) .model -> Qwen3TTSTokenizerV2Model
    .encoder.quantizer                              MimiSplitResidualVectorQuantizer
      .semantic_residual_vector_quantizer           MimiResidualVectorQuantizer  <-- semantic
        .input_proj  Conv1d(512, 256)               (frozen projection, §5.2)
        .layers[0].codebook  MimiEuclideanCodebook  .embed -> [2048, 256]
      .acoustic_residual_vector_quantizer           (31 layers, ignored here)

  tokenizer.encode(x) -> {'audio_codes': [Tensor[T, 16]]};  column 0 is the semantic id.
  srvq.encode(pre_proj_latent) reproduces that column-0 id  (the E0 invariant, §11).

The semantic codebook size is read from the model (2048 here), NOT hardcoded.
torch is imported lazily so the rest of the package stays importable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Graded acceptance policy (§11). Pure; unit-testable without torch.            #
# --------------------------------------------------------------------------- #
def graded_agreement_decision(
    rate: float, pass_min: float = 0.999, investigate_min: float = 0.95
) -> str:
    """Return 'PASS' | 'INVESTIGATE' | 'HARD_STOP' per Notion §11.

    NEVER replace this with a hard threshold check on an agreement rate directly in
    experiment code (tests enforce the absence of such a check).
    """
    if rate >= pass_min:
        return "PASS"
    if rate >= investigate_min:
        return "INVESTIGATE"
    return "HARD_STOP"


@dataclass
class QuantizerHandles:
    semantic_rvq: Any            # MimiResidualVectorQuantizer (the semantic stream)
    first_layer: Any             # semantic_rvq.layers[0]  (MimiVectorQuantization)
    codebook_module: Any         # MimiEuclideanCodebook (.embed -> [V, d])
    input_proj: Optional[Any]    # Conv1d(512, 256) or None
    quantize_fn: Any             # srvq.encode : pre-proj latent -> ids
    discovered_path: str
    notes: list = field(default_factory=list)


def _module_tree(model, max_depth: int = 4) -> str:
    lines = []
    for name, mod in model.named_modules():
        if name.count(".") <= max_depth:
            lines.append(f"{'  ' * name.count('.')}{name or '<root>'}: {type(mod).__name__}")
    return "\n".join(lines)


def _nn_root(obj):
    """Return the underlying nn.Module for a wrapper object (or obj itself)."""
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        return obj
    inner = getattr(obj, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    for _, v in vars(obj).items():
        if isinstance(v, nn.Module):
            return v
    return None


def codebook_embed(cbmod):
    """Read the [V, d] centroids from a MimiEuclideanCodebook (property or buffers)."""
    if hasattr(cbmod, "embed"):
        return cbmod.embed
    if hasattr(cbmod, "embed_sum") and hasattr(cbmod, "cluster_usage"):
        return cbmod.embed_sum / cbmod.cluster_usage.clamp(min=1e-8)[:, None]
    if hasattr(cbmod, "weight"):
        return cbmod.weight
    raise RuntimeError(f"cannot read codebook centroids from {type(cbmod).__name__}")


def locate_semantic_quantizer(tokenizer_wrapper) -> QuantizerHandles:
    """Discover the encode-path semantic quantizer. Fails loudly with a module dump."""
    import torch  # noqa: F401  (lazy; GPU host only)

    root = _nn_root(tokenizer_wrapper)
    if root is None:
        raise RuntimeError(
            f"tokenizer wrapper {type(tokenizer_wrapper).__name__} exposes no nn.Module "
            "(expected a `.model` attribute). Cannot locate the semantic quantizer.")

    # confirmed path, with a search fallback
    srvq = None
    enc = getattr(root, "encoder", None)
    q = getattr(enc, "quantizer", None) if enc is not None else None
    if q is not None:
        srvq = getattr(q, "semantic_residual_vector_quantizer", None)
    if srvq is None:
        for name, mod in root.named_modules():
            if name.endswith("semantic_residual_vector_quantizer"):
                srvq = mod
                break
    if srvq is None:
        raise RuntimeError(
            "Could not find `semantic_residual_vector_quantizer` (§20 stop #1/#2).\n"
            + _module_tree(root))

    layers = list(getattr(srvq, "layers", []))
    if not layers:
        raise RuntimeError("Semantic RVQ has no `.layers`.\n" + _module_tree(srvq))
    first = layers[0]
    cbmod = getattr(first, "codebook", None) or getattr(first, "_codebook", None)
    if cbmod is None:
        raise RuntimeError("No `.codebook` on the semantic VQ layer.\n" + _module_tree(first))
    _ = codebook_embed(cbmod)  # validate readable now

    input_proj = getattr(srvq, "input_proj", None)
    if not hasattr(srvq, "encode") or not callable(srvq.encode):
        raise RuntimeError("Semantic RVQ has no real `.encode()`; §5 forbids substituting "
                           "a hand-written nearest-centroid.")

    notes = [
        "path: model.encoder.quantizer.semantic_residual_vector_quantizer",
        f"input_proj present: {input_proj is not None}",
        "quantize via srvq.encode(pre_proj_latent); latents via input_proj forward hook",
    ]
    return QuantizerHandles(
        semantic_rvq=srvq, first_layer=first, codebook_module=cbmod,
        input_proj=input_proj, quantize_fn=srvq.encode,
        discovered_path="model.encoder.quantizer.semantic_residual_vector_quantizer",
        notes=notes)
