"""E0-c discovery probe (spec §2). RUNS ON THE GPU HOST.

Goal: establish the exact talker(3072) -> codec-semantic(2048) map from the REAL generation
path, using the codec ids Qwen sends DOWNSTREAM (not waveform re-encoding). This script is a
DISCOVERY probe: it prints the generation source, special-token config, and empirical
(talker_id, codec_id) observations so the map rule can be pinned. It writes a candidate
artifacts/e0c/map.json only if the per-frame alignment is clean and unambiguous.

Do not trust a map inferred from constants — it must be observed from the generation path.
"""
from __future__ import annotations

import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _src(obj, limit=6000):
    try:
        return inspect.getsource(obj)[:limit]
    except Exception as e:
        return f"<no source: {e}>"


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    adapter = SpeechLMAdapter.load(cfg, device="cuda:0")
    probe(adapter)


def probe(adapter):
    """Run the E0-c discovery probe on an already-loaded adapter (no model reload)."""
    import torch
    talker = adapter.model.model.talker
    head = adapter._codec_head()

    print("=" * 70, "\n[1] CONFIG / SPECIAL TOKENS")
    for name in ("config", "generation_config"):
        obj = getattr(adapter.model, name, None) or getattr(adapter.model.model, name, None)
        if obj is not None:
            for k in ("vocab_size", "codec_vocab_size", "semantic_vocab_size", "num_codebooks",
                      "bos_token_id", "eos_token_id", "pad_token_id", "codec_bos_token_id",
                      "codec_eos_token_id", "codec_pad_token_id", "codec_nothing_id",
                      "codec_special_tokens", "semantic_codebook_size", "codec_offset"):
                if hasattr(obj, k):
                    print(f"  {name}.{k} = {getattr(obj, k)}")
    print("  head.out_features =", head.out_features)

    print("=" * 70, "\n[2] generate_voice_clone SOURCE (truncated)")
    print(_src(adapter.model.generate_voice_clone))
    print("=" * 70, "\n[3] talker class + generate/forward SOURCE (truncated)")
    print("talker type:", type(talker).__name__)
    for meth in ("generate", "forward", "_sample_codec", "sample"):
        if hasattr(talker, meth):
            print(f"--- talker.{meth} ---")
            print(_src(getattr(talker, meth), 3500))

    print("=" * 70, "\n[4] EMPIRICAL (sampled talker id -> codec id downstream)")
    # capture per-step semantic logits (argmax as a proxy for the sampled id) ...
    steps = []
    def head_hook(_m, _i, out):
        lg = out[0] if isinstance(out, tuple) else out
        steps.append(int(lg.detach()[..., -1, :].argmax().item()))
    # ... and the codes tensor handed to the codec decoder.
    codes_cap = {}
    dec = None
    for n, m in adapter.tokenizer.model.named_modules():
        if n.endswith("decoder") and hasattr(m, "forward"):
            dec = m; break
    def dec_pre(_m, a, k):
        codes_cap.setdefault("codes", a[0] if a else next(iter(k.values()), None))
    h1 = head.register_forward_hook(head_hook)
    h2 = dec.register_forward_pre_hook(dec_pre, with_kwargs=True) if dec is not None else None
    import urllib.request
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")
    torch.manual_seed(0)
    wavs, sr = adapter.model.generate_voice_clone(
        text="The morning train arrived exactly on time.", language="English", voice_clone_prompt=vp)
    h1.remove()
    if h2: h2.remove()

    print("talker head argmax per step (first 24):", steps[:24])
    codes = codes_cap.get("codes")
    if codes is not None:
        c = codes.detach().cpu().numpy() if hasattr(codes, "detach") else np.asarray(codes)
        print("codes-at-decoder shape:", c.shape)
        # try to locate the semantic column (values in 0..2047) — usually column 0
        print("codes sample [:, first few]:", c.reshape(c.shape[-2], c.shape[-1])[:6] if c.ndim >= 2 else c[:6])
    else:
        print("!! did not capture codes at the decoder — inspect [2]/[3] source to find the decode call.")

    # ALSO: the authoritative internal path is what the talker emits + how it is remapped.
    # Re-encoding the OUTPUT waveform is NOT authoritative (only ~96% exact) and is not used here.
    os.makedirs("artifacts/e0c", exist_ok=True)
    json.dump({"talker_head_argmax_first24": steps[:24],
               "note": "discovery only; map.json is written after the flow is confirmed"},
              open("artifacts/e0c/probe_raw.json", "w"), indent=2)
    print("\nWrote artifacts/e0c/probe_raw.json. Paste sections [1]-[4] back for map finalization.")


if __name__ == "__main__":
    main()
