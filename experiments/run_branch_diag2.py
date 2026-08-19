"""Test the codec_head force lever (spec §5.1, corrected mechanism). GPU host via diag2(adapter).

`prefix_allowed_tokens_fn` is ignored by Qwen3-TTS; this tests forcing at talker.codec_head
directly. If the constant and targeted forces take effect (and reproduce base when forcing the
natural token), branch rollouts are trustworthy and the pilot can be rebuilt on this lever.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prewm.forcing import (CodecHeadForcer, first_k_schedule,  # noqa: E402
                               prefix_then_force_schedule)


def _gen(adapter, vp, text, seed, forcer_schedule=None, max_new_tokens=None):
    import torch
    talker = adapter.model.model.talker
    toks = []

    def pre(_m, a, k):
        ids = k.get("input_ids", a[0] if a else None)
        if ids is not None and ids.detach().shape[-1] == 1:
            toks.append(int(ids.detach().reshape(-1)[-1].item()))

    h = talker.register_forward_pre_hook(pre, with_kwargs=True)
    torch.manual_seed(seed)
    kw = dict(text=text, language="English", voice_clone_prompt=vp)
    if max_new_tokens is not None:                      # cap: forced runs may never hit eos
        kw["max_new_tokens"] = int(max_new_tokens)
    try:
        if forcer_schedule is not None:
            with CodecHeadForcer(adapter, forcer_schedule) as f:
                wavs, sr = adapter.model.generate_voice_clone(**kw)
                nforced = len(f.forced_log)
        else:
            wavs, sr = adapter.model.generate_voice_clone(**kw)
            nforced = 0
    finally:
        h.remove()
    return np.array(toks, dtype=np.int64), np.asarray(wavs[0], np.float32), nforced


def diag2(adapter, text="Rain fell steadily across the quiet valley throughout the afternoon."):
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        import urllib.request
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")

    base, base_wav, _ = _gen(adapter, vp, text, seed=0)
    print(f"[diag2] base tokens: {base.size}, first 10: {base[:10].tolist()}")

    # A2 — force token 100 for the FIRST 5 steps only, then free (so generation still ends at eos)
    a_tok, _, na = _gen(adapter, vp, text, seed=0, forcer_schedule=first_k_schedule(5, 100),
                        max_new_tokens=int(base.size + 15))
    a_ok = bool(a_tok.size >= 5 and np.all(a_tok[:5] == 100))
    print(f"[diag2] A2 force-first-5=100 (n_forced={na}): first5-all-100={a_ok}; first 10: {a_tok[:10].tolist()}")

    # B2 — teacher-force base[:t], force ALT at t, free after
    t = 6 if base.size > 12 else max(1, base.size // 2)
    alt = int((base[t] + 500) % adapter.semantic_vocab_size)
    cap = int(base.size + 15)
    b_tok, b_wav, _ = _gen(adapter, vp, text, seed=0,
                           forcer_schedule=prefix_then_force_schedule(base, t, alt), max_new_tokens=cap)
    prefix_held = bool(b_tok.size > t and np.array_equal(b_tok[:t], base[:t]))
    tok_forced = bool(b_tok.size > t and b_tok[t] == alt)
    tail_div = bool(not np.array_equal(b_tok[:min(len(b_tok), len(base))],
                                       base[:min(len(b_tok), len(base))]))
    print(f"[diag2] B2 force base[:{t}]+{alt} @t={t} (base[t]={int(base[t])}): "
          f"prefix_held={prefix_held} token_forced={tok_forced} tail_diverges={tail_div}")

    # C2 — natural-token teacher-force must reproduce base (real §5.1 check under a DIFFERENT seed)
    nat = int(base[t])
    n_tok, _, _ = _gen(adapter, vp, text, seed=123,
                       forcer_schedule=prefix_then_force_schedule(base, t, nat), max_new_tokens=cap)
    # prefix + forced natural token must match base up to t (even under a different seed)
    natural_prefix_ok = bool(n_tok.size > t and np.array_equal(n_tok[:t + 1], base[:t + 1]))
    print(f"[diag2] C2 natural teacher-force under seed 123 reproduces base[:{t+1}]: {natural_prefix_ok}")

    # D2 — two different forced tokens -> different re-encoded latent at t
    def zt(wav):
        z = adapter.encode_semantic_space(wav).frame_latents
        return z[t] if t < len(z) else None
    alt2 = int((base[t] + 900) % adapter.semantic_vocab_size)
    _, w1, _ = _gen(adapter, vp, text, seed=42, forcer_schedule=prefix_then_force_schedule(base, t, alt), max_new_tokens=cap)
    _, w2, _ = _gen(adapter, vp, text, seed=42, forcer_schedule=prefix_then_force_schedule(base, t, alt2), max_new_tokens=cap)
    z1, z2 = zt(w1), zt(w2)
    ld = float(np.linalg.norm(z1 - z2)) if (z1 is not None and z2 is not None) else None
    print(f"[diag2] D2 ||z(t;alt) - z(t;alt2)|| = {ld}")

    verdict = {"constant_force_works": a_ok, "prefix_held": prefix_held, "token_forced": tok_forced,
               "tail_diverges": tail_div, "natural_teacherforce_reproduces": natural_prefix_ok,
               "latent_diff_across_tokens": ld,
               "forcing_effective": bool(a_ok and tok_forced and prefix_held and natural_prefix_ok
                                         and (ld or 0) > 1e-4)}
    print("\n[diag2] VERDICT:", verdict)
    print("[diag2] =>", "codec_head forcing WORKS — rebuild the pilot on this lever."
          if verdict["forcing_effective"] else
          "codec_head forcing still not faithful; inspect where semantic sampling actually happens.")
    return verdict


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    diag2(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
