"""Branch-forcing DISCRIMINATING diagnostic (spec §5.1, corrected). GPU host via diag(adapter).

The earlier §5.1 check was too weak: forcing the NATURAL token under the BASE seed reproduces
the base path whether or not forcing does anything (same seed -> same free generation). So a
no-op forcing passed it, and the pilot's J_clean=0 (machine-zero psi) is a forcing artifact,
not a result. This diagnostic proves whether forcing actually changes the output:

  Test A (constant force): force a FIXED ordinary token at every step -> do the emitted tokens
    become that constant? If not, prefix_allowed_tokens_fn is being ignored.
  Test B (counterfactual): force base[:t] + ALT at t (ALT != base[t]) -> does token t become
    ALT, does the prefix stay, and does the tail diverge from base?
  Test C (frame-in-range): after forcing, is the re-encoded frame t in range, and does the
    detector latent at t actually differ across two different forced tokens?

Prints a verdict. Only if A/B/C pass is branch intervention trustworthy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _force_fn(prefix, v, t, talker_vocab):
    state = {"c": 0}
    allow_all = list(range(talker_vocab))

    def fn(_b, _ids):
        c = state["c"]; state["c"] += 1
        if t is None:
            return [int(v)]                 # constant force at every step
        if c < t:
            return [int(prefix[c])]
        if c == t:
            return [int(v)]
        return allow_all
    return fn


def _gen_capture(adapter, vp, text, seed, force=None):
    import torch
    talker = adapter.model.model.talker
    toks = []

    def pre(_m, a, k):
        ids = k.get("input_ids", a[0] if a else None)
        if ids is not None and ids.detach().shape[-1] == 1:
            toks.append(int(ids.detach().reshape(-1)[-1].item()))

    h = talker.register_forward_pre_hook(pre, with_kwargs=True)
    torch.manual_seed(seed)
    try:
        kw = dict(text=text, language="English", voice_clone_prompt=vp)
        if force is not None:
            kw["prefix_allowed_tokens_fn"] = force
        wavs, sr = adapter.model.generate_voice_clone(**kw)
    finally:
        h.remove()
    return np.array(toks, dtype=np.int64), np.asarray(wavs[0], np.float32)


def diag(adapter, text="Rain fell steadily across the quiet valley throughout the afternoon."):
    tv = adapter.talker_vocab_size
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        import urllib.request
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")

    base, base_wav = _gen_capture(adapter, vp, text, seed=0)
    print(f"[diag] base tokens: {base.size}, first 10: {base[:10].tolist()}")

    # Test A — constant force to token 100 at every step
    a_tok, _ = _gen_capture(adapter, vp, text, seed=0, force=_force_fn(None, 100, None, tv))
    a_const = bool(a_tok.size and np.all(a_tok == 100))
    print(f"[diag] TEST A constant-force==100: all-equal-100={a_const}; first 10: {a_tok[:10].tolist()}")

    # Test B — counterfactual at t=6
    t = 6 if base.size > 12 else max(1, base.size // 2)
    alt = int((base[t] + 500) % adapter.semantic_vocab_size)
    b_tok, b_wav = _gen_capture(adapter, vp, text, seed=0, force=_force_fn(base, alt, t, tv))
    prefix_held = bool(np.array_equal(b_tok[:t], base[:t]))
    tok_t_forced = bool(b_tok.size > t and b_tok[t] == alt)
    tail_diverges = bool(b_tok.size != base.size or not np.array_equal(b_tok[:min(len(b_tok), len(base))],
                                                                       base[:min(len(b_tok), len(base))]))
    print(f"[diag] TEST B force base[:{t}]+{alt} @t={t} (base[t]={int(base[t])}): "
          f"prefix_held={prefix_held} token_t_forced={tok_t_forced} tail_diverges={tail_diverges}")

    # Test C — does the re-encoded detector latent at t differ across two forced tokens?
    def zt(wav):
        z = adapter.encode_semantic_space(wav).frame_latents
        return z[t] if t < len(z) else None
    alt2 = int((base[t] + 900) % adapter.semantic_vocab_size)
    _, w1 = _gen_capture(adapter, vp, text, seed=42, force=_force_fn(base, alt, t, tv))
    _, w2 = _gen_capture(adapter, vp, text, seed=42, force=_force_fn(base, alt2, t, tv))
    z1, z2 = zt(w1), zt(w2)
    in_range = z1 is not None and z2 is not None
    latent_diff = float(np.linalg.norm(z1 - z2)) if in_range else None
    print(f"[diag] TEST C frame t in range: {in_range}; ||z(t;alt) - z(t;alt2)|| = {latent_diff}")

    verdict = {"constant_force_works": a_const, "prefix_held": prefix_held,
               "token_t_forced": tok_t_forced, "tail_diverges": tail_diverges,
               "frame_t_in_range": in_range, "latent_diff_across_tokens": latent_diff,
               "forcing_effective": bool(a_const and tok_t_forced and prefix_held
                                         and in_range and (latent_diff or 0) > 1e-4)}
    print("\n[diag] VERDICT:", verdict)
    if not verdict["forcing_effective"]:
        print("[diag] => prefix_allowed_tokens_fn is NOT a faithful force lever here. Branch C/J = "
              "NOT IDENTIFIED until a real force hook (logits_processor / direct codec_head "
              "intervention) is found. Paste this verdict back.")
    else:
        print("[diag] => forcing is real; the pilot can be trusted (rebuild with the fixed §5.1 check).")
    return verdict


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    diag(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
