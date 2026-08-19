"""Locate where a forced semantic token shows up in the re-encode (spec §5 alignment). GPU host.

Diagnoses why psi was constant across candidates: forces 3 distinct tokens at one anchor with
FULL right-context (free continuation, capped — no eos boundary near t), re-encodes, and reports
(a) at which re-encode frame index the candidates diverge, and (b) the re-encoded semantic ids
around t. This tells us the semantic-token -> re-encode-frame offset and whether forcing changes
the audio at all.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prewm.forcing import CodecHeadForcer, prefix_then_force_schedule  # noqa: E402


def align(adapter, text="Rain fell steadily across the quiet valley throughout the afternoon."):
    import torch
    talker = adapter.model.model.talker
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        import urllib.request
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")

    def gen(sched=None, seed=0, mnt=None):
        toks = []
        def pre(_m, a, k):
            ids = k.get("input_ids", a[0] if a else None)
            if ids is not None and ids.detach().shape[-1] == 1:
                toks.append(int(ids.detach().reshape(-1)[-1].item()))
        h = talker.register_forward_pre_hook(pre, with_kwargs=True); torch.manual_seed(seed)
        kw = dict(text=text, language="English", voice_clone_prompt=vp)
        if mnt is not None:
            kw["max_new_tokens"] = int(mnt)
        try:
            if sched is not None:
                with CodecHeadForcer(adapter, sched):
                    wavs, _ = adapter.model.generate_voice_clone(**kw)
            else:
                wavs, _ = adapter.model.generate_voice_clone(**kw)
        finally:
            h.remove()
        return np.asarray(wavs[0], np.float32), np.array(toks, np.int64)

    base_wav, base_tok = gen()
    base_frames = base_tok[1:]
    base_ids = np.asarray(adapter.encode_semantic_space(base_wav).semantic_ids)
    T = len(base_frames)
    t = min(T // 2, T - 8)                        # a mid interior anchor
    cap = int(T + 30)
    three = [int((base_frames[t] + d) % adapter.semantic_vocab_size) for d in (200, 500, 900)]
    print(f"[align] base frames={T}, anchor t={t}, base_frames[t]={int(base_frames[t])}, forcing {three}")

    Zs, IDs = [], []
    for v in three:
        wav, _ = gen(prefix_then_force_schedule(base_frames, t, v), seed=7, mnt=cap)  # full context
        sp = adapter.encode_semantic_space(wav)
        Zs.append(sp.frame_latents); IDs.append(np.asarray(sp.semantic_ids))
    L = min(len(z) for z in Zs)
    # per-frame max latent difference across the 3 candidate re-encodes
    dif = np.zeros(L)
    for i in (1, 2):
        dif = np.maximum(dif, np.linalg.norm(Zs[0][:L] - Zs[i][:L], axis=1))
    idx = int(np.argmax(dif))
    print(f"[align] per-frame latent diff across candidates: argmax@frame {idx} (value {dif[idx]:.3f})")
    print(f"[align] diff near t: {np.round(dif[max(0,t-3):t+8], 2).tolist()} (indices {max(0,t-3)}..{t+7})")
    for v, ids in zip(three, IDs):
        seg = ids[max(0, t - 2):t + 5].tolist()
        print(f"[align] v={v}: reenc semantic ids[t-2:t+5]={seg}")
    print(f"[align] base reenc ids[t-2:t+5]={base_ids[max(0,t-2):t+5].tolist()}")
    print("[align] => offset = (argmax frame) - t =", idx - t,
          "| forcing changes audio:", bool(dif[idx] > 1e-3))
    return {"t": t, "argmax_frame": idx, "offset": idx - t, "max_diff": float(dif[idx])}


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    align(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
