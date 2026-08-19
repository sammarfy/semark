"""Branch-intervention FEASIBILITY probe (spec §5.1, Phase D). RUNS ON THE GPU HOST.

Before any C/J estimate we must be able to (a) force a chosen semantic candidate at a frozen
prefix and (b) reproduce the natural path when the forced token equals the sampled one. This
probe tests whether Qwen's generation exposes a faithful forcing lever (HF
`prefix_allowed_tokens_fn` / logits processor), and reports feasibility. If forcing cannot
reproduce the base path, branch rollouts are untrusted and C/J = NOT IDENTIFIED (spec §1C/§5.6)
— never substitute pooled covariance.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prewm.branch import natural_reproduction_ok  # noqa: E402


def _capture_sampled(adapter):
    """Return a hook handle list + a growing list that records decode-step semantic tokens."""
    talker = adapter.model.model.talker
    seq = []

    def pre(_m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.detach().shape[-1] == 1:
            seq.append(int(ids.detach().reshape(-1)[-1].item()))

    return talker.register_forward_pre_hook(pre, with_kwargs=True), seq


def probe(adapter, text="The morning train arrived exactly on time.", seed=0):
    import torch
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        import urllib.request
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")

    # 1) base run, capture sampled semantic token sequence
    h, base_seq = _capture_sampled(adapter)
    torch.manual_seed(seed)
    try:
        adapter.model.generate_voice_clone(text=text, language="English", voice_clone_prompt=vp)
    finally:
        h.remove()
    base = np.array(base_seq, dtype=np.int64)
    print(f"[branch-probe] base sampled {base.size} semantic tokens; first 12: {base[:12].tolist()}")

    # 2) try to force the SAME tokens via prefix_allowed_tokens_fn and check reproduction
    forced_seq_holder = {}

    def make_force_fn(target):
        # HF calls fn(batch_id, input_ids) -> list of allowed next-token ids.
        # We index the target by how many semantic tokens have been produced so far. The exact
        # index mapping is model-specific; RUNTIME-VERIFY against base reproduction below.
        def fn(_batch_id, input_ids):
            step = int(input_ids.shape[-1])   # RUNTIME-VERIFY: is this the semantic step index?
            i = step - (input_ids.shape[-1] - len(forced_seq_holder.get("seen", [])))
            idx = len(forced_seq_holder.setdefault("seen", []))
            forced_seq_holder["seen"].append(step)
            return [int(target[idx])] if idx < len(target) else list(range(adapter.talker_vocab_size))
        return fn

    feasible, err = True, None
    h2, forced_seq = _capture_sampled(adapter)
    torch.manual_seed(seed)
    try:
        adapter.model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=vp,
            prefix_allowed_tokens_fn=make_force_fn(base))
    except TypeError as e:
        feasible, err = False, f"generate_voice_clone rejects prefix_allowed_tokens_fn: {e}"
    except Exception as e:
        feasible, err = False, f"forcing raised: {type(e).__name__}: {e}"
    finally:
        h2.remove()

    verdict = {"forcing_kwarg_accepted": feasible, "error": err}
    if feasible and forced_seq:
        rep = natural_reproduction_ok(base, np.array(forced_seq, dtype=np.int64))
        verdict.update(rep)
        print(f"[branch-probe] forced-natural reproduction: token_match={rep['token_match']:.3f} ok={rep['ok']}")
    else:
        print(f"[branch-probe] forcing not usable via this lever: {err}")

    print("\n[branch-probe] FEASIBILITY:",
          "branch rollouts look viable — build the pilot" if verdict.get("ok")
          else "prefix_allowed_tokens_fn path did NOT reproduce base; inspect talker.generate for a "
               "lower-level force hook (logits_processor / custom sampling). If none is faithful, "
               "report C = NOT IDENTIFIED (spec §5.6).")
    print("Paste this verdict back:", verdict)
    return verdict


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    probe(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
