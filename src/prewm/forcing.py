"""Real force lever: overwrite talker.codec_head logits at decode steps (spec §5).

`prefix_allowed_tokens_fn` is ignored by Qwen3-TTS generation, so we force the semantic token
at its source. A forward hook on `codec_head` replaces the last-position logits at each DECODE
step (seq length 1) with a spike on the forced token; prefill calls (seq length > 1) are left
untouched and not counted. This is upstream of HF sampling, so the forced token is what gets
sampled — verified by run_branch_diag2 before any C/J estimate.
"""
from __future__ import annotations

from typing import Callable, Optional


class CodecHeadForcer:
    """Context manager. schedule(decode_step) -> forced token id, or None to leave free."""

    def __init__(self, adapter, schedule: Callable[[int], Optional[int]], spike: float = 30.0):
        self.head = adapter._codec_head()
        self.schedule = schedule
        self.spike = spike
        self.decode_step = 0
        self.handle = None
        self.forced_log: list[tuple[int, int]] = []   # (decode_step, forced_token)

    def _hook(self, _module, _inp, out):
        lg = out[0] if isinstance(out, tuple) else out
        # decode step: the model emits one next-token position (seq length 1 with kv-cache)
        if lg.shape[-2] == 1:
            tok = self.schedule(self.decode_step)
            if tok is not None:
                lg[..., -1, :] = -1e9
                lg[..., -1, int(tok)] = self.spike
                self.forced_log.append((self.decode_step, int(tok)))
            self.decode_step += 1
        return out

    def __enter__(self):
        self.decode_step = 0
        self.forced_log = []
        self.handle = self.head.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


def const_schedule(token: int):
    return lambda _step: int(token)


def first_k_schedule(k: int, token: int):
    """Force `token` for the first k decode steps, then free (so generation still hits eos)."""
    return lambda step: int(token) if step < k else None


def single_step_schedule(t: int, token: int):
    """Force `token` at decode step t only; free elsewhere."""
    return lambda step: int(token) if step == t else None


def branch_schedule(prefix_tokens, t: int, token: int, margin: int, eos: int):
    """Branch rollout schedule: teacher-force prefix[:t], force `token` at t, free for `margin`
    frames (to give the encoder right-context at frame t), then force eos to TERMINATE. Bounds
    every generation to ~t+margin+1 tokens regardless of max_new_tokens (spec §5)."""
    def sched(step):
        if step < t:
            return int(prefix_tokens[step])
        if step == t:
            return int(token)
        if step >= t + margin:
            return int(eos)
        return None
    return sched


def prefix_then_force_schedule(prefix_tokens, t: int, token: int):
    """Force prefix_tokens[:t] for steps < t, `token` at t, free after (teacher-forced prefix)."""
    def sched(step):
        if step < t:
            return int(prefix_tokens[step])
        if step == t:
            return int(token)
        return None
    return sched
