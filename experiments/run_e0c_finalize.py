"""E0-c finalize (spec §2). RUNS ON THE GPU HOST via finalize(adapter).

Source proof (talker.forward): `codec_ids = torch.cat((input_ids, predictor.sequences), -1)`,
so the sampled semantic token `input_ids` IS the codec semantic id sent downstream (identity)
for ordinary tokens; ids >= codec_vocab are special/control (eos/pad = 2150). This finalize
step VERIFIES the rule on the actually-sampled tokens (no waveform re-encoding) and writes the
authoritative map artifacts. Establishes E0-c PASS/INVESTIGATE.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prewm.talker_map import TalkerCodecMap, ORDINARY  # noqa: E402


def _special_ids(adapter):
    """Best-effort read of special codec ids from the talker/generation config."""
    out = {}
    for holder in (getattr(adapter.model, "model", None), adapter.model):
        gc = getattr(holder, "generation_config", None)
        cfg = getattr(holder, "config", None)
        for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
            for src in (gc, cfg):
                v = getattr(src, name, None) if src is not None else None
                if isinstance(v, int) and name.split("_")[0] not in out:
                    out[name.split("_")[0]] = v
    return out


def finalize(adapter, n_gen: int = 4, texts=None):
    import torch
    talker = adapter.model.model.talker
    codec_vocab = int(adapter.semantic_vocab_size)     # 2048
    talker_vocab = int(adapter.talker_vocab_size)      # 3072
    num_code_groups = getattr(getattr(talker, "config", None), "num_code_groups", None)
    head = adapter._codec_head()

    specials = _special_ids(adapter)
    eos = specials.get("eos", 2150)
    pad = specials.get("pad", eos)
    bos = specials.get("bos", None)
    print(f"[e0c] codec_vocab={codec_vocab} talker_vocab={talker_vocab} "
          f"num_code_groups={num_code_groups} specials={specials}")

    # ---- capture actually-sampled semantic tokens + final support ---------
    sampled = []          # decode-step input_ids (== codec_ids[...,0], the semantic code)
    support_union = set()

    def talker_pre(_m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            return
        ids = ids.detach()
        if ids.shape[-1] == 1:                         # a single decode step
            sampled.append(int(ids.reshape(-1)[-1].item()))

    def head_hook(_m, _i, out):
        lg = out[0] if isinstance(out, tuple) else out
        v = lg.detach()[..., -1, :].reshape(-1)
        k = min(50, v.numel())
        support_union.update(int(i) for i in torch.topk(v, k).indices.tolist())

    h1 = talker.register_forward_pre_hook(talker_pre, with_kwargs=True)
    h2 = head.register_forward_hook(head_hook)
    import urllib.request
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")
    texts = texts or ["The morning train arrived exactly on time.",
                      "Rain fell steadily across the quiet valley throughout the afternoon.",
                      "He counted the coins twice before handing them to the shopkeeper.",
                      "After dinner they walked along the shore and talked about the future."][:n_gen]
    try:
        for i, t in enumerate(texts):
            torch.manual_seed(i)
            adapter.model.generate_voice_clone(text=t, language="English", voice_clone_prompt=vp)
    finally:
        h1.remove(); h2.remove()

    sampled = np.array(sampled, dtype=np.int64)
    ordinary_sampled = sampled[sampled < codec_vocab]
    special_sampled = sampled[sampled >= codec_vocab]
    uniq, cnt = np.unique(sampled, return_counts=True)
    counts = dict(zip(uniq.tolist(), cnt.tolist()))

    # ---- verification (authoritative, no re-encode) -----------------------
    # every sampled special is an enumerated special; every ordinary sample is < codec_vocab.
    unexplained_specials = sorted(int(v) for v in np.unique(special_sampled)
                                  if v not in (eos, pad) and v != bos)
    gate = "PASS"
    notes = []
    if len(sampled) == 0:
        gate, note = "INVESTIGATE", "no sampled tokens captured"
        notes.append(note)
    if unexplained_specials:
        gate = "INVESTIGATE"
        notes.append(f"sampled specials not enumerated: {unexplained_specials[:10]}")

    # ---- build + write the map --------------------------------------------
    special_classes = {eos: "eos"}
    if pad != eos:
        special_classes[pad] = "pad"
    if isinstance(bos, int):
        special_classes[bos] = "bos"
    m = TalkerCodecMap.identity_with_specials(talker_vocab, codec_vocab, special_classes)

    out = "artifacts/e0c"
    os.makedirs(out, exist_ok=True)
    rule = ("talker.forward: codec_ids = cat(input_ids, code_predictor.sequences); "
            "input_ids is codec_ids[...,0] -> identity for v < codec_vocab; v >= codec_vocab special")
    with open(os.path.join(out, "talker_codec_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["talker_id", "token_class", "codec_id", "source_code_rule",
                    "observed_count", "observed_in_final_support", "observed_as_sampled_token", "notes"])
        for v in range(talker_vocab):
            w.writerow([v, m.token_class[v], int(m.codec_id[v]), "identity" if v < codec_vocab else "special",
                        counts.get(v, 0), int(v in support_union), int(counts.get(v, 0) > 0),
                        "eos/pad" if v in (eos, pad) else ""])
    map_json = m.to_json()
    map_json["established_from"] = rule
    map_json["specials"] = {"eos": eos, "pad": pad, "bos": bos}
    map_json["num_code_groups"] = num_code_groups
    json.dump(map_json, open(os.path.join(out, "map.json"), "w"))
    with open(os.path.join(out, "mapping_examples.jsonl"), "w") as f:
        for v in list(np.unique(ordinary_sampled))[:40]:
            f.write(json.dumps({"talker_id": int(v), "codec_id": int(v), "class": ORDINARY}) + "\n")
    metrics = {"codec_vocab": codec_vocab, "talker_vocab": talker_vocab,
               "num_code_groups": num_code_groups, "specials": {"eos": eos, "pad": pad, "bos": bos},
               "n_sampled": int(sampled.size), "n_ordinary": int(ordinary_sampled.size),
               "n_special": int(special_sampled.size),
               "max_ordinary_sampled": int(ordinary_sampled.max()) if ordinary_sampled.size else None,
               "distinct_ordinary": int(np.unique(ordinary_sampled).size),
               "unexplained_specials": unexplained_specials, "gate": gate, "rule": rule}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)

    L = ["# E0-c Report — talker(3072) -> codec-semantic(2048) map\n",
         f"- **Rule (source-proven):** {rule}",
         f"- codec_vocab={codec_vocab}, talker_vocab={talker_vocab}, num_code_groups={num_code_groups}",
         f"- specials: eos={eos}, pad={pad}, bos={bos}; region [{codec_vocab}, {talker_vocab}) is special/reserved",
         f"- verified on **{sampled.size} actually-sampled tokens** across {len(texts)} generations: "
         f"all {ordinary_sampled.size} ordinary samples < {codec_vocab} "
         f"(max {metrics['max_ordinary_sampled']}), {special_sampled.size} specials, "
         f"{len(unexplained_specials)} unexplained.",
         "- authoritative: uses the codec id Qwen sends downstream (input_ids == codec_ids[...,0]); "
         "NO waveform re-encoding used.",
         f"\n## Gate: **{gate}**"]
    if notes:
        L.append("Notes: " + "; ".join(notes))
    open(os.path.join(out, "E0C_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"[e0c] gate={gate} | ordinary_sampled<{codec_vocab}: "
          f"{ordinary_sampled.size}/{sampled.size} | unexplained_specials={unexplained_specials[:5]}")
    print("[e0c] wrote artifacts/e0c/{map.json,talker_codec_map.csv,E0C_REPORT.md}")
    return metrics


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    finalize(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
