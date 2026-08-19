"""E0 stage 1: generate the 32 utterances and cache immutable artifacts (§9A, §17).

    8 texts x 4 seeds x 1 fixed voice = 32.

Runs on the GPU host (Colab). Saves waveform + emitted codes + p_t trace per sample,
a manifest (failed samples preserved with a reason), and metadata.json.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io  # noqa: E402
from src.io import ManifestEntry, make_sample_id, config_hash  # noqa: E402


def _git_sha_and_dirty(repo_root):
    try:
        sha = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", repo_root, "status", "--porcelain"]).strip())
        return sha, dirty
    except Exception:
        return None, None


def _wav_duration_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--ref_audio", required=True, help="repo default reference clip (§8.1)")
    ap.add_argument("--ref_text", required=True, help="transcript of the reference clip")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    cfg = yaml.safe_load(open(args.config))
    prompts = io.read_json(cfg["design"]["prompts_file"])["prompts"]
    seeds = cfg["design"]["seeds"]
    assert len(prompts) == cfg["design"]["n_texts"], "prompt count != n_texts"
    assert len(seeds) == cfg["design"]["n_seeds"], "seed count != n_seeds"

    art = cfg["paths"]["artifacts_root"]
    sample_root = os.path.join(art, "samples")
    os.makedirs(sample_root, exist_ok=True)
    manifest_path = os.path.join(art, "manifest.jsonl")
    open(manifest_path, "w").close()  # fresh manifest

    ch = config_hash(cfg)

    # ---- load model via the adapter (real qwen-tts API) --------------------
    import torch  # noqa: F401
    from src.adapters.qwen3_tts import SpeechLMAdapter
    adapter = SpeechLMAdapter.load(cfg, device=args.device)

    # §3 FIRST: dump discovered structure so any hook mismatch surfaces loudly.
    structure = adapter.dump_structure()

    voice_prompt = adapter.create_voice_prompt(args.ref_audio, args.ref_text)
    voice_id = cfg["design"]["voice_id"]

    # ---- metadata.json (§18) ----------------------------------------------
    sha, dirty = _git_sha_and_dirty(repo_root)
    ref_sha = io.sha256_file(args.ref_audio)
    try:
        ref_dur = _wav_duration_seconds(args.ref_audio)
    except Exception:
        ref_dur = None
    md = {
        "experiment_name": cfg["experiment_name"],
        "git_sha": sha, "git_dirty": dirty,
        "config_hash": ch,
        "prompts_file_sha256": io.sha256_file(cfg["design"]["prompts_file"]),
        "seeds": seeds,
        "voice_id": voice_id, "voice_ref_sha256": ref_sha, "voice_ref_duration_s": ref_dur,
        "sampling_config_actual": adapter.get_sampling_config(),
        "discovered_structure": structure,
        **adapter.get_model_metadata(),
        "tokenizer_facts_config": cfg["tokenizer_facts"],
        "warnings": [],
    }
    if not structure.get("semantic_vocab_matches_config", True):
        md["warnings"].append(structure.get("WARNING", "semantic vocab mismatch"))
    if "NOTE_D4" in structure:
        md["warnings"].append(structure["NOTE_D4"])
    io.write_json(os.path.join(art, "metadata.json"), md)
    io.write_json(os.path.join(art, "prompts.json"), {"prompts": prompts})

    # ---- generation loop ---------------------------------------------------
    n_ok = 0
    for p in prompts:
        for seed in seeds:
            sid = make_sample_id(p["id"], voice_id, seed, ch)
            sdir = os.path.join(sample_root, sid)
            os.makedirs(sdir, exist_ok=True)
            try:
                tr = adapter.generate_trace(
                    text=p["text"], seed=seed, sample_id=sid, voice_id=voice_id,
                    voice_clone_prompt=voice_prompt)

                # persist waveform as wav
                wav_path = os.path.join(sdir, "waveform.wav")
                _write_wav(wav_path, tr.waveform, tr.sample_rate)
                torch.save(torch.as_tensor(np.asarray(tr.semantic_ids)),
                           os.path.join(sdir, "semantic_ids.pt"))
                if tr.all_codec_ids is not None:
                    torch.save(torch.as_tensor(np.asarray(tr.all_codec_ids)),
                               os.path.join(sdir, "codec_ids.pt"))
                torch.save({"raw_semantic_probs": tr.raw_semantic_probs,
                            "support_ids": tr.support_ids,
                            "support_logits": tr.support_logits,
                            "sampling_metadata": tr.sampling_metadata},
                           os.path.join(sdir, "generation_trace.pt"))

                tensors = {"waveform": "waveform.wav", "semantic_ids": "semantic_ids.pt",
                           "generation_trace": "generation_trace.pt"}
                shapes = {"waveform": [int(len(tr.waveform))],
                          "semantic_ids": [int(np.asarray(tr.semantic_ids).size)],
                          "generation_trace": list(np.asarray(tr.raw_semantic_probs).shape)
                          if tr.raw_semantic_probs is not None else []}
                if tr.all_codec_ids is not None:
                    tensors["codec_ids"] = "codec_ids.pt"
                    shapes["codec_ids"] = list(np.asarray(tr.all_codec_ids).shape)

                io.append_manifest(manifest_path, ManifestEntry(
                    sample_id=sid, prompt_id=p["id"], text=p["text"], voice_id=voice_id,
                    seed=seed, status="ok", tensors=tensors, shapes=shapes))
                n_ok += 1
                print(f"[ok] {sid}  T={shapes['semantic_ids'][0]}")
            except Exception as e:  # preserve failures (§22)
                io.append_manifest(manifest_path, ManifestEntry(
                    sample_id=sid, prompt_id=p["id"], text=p["text"], voice_id=voice_id,
                    seed=seed, status="failed", failure_reason=f"{type(e).__name__}: {e}"))
                print(f"[FAILED] {sid}: {e}", file=sys.stderr)

    print(f"\nE0 generate done: {n_ok}/{len(prompts)*len(seeds)} ok. Manifest: {manifest_path}")


def _write_wav(path, waveform, sr):
    x = np.asarray(waveform, dtype=np.float32).ravel()
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
