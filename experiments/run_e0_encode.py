"""E0 stage 2: clean encode -> decode -> re-encode round trip + metrics (§9C, §12).

The high-level generate API does not return the LM's emitted codes, so E0 measures the
codec's own round-trip stability: encode(wav) -> codes -> decode(codes) -> wav2 ->
encode(wav2), comparing codes/latents after deterministic offset alignment. The §11
coordinate invariant Q(h)==tokenizer-id is measured directly inside encode. No attacks.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io, metrics  # noqa: E402
from src.alignment import align_after_offset, crosscorr_peak_lag, norm_sequence  # noqa: E402
from src.semantic_space import graded_agreement_decision  # noqa: E402
from src.transforms import whitening_from_codebook, transform_shared_space  # noqa: E402


def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32767.0, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    cfg = yaml.safe_load(open(args.config))
    art = cfg["paths"]["artifacts_root"]
    sample_root = os.path.join(art, "samples")
    metrics_dir = os.path.join(art, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    import torch  # noqa: F401
    from src.adapters.qwen3_tts import SpeechLMAdapter
    adapter = SpeechLMAdapter.load(cfg, device=args.device)

    codebook = adapter.get_semantic_codebook()
    wp = whitening_from_codebook(codebook, ridge=cfg["transform"]["ridge"])
    io.write_json(os.path.join(metrics_dir, "whitening_params.json"), wp.to_metadata())
    np.save(os.path.join(metrics_dir, "semantic_codebook.npy"), codebook)

    entries = [e for e in io.read_manifest(os.path.join(art, "manifest.jsonl"))
               if e.get("status") == "ok"]
    rows, determinism_rows = [], []
    probe_budget = cfg["encoder_probe"]["n_waveforms"]

    for i, e in enumerate(entries):
        sid = e["sample_id"]
        sdir = os.path.join(sample_root, sid)
        wav, sr = _read_wav(os.path.join(sdir, "waveform.wav"))

        # encode #1 (+ coordinate invariant), then decode -> re-encode #2
        enc1 = adapter.encode_raw(wav)
        space1 = adapter.encode_semantic_space(wav)
        ids1, lat1 = np.asarray(space1.semantic_ids), space1.frame_latents
        coord_agree = float(space1.hook_metadata["coordinate_agreement"])

        try:
            wav2, _ = adapter.decode_codes(enc1)
            space2 = adapter.encode_semantic_space(wav2)
            ids2, lat2 = np.asarray(space2.semantic_ids), space2.frame_latents

            offset = crosscorr_peak_lag(norm_sequence(lat1), norm_sequence(lat2), max_lag=8)
            o_al, r_al, align = align_after_offset(lat1, lat2, frame_offset=offset, max_lag=8)
            L = align.aligned_length
            i1 = ids1[max(0, offset):][:L]
            i2 = ids2[max(0, -offset):][:L]
            k = min(len(i1), len(i2), L)
            token_match = metrics.exact_match_rate(i1[:k], i2[:k]) if k else float("nan")
            cos = metrics.rowwise_cosine(o_al[:k], r_al[:k])
            l2 = metrics.rowwise_l2(o_al[:k], r_al[:k])
            wc = metrics.rowwise_cosine(
                transform_shared_space(o_al[:k], wp.center, wp.whitening_matrix),
                transform_shared_space(r_al[:k], wp.center, wp.whitening_matrix))
            rows.append({
                "sample_id": sid, "prompt_id": e["prompt_id"], "seed": e["seed"],
                "coordinate_agreement": coord_agree,
                "coordinate_decision": graded_agreement_decision(
                    coord_agree, cfg["acceptance"]["pass_min"], cfg["acceptance"]["investigate_min"]),
                "T1": int(len(ids1)), "T2": int(len(ids2)),
                "frame_offset": align.frame_offset,
                "frame_offset_crosscorr_peak": align.frame_offset_crosscorr_peak,
                "offset_verified": align.offset_verified, "aligned_length": L,
                "roundtrip_token_match": token_match,
                "cos_mean": float(np.mean(cos)) if k else float("nan"),
                "cos_median": float(np.median(cos)) if k else float("nan"),
                "l2_mean": float(np.mean(l2)) if k else float("nan"),
                "whitened_cos_mean": float(np.mean(wc)) if k else float("nan")})
        except AssertionError as exc:  # §20 stop #10
            rows.append({"sample_id": sid, "prompt_id": e["prompt_id"], "seed": e["seed"],
                         "coordinate_agreement": coord_agree, "offset_verified": False,
                         "error": str(exc)})

        if i < probe_budget:
            a = adapter.encode_semantic_space(wav).frame_latents
            b = adapter.encode_semantic_space(wav).frame_latents
            repeat_identical = bool(np.allclose(a, b, atol=1e-5))
            half = len(wav) // 2
            left = adapter.encode_semantic_space(wav[:half]).frame_latents
            right = adapter.encode_semantic_space(wav[half:]).frame_latents
            split = np.concatenate([left, right], axis=0)[: len(a)]
            cos_seam = metrics.rowwise_cosine(a[: len(split)], split)
            below = np.where(cos_seam < 0.99)[0]
            seam_len = 0 if below.size == 0 else int(below.max() - below.min() + 1)
            determinism_rows.append({"sample_id": sid, "repeat_identical": repeat_identical,
                                     "seam_contamination_frames": seam_len})

        torch.save(torch.as_tensor(lat1), os.path.join(sdir, "encoder_semantic_latents.pt"))
        torch.save(torch.as_tensor(ids1), os.path.join(sdir, "reencoded_semantic_ids.pt"))
        io.write_json(os.path.join(sdir, "alignment.json"), rows[-1])

    _write_csv(os.path.join(metrics_dir, "per_sample.csv"), rows)
    _write_csv(os.path.join(metrics_dir, "encoder_determinism.csv"), determinism_rows)
    print(f"E0 encode done: {len(rows)} samples, {len(determinism_rows)} determinism probes.")


def _write_csv(path, rows):
    import csv
    if not rows:
        open(path, "w").close()
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
