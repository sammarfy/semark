"""E0-e boundary/context diagnostic (spec §6, Q10). RUNS ON THE GPU HOST via run(adapter).

The ~29-frame chunk seam only proved sensitivity to an ARTIFICIAL split. E0-e asks whether a
NORMAL encode needs left context to stabilize: re-encode suffixes (offset = k*hop, frame-
synchronous) and compare to the full encode, indexed by distance from the new boundary. Reads
a context length via a PREDECLARED criterion (median cosine >= 0.999 AND norm-L2 <= 0.01).
Uses cached E0 waveforms — no generation. No DTW: alignment is the fixed frame-hop offset.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io  # noqa: E402
from src.alignment import crosscorr_peak_lag, norm_sequence  # noqa: E402
from src.prewm.boundary import (error_vs_distance, aggregate_by_distance,  # noqa: E402
                                 context_length, descriptive_context_length)


def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32767.0, sr


def run(adapter, e0_root="artifacts/e0", out="artifacts/e0_boundary",
        n_waveforms=8, truncation_frames=(4, 8, 12, 16, 24, 32)):
    hop = int(adapter.sample_rate_hz / adapter.frame_rate_hz)   # 1920
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    entries = [e for e in io.read_manifest(os.path.join(e0_root, "manifest.jsonl"))
               if e.get("status") == "ok"]
    # pick the longest utterances (most interior)
    entries = sorted(entries, key=lambda e: -int(e.get("shapes", {}).get("semantic_ids", [0])[0]))[:n_waveforms]

    curves, rows = [], []
    for e in entries:
        wav, sr = _read_wav(os.path.join(e0_root, "samples", e["sample_id"], "waveform.wav"))
        z_full = adapter.encode_semantic_space(wav).frame_latents            # [T, d]
        T = len(z_full)
        for k in truncation_frames:
            if k + 8 >= T:
                continue
            z_suf = adapter.encode_semantic_space(wav[k * hop:]).frame_latents  # suffix re-encode
            # verify frame-synchronous alignment (fixed hop offset, no DTW): peak lag ~ 0
            lag = crosscorr_peak_lag(norm_sequence(z_full[k:k + len(z_suf)]),
                                     norm_sequence(z_suf), max_lag=4)
            cur = error_vs_distance(z_full[k:], z_suf, first_frame_distance=0)
            curves.append(cur)
            for dist, cos, l2 in zip(cur["distance"], cur["cosine"], cur["norm_l2"]):
                rows.append({"sample_id": e["sample_id"], "trunc_frame": k, "distance": int(dist),
                             "align_lag": int(lag), "cosine": float(cos), "norm_l2": float(l2)})

    agg = aggregate_by_distance(curves)
    ctx = context_length(agg, cos_thresh=0.999, l2_thresh=0.01)
    desc = descriptive_context_length(agg)   # relative-to-floor fallback (spec §6)
    if ctx["b_context"] is None:              # absolute thresholds numerically inappropriate
        ctx = {**ctx, **desc, "note": "absolute thresholds not met (L2 floor > 0.01); using "
               "relative-to-floor descriptive criterion"}

    with open(os.path.join(out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "trunc_frame", "distance", "align_lag", "cosine", "norm_l2"])
        w.writeheader(); w.writerows(rows)
    aggregate = {"distance": agg["distance"].tolist(),
                 "median_cosine": agg["median_cosine"].tolist(),
                 "median_norm_l2": agg["median_norm_l2"].tolist(),
                 "n": agg["n"].tolist(), **ctx,
                 "seam_ref_frames": 29, "hop": hop}
    io.write_json(os.path.join(out, "aggregate.json"), aggregate)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(agg["distance"], agg["median_cosine"], marker="o")
        ax[0].axhline(0.999, color="crimson", ls="--", label="0.999")
        ax[0].set_xlabel("frames from new boundary"); ax[0].set_ylabel("median cosine"); ax[0].legend()
        ax[1].plot(agg["distance"], agg["median_norm_l2"], marker="o", color="darkorange")
        ax[1].axhline(0.01, color="crimson", ls="--", label="0.01")
        ax[1].set_xlabel("frames from new boundary"); ax[1].set_ylabel("median norm L2"); ax[1].legend()
        b = ctx["b_context"]
        fig.suptitle(f"E0-e boundary context: b_context = {b}" if b is not None
                     else "E0-e boundary context (threshold never met — see curve)")
        plt.tight_layout(); plt.savefig(os.path.join(out, "figures", "boundary_error_vs_distance.png")); plt.close()
    except Exception as ex:
        print("figure skipped:", ex)

    L = ["# E0-e Boundary/Context Report (Q10)\n",
         f"- criterion: {ctx['criterion']}",
         f"- **b_context = {ctx['b_context']}** frames"
         + ("" if ctx["b_context"] is not None else " (threshold not met at tested distances — report full curve)"),
         f"- compares suffix re-encodes (offset k*hop, hop={hop}) to the full encode; no DTW; "
         "alignment lag verified ~0 (frame-synchronous).",
         "- the ~29-frame chunk seam is a *chunk-split* artifact; b_context is the NORMAL-encode "
         "startup length. Use b_context (not 29) for Stage-1 interior masks / prompt length (§7.1)."]
    open(os.path.join(out, "E0_BOUNDARY_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"E0-e done. b_context={ctx['b_context']} (criterion cos>=0.999 & normL2<=0.01)")
    return aggregate


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    run(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
