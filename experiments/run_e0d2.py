"""E0-d2: corrected mapped V on the E0-c map (spec §4, Q7). Cached-data only, no regen.

Uses the E0-c source-proven map (identity for ordinary talker ids < codec_vocab; specials
above get phi=0 but KEEP their probability mass). For each cached final p_t and each of 64
fixed diagnostic keys:

    phi_t(v) = u^T z_tilde_{m(v)}   (ordinary v);   0 (special/control v)
    mu_t = sum_v p_t(v) phi_t(v)
    V_t  = sum_v p_t(v) (phi_t(v) - mu_t)^2

The old V ~= 0.174 is OBSOLETE after this (it predated the verified map). V/sigma^2 is a
SCALE DIAGNOSTIC only; the real J needs C (branch rollouts, Phase D).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io  # noqa: E402
from src.sampling_trace import softmax  # noqa: E402
from src.transforms import whitening_from_codebook, transform_shared_space  # noqa: E402
from src.prewm.keys import diagnostic_keys  # noqa: E402
from src.prewm.talker_map import TalkerCodecMap  # noqa: E402
from src.metrics import entropy as _entropy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0.yaml")
    ap.add_argument("--e0_root", default="artifacts/e0")
    ap.add_argument("--out", default="artifacts/e0d")
    ap.add_argument("--n_keys", type=int, default=64)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)

    import torch
    e0 = args.e0_root
    md = io.read_json(os.path.join(e0, "metadata.json"))
    temp = (md.get("sampling_config_actual") or {}).get("temperature") or 1.0
    codebook = np.load(os.path.join(e0, "metrics", "semantic_codebook.npy"))
    codec_vocab, dim = codebook.shape                    # 2048, 256
    talker_vocab = 3072

    # map: prefer artifacts/e0c/map.json if finalize wrote it; else source-proven identity.
    map_path = "artifacts/e0c/map.json"
    if os.path.exists(map_path):
        m = TalkerCodecMap.from_json(json.load(open(map_path)))
        map_source = "artifacts/e0c/map.json"
    else:
        m = TalkerCodecMap.identity_with_specials(talker_vocab, codec_vocab, {2150: "eos"})
        map_source = "E0-c source-proven identity (map.json pending finalize on GPU)"

    wp = whitening_from_codebook(codebook, ridge=cfg["transform"]["ridge"])
    z_tilde = transform_shared_space(codebook, wp.center, wp.whitening_matrix)   # [2048, dim]
    U = diagnostic_keys(args.n_keys, dim)                                        # [n_keys, dim]
    codec_feat = z_tilde @ U.T                                                   # [2048, n_keys]

    entries = [e for e in io.read_manifest(os.path.join(e0, "manifest.jsonl")) if e.get("status") == "ok"]
    per_key_V = {k: [] for k in range(args.n_keys)}
    per_text_V = {}
    frame_rows = []       # V (key 0) vs entropy vs p_max, ordinary mass
    ordinary_mass_all, special_mass_all, support_all = [], [], []

    for e in entries:
        tr = torch.load(os.path.join(e0, "samples", e["sample_id"], "generation_trace.pt"),
                        weights_only=False)
        sup_ids, sup_lg = tr["support_ids"], tr["support_logits"]
        for t in range(len(sup_ids)):
            ids = np.asarray(sup_ids[t]).astype(np.int64)
            p = softmax(np.asarray(sup_lg[t], dtype=np.float64) / temp)     # sums to 1 over support
            ordinary = ids < codec_vocab
            ord_mass = float(p[ordinary].sum())
            ordinary_mass_all.append(ord_mass)
            special_mass_all.append(1.0 - ord_mass)
            support_all.append(int(ids.size))
            # phi over the FULL support for each key (special candidates -> 0)
            phi_full = np.zeros((ids.size, args.n_keys), dtype=np.float64)
            phi_full[ordinary] = codec_feat[ids[ordinary]]                 # [n_ord, n_keys]
            mu = (p[:, None] * phi_full).sum(axis=0)                       # [n_keys]
            V = (p[:, None] * (phi_full - mu) ** 2).sum(axis=0)            # [n_keys]
            for k in range(args.n_keys):
                per_key_V[k].append(float(V[k]))
            per_text_V.setdefault(e["prompt_id"], []).append(float(V.mean()))
            frame_rows.append({"V_mean_over_keys": float(V.mean()),
                               "entropy": float(_entropy(p)), "p_max": float(p.max()),
                               "ordinary_mass": ord_mass, "support": int(ids.size)})

    V_per_frame = np.array([r["V_mean_over_keys"] for r in frame_rows])
    ent = np.array([r["entropy"] for r in frame_rows])
    pmax = np.array([r["p_max"] for r in frame_rows])

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan")

    metrics = {
        "map_source": map_source, "n_keys": args.n_keys, "n_frames": len(frame_rows),
        "V_mean": float(V_per_frame.mean()), "V_median": float(np.median(V_per_frame)),
        "V_p10": float(np.percentile(V_per_frame, 10)), "V_p90": float(np.percentile(V_per_frame, 90)),
        "V_per_key_mean": {int(k): float(np.mean(v)) for k, v in per_key_V.items()},
        "V_per_text_mean": {t: float(np.mean(v)) for t, v in per_text_V.items()},
        "ordinary_mass_mean": float(np.mean(ordinary_mass_all)),
        "special_mass_mean": float(np.mean(special_mass_all)),
        "support_size_mean": float(np.mean(support_all)),
        "corr_V_entropy": corr(V_per_frame, ent), "corr_V_pmax": corr(V_per_frame, pmax),
        "obsolete_V_note": "supersedes the provisional V~=0.174 (pre-map).",
        "scale_diagnostic": {"sqrt_V_over_sigma_matched": None},
    }
    # scale diagnostic vs sigma_matched from E0-d1 if present
    d1 = os.path.join(args.out, "metrics.json")
    if os.path.exists(d1):
        sm = io.read_json(d1).get("sigma_matched")
        if sm:
            metrics["scale_diagnostic"]["sqrt_V_over_sigma_matched"] = float(np.sqrt(metrics["V_mean"]) / sm)
            metrics["scale_diagnostic"]["note"] = "SCALE DIAGNOSTIC ONLY — NOT J_max; real J needs C (Phase D)."

    with open(os.path.join(args.out, "mapped_V.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["V_mean_over_keys", "entropy", "p_max", "ordinary_mass", "support"])
        w.writeheader(); w.writerows(frame_rows)
    io.write_json(os.path.join(args.out, "mapped_V_metrics.json"), metrics)

    L = ["# E0-d2 Report — corrected mapped V (Q7)\n",
         f"- map: {map_source}",
         f"- **V (mapped) = {metrics['V_mean']:.4f}** mean (median {metrics['V_median']:.4f}, "
         f"p10 {metrics['V_p10']:.4f}, p90 {metrics['V_p90']:.4f}), 64 keys, {metrics['n_frames']} frames",
         f"- ordinary-candidate probability mass: **{metrics['ordinary_mass_mean']:.4f}** "
         f"(special/control mass {metrics['special_mass_mean']:.4f}); support size {metrics['support_size_mean']:.1f}",
         f"- corr(V, entropy) = {metrics['corr_V_entropy']:.3f}; corr(V, p_max) = {metrics['corr_V_pmax']:.3f}",
         f"- **the provisional V ≈ 0.174 is OBSOLETE.**",
         (f"- scale diagnostic sqrt(V)/sigma_matched = {metrics['scale_diagnostic']['sqrt_V_over_sigma_matched']:.4f} "
          "— NOT a J_max; the real J requires C from branch rollouts (Phase D)."
          if metrics["scale_diagnostic"]["sqrt_V_over_sigma_matched"] else "")]
    open(os.path.join(args.out, "E0D2_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"E0-d2 done. mapped V={metrics['V_mean']:.4f} (median {metrics['V_median']:.4f}), "
          f"ordinary_mass={metrics['ordinary_mass_mean']:.4f}, map={map_source}")


if __name__ == "__main__":
    main()
