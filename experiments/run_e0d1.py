"""E0-d1: local same-context stochasticity (D_local) + matched detector null scale
(sigma_matched), from CACHED E0 traces only. No audio regeneration (spec §3, Q3/Q4/Q5).

D_local uses the actual final p_t (post temperature/top-k) reconstructed from the cached
support logits. sigma_matched uses the cached clean re-encoded semantic latents in the
shared whitened detector space, scored by 64 fixed diagnostic keys, bootstrapped over TEXTS.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io  # noqa: E402
from src.sampling_trace import softmax  # noqa: E402
from src.transforms import whitening_from_codebook, transform_shared_space  # noqa: E402
from src.prewm.stochasticity import collision, local_disagreement, d_local  # noqa: E402
from src.prewm.keys import diagnostic_keys  # noqa: E402
from src.prewm.bootstrap import bootstrap_statistic  # noqa: E402


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
    entries = [e for e in io.read_manifest(os.path.join(e0, "manifest.jsonl")) if e.get("status") == "ok"]

    codebook = np.load(os.path.join(e0, "metrics", "semantic_codebook.npy"))
    wp = whitening_from_codebook(codebook, ridge=cfg["transform"]["ridge"])
    d = codebook.shape[1]
    U = diagnostic_keys(args.n_keys, d)   # [n_keys, 256]

    # ---- D_local from cached final p_t ------------------------------------
    local_rows, all_frame_disagree = [], []
    per_text_frames: dict[str, list[np.ndarray]] = {}
    for e in entries:
        tr = torch.load(os.path.join(e0, "samples", e["sample_id"], "generation_trace.pt"),
                        weights_only=False)
        sup_ids, sup_lg = tr["support_ids"], tr["support_logits"]
        for t in range(len(sup_ids)):
            pf = softmax(np.asarray(sup_lg[t], dtype=np.float64) / temp)  # final p_t on support
            ld = float(local_disagreement(pf))
            all_frame_disagree.append(ld)
            per_text_frames.setdefault(e["prompt_id"], []).append(pf)
        local_rows.append({"sample_id": e["sample_id"], "prompt_id": e["prompt_id"], "seed": e["seed"],
                           "T": len(sup_ids)})
    dloc = d_local([p for ps in per_text_frames.values() for p in ps])
    # per-text D_local (text-level so bootstrap is over texts)
    per_text_dlocal = {tid: float(np.mean([local_disagreement(p) for p in ps]))
                       for tid, ps in per_text_frames.items()}
    dloc_boot = bootstrap_statistic(list(per_text_dlocal), lambda ts: float(np.mean([per_text_dlocal[t] for t in ts])),
                                    n_replicates=500)

    with open(os.path.join(args.out, "local_stochasticity.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prompt_id", "d_local_text"])
        for tid, v in sorted(per_text_dlocal.items()):
            w.writerow([tid, v])

    # ---- sigma_matched from cached clean re-encoded latents ---------------
    sig_rows = []
    per_key_text_scores: dict[tuple, np.ndarray] = {}
    for e in entries:
        z = torch.load(os.path.join(e0, "samples", e["sample_id"], "encoder_semantic_latents.pt"),
                       weights_only=False)
        z = z.numpy() if hasattr(z, "numpy") else np.asarray(z)
        zt = transform_shared_space(z, wp.center, wp.whitening_matrix)   # [T, d]
        s = zt @ U.T                                                     # [T, n_keys]
        for k in range(args.n_keys):
            per_key_text_scores.setdefault((k, e["prompt_id"]), [])
            per_key_text_scores[(k, e["prompt_id"])].append(s[:, k])
    # per (key, text): mean and var of the score over that text's frames
    for (k, tid), chunks in per_key_text_scores.items():
        sk = np.concatenate(chunks)
        sig_rows.append({"key_id": k, "text_id": tid, "channel": "clean_reencode",
                         "n_frames": int(sk.size), "mean_s": float(sk.mean()), "var_s": float(sk.var())})
    with open(os.path.join(args.out, "sigma_matched.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key_id", "text_id", "channel", "n_frames", "mean_s", "var_s"])
        w.writeheader(); w.writerows(sig_rows)

    # aggregate sigma_matched^2 = mean over keys of (text-averaged frame variance); bootstrap texts
    texts = sorted({r["text_id"] for r in sig_rows})
    by_kt = {(r["key_id"], r["text_id"]): r["var_s"] for r in sig_rows}
    def sigma2_over_texts(ts):
        return float(np.mean([by_kt[(k, t)] for k in range(args.n_keys) for t in ts]))
    sig_boot = bootstrap_statistic(texts, sigma2_over_texts, n_replicates=500)
    sigma_matched = float(np.sqrt(sig_boot["point"]))

    # operational null: not available at this milestone
    open(os.path.join(args.out, "sigma_operational.csv"), "w").write("channel\nDEFERRED\n")

    metrics = {
        "temperature": temp,
        "D_local": dloc, "D_local_bootstrap_texts": dloc_boot,
        "trajectory_seed_disagreement_ref": 0.867,
        "cascade_share_est": float(1 - dloc["mean"] / 0.867) if dloc["mean"] == dloc["mean"] else None,
        "sigma_matched": sigma_matched,
        "sigma_matched_sq_bootstrap_texts": sig_boot,
        "sigma_operational": "DEFERRED",
        "n_keys": args.n_keys, "n_texts": len(texts), "n_frames_total": len(all_frame_disagree),
        "whitening": wp.to_metadata(),
        "boundary_caveat": "interior-frame mask pending E0-e; all frames used here",
    }
    io.write_json(os.path.join(args.out, "metrics.json"), metrics)
    _report(args.out, md, metrics)
    print(f"E0-d1 done. D_local={dloc['mean']:.4f} (median {dloc['median']:.4f}), "
          f"sigma_matched={sigma_matched:.4f}. cascade_share≈{metrics['cascade_share_est']:.3f}")


def _report(out, md, m):
    L = []
    L.append("# E0-d1 Report — local stochasticity + matched null scale\n")
    L.append(f"- model: {md.get('model') or 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'}, temp={m['temperature']}")
    L.append(f"- source: cached E0 traces, {m['n_texts']} texts, {m['n_frames_total']} frames. No regeneration.\n")
    L.append("## Q3 — local same-context stochastic freedom  D_local = E_t[1 - Σ p_t²]")
    d = m["D_local"]
    L.append(f"- **D_local = {d['mean']:.4f}** (median {d['median']:.4f}, p10 {d['p10']:.4f}, p90 {d['p90']:.4f})")
    b = m["D_local_bootstrap_texts"]
    L.append(f"- text-bootstrap 95% CI: [{b['ci_low']:.4f}, {b['ci_high']:.4f}]\n")
    L.append("## Q4 — how much of the 87% seed disagreement is cascade?")
    L.append(f"- full-trajectory seed disagreement ≈ 0.867; local same-context freedom = {d['mean']:.4f}")
    L.append(f"- **so ≈ {m['cascade_share_est']*100:.0f}% of the trajectory disagreement is autoregressive cascade**, "
             "not per-frame sampling freedom. Do NOT call 0.87 per-frame capacity.\n")
    L.append("## Q5 — matched detector null scale")
    L.append(f"- **sigma_matched = {m['sigma_matched']:.4f}** (whitened detector score, 64 fixed keys, clean re-encode)")
    sb = m["sigma_matched_sq_bootstrap_texts"]
    L.append(f"- sigma_matched² text-bootstrap 95% CI: [{sb['ci_low']:.4f}, {sb['ci_high']:.4f}]")
    L.append(f"- sigma_operational: **DEFERRED** (no legitimate operational-null population available yet)\n")
    L.append("> Caveat: interior-frame mask pending E0-e; all frames used. sigma_matched is on the E0 "
             "smoke population (8 texts × 4 seeds); E1 will re-estimate on the 800-realization corpus.")
    open(os.path.join(out, "E0D1_REPORT.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
