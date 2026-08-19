"""E0 stage 3: aggregate metrics, E0-b diagnostics, figures, and E0_REPORT.md (§9, §19).

Operates ONLY on cached artifacts (§22): never regenerates audio. Produces:
  metrics/entropy_metrics.csv (raw AND final), seed_diversity.csv, aggregate.json,
  figures/*.png, and E0_REPORT.md with the graded gate decision.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io, metrics  # noqa: E402
from src.sampling_trace import softmax  # noqa: E402
from src.semantic_space import graded_agreement_decision  # noqa: E402
from src.transforms import whitening_from_codebook, transform_shared_space  # noqa: E402


def _load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _reconstruct_final_probs(support_ids, support_logits, temperature=1.0):
    """p_final over the retained support (renormalized), with the SAME temperature the
    generator used. The adapter stores RAW (un-tempered) support logits, so temperature
    must be applied here to keep H(p_final) comparable to H(p_raw) (which is temp-scaled).
    Without this, p_final looks less peaked than p_raw and the invariant H_raw>=H_final
    is violated (the original E0-b bug)."""
    t = float(temperature) if temperature else 1.0
    p = softmax(np.asarray(support_logits, dtype=np.float64) / t)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0.yaml")
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    cfg = yaml.safe_load(open(args.config))
    art = cfg["paths"]["artifacts_root"]
    sample_root = os.path.join(art, "samples")
    metrics_dir = os.path.join(art, "metrics")
    fig_dir = os.path.join(art, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = io.read_manifest(os.path.join(art, "manifest.jsonl"))
    ok = [e for e in manifest if e.get("status") == "ok"]
    failed = [e for e in manifest if e.get("status") == "failed"]

    # frozen-codebook whitening for x = W(c - c_bar); V_t direction bank
    md = io.read_json(os.path.join(art, "metadata.json"))
    gen_temperature = (md.get("sampling_config_actual") or {}).get("temperature") or 1.0
    # codebook is needed for x_v; reload it from a cached file if present, else skip V_t
    cb_path = os.path.join(metrics_dir, "semantic_codebook.npy")
    codebook = np.load(cb_path) if os.path.exists(cb_path) else None
    wp = whitening_from_codebook(codebook, ridge=cfg["transform"]["ridge"]) if codebook is not None else None
    if wp is not None:
        x_v = transform_shared_space(codebook, wp.center, wp.whitening_matrix)
        u = metrics.random_unit_directions(cfg["vt_directions"]["n_directions"],
                                            x_v.shape[1], cfg["vt_directions"]["seed"])

    # ---- E0-b: per-step entropy RAW vs FINAL, N_eff, p_max, V_t ------------
    ent_rows = []
    per_sample_stoch = {}
    for e in ok:
        sid = e["sample_id"]
        tr = torch.load(os.path.join(sample_root, sid, "generation_trace.pt"),
                        weights_only=False)
        p_raw = np.asarray(tr["raw_semantic_probs"], dtype=np.float64)  # [T, V]
        sup_ids = tr["support_ids"]
        sup_logits = tr["support_logits"]
        T = p_raw.shape[0]
        H_raw = metrics.entropy(p_raw, axis=-1)
        Neff_raw = np.exp(H_raw)
        pmax_raw = p_raw.max(axis=-1)
        H_fin, sup_sz, vt_trunc, vt_full = [], [], [], []
        V = codebook.shape[0] if codebook is not None else None
        for t in range(T):
            pf = _reconstruct_final_probs(sup_ids[t], sup_logits[t], gen_temperature)
            H_fin.append(metrics.entropy(pf))
            sup_sz.append(int(len(pf)))
            if wp is not None:
                # D4: p_t is over the talker vocab (3072) but codec centroids index 0..V-1.
                # For V_t we restrict to the codec-code region, assuming talker tokens < V
                # map to codec codes identically. Specials (>=V) are excluded and flagged.
                sup = np.asarray(sup_ids[t])
                m = sup < V
                if m.any():
                    pf_sub = pf[m]; s = pf_sub.sum()
                    pf_sub = pf_sub / s if s > 0 else pf_sub
                    vt_trunc.append(float(metrics.vt_directions(u, x_v[sup[m]], pf_sub).mean()))
                else:
                    vt_trunc.append(float("nan"))
                pr = p_raw[t][:V]; s = pr.sum()
                vt_full.append(float(metrics.vt_directions(u, x_v, pr / s).mean()) if s > 0 else float("nan"))
        H_fin = np.asarray(H_fin)
        row = {"sample_id": sid, "prompt_id": e["prompt_id"], "seed": e["seed"], "T": T,
               "H_raw_mean": float(H_raw.mean()), "H_final_mean": float(H_fin.mean()),
               "Neff_raw_mean": float(Neff_raw.mean()),
               "Neff_final_mean": float(np.exp(H_fin).mean()),
               "pmax_raw_mean": float(pmax_raw.mean()),
               "support_size_mean": float(np.mean(sup_sz)),
               "Vt_trunc_mean": float(np.mean(vt_trunc)) if vt_trunc else float("nan"),
               "Vt_full_mean": float(np.mean(vt_full)) if vt_full else float("nan")}
        ent_rows.append(row)
        per_sample_stoch[sid] = {"H_raw": H_raw, "H_final": H_fin}
    _write_csv(os.path.join(metrics_dir, "entropy_metrics.csv"), ent_rows)

    # ---- §14 seed diversity + §15 crude R_trace ---------------------------
    by_text = {}
    for e in ok:
        by_text.setdefault(e["prompt_id"], []).append(e)
    seed_rows, deltaR = [], []
    for pid, group in by_text.items():
        ids = {g["seed"]: torch.load(os.path.join(sample_root, g["sample_id"],
               "semantic_ids.pt")).numpy().ravel() for g in group}
        seeds = sorted(ids)
        durations = [len(ids[s]) for s in seeds]
        disagreements = []
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                ia, ib = ids[seeds[a]], ids[seeds[b]]
                L = min(len(ia), len(ib))
                if L:
                    disagreements.append(float((ia[:L] != ib[:L]).mean()))
                    if wp is not None:
                        xa = transform_shared_space(codebook[np.clip(ia[:L], 0, codebook.shape[0]-1)],
                                                    wp.center, wp.whitening_matrix)
                        xb = transform_shared_space(codebook[np.clip(ib[:L], 0, codebook.shape[0]-1)],
                                                    wp.center, wp.whitening_matrix)
                        deltaR.append(xa - xb)
        seed_rows.append({"prompt_id": pid,
                          "seed_token_disagreement_mean": float(np.mean(disagreements)) if disagreements else float("nan"),
                          "duration_frames_min": int(min(durations)),
                          "duration_frames_max": int(max(durations)),
                          "duration_spread_frames": int(max(durations) - min(durations))})
    _write_csv(os.path.join(metrics_dir, "seed_diversity.csv"), seed_rows)

    # Sigma_D from clean round-trip diffs (per_sample.csv has whitened cos, but we need
    # the diffs); approximate tr(Sigma_D) from stored latents where available.
    R_trace = None
    if wp is not None and deltaR:
        sigma_r = metrics.cov_from_diffs(np.concatenate(deltaR, axis=0))
        # Sigma_D: recompute clean round-trip diffs in whitened space
        dD = []
        for e in ok:
            sid = e["sample_id"]
            lat_path = os.path.join(sample_root, sid, "encoder_semantic_latents.pt")
            if not os.path.exists(lat_path):
                continue
            reenc = torch.load(lat_path, weights_only=False)
            reenc = reenc.numpy() if hasattr(reenc, "numpy") else np.asarray(reenc)
            emit = torch.load(os.path.join(sample_root, sid, "semantic_ids.pt")).numpy().ravel()
            orig = codebook[np.clip(emit, 0, codebook.shape[0]-1)]
            L = min(len(orig), len(reenc))
            xo = transform_shared_space(orig[:L], wp.center, wp.whitening_matrix)
            xr = transform_shared_space(reenc[:L], wp.center, wp.whitening_matrix)
            dD.append(xo - xr)
        if dD:
            sigma_d = metrics.cov_from_diffs(np.concatenate(dD, axis=0))
            R_trace = metrics.trace_ratio(sigma_r, sigma_d)

    # ---- gate decision (§21) ----------------------------------------------
    per_sample = _load_csv(os.path.join(metrics_dir, "per_sample.csv"))
    det = _load_csv(os.path.join(metrics_dir, "encoder_determinism.csv"))

    def _floats(col):
        return [float(r[col]) for r in per_sample if r.get(col) not in (None, "", "nan")]

    # E0-A coordinate invariant Q(h)==tokenizer-id drives the §21.B gate.
    coord = _floats("coordinate_agreement")
    overall_agree = float(np.mean(coord)) if coord else float("nan")
    agree_decision = graded_agreement_decision(
        overall_agree, cfg["acceptance"]["pass_min"], cfg["acceptance"]["investigate_min"]) \
        if coord else "NA"
    # §12 clean decode->re-encode round-trip token match (reported alongside).
    rt = _floats("roundtrip_token_match")
    roundtrip_match = float(np.mean(rt)) if rt else float("nan")
    offsets_ok = all(str(r.get("offset_verified")) == "True" for r in per_sample) if per_sample else False
    repeat_ok = all(str(r.get("repeat_identical")) == "True" for r in det) if det else False
    seam_lengths = [int(r["seam_contamination_frames"]) for r in det if r.get("seam_contamination_frames")]

    H_raw_all = float(np.mean([r["H_raw_mean"] for r in ent_rows])) if ent_rows else float("nan")
    H_fin_all = float(np.mean([r["H_final_mean"] for r in ent_rows])) if ent_rows else float("nan")
    seed_disagree_all = float(np.nanmean([r["seed_token_disagreement_mean"] for r in seed_rows])) if seed_rows else float("nan")

    # §21.F stochasticity check (heuristic thresholds; ablation is the arbiter)
    weak_stoch = (not np.isnan(H_raw_all) and H_raw_all < 0.2) or \
                 (not np.isnan(seed_disagree_all) and seed_disagree_all < 0.02)
    binding_constraint = ("truncation" if (H_raw_all - H_fin_all) > 0.5 and H_fin_all < 0.2
                          else "base_distribution" if weak_stoch else "neither/healthy")

    decision, recommend = _decide(agree_decision, offsets_ok, repeat_ok, weak_stoch)

    aggregate = {
        "n_ok": len(ok), "n_failed": len(failed),
        "coordinate_agreement_overall": overall_agree,
        "coordinate_agreement_decision": agree_decision,
        "roundtrip_token_match_overall": roundtrip_match,
        "offsets_all_verified": offsets_ok,
        "encoder_repeat_deterministic": repeat_ok,
        "seam_contamination_frames_max": int(max(seam_lengths)) if seam_lengths else None,
        "H_raw_mean": H_raw_all, "H_final_mean": H_fin_all,
        "seed_token_disagreement_mean": seed_disagree_all,
        "binding_constraint": binding_constraint,
        "R_trace_crude_upper_bound": R_trace,
        "weak_stochasticity": weak_stoch,
        "gate_decision": decision, "recommendation": recommend,
    }
    io.write_json(os.path.join(metrics_dir, "aggregate.json"), aggregate)

    _figures(fig_dir, ent_rows, seed_rows, per_sample, det, plt)
    _write_report(art, cfg, md, aggregate, ent_rows, seed_rows, per_sample, det, failed, R_trace)
    print("E0 analyze done. Gate decision:", decision)
    print("Report:", os.path.join(art, "E0_REPORT.md"))


def _decide(agree_decision, offsets_ok, repeat_ok, weak_stoch):
    if agree_decision == "HARD_STOP" or not repeat_ok:
        return "HARD STOP", "Debug E0 (coordinate system / encoder determinism). Do not proceed."
    if agree_decision in ("INVESTIGATE", "NA") or not offsets_ok:
        return "INVESTIGATE", "Resolve agreement/offset issues before E1."
    if weak_stoch:
        return "INVESTIGATE", ("Run the §21.F raised-temperature/top-k ablation. If H_t, "
                               "seed disagreement and R_trace recover -> proceed with ablation "
                               "documented; if not -> HARD STOP.")
    return "PASS", "Proceed to E1 (with explicit approval)."


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _figures(fig_dir, ent_rows, seed_rows, per_sample, det, plt):
    if ent_rows:
        hr = [r["H_raw_mean"] for r in ent_rows]
        hf = [r["H_final_mean"] for r in ent_rows]
        plt.figure(figsize=(7, 4))
        plt.hist(hr, bins=12, alpha=0.6, label="H(p_raw)")
        plt.hist(hf, bins=12, alpha=0.6, label="H(p_final)")
        plt.xlabel("per-sample mean entropy (nats)"); plt.ylabel("count"); plt.legend()
        plt.title("E0-b: generation entropy, raw vs final")
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "entropy_hist_raw_vs_final.png")); plt.close()

        nf = [r["Neff_final_mean"] for r in ent_rows]
        plt.figure(figsize=(7, 4)); plt.hist(nf, bins=12, color="steelblue")
        plt.xlabel("effective vocabulary size (final)"); plt.ylabel("count")
        plt.title("E0-b: effective vocabulary (final)")
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "effective_vocab_hist.png")); plt.close()
    if per_sample:
        tm = [float(r["coordinate_agreement"]) for r in per_sample
              if r.get("coordinate_agreement") not in (None, "", "nan")]
        if tm:
            plt.figure(figsize=(7, 4)); plt.hist(tm, bins=12, color="seagreen")
            plt.xlabel("coordinate agreement  Q(h)==tokenizer id"); plt.ylabel("count")
            plt.title("E0-A: coordinate-system agreement")
            plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "quantizer_agreement.png")); plt.close()
    if seed_rows:
        sd = [r["seed_token_disagreement_mean"] for r in seed_rows]
        plt.figure(figsize=(7, 4)); plt.bar(range(len(sd)), sd, color="indianred")
        plt.xlabel("text index"); plt.ylabel("seed token disagreement")
        plt.title("§14: same-text/same-voice seed diversity")
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "seed_diversity.png")); plt.close()
    if det:
        sc = [int(r["seam_contamination_frames"]) for r in det if r.get("seam_contamination_frames")]
        if sc:
            plt.figure(figsize=(7, 4)); plt.hist(sc, bins=8, color="slateblue")
            plt.xlabel("seam contamination (frames)"); plt.ylabel("count")
            plt.title("§9.5: encoder seam contamination")
            plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "seam_contamination.png")); plt.close()


def _write_report(art, cfg, md, agg, ent_rows, seed_rows, per_sample, det, failed, R_trace):
    from datetime import datetime, timezone  # allowed here (script host, not workflow)
    lines = []
    A = lines.append
    A(f"# E0 Report — SEMark representation gate\n")
    A(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n")
    A("## 1. Environment and revisions")
    for k in ("model", "torch_version", "transformers_version", "cuda_version", "gpu",
              "git_sha", "git_dirty"):
        A(f"- **{k}**: {md.get(k) if k != 'model' else cfg['model']['id']}")
    A(f"- **tokenizer**: {cfg['model']['tokenizer_id']}")
    A(f"- **sampling (actual)**: {md.get('sampling_config_actual')}\n")
    A("## 2. Discovered hook locations")
    A(f"```\n{md.get('discovered_structure')}\n```\n")
    A("## 3. Semantic-space bridge")
    A("model semantic id ↔ first semantic codebook ↔ encoder pre-quant latent ↔ semantic quantizer\n")
    A("## 4. Whitening (W, c_bar) — §6.1")
    A("Defined from the FROZEN semantic codebook (public object). **E1.5 may refit W/c_bar "
      "on fit-split latents; if it does, ALL E0 representation metrics must be recomputed "
      "under the new transform, not carried forward.**\n")
    A("## 5. Coordinate-system agreement Q(h)==tokenizer-id (valid frames)")
    A(f"- overall coordinate agreement: **{agg['coordinate_agreement_overall']}** "
      f"→ **{agg['coordinate_agreement_decision']}**")
    A(f"- clean decode→re-encode round-trip token match: **{agg['roundtrip_token_match_overall']}**")
    A(f"- offsets all verified by cross-correlation: **{agg['offsets_all_verified']}**")
    A("- NOTE: state the valid-frame exclusion definition explicitly once the runtime "
      "padding/BOS/EOS frames are known (§11).\n")
    A("## 6. Clean round-trip stability")
    A(_mini_table(per_sample, ["sample_id", "coordinate_agreement", "roundtrip_token_match",
                               "cos_mean", "l2_mean", "whitened_cos_mean", "frame_offset",
                               "offset_verified"]))
    A("## 7. Encoder determinism and seam contamination (§9.5)")
    A(f"- repeat-encode deterministic: **{agg['encoder_repeat_deterministic']}**")
    A(f"- max seam contamination: **{agg['seam_contamination_frames_max']}** frames "
      "(bounds the shortest crop the detector can ever score; constrains §7.3)\n")
    A("## 8. Generation stochasticity — RAW vs FINAL (§7.1)")
    A(f"- mean H(p_raw): **{agg['H_raw_mean']:.4f}** nats | mean H(p_final): "
      f"**{agg['H_final_mean']:.4f}** nats")
    A(f"- **binding constraint**: {agg['binding_constraint']} "
      "(truncation → raise top_k; base_distribution → §21.F ablation)")
    A(f"- headline V_t is the TRUNCATED value (under p_final); full-codebook V_t (p_raw) "
      "is the secondary number in entropy_metrics.csv\n")
    A("## 9. Same-text/same-voice seed diversity (§14)")
    A(_mini_table(seed_rows, ["prompt_id", "seed_token_disagreement_mean",
                              "duration_spread_frames"]))
    A("## 10. Trace ratio R_trace (§15) — CRUDE, UPPER BOUND")
    A(f"- R_trace = tr(Σ_R)/(tr(Σ_D)+ε) = **{R_trace}**")
    A("- CRUDE DIAGNOSTIC: naive index alignment, NOT comparable to Stage-1 MFA-aligned Σ_R.")
    A("- SYSTEMATIC UPPER BOUND: Σ_D here is clean round trip only; Stage-1 adds MP3/Opus/"
      "resampling/noise/filtering, which only increase tr(Σ_D). Read as weaker than it looks.\n")
    A("## 11. Known uncontrolled factors")
    A("- single fixed voice conditioning (§8.1): cannot separate 'model is deterministic' "
      "from 'this reference clip yields low variance';")
    A("- read-speech-only prompts; clean-channel-only Σ_D.\n")
    A("## 12. Discrepancies from the Notion spec (confirmed at runtime)")
    A("- **D1**: the codec semantic codebook is **2048×256** (read from the model), not the "
      "4096 in the config JSON nor the 2048/16 acoustic assumption of §7.")
    A("- **D2**: the encode-path semantic quantizer lives inside transformers' Mimi under "
      "`model.encoder.quantizer.semantic_residual_vector_quantizer`; hooks discovered, not assumed.")
    A("- **D3**: semantic latent dim = **256** (post `input_proj`); pre-proj is 512.")
    A("- **D4**: the talker semantic head (`talker.codec_head`) is **3072-wide** while the codec "
      "codebook is 2048 — an index-space gap. p_t is over 3072; V_t restricts to the codec-code "
      "region assuming an identity map for tokens < 2048 (flagged, not assumed silently).\n")
    A(f"## 13. Gate decision: **{agg['gate_decision']}**")
    A(f"## 14. Recommended next step: {agg['recommendation']}")
    if failed:
        A(f"\n> {len(failed)} sample(s) FAILED and are preserved in the manifest with reasons.")
    with open(os.path.join(art, "E0_REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _mini_table(rows, cols):
    if not rows:
        return "_(no rows)_\n"
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows[:40]:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join([head, sep] + body) + "\n"


if __name__ == "__main__":
    main()
