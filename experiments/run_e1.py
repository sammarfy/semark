"""E1 provenance/channel probe (spec §7-§9). RUNS ON THE GPU HOST via run(adapter).

GATED: only run after E0-e passes and the E0 detector gates hold. No watermarking. Estimates,
per synchronous derivative family c, the regularized generalized-eigenvalue spectrum of
re-performance variance (Sigma_R) against derivative variance (Sigma_D^(c)), with text-level
bootstrap. Sigma_D/Sigma_R are second moments of DIFFERENCE vectors (pooling frames is correct
here) — this is NOT the within-frame phi/psi channel covariance C_raw, which stays in branch.py.

Re-performance alignment here is naive-prefix (E1 DIAGNOSTIC). MFA phone-occurrence alignment is
required for E1.5 Stage-1; if the per-text duration spread exceeds the config bound, STOP and
report the alignment failure rather than trusting Sigma_R.
"""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io  # noqa: E402
from src.transforms import whitening_from_codebook, transform_shared_space  # noqa: E402
from src.prewm.splits import text_splits  # noqa: E402
from src.prewm.channels import apply_synchronous, codec_roundtrip, CODEC_FAMILIES  # noqa: E402
from src.alignment import crosscorr_peak_lag, norm_sequence  # noqa: E402
from src.prewm import spectrum as sp  # noqa: E402
from src.prewm.covariance import pooled_covariance  # noqa: F401  (kept out of the C path)
from src.prewm.bootstrap import bootstrap_text_indices  # noqa: E402
from src.metrics import cov_from_diffs  # noqa: E402


def _write_wav_tmp(wav, sr):
    import tempfile, soundfile as sf
    t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(t.name, np.asarray(wav, np.float32).ravel(), sr)
    return t.name


def run(adapter, config="configs/e1.yaml", max_texts=None, n_seeds=None, families=None, progress=10):
    cfg = yaml.safe_load(open(config))
    art = cfg["paths"]["artifacts_root"]
    os.makedirs(os.path.join(art, "spectra"), exist_ok=True)
    os.makedirs(os.path.join(art, "figures"), exist_ok=True)

    # b_context from E0-e (spec §7.1); default modest interior if not present.
    b_ctx = 8
    bpath = "artifacts/e0_boundary/aggregate.json"
    if os.path.exists(bpath):
        b_ctx = json.load(open(bpath)).get("b_context") or 8

    prompts = io.read_json(cfg["design"]["prompts_file"])["prompts"]
    if max_texts:
        prompts = prompts[:max_texts]
    seeds = cfg["design"]["seeds"][:n_seeds] if n_seeds else cfg["design"]["seeds"]
    sd = cfg["design"]["splits"]
    splits = text_splits([p["id"] for p in prompts], sd["fit"] if not max_texts else max(2, max_texts // 2),
                         sd["dev"] if not max_texts else 1, sd["test"] if not max_texts else 1, sd["seed"])
    io.write_json(os.path.join(art, "splits.json"), splits)

    codebook = adapter.get_semantic_codebook()
    wp = whitening_from_codebook(codebook, ridge=cfg["transform"]["ridge"])
    families = families or cfg["channels"]["synchronous"]
    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        import urllib.request
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")
    sr = adapter.sample_rate_hz

    def whitened(wav):
        sp_space = adapter.encode_semantic_space(wav)
        z = sp_space.frame_latents
        return transform_shared_space(z, wp.center, wp.whitening_matrix)   # [T, d]

    # ---- generate 800 realizations; cache clean + per-channel whitened latents ----
    # (this is the compute-heavy Colab step; interior-trim b_ctx frames each side)
    per_text_seed_clean = {}     # (text_id, seed) -> z_clean [Ti, d]
    deltaD = {f: [] for f in families if f != "clean"}
    rng = np.random.default_rng(0)
    import torch
    total = len(prompts) * len(seeds); count = 0; kept = 0
    print(f"E1: generating {total} realizations x {len(families)} channels (b_context={b_ctx})")
    for p in prompts:
        for s in seeds:
            torch.manual_seed(s)
            wavs, _ = adapter.model.generate_voice_clone(text=p["text"], language="English",
                                                         voice_clone_prompt=vp)
            wav = np.asarray(wavs[0], np.float32)
            z_clean = whitened(wav)
            count += 1
            lo, hi = b_ctx, len(z_clean) - b_ctx
            if hi - lo >= 8:
                kept += 1
                per_text_seed_clean[(p["id"], s)] = z_clean[lo:hi]
                for fam in families:
                    if fam == "clean":
                        continue
                    wav_c = (codec_roundtrip(fam, wav, sr) if fam in CODEC_FAMILIES
                             else apply_synchronous(fam, wav, sr, rng))
                    z_c = whitened(wav_c)
                    # codecs add a fixed encoder delay -> offset-align latents before differencing
                    off = crosscorr_peak_lag(norm_sequence(z_clean), norm_sequence(z_c), max_lag=6)
                    a, b = (z_clean, z_c[off:]) if off >= 0 else (z_clean[-off:], z_c)
                    mfr = min(len(a), len(b)); lo2, hi2 = b_ctx, mfr - b_ctx
                    if hi2 - lo2 >= 8:
                        deltaD[fam].append((a[:mfr] - b[:mfr])[lo2:hi2])
            if progress and count % progress == 0:
                print(f"  {count}/{total} generated ({kept} with usable interior)")

    # ---- cache whitened latents so re-analysis needs no regeneration ----
    import pickle
    os.makedirs(os.path.join(art, "cache"), exist_ok=True)
    pickle.dump({"per_text_seed_clean": {f"{k[0]}|{k[1]}": v for k, v in per_text_seed_clean.items()},
                 "deltaD": deltaD, "b_ctx": b_ctx, "families": families},
                open(os.path.join(art, "cache", "latents.pkl"), "wb"))
    print("cached latents ->", os.path.join(art, "cache", "latents.pkl"))

    # ---- Sigma_D^(c) (pooled difference second-moment — correct for Sigma_D) ----
    SigmaD = {f: cov_from_diffs(np.concatenate(v)) for f, v in deltaD.items() if v}

    # ---- Sigma_R: per-text across-seed scatter (naive prefix), averaged over texts ----
    dur_spread = []
    scatters = []
    by_text = {}
    for (tid, s), z in per_text_seed_clean.items():
        by_text.setdefault(tid, []).append(z)
    for tid, zs in by_text.items():
        L = min(len(z) for z in zs)
        dur_spread.append(max(len(z) for z in zs) - L)
        if L < 4 or len(zs) < 2:
            continue
        stack = np.stack([z[:L] for z in zs])       # [n_seed, L, d]
        mean = stack.mean(axis=0, keepdims=True)
        dev = (stack - mean).reshape(-1, stack.shape[-1])
        scatters.append(cov_from_diffs(dev))
    alignment_ok = (np.median(dur_spread) <= cfg["reperformance"]["max_duration_spread_frames"]) if dur_spread else False
    SigmaR = np.mean(scatters, axis=0) if scatters else None

    # ARTIFACT CONTROL: recompute Sigma_R using only near-matched-duration realizations (spread<=1),
    # so naive-prefix misalignment cannot inflate it. If the tail vanishes here, it was misalignment.
    scatters_md = []
    for tid, zs in by_text.items():
        lens = [len(z) for z in zs]
        Lmed = int(np.median(lens))
        matched = [z for z in zs if abs(len(z) - Lmed) <= 1]
        if len(matched) < 2:
            continue
        Lm = min(len(z) for z in matched)
        st = np.stack([z[:Lm] for z in matched]); dv = (st - st.mean(0, keepdims=True)).reshape(-1, st.shape[-1])
        scatters_md.append(cov_from_diffs(dv))
    SigmaR_md = np.mean(scatters_md, axis=0) if scatters_md else None

    # ---- regularized spectra + TWO-OBJECTIVE PLANE (exposes tiny-denominator ratios) ----
    def two_objective(Sd, Sr, eps, r=8):
        M = sp.regularized_M(Sd, Sr, eps=eps); V = sp.top_eigvecs(M, r)
        # report v^T Sigma_D v and v^T Sigma_R v for top-r M-eigenvectors (in ORIGINAL space)
        Bi = sp.inv_sqrt_psd(Sd + eps*np.eye(Sd.shape[0])); W = Bi @ V   # back to original coords
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        vd = np.array([w @ Sd @ w for w in W.T]); vr = np.array([w @ Sr @ w for w in W.T])
        return vd.tolist(), vr.tolist()

    eps_mid = cfg["spectrum"]["eps_grid"][2]
    results = {}
    for f, Sd in SigmaD.items():
        if SigmaR is None:
            continue
        per_eps = {}
        for eps in cfg["spectrum"]["eps_grid"]:
            w = sp.spectrum(sp.regularized_M(Sd, SigmaR, eps=eps))
            per_eps[str(eps)] = {"top32": w[:32].tolist(), "n_above_1": int((w > 1).sum())}
        vd, vr = two_objective(Sd, SigmaR, eps_mid)
        md = {}
        if SigmaR_md is not None:
            w_md = sp.spectrum(sp.regularized_M(Sd, SigmaR_md, eps=eps_mid))
            md = {"matched_dur_top6": w_md[:6].tolist(), "matched_dur_n_above_1": int((w_md > 1).sum())}
        results[f] = {"eps_sweep": per_eps,
                      "two_objective_top8": {"Sigma_D": vd, "Sigma_R": vr},
                      "matched_duration": md,
                      "eff_rank_Sd": sp.effective_rank(Sd), "eff_rank_Sr": sp.effective_rank(SigmaR),
                      "cond_B": sp.condition_number(Sd, eps=eps_mid),
                      "mean_trace_Sd": float(np.trace(Sd)/Sd.shape[0]),
                      "mean_trace_Sr": float(np.trace(SigmaR)/SigmaR.shape[0])}
        json.dump(results[f], open(os.path.join(art, "spectra", f"spectrum_{f}.json"), "w"), indent=2)

    dim = codebook.shape[1]
    # ---- honest gate: a real tail is LOW-RANK and not a tiny-denominator/alignment artifact ----
    def family_verdict(r):
        mid = r["eps_sweep"][str(eps_mid)]
        top = mid["top32"][0]; nabove = mid["n_above_1"]
        tiny_denom = r["mean_trace_Sd"] < 0.05 * r["mean_trace_Sr"]      # Sigma_D negligible vs Sigma_R
        pervasive = nabove > 0.5 * dim                                    # not low-rank
        md = r.get("matched_duration", {})
        collapses = bool(md) and md.get("matched_dur_n_above_1", nabove) < 0.3 * nabove
        return {"top": top, "n_above_1": nabove, "tiny_denominator": tiny_denom,
                "pervasive_not_lowrank": pervasive, "tail_collapses_when_aligned": collapses,
                "credible_lowrank_tail": bool(not tiny_denom and not pervasive and not collapses and top > 1.5)}
    verdicts = {f: family_verdict(r) for f, r in results.items() if f != "clean"}
    credible = [f for f, v in verdicts.items() if v["credible_lowrank_tail"]]
    if not alignment_ok:
        gate, why = "STOP", f"naive-prefix alignment unreliable (median duration spread {np.median(dur_spread):.0f}); MFA required"
    elif any(v["tiny_denominator"] or v["pervasive_not_lowrank"] for v in verdicts.values()):
        gate, why = "INVESTIGATE", ("eigen-tail is a tiny-Sigma_D / alignment artifact (pervasive, huge "
                                    "ratios) — DSP channels too weak and/or Sigma_R inflated by misalignment. "
                                    "Need realistic codec channels (MP3/Opus) + matched/MFA alignment.")
    elif len(credible) >= 2:
        gate, why = "PROCEED", "multiple realistic families show a credible LOW-RANK tail (verify bootstrap/held-out)"
    else:
        gate, why = "INVESTIGATE", "no credible low-rank tail yet"

    metrics = {"n_texts": len(by_text), "families": list(results), "b_context": b_ctx,
               "verdicts": verdicts,
               "alignment_ok": bool(alignment_ok), "median_duration_spread": float(np.median(dur_spread)) if dur_spread else None,
               "credible_lowrank_families": credible, "gate": gate, "why": why,
               "note": "codec (MP3/Opus) families need ffmpeg on host; add to complete the realistic set."}
    io.write_json(os.path.join(art, "metrics", "e1_metrics.json") if os.path.isdir(os.path.join(art, "metrics"))
                  else _mkdir_json(art), metrics)
    _report(art, metrics, results)
    print(f"E1 done. gate={gate} | credible_lowrank={credible} | why={why}")
    return metrics


def _mkdir_json(art):
    os.makedirs(os.path.join(art, "metrics"), exist_ok=True)
    return os.path.join(art, "metrics", "e1_metrics.json")


def _report(art, m, results):
    L = ["# E1 Report — derivative vs re-performance (Q11-Q13)\n",
         f"- texts={m['n_texts']}, b_context={m['b_context']}, alignment_ok={m['alignment_ok']} "
         f"(median duration spread {m['median_duration_spread']})",
         "- Sigma_D/Sigma_R are difference second-moments (pooling frames is correct here); the "
         "within-frame phi/psi channel covariance C stays in the branch-rollout path.",
         "- re-performance alignment is naive-prefix (E1 DIAGNOSTIC); MFA required for E1.5.\n",
         "## Regularized eigen-tail by family (eigenvalues > 1 mean re-perf variance exceeds derivative)"]
    for f, r in results.items():
        mid = r["eps_sweep"][list(r["eps_sweep"])[len(r["eps_sweep"]) // 2]]
        L.append(f"- **{f}**: top eigenvalue {mid['top32'][0]:.3f}, {mid['n_above_1']} above 1 "
                 f"(effrank Sd {r['eff_rank_Sd']:.1f}, cond B {r['cond_B']:.1f})")
    L.append(f"\n## E1 gate: **{m['gate']}** — {m['why']}")
    L.append("\n> Codec families (MP3/Opus) need ffmpeg on the host; add them for the realistic set "
             "before a PROCEED is trusted. Bootstrap principal-angle stability is computed on the full run.")
    open(os.path.join(art, "E1_REPORT.md"), "w").write("\n".join(L) + "\n")


def main():
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e1.yaml"))
    run(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
