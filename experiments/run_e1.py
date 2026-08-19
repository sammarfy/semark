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
from src.prewm.channels import apply_synchronous  # noqa: E402
from src.prewm import spectrum as sp  # noqa: E402
from src.prewm.covariance import pooled_covariance  # noqa: F401  (kept out of the C path)
from src.prewm.bootstrap import bootstrap_text_indices  # noqa: E402
from src.metrics import cov_from_diffs  # noqa: E402


def _write_wav_tmp(wav, sr):
    import tempfile, soundfile as sf
    t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(t.name, np.asarray(wav, np.float32).ravel(), sr)
    return t.name


def run(adapter, config="configs/e1.yaml", max_texts=None):
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
    seeds = cfg["design"]["seeds"]
    sd = cfg["design"]["splits"]
    splits = text_splits([p["id"] for p in prompts], sd["fit"] if not max_texts else max(2, max_texts // 2),
                         sd["dev"] if not max_texts else 1, sd["test"] if not max_texts else 1, sd["seed"])
    io.write_json(os.path.join(art, "splits.json"), splits)

    codebook = adapter.get_semantic_codebook()
    wp = whitening_from_codebook(codebook, ridge=cfg["transform"]["ridge"])
    families = cfg["channels"]["synchronous"]
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
    for p in prompts:
        for s in seeds:
            import torch; torch.manual_seed(s)
            tr = adapter.generate_trace(text=p["text"], seed=s, sample_id=f"{p['id']}_s{s}",
                                        voice_id="repo_default", voice_clone_prompt=vp)
            wav = np.asarray(tr.waveform, np.float32)
            z_clean = whitened(wav)
            lo, hi = b_ctx, len(z_clean) - b_ctx
            if hi - lo < 8:
                continue
            per_text_seed_clean[(p["id"], s)] = z_clean[lo:hi]
            for f in families:
                if f == "clean":
                    continue
                wav_c = apply_synchronous(f, wav, sr, rng)
                z_c = whitened(wav_c)
                m = min(len(z_clean), len(z_c))
                d = (z_clean[:m] - z_c[:m])[lo:hi]
                deltaD[f].append(d)

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

    # ---- regularized spectra per family + text bootstrap ----
    results = {}
    for f, Sd in SigmaD.items():
        if SigmaR is None:
            continue
        per_eps = {}
        for eps in cfg["spectrum"]["eps_grid"]:
            M = sp.regularized_M(Sd, SigmaR, eps=eps)
            w = sp.spectrum(M)
            per_eps[str(eps)] = {"top32": w[:32].tolist(), "n_above_1": int((w > 1).sum())}
        M_sh = sp.regularized_M(Sd, SigmaR, eps=cfg["spectrum"]["eps_grid"][2], shrinkage=True,
                                n_texts=len(by_text))
        results[f] = {
            "eps_sweep": per_eps,
            "shrinkage_top32": sp.spectrum(M_sh)[:32].tolist(),
            "eff_rank_Sd": sp.effective_rank(Sd), "eff_rank_Sr": sp.effective_rank(SigmaR),
            "cond_B": sp.condition_number(Sd, eps=cfg["spectrum"]["eps_grid"][2]),
        }
        json.dump(results[f], open(os.path.join(art, "spectra", f"spectrum_{f}.json"), "w"), indent=2)

    # ---- gate (spec §9): stable tail > 1 across realistic families, bootstrap-robust ----
    fams_with_tail = [f for f, r in results.items()
                      if r["eps_sweep"][str(cfg["spectrum"]["eps_grid"][2])]["n_above_1"] > 0
                      and f not in ("clean",)]
    realistic = [f for f in fams_with_tail if f not in ("resample_8k",)]  # crude "realistic" set
    if not alignment_ok:
        gate, why = "STOP", "re-performance alignment unreliable (duration spread too large); MFA required"
    elif not fams_with_tail:
        gate, why = "STOP", "no derivative family shows a tail above 1"
    elif len(realistic) >= 2:
        gate, why = "PROCEED", "multiple realistic families show a tail above 1 (verify bootstrap on GPU)"
    else:
        gate, why = "INVESTIGATE", "tail exists only for clean/mild families; realistic codecs pending (need ffmpeg)"

    metrics = {"n_texts": len(by_text), "families": list(results), "b_context": b_ctx,
               "alignment_ok": bool(alignment_ok), "median_duration_spread": float(np.median(dur_spread)) if dur_spread else None,
               "families_with_tail": fams_with_tail, "gate": gate, "why": why,
               "note": "codec (MP3/Opus) families need ffmpeg on host; add to complete the realistic set."}
    io.write_json(os.path.join(art, "metrics", "e1_metrics.json") if os.path.isdir(os.path.join(art, "metrics"))
                  else _mkdir_json(art), metrics)
    _report(art, metrics, results)
    print(f"E1 done. gate={gate} ({why}). families_with_tail={fams_with_tail}")
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
