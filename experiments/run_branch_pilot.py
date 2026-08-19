"""Branch-rollout pilot — within-frame C_raw and J_clean (spec §5, Phase D). GPU host.

Forcing lever: overwrite talker.codec_head logits at decode steps (CodecHeadForcer) — the only
faithful lever (prefix_allowed_tokens_fn is ignored). Every rollout forces base[:t] + candidate
v at t, a small free margin (encoder right-context), then eos to TERMINATE — so each generation
is ~t+margin+1 tokens (no derailment, no hang).

For each high-entropy interior anchor t:
  psi_bar(v) = whitened detector latent at frame t after decode->reencode (mean over CRN seeds)
  phi(v)     = u^T z_tilde_{codebook[v]}   (ordinary; identity map)
  C_t        = Cov_{v~p_t}[phi, psi_bar]   (within-frame; NEVER pooled)
aggregate -> J_clean. Guards: psi must vary across candidates; forcing the natural token must
reproduce the base frame-t latent; |J_t| <= 1 (Cauchy-Schwarz).
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.transforms import whitening_from_codebook, transform_shared_space  # noqa: E402
from src.sampling_trace import reconstruct_raw_and_final  # noqa: E402
from src.prewm.keys import diagnostic_keys  # noqa: E402
from src.prewm.candidates import select_candidates  # noqa: E402
from src.prewm.stochasticity import collision  # noqa: E402
from src.prewm.talker_map import TalkerCodecMap  # noqa: E402
from src.prewm.branch import BranchFrame, aggregate_frames  # noqa: E402
from src.prewm.forcing import CodecHeadForcer, isolate_schedule  # noqa: E402

EOS = 2150
OFFSET = 1   # talker emitted frame t -> re-encoded detector frame t+1 (measured, run_branch_align)


def pilot(adapter, n_anchors=8, k_max=6, n_cont=2, n_keys=64, b_context=10, margin=14,
          texts=None, out="artifacts/e0d/branch_rollouts"):
    import torch
    os.makedirs(out, exist_ok=True)
    codebook = adapter.get_semantic_codebook()
    codec_vocab, dim = codebook.shape
    talker_vocab = adapter.talker_vocab_size
    talker = adapter.model.model.talker
    head = adapter._codec_head()
    m = (TalkerCodecMap.from_json(json.load(open("artifacts/e0c/map.json")))
         if os.path.exists("artifacts/e0c/map.json")
         else TalkerCodecMap.identity_with_specials(talker_vocab, codec_vocab, {2150: "eos"}))
    ordinary_mask = m.ordinary_mask()

    wp = whitening_from_codebook(codebook, ridge=1e-3)
    U = diagnostic_keys(n_keys, dim)
    codec_feat = transform_shared_space(codebook, wp.center, wp.whitening_matrix) @ U.T   # [codec_vocab, n_keys]
    temp, topk, topp = (adapter.get_sampling_config()[k] for k in ("temperature", "top_k", "top_p"))

    ref = "/content/clone.wav"
    if not os.path.exists(ref):
        import urllib.request
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav", ref)
    vp = adapter.create_voice_prompt(ref, "Okay. Yeah. I resent you. I love you. I respect you. "
                                          "But you know what? You blew it! And thanks to you.")
    texts = texts or ["Rain fell steadily across the quiet valley throughout the afternoon.",
                      "After dinner they walked along the shore and talked about the future.",
                      "The committee agreed to postpone the decision until the following week."]

    def z_tilde(wav):
        z = adapter.encode_semantic_space(np.asarray(wav, np.float32)).frame_latents
        return transform_shared_space(z, wp.center, wp.whitening_matrix)   # [T, dim]

    def gen_forced(sched, seed):
        toks = []
        def pre(_m, a, k):
            ids = k.get("input_ids", a[0] if a else None)
            if ids is not None and ids.detach().shape[-1] == 1:
                toks.append(int(ids.detach().reshape(-1)[-1].item()))
        h = talker.register_forward_pre_hook(pre, with_kwargs=True)
        torch.manual_seed(seed)
        try:
            with CodecHeadForcer(adapter, sched):
                wavs, _ = adapter.model.generate_voice_clone(text=text, language="English",
                                                             voice_clone_prompt=vp)
        finally:
            h.remove()
        return np.asarray(wavs[0], np.float32), np.array(toks, np.int64)

    frames: list[BranchFrame] = []
    rows, natural_dev = [], []
    anchors_done = 0
    for text in texts:
        if anchors_done >= n_anchors:
            break
        # base generation: capture tokens + per-step codec_head logits (CAPTURE ONLY)
        base_tokens, logits_steps = [], []
        def base_pre(_m, a, k):
            ids = k.get("input_ids", a[0] if a else None)
            if ids is not None and ids.detach().shape[-1] == 1:
                base_tokens.append(int(ids.detach().reshape(-1)[-1].item()))
        def base_head(_m, _i, o):
            lg = o[0] if isinstance(o, tuple) else o
            if lg.shape[-2] == 1:
                logits_steps.append(lg.detach()[..., -1, :].float().cpu().numpy().reshape(-1))
        h1 = talker.register_forward_pre_hook(base_pre, with_kwargs=True)
        h2 = head.register_forward_hook(base_head)
        torch.manual_seed(0)
        try:
            base_wavs, _ = adapter.model.generate_voice_clone(text=text, language="English",
                                                              voice_clone_prompt=vp)
        finally:
            h1.remove(); h2.remove()
        base_tokens = np.array(base_tokens, np.int64)
        # The pre-hook captures the INITIAL input token first, so emitted frame s == base_tokens[s+1]
        # (confirmed +1 offset). logits_steps[s] is already frame-indexed (codec_head at step s ->
        # frame s). Align the forcing prefix to true emitted frames.
        base_frames = base_tokens[1:]
        T = min(len(base_frames), len(logits_steps))
        base_frames, logits_steps = base_frames[:T], logits_steps[:T]
        base_z = z_tilde(base_wavs[0])

        # p_t and local disagreement per step; pick HIGH-entropy interior anchors
        p_finals, disagree = [], []
        for lg in logits_steps:
            _, pf = reconstruct_raw_and_final(lg, temp, topk, topp)
            p_finals.append(pf); disagree.append(1.0 - float(collision(pf)))
        disagree = np.array(disagree)
        interior = np.arange(b_context, T - 1)
        interior = interior[interior < len(base_z)]
        if interior.size == 0:
            continue
        # most stochastic frames first (avoid repeated-token runs)
        order = interior[np.argsort(-disagree[interior])]
        take = min(max(1, n_anchors // len(texts) + 1), order.size)

        for t in order[:take]:
            if anchors_done >= n_anchors:
                break
            t = int(t)
            pf = p_finals[t]
            cs = select_candidates(pf, ordinary_mask, target_mass=0.995, k_max=k_max)
            if cs.ids.size < 2:
                continue
            cand = cs.ids
            psi = np.zeros((cand.size, n_keys))
            for ci, v in enumerate(cand):
                acc = []
                for s in range(n_cont):
                    wav, _ = gen_forced(isolate_schedule(base_frames, t, int(v), margin, EOS),
                                        seed=90000 + t * 100 + s)
                    zt = z_tilde(wav)
                    if t + OFFSET < len(zt):
                        acc.append(zt[t + OFFSET] @ U.T)
                if acc:
                    psi[ci] = np.mean(acc, axis=0)
            # natural-reproduction guard: forcing base[t] should ~reproduce the base frame-t latent
            wav_nat, _ = gen_forced(isolate_schedule(base_frames, t, int(base_frames[t]), margin, EOS),
                                    seed=90000 + t * 100)
            zt_nat = z_tilde(wav_nat)
            if t + OFFSET < len(zt_nat) and t + OFFSET < len(base_z):
                natural_dev.append(float(np.linalg.norm(zt_nat[t + OFFSET] - base_z[t + OFFSET])))

            phi = codec_feat[cand]
            for k in range(n_keys):
                frames.append(BranchFrame(anchors_done, text[:20], t, cand, pf[cand],
                                          phi[:, k], psi[:, k], covered_mass=cs.covered_mass))
            rows.append({"anchor": anchors_done, "t": t, "n_cand": int(cand.size),
                         "disagree": float(disagree[t]), "covered_mass": float(cs.covered_mass)})
            anchors_done += 1

    if not frames:
        print("[branch-pilot] no anchors — NOT IDENTIFIED"); return {"gate": "NOT_IDENTIFIED"}

    aggs = [aggregate_frames(frames[k::n_keys], equal="state") for k in range(n_keys)]
    J = np.array([a.J_clean for a in aggs])
    sigma_within = float(np.mean([a.sigma_matched for a in aggs]))
    psi_has_variance = sigma_within > 1e-6
    cs_ok = all(abs(j) <= 1 + 1e-6 for a in aggs for j in a.per_state_j)
    natural_ok = (np.median(natural_dev) < 5.0) if natural_dev else False   # base-latent reproduced
    identified = bool(psi_has_variance and cs_ok)

    with open(os.path.join(out, "..", "branch_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anchor", "t", "n_cand", "disagree", "covered_mass"])
        w.writeheader(); w.writerows(rows)
    metrics = {"n_anchors": anchors_done, "n_keys": n_keys, "n_cont": n_cont, "margin": margin,
               "J_clean_mean": float(J.mean()), "J_clean_median": float(np.median(J)),
               "J_clean_p10": float(np.percentile(J, 10)), "J_clean_p90": float(np.percentile(J, 90)),
               "V_mean": float(np.mean([a.V for a in aggs])),
               "C_raw_mean": float(np.mean([a.C_raw for a in aggs])),
               "sigma_matched_within": sigma_within, "psi_has_variance": psi_has_variance,
               "cauchy_schwarz_ok": cs_ok, "natural_repro_latent_dev_median": float(np.median(natural_dev)) if natural_dev else None,
               "natural_repro_ok": bool(natural_ok), "estimator": "within-frame mean_t(cov); NOT pooled",
               "gate": "IDENTIFIED" if identified else "NOT_IDENTIFIED",
               "reason": None if identified else "psi ~zero variance (forcing ineffective) or CS violated"}
    json.dump(metrics, open(os.path.join(out, "..", "branch_pilot_metrics.json"), "w"), indent=2)
    L = ["# Branch-rollout pilot — C_raw / J_clean (Q8/Q9)\n",
         f"- anchors={anchors_done} (high-entropy interior), keys={n_keys}, CRN seeds={n_cont}, margin={margin}",
         f"- **J_clean = {metrics['J_clean_mean']:.4f}** mean (median {metrics['J_clean_median']:.4f}, "
         f"p10 {metrics['J_clean_p10']:.4f}, p90 {metrics['J_clean_p90']:.4f})",
         f"- C_raw={metrics['C_raw_mean']:.4f}, V={metrics['V_mean']:.4f}, sigma_within={sigma_within:.4f}",
         f"- psi varies across candidates: {psi_has_variance}; natural-token reproduces base latent "
         f"(median dev {metrics['natural_repro_latent_dev_median']}): {natural_ok}",
         f"- Cauchy-Schwarz |J_t|<=1: {cs_ok}",
         f"\n## Identification: **{metrics['gate']}**",
         "J_clean is clean-channel recoverable-evidence efficiency, NOT a latency claim (needs E2 B)."]
    open(os.path.join(out, "..", "E0D_BRANCH_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"[branch-pilot] gate={metrics['gate']} | J_clean={metrics['J_clean_mean']:.4f} "
          f"| psi_var={psi_has_variance} | natural_ok={natural_ok} | CS_ok={cs_ok} | anchors={anchors_done}")
    return metrics


def main():
    import yaml
    from src.adapters.qwen3_tts import SpeechLMAdapter
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    pilot(SpeechLMAdapter.load(cfg, device="cuda:0"))


if __name__ == "__main__":
    main()
