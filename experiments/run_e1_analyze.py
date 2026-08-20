"""Deep re-analysis of the cached E1 latents (spec §5.0.2, §10). No GPU, no regeneration.

Answers the questions the pilot gate could not: is the huge RTD ratio a real LOW-RANK
performance-bound subspace, or pervasive tiny-denominator structure? Is there a low-low control
subspace? Does the fitted subspace separate re-performance on HELD-OUT texts? Is it bootstrap-
stable? Reads artifacts/e1/cache/latents.pkl (whitened latents keyed by text|seed + pooled
channel diffs). Pure numpy; prints a paste-able summary.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prewm import spectrum as sp  # noqa: E402
from src.metrics import cov_from_diffs  # noqa: E402
from src.prewm.splits import text_splits  # noqa: E402


def _sigma_r(text_latents: dict, texts, matched=False):
    """Per-text across-seed scatter, averaged over the given texts. matched=True keeps only
    near-median-duration seeds (no naive-prefix drift)."""
    scat = []
    for tid in texts:
        zs = text_latents.get(tid, [])
        if len(zs) < 2:
            continue
        if matched:
            med = int(np.median([len(z) for z in zs]))
            zs = [z for z in zs if abs(len(z) - med) <= 1]
            if len(zs) < 2:
                continue
        L = min(len(z) for z in zs)
        st = np.stack([z[:L] for z in zs])
        dv = (st - st.mean(0, keepdims=True)).reshape(-1, st.shape[-1])
        scat.append(cov_from_diffs(dv))
    return np.mean(scat, axis=0) if scat else None


def analyze(art="artifacts/e1", eps=1e-2, r=8):
    cache = pickle.load(open(os.path.join(art, "cache", "latents.pkl"), "rb"))
    # regroup latents by text
    text_latents = {}
    for key, z in cache["per_text_seed_clean"].items():
        tid = key.split("|")[0]
        text_latents.setdefault(tid, []).append(np.asarray(z))
    texts = sorted(text_latents)
    SigmaD = {f: cov_from_diffs(np.concatenate(v)) for f, v in cache["deltaD"].items() if v}
    dim = next(iter(SigmaD.values())).shape[0]
    SigmaR = _sigma_r(text_latents, texts)

    print(f"=== E1 cached re-analysis: {len(texts)} texts, dim {dim}, eps {eps} ===")
    print(f"Sigma_R eff-rank {sp.effective_rank(SigmaR):.1f}, mean-trace {np.trace(SigmaR)/dim:.4f}")

    for fam, Sd in SigmaD.items():
        M = sp.regularized_M(Sd, SigmaR, eps=eps)
        w = sp.spectrum(M)
        # two-objective over the WHOLE M-eigenbasis (not just top r)
        _, Vall = np.linalg.eigh(0.5 * (M + M.T))
        Bi = sp.inv_sqrt_psd(Sd + eps * np.eye(dim))
        Worig = Bi @ Vall
        Worig /= (np.linalg.norm(Worig, axis=0, keepdims=True) + 1e-12)
        sd_dir = np.einsum('ij,jk,ik->i', Worig.T, Sd, Worig.T)
        sr_dir = np.einsum('ij,jk,ik->i', Worig.T, SigmaR, Worig.T)
        # low-low control: low Sigma_D AND low Sigma_R (bottom quartiles of both)
        lowD = sd_dir <= np.percentile(sd_dir, 25)
        lowR = sr_dir <= np.percentile(sr_dir, 25)
        n_lowlow = int((lowD & lowR).sum())
        # is the ratio a low-rank tail or pervasive? gap between top and the bulk
        ratio_sorted = np.sort(w)[::-1]
        print(f"\n--- {fam} ---")
        print(f"  spectrum: top {ratio_sorted[0]:.1f}, r8 {ratio_sorted[7]:.1f}, median {np.median(w):.2f}, "
              f"min {w.min():.3f}; n>1 {(w>1).sum()}/{dim}")
        print(f"  Sigma_D mean-trace {np.trace(Sd)/dim:.4f} | Sigma_D per-dir quartiles "
              f"{np.round(np.percentile(sd_dir,[25,50,75]),4).tolist()}")
        print(f"  Sigma_R per-dir quartiles {np.round(np.percentile(sr_dir,[25,50,75]),3).tolist()}")
        print(f"  low-low control directions (lowD & lowR): {n_lowlow}/{dim}")

    # held-out separation (spec §10.2): fit M on FIT texts' Sigma_R, evaluate the top-r subspace's
    # generalized Rayleigh ratio on TEST texts' Sigma_R (pooled Sigma_D). Use the most realistic codec.
    fam = "mp3_64" if "mp3_64" in SigmaD else next(iter(SigmaD))
    Sd = SigmaD[fam]
    n = len(texts)
    spl = text_splits(texts, max(2, int(0.6 * n)), max(1, int(0.2 * n)), max(1, n - int(0.6 * n) - int(0.2 * n)))
    Sr_fit = _sigma_r(text_latents, spl["fit"]); Sr_test = _sigma_r(text_latents, spl["test"])
    if Sr_fit is not None and Sr_test is not None:
        M_fit = sp.regularized_M(Sd, Sr_fit, eps=eps)
        P = sp.top_eigvecs(M_fit, r)                 # top-r performance-bound subspace (fit)
        Bi = sp.inv_sqrt_psd(Sd + eps * np.eye(dim))
        Porig = Bi @ P; Porig /= np.linalg.norm(Porig, axis=0, keepdims=True) + 1e-12
        # held-out generalized Rayleigh: Sigma_R_test vs Sigma_D on the fit subspace vs random
        def gr(cov, W):
            num = np.einsum('ij,jk,ik->i', W.T, cov, W.T)
            den = np.einsum('ij,jk,ik->i', W.T, Sd, W.T)
            return float(np.mean(num / (den + eps)))
        rng = np.random.default_rng(0)
        Rrand = rng.standard_normal((dim, r)); Rrand /= np.linalg.norm(Rrand, axis=0, keepdims=True)
        print(f"\n=== HELD-OUT ({fam}): fit P on {len(spl['fit'])} texts, eval on {len(spl['test'])} ===")
        print(f"  held-out Rayleigh(Sigma_R_test/Sigma_D) on fit-P: {gr(Sr_test, Porig):.1f}  vs random dirs: {gr(Sr_test, Rrand):.1f}")
        print("  (P generalizes if held-out ratio on P >> random; both huge => tiny-Sigma_D, not a real subspace)")

    # bootstrap stability of top-r subspace across text resamples (spec §10.2 principal angles)
    from src.prewm.bootstrap import bootstrap_text_indices
    reps = bootstrap_text_indices(texts, 30, seed=1)
    Mfull = sp.regularized_M(Sd, SigmaR, eps=eps); Pfull = sp.top_eigvecs(Mfull, r)
    angs = []
    for rp in reps[:20]:
        Sr_b = _sigma_r(text_latents, rp)
        if Sr_b is None:
            continue
        Pb = sp.top_eigvecs(sp.regularized_M(Sd, Sr_b, eps=eps), r)
        angs.append(np.degrees(sp.principal_angles(Pfull, Pb)).max())
    if angs:
        print(f"\n=== bootstrap top-{r} subspace stability ({fam}) ===")
        print(f"  max principal angle across resamples: median {np.median(angs):.1f} deg, p90 {np.percentile(angs,90):.1f} deg")
        print("  (<~15 deg = stable low-rank geometry; >~45 deg = unstable/not reproducible)")

    print("\nVERDICT INPUTS: compare (a) low-rank vs pervasive n>1, (b) low-low control size, "
          "(c) held-out P ratio vs random, (d) bootstrap angle. A real performance-bound subspace "
          "needs a low-rank tail, a non-trivial low-low control, held-out P >> random, and stable angles.")


def main():
    analyze()


if __name__ == "__main__":
    main()
