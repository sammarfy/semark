# SEMark pre-watermark gate report

Status date: 2026-08-19. Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (bfloat16, Colab T4),
torch 2.11.0+cu128 / transformers 4.57.3 for the E0 traces; analysis torch 2.13.0 CPU.

This milestone is **staged and gated** (spec §15). All model-side phases need the GPU host
and runtime discovery of Qwen's generation path, exactly as E0 did. This report records what
is **answered now** (scientific backbone + E0-d1 on cached traces) and what is **pending the
Colab gate** (E0-c → E0-d2 → branch C/J → E0-e → E1 → E1.5). No watermark, no E2.

## Question table

| Q | Question | Result | Evidence | Gate |
|---|----------|--------|----------|------|
| — | Tokenizer shared space valid | YES | E0 §14 (agreement 1.000) | PASS |
| Q1 | 3072→2048 mapping exact | **RESOLVED: identity for [0,2048); [2048,3072) special** | talker.forward source: `codec_ids=cat(input_ids,predictor.seq)`; `e0c/E0C_REPORT.md` | PASS* |
| Q2 | Real special/control tokens | eos=pad=**2150**; [2048,3072) special/reserved | probe [4] + config | PASS* |
| Q3 | Local D_local = E[1−Σp²] | **0.226** (median 0.141) | `e0d/E0D1_REPORT.md` | DONE |
| Q4 | Share of 87% that is cascade | **≈74%** cascade, ≈26% local | `e0d/metrics.json` | DONE |
| Q5 | sigma_matched | **1.597** (64 keys, clean re-encode) | `e0d/sigma_matched.csv` | DONE |
| Q6 | sigma_operational differs? | DEFERRED (no operational null yet) | `e0d/sigma_operational.csv` | DEFERRED |
| Q7 | Correctly mapped V | **0.174** mean (median 0.081); ordinary mass 0.979 | `e0d/E0D2_REPORT.md` | DONE |

\* PASS is source-proven; the on-hardware confirmation artifact (`map.json`, sampled-token
verification) is written by `run_e0c_finalize.finalize(adapter)` on the GPU host.
| Q8 | Branch rollouts identify C_raw | PENDING (needs GPU branch intervention) | `branch.py` framework + tests | GATE |
| Q9 | J_clean (matched) | NOT IDENTIFIED yet | requires Q8 | GATE |
| Q10 | 29-frame seam = startup context? | PENDING | `run_e0e_boundary.py` (Colab) | GATE |
| Q11 | Derivative eigen-tail vs re-performance | PENDING | E1 (Colab, 800 gen) | GATE |
| Q12 | Stable low-rank RTD subspace | PENDING | E1 / E1.5 | GATE |
| Q13 | High-RTD preserves recoverable evidence | PENDING | E1.5 branch-J(α) | GATE |
| Q14 | Ready for E2 | **NO — INVESTIGATE** | this report | — |
| Q15 | Quality-safe B_headline | DEFERRED PENDING QUALITY-CALIBRATED E2 | (out of scope) | — |
| Q16 | 5/10/30 s latency claim | DEFERRED PENDING B_HEADLINE | (out of scope) | — |

## What is delivered now (runnable + tested)

- **Scientific backbone** (`src/prewm/`): within-frame covariance estimator with the
  mandatory pooled-covariance contrast test; identity-channel J=1; regularized spectrum
  `M=B^{-1/2}ΣR B^{-1/2}` (no direct inverse) + Ledoit-Wolf; text-level bootstrap; text
  splits with leakage guards; D_local; boundary-curve; branch candidate selection;
  talker→codec map consumer. 67 unit tests pass, 10 model-gated skips.
- **E0-d1 executed on cached E0 traces** (no regeneration): D_local, cascade share,
  sigma_matched, with text-level bootstrap CIs. See `e0d/E0D1_REPORT.md`.
- **E0-c probe** (`run_e0c_probe.py`) ready for the GPU host — the gate for all model-side
  V/C/J. It establishes the map from the codec ids Qwen sends downstream, NOT re-encoding.

## Key early scientific signal

Per-token watermark capacity looks **small and cascade-dominated**: only ≈26% of the 87%
seed disagreement is local same-context freedom (D_local≈0.23). Combined with the peaked
per-step distribution from E0 (N_eff≈1.75), the budget must come from **length**, not
per-token entropy. This does not kill the project but sharpens the §5–6 strength/duration
tradeoff and the §20 "entropy is the budget" concern.

## Final readiness

**INVESTIGATE** — the backbone is in place and the cached-data questions (Q3–Q5) are
answered, but the model-side gate (E0-c) and E1 have not yet run on the GPU host, so
V/C/J and RTD geometry are not yet identified.

Low-latency claim: **DEFERRED_PENDING_B_HEADLINE** (quality-safe B not measured; belongs to E2).
