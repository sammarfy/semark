# E0-d1 Report — local stochasticity + matched null scale

- model: Qwen/Qwen3-TTS-12Hz-0.6B-Base, temp=0.9
- source: cached E0 traces, 8 texts, 1504 frames. No regeneration.

## Q3 — local same-context stochastic freedom  D_local = E_t[1 - Σ p_t²]
- **D_local = 0.2260** (median 0.1406, p10 0.0017, p90 0.5615)
- text-bootstrap 95% CI: [0.2140, 0.2410]

## Q4 — how much of the 87% seed disagreement is cascade?
- full-trajectory seed disagreement ≈ 0.867; local same-context freedom = 0.2260
- **so ≈ 74% of the trajectory disagreement is autoregressive cascade**, not per-frame sampling freedom. Do NOT call 0.87 per-frame capacity.

## Q5 — matched detector null scale
- **sigma_matched = 1.5967** (whitened detector score, 64 fixed keys, clean re-encode)
- sigma_matched² text-bootstrap 95% CI: [2.4259, 2.7004]
- sigma_operational: **DEFERRED** (no legitimate operational-null population available yet)

> Caveat: interior-frame mask pending E0-e; all frames used. sigma_matched is on the E0 smoke population (8 texts × 4 seeds); E1 will re-estimate on the 800-realization corpus.
