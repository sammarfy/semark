# SEMark — E0 milestone (adapter + tests + 32-utterance smoke test)

Scope of this milestone (ONLY): the Qwen3-TTS adapter, unit tests, and the 32-utterance
E0 smoke test with E0-b stochastic-realization diagnostics. **No** watermark sampler,
RTD fitting, E1, E1.5, baselines, or attacks. Source of truth: the Notion draft
"08 — Draft Paper: SEMark".

E0 answers one question before more compute is spent:

> Can Qwen3-TTS expose a generation-side semantic codebook representation and a
> detector-side continuous semantic representation that live in the SAME coordinate
> system and remain meaningful after waveform decode → re-encode?

## What runs where

- **Model-agnostic core + unit tests** run anywhere (pure numpy). `pytest tests/`.
- **The 32-utterance E0 run** needs a CUDA GPU (free Colab T4 is enough for the 0.6B
  model). Use the Colab notebook `run_e0_colab.ipynb`.

## Layout

```
configs/     e0.yaml, e0_prompts.json          # frozen config + 8 prompts
src/         transforms, metrics, alignment, io, sampling_trace, semantic_space
src/adapters/qwen3_tts.py                       # the model-specific adapter
experiments/ run_e0_generate | run_e0_encode | run_e0_analyze
tests/       7 test modules (model-agnostic pass locally; model tests skip)
artifacts/e0/  raw samples, metrics, figures, E0_REPORT.md   (git-ignored)
```

## Confirmed facts (inspected offline) and discrepancies from the draft

- Model `Qwen/Qwen3-TTS-12Hz-0.6B-Base` + `Qwen/Qwen3-TTS-Tokenizer-12Hz` are real.
- **D1**: semantic vocab is **4096** (`semantic_codebook_size`), NOT the 2048 assumed in
  Notion §7 (2048/16 codebooks are the *acoustic* decoder stream).
- **D2**: the encode-path semantic quantizer lives inside transformers' Mimi (the Qwen
  encoder subclasses `MimiModel`); adapter hooks are **discovered at runtime**, never
  assumed. Run `adapter.dump_structure()` first.
- **D3**: semantic latent dim is confirmed at runtime from the discovered codebook.

## Run E0 (Colab)

1. Open `run_e0_colab.ipynb` in Colab, set runtime to a T4 GPU.
2. Run all cells. The notebook installs `qwen-tts`, locates the repo's default reference
   clip, runs hook discovery, generates 32 utterances, re-encodes them, and produces
   `artifacts/e0/E0_REPORT.md`.

## Run tests

```bash
pytest tests/ -q                       # model-agnostic (local)
SEMARK_MODEL_READY=1 pytest tests/ -q  # + model tests (on the GPU host)
```

## Gate

E0 PASS is necessary infrastructure trust, not evidence SEMark works. Do not proceed to
E1 without explicit approval (per the implementation prompt).
