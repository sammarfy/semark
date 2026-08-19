# E0-c Report — talker(3072) -> codec-semantic(2048) map

- **Rule (source-proven):** talker.forward: codec_ids = cat(input_ids, code_predictor.sequences); input_ids is codec_ids[...,0] -> identity for v < codec_vocab; v >= codec_vocab special
- codec_vocab=2048, talker_vocab=3072, num_code_groups=16
- specials: eos=2150, pad=2150, bos=None; region [2048, 3072) is special/reserved
- verified on **180 actually-sampled tokens** across 4 generations: all 180 ordinary samples < 2048 (max 2026), 0 specials, 0 unexplained.
- authoritative: uses the codec id Qwen sends downstream (input_ids == codec_ids[...,0]); NO waveform re-encoding used.

## Gate: **PASS**
