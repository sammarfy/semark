# E0 Report — SEMark representation gate

Generated (UTC): 2026-08-19T11:38:52.575083+00:00

## 1. Environment and revisions
- **model**: Qwen/Qwen3-TTS-12Hz-0.6B-Base
- **torch_version**: 2.11.0+cu128
- **transformers_version**: 4.57.3
- **cuda_version**: 12.8
- **gpu**: Tesla T4
- **git_sha**: None
- **git_dirty**: None
- **tokenizer**: Qwen/Qwen3-TTS-Tokenizer-12Hz
- **sampling (actual)**: {'do_sample': True, 'repetition_penalty': 1.05, 'temperature': 0.9, 'top_k': 50, 'top_p': 1.0}

## 2. Discovered hook locations
```
{'NOTE_D4': 'talker semantic head is 3072-wide but codec codebook is 2048; p_t is over the talker vocab.', 'codebook_shape': [2048, 256], 'input_proj_present': True, 'notes': ['path: model.encoder.quantizer.semantic_residual_vector_quantizer', 'input_proj present: True', 'quantize via srvq.encode(pre_proj_latent); latents via input_proj forward hook'], 'quantizer_path': 'model.encoder.quantizer.semantic_residual_vector_quantizer', 'sampling': {'do_sample': True, 'repetition_penalty': 1.05, 'temperature': 0.9, 'top_k': 50, 'top_p': 1.0}, 'semantic_codec_vocab': 2048, 'semantic_dim': 256, 'talker_codec_head_out': 3072}
```

## 3. Semantic-space bridge
model semantic id ↔ first semantic codebook ↔ encoder pre-quant latent ↔ semantic quantizer

## 4. Whitening (W, c_bar) — §6.1
Defined from the FROZEN semantic codebook (public object). **E1.5 may refit W/c_bar on fit-split latents; if it does, ALL E0 representation metrics must be recomputed under the new transform, not carried forward.**

## 5. Coordinate-system agreement Q(h)==tokenizer-id (valid frames)
- overall coordinate agreement: **1.0** → **PASS**
- clean decode→re-encode round-trip token match: **0.9590824731251875**
- offsets all verified by cross-correlation: **True**
- NOTE: state the valid-frame exclusion definition explicitly once the runtime padding/BOS/EOS frames are known (§11).

## 6. Clean round-trip stability
| sample_id | coordinate_agreement | roundtrip_token_match | cos_mean | l2_mean | whitened_cos_mean | frame_offset | offset_verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p0_repo_default_s101_16ef67cbcfec | 1.0 | 1.0 | 0.9973011788278427 | 15.037964878542448 | 0.9960053215522202 | 0 | True |
| p0_repo_default_s202_0292cd8ec9d9 | 1.0 | 0.96875 | 0.9952471306440509 | 18.995371541498617 | 0.9922452021585177 | 0 | True |
| p0_repo_default_s303_1f2fe94b1a6a | 1.0 | 0.9333333333333333 | 0.9952422832831956 | 18.893372264334168 | 0.9924670254995858 | 0 | True |
| p0_repo_default_s404_fc8826df4f7b | 1.0 | 0.9428571428571428 | 0.9983134795249226 | 13.087062118307959 | 0.9968259688994654 | 0 | True |
| p1_repo_default_s101_fbbf3aa8c6e1 | 1.0 | 0.96 | 0.9982654402691712 | 12.835307434048094 | 0.9966381501624695 | 0 | True |
| p1_repo_default_s202_ad3d6206b189 | 1.0 | 0.9761904761904762 | 0.9970761283183522 | 15.236887622637337 | 0.9959721197459034 | 0 | True |
| p1_repo_default_s303_f3e3e5960de1 | 1.0 | 0.9655172413793104 | 0.9965113102695359 | 18.685760430186146 | 0.9933345223911481 | 0 | True |
| p1_repo_default_s404_ee537186a5a1 | 1.0 | 0.9772727272727273 | 0.9978010451899223 | 13.79860747486553 | 0.9964610633148656 | 0 | True |
| p2_repo_default_s101_c0b1e704741b | 1.0 | 0.9090909090909091 | 0.9947390585065023 | 19.969192195783034 | 0.9901020026262384 | 0 | True |
| p2_repo_default_s202_d6f66fba357b | 1.0 | 1.0 | 0.9968220264964559 | 15.563091371005694 | 0.995066285331826 | 0 | True |
| p2_repo_default_s303_207f49bac26e | 1.0 | 0.9347826086956522 | 0.9971317309472715 | 16.185115056936063 | 0.9950581998870267 | 0 | True |
| p2_repo_default_s404_1897b9d15e39 | 1.0 | 0.9583333333333334 | 0.9952479242583651 | 17.872316936263918 | 0.9923155787253037 | 0 | True |
| p3_repo_default_s101_c9d3e2125a25 | 1.0 | 0.9761904761904762 | 0.997484486172062 | 14.58909393506927 | 0.9962536664505011 | 0 | True |
| p3_repo_default_s202_1f21ae948cd2 | 1.0 | 0.9230769230769231 | 0.9957472690352338 | 16.59700370505472 | 0.9937510593591333 | 0 | True |
| p3_repo_default_s303_653aa1a906f8 | 1.0 | 0.9512195121951219 | 0.989168077583212 | 20.7025535936951 | 0.9833789302030707 | 0 | True |
| p3_repo_default_s404_438d2fc78684 | 1.0 | 0.9183673469387755 | 0.9945854839499052 | 16.564954681596582 | 0.9931427288729875 | 0 | True |
| p4_repo_default_s101_0f2ce93cd170 | 1.0 | 0.9565217391304348 | 0.9963970061440313 | 14.448815940737301 | 0.9943810478966049 | 0 | True |
| p4_repo_default_s202_fae2b2b8c176 | 1.0 | 0.9782608695652174 | 0.9978219036008893 | 13.564847561477423 | 0.9960897502805593 | 0 | True |
| p4_repo_default_s303_405eae3a1111 | 1.0 | 0.9565217391304348 | 0.9969737375082445 | 15.040263057981452 | 0.9949038745639116 | 0 | True |
| p4_repo_default_s404_6783a23a51a7 | 1.0 | 0.9583333333333334 | 0.9929391828105963 | 20.592534467738783 | 0.9887869758073652 | 0 | True |
| p5_repo_default_s101_2e09396abae4 | 1.0 | 0.975 | 0.9942358043860455 | 21.350236046223422 | 0.9908255730361765 | 0 | True |
| p5_repo_default_s202_2ec04a0dbb60 | 1.0 | 0.975 | 0.9951864783870009 | 18.609037783951553 | 0.9926033992164349 | 0 | True |
| p5_repo_default_s303_f214912b4437 | 1.0 | 0.9782608695652174 | 0.9973373736911585 | 14.255376999280728 | 0.9960533850536905 | 0 | True |
| p5_repo_default_s404_162b0dac4264 | 1.0 | 0.9591836734693877 | 0.9951904355935454 | 17.563450385166952 | 0.9904484632995832 | 0 | True |
| p6_repo_default_s101_4b18773fd91c | 1.0 | 0.9629629629629629 | 0.9968005092668082 | 15.931427586936415 | 0.994452265880688 | 0 | True |
| p6_repo_default_s202_918b9a87b276 | 1.0 | 0.9672131147540983 | 0.9954901036559178 | 16.726931596926097 | 0.9938550887197719 | 0 | True |
| p6_repo_default_s303_419ceaee248d | 1.0 | 0.9491525423728814 | 0.99780463445761 | 13.522321554579744 | 0.9961262256831288 | 0 | True |
| p6_repo_default_s404_f3aa73e9bdf4 | 1.0 | 0.9104477611940298 | 0.9964009835787087 | 16.865251688428884 | 0.9937931651575469 | 0 | True |
| p7_repo_default_s101_99aa7bd58829 | 1.0 | 0.94 | 0.9970503326452387 | 15.296960909120958 | 0.9955373365523421 | 0 | True |
| p7_repo_default_s202_47a7c467c6e5 | 1.0 | 0.9838709677419355 | 0.9976460259994815 | 13.766202072305555 | 0.9963782466724084 | 0 | True |
| p7_repo_default_s303_e94502541332 | 1.0 | 0.9782608695652174 | 0.9978898193363167 | 13.747067636510016 | 0.9967397445816716 | 0 | True |
| p7_repo_default_s404_3bbd5e38f6c4 | 1.0 | 0.9666666666666667 | 0.9984080165862281 | 11.963538598591551 | 0.9964264916172118 | 0 | True |

## 7. Encoder determinism and seam contamination (§9.5)
- repeat-encode deterministic: **True**
- max seam contamination: **29** frames (bounds the shortest crop the detector can ever score; constrains §7.3)

## 8. Generation stochasticity — RAW vs FINAL (§7.1)
- mean H(p_raw): **0.4442** nats | mean H(p_final): **0.4440** nats
- **binding constraint**: neither/healthy (truncation → raise top_k; base_distribution → §21.F ablation)
- headline V_t is the TRUNCATED value (under p_final); full-codebook V_t (p_raw) is the secondary number in entropy_metrics.csv

## 9. Same-text/same-voice seed diversity (§14)
| prompt_id | seed_token_disagreement_mean | duration_spread_frames |
| --- | --- | --- |
| p0 | 0.8353670634920635 | 5 |
| p1 | 0.88988455988456 | 16 |
| p2 | 0.8288180720245936 | 5 |
| p3 | 0.876046040680187 | 10 |
| p4 | 0.7681159420289855 | 2 |
| p5 | 0.9230072463768115 | 9 |
| p6 | 0.9077135624093112 | 13 |
| p7 | 0.9082608695652173 | 16 |

## 10. Trace ratio R_trace (§15) — CRUDE, UPPER BOUND
- R_trace = tr(Σ_R)/(tr(Σ_D)+ε) = **1.7399898984557394**
- CRUDE DIAGNOSTIC: naive index alignment, NOT comparable to Stage-1 MFA-aligned Σ_R.
- SYSTEMATIC UPPER BOUND: Σ_D here is clean round trip only; Stage-1 adds MP3/Opus/resampling/noise/filtering, which only increase tr(Σ_D). Read as weaker than it looks.

## 11. Known uncontrolled factors
- single fixed voice conditioning (§8.1): cannot separate 'model is deterministic' from 'this reference clip yields low variance';
- read-speech-only prompts; clean-channel-only Σ_D.

## 12. Discrepancies from the Notion spec (confirmed at runtime)
- **D1**: the codec semantic codebook is **2048×256** (read from the model), not the 4096 in the config JSON nor the 2048/16 acoustic assumption of §7.
- **D2**: the encode-path semantic quantizer lives inside transformers' Mimi under `model.encoder.quantizer.semantic_residual_vector_quantizer`; hooks discovered, not assumed.
- **D3**: semantic latent dim = **256** (post `input_proj`); pre-proj is 512.
- **D4**: the talker semantic head (`talker.codec_head`) is **3072-wide** while the codec codebook is 2048 — an index-space gap. p_t is over 3072; V_t restricts to the codec-code region assuming an identity map for tokens < 2048 (flagged, not assumed silently).

## 13. Gate decision: **PASS**
## 14. Recommended next step: Proceed to E1 (with explicit approval).
