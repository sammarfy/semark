"""Qwen3-TTS-12Hz adapter (Notion §3, §4, §5, §7). All hooks confirmed at runtime.

Detection side (tokenizer):
  adapter.tokenizer.model.encoder.quantizer.semantic_residual_vector_quantizer
  - input_proj Conv1d(512,256); layers[0].codebook.embed -> [2048, 256]
  - tokenizer.encode(x) -> {'audio_codes':[Tensor[T,16]]}, column 0 = semantic id
  - srvq.encode(pre_proj_latent) reproduces column 0 (E0 invariant, §11)

Generation side (talker):
  adapter.model.model.talker.codec_head  Linear(1024 -> 3072)   = semantic p_t head
  generation params: temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05

DISCREPANCIES vs the Notion draft (documented, not silently changed):
  D1: codec semantic codebook is 2048x256 (config's 4096 is not this object).
  D2: semantic quantizer lives in transformers Mimi under the encoder (found at runtime).
  D3: semantic latent dim = 256 (post input_proj); pre-proj is 512.
  D4: the talker's semantic head is 3072-wide while the codec codebook is 2048 -- an
      index-space gap (specials/offset). p_t is over 3072; the codec code space is 2048.
      Sizes are READ at runtime, never hardcoded.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

import numpy as np

from src.sampling_trace import GenerationTrace, SemanticSpace, reconstruct_raw_and_final
from src.semantic_space import locate_semantic_quantizer, codebook_embed, QuantizerHandles


class SpeechLMAdapter:
    def __init__(self, model, tokenizer, device: str, dtype: str,
                 frame_rate_hz: float = 12.5, sample_rate_hz: int = 24000):
        self.model = model            # Qwen3TTSModel wrapper
        self.tokenizer = tokenizer    # Qwen3TTSTokenizer wrapper
        self.device = device
        self.dtype = dtype
        self.frame_rate_hz = frame_rate_hz
        self.sample_rate_hz = sample_rate_hz
        self._q: Optional[QuantizerHandles] = None
        self._codebook_np: Optional[np.ndarray] = None
        # sizes read at load()
        self.semantic_vocab_size: int = 0     # codec semantic codebook (2048)
        self.semantic_dim: int = 0            # 256
        self.talker_vocab_size: int = 0       # codec_head out (3072)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, cfg: dict, device: str = "cuda:0"):
        import torch
        from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[cfg["model"]["dtype"]]
        model = Qwen3TTSModel.from_pretrained(
            cfg["model"]["id"], device_map=device, dtype=dtype, attn_implementation="eager")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            cfg["model"]["tokenizer_id"], device_map=device)
        facts = cfg["tokenizer_facts"]
        self = cls(model, tokenizer, device=device, dtype=cfg["model"]["dtype"],
                   frame_rate_hz=facts["frame_rate_hz"], sample_rate_hz=facts["sample_rate_hz"])
        cb = self.get_semantic_codebook()
        self.semantic_vocab_size, self.semantic_dim = int(cb.shape[0]), int(cb.shape[1])
        head = self._codec_head()
        self.talker_vocab_size = int(head.out_features)
        return self

    # ------------------------------------------------------------------ #
    def quantizer(self) -> QuantizerHandles:
        if self._q is None:
            self._q = locate_semantic_quantizer(self.tokenizer)
        return self._q

    def _codec_head(self):
        """Locate the talker semantic-logits head (Linear -> talker semantic vocab)."""
        import torch.nn as nn
        root = getattr(self.model, "model", self.model)
        cand = None
        for name, m in root.named_modules():
            if name.endswith("codec_head") and isinstance(m, nn.Linear):
                cand = m
        if cand is None:
            raise RuntimeError("Could not find `talker.codec_head` (semantic p_t head).")
        return cand

    def get_semantic_codebook(self) -> np.ndarray:
        if self._codebook_np is None:
            q = self.quantizer()
            emb = codebook_embed(q.codebook_module)
            self._codebook_np = emb.detach().float().cpu().numpy()
        return self._codebook_np

    def dump_structure(self) -> dict:
        q = self.quantizer()
        cb = self.get_semantic_codebook()
        head = self._codec_head()
        info = {
            "quantizer_path": q.discovered_path,
            "codebook_shape": list(cb.shape),
            "input_proj_present": q.input_proj is not None,
            "semantic_codec_vocab": int(cb.shape[0]),
            "semantic_dim": int(cb.shape[1]),
            "talker_codec_head_out": int(head.out_features),
            "sampling": self.get_sampling_config(),
            "notes": q.notes,
        }
        if int(cb.shape[0]) != int(head.out_features):
            info["NOTE_D4"] = (f"talker semantic head is {head.out_features}-wide but codec "
                               f"codebook is {cb.shape[0]}; p_t is over the talker vocab.")
        for k, v in info.items():
            print(f"[dump_structure] {k}: {v}")
        return info

    def get_sampling_config(self) -> dict:
        gc = getattr(self.model, "generation_config", None) \
            or getattr(getattr(self.model, "model", None), "generation_config", None)
        return {"temperature": getattr(gc, "temperature", None),
                "top_k": getattr(gc, "top_k", None),
                "top_p": getattr(gc, "top_p", None),
                "repetition_penalty": getattr(gc, "repetition_penalty", None),
                "do_sample": getattr(gc, "do_sample", None)}

    def get_model_metadata(self) -> dict:
        import torch, transformers
        return {"semantic_vocab_size": self.semantic_vocab_size, "semantic_dim": self.semantic_dim,
                "talker_vocab_size": self.talker_vocab_size,
                "frame_rate_hz": self.frame_rate_hz, "sample_rate_hz": self.sample_rate_hz,
                "dtype": self.dtype, "device": self.device,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}

    # ------------------------------------------------------------------ #
    # Encode: waveform -> pre-quant latent -> REAL semantic quantizer      #
    # ------------------------------------------------------------------ #
    def _to_encode_input(self, wav):
        """tokenizer.encode wants a path; write arrays to a temp 24k wav."""
        if isinstance(wav, str):
            return wav, None
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, np.asarray(wav, dtype=np.float32).ravel(), self.sample_rate_hz)
        return tmp.name, tmp.name

    def _encode_capture(self, wav):
        """Run tokenizer.encode, capture (pre,proj) latent via input_proj hook, and the
        real quantizer's ids. Returns (codes16 [T,16], proj_latents [T,256], our_ids [T],
        agreement float)."""
        import torch
        q = self.quantizer()
        path, tmp = self._to_encode_input(wav)
        cap = {}

        def hook(_m, inp, out):
            cap["pre"], cap["proj"] = inp[0].detach(), out.detach()

        h = q.input_proj.register_forward_hook(hook) if q.input_proj is not None else None
        try:
            enc_out = self.tokenizer.encode(path)
        finally:
            if h is not None:
                h.remove()
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

        codes16 = self._audio_codes(enc_out)                 # [T, 16] int
        tok_sem = codes16[:, 0].astype(np.int64)             # tokenizer's semantic id
        if "pre" not in cap:
            raise RuntimeError("input_proj hook did not fire during tokenizer.encode "
                               "(§20 stop #1). The encode path changed.")
        proj = cap["proj"][0].detach().float().cpu().numpy()  # [256, T]
        if proj.shape[0] == self.semantic_dim:
            proj = proj.T                                     # -> [T, 256]
        with torch.no_grad():
            our = q.quantize_fn(cap["pre"])                   # srvq.encode(pre) -> [1,1,T]
        our_ids = np.asarray(our.detach().cpu()).reshape(-1).astype(np.int64)
        T = min(len(tok_sem), len(our_ids), proj.shape[0])
        agree = float((our_ids[:T] == tok_sem[:T]).mean()) if T else float("nan")
        return codes16[:T], proj[:T], our_ids[:T], agree, tok_sem[:T]

    @staticmethod
    def _audio_codes(enc_out) -> np.ndarray:
        import torch
        codes = enc_out["audio_codes"] if isinstance(enc_out, dict) else enc_out
        if isinstance(codes, (list, tuple)):
            codes = codes[0]
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().numpy()
        codes = np.asarray(codes)
        if codes.ndim == 3:
            codes = codes[0]
        return codes.astype(np.int64)                         # [T, K]

    def encode_semantic_space(self, wav) -> SemanticSpace:
        codes16, proj, our_ids, agree, tok_sem = self._encode_capture(wav)
        T = proj.shape[0]
        return SemanticSpace(
            codebook=self.get_semantic_codebook(), frame_latents=proj,
            semantic_ids=tok_sem, valid_frame_mask=np.ones(T, dtype=bool),
            coordinate_system_name="encoder_first_semantic_quantizer",
            hook_metadata={"discovered_path": self.quantizer().discovered_path,
                           "coordinate_agreement": agree,
                           "requantized_ids": our_ids, "all_codec_ids": codes16})

    def decode_codes(self, enc_out):
        """tokenizer.decode(codes) -> waveform array (for the clean round-trip)."""
        wavs, sr = self.tokenizer.decode(enc_out)
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def encode_raw(self, wav):
        """Return the raw tokenizer.encode() output (for decode round-trips)."""
        path, tmp = self._to_encode_input(wav)
        try:
            return self.tokenizer.encode(path)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------------ #
    # Generation with p_t tracing (§7, §7.1)                              #
    # ------------------------------------------------------------------ #
    def create_voice_prompt(self, ref_audio, ref_text):
        return self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

    def generate_trace(self, text, seed, sample_id, voice_id, voice_clone_prompt=None,
                       ref_audio=None, ref_text=None, language="English") -> GenerationTrace:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        head = self._codec_head()
        captured: list = []

        def hook(_m, _i, out):
            logits = out[0] if isinstance(out, tuple) else out
            captured.append(logits.detach()[..., -1, :].float().cpu().reshape(-1))

        hh = head.register_forward_hook(hook)
        try:
            kw = dict(text=text, language=language)
            if voice_clone_prompt is not None:
                kw["voice_clone_prompt"] = voice_clone_prompt
            else:
                kw["ref_audio"], kw["ref_text"] = ref_audio, ref_text
            wavs, sr = self.model.generate_voice_clone(**kw)
        finally:
            hh.remove()

        waveform = np.asarray(wavs[0], dtype=np.float32)
        # emitted semantic ids: re-encode the generated waveform (codes not returned by API)
        space = self.encode_semantic_space(waveform)
        semantic_ids = np.asarray(space.semantic_ids)
        T = len(semantic_ids)

        # p_t: keep the last T captured codec_head distributions (drop prompt-prefill calls)
        cfg = self.get_sampling_config()
        logits_steps = captured[-T:] if len(captured) >= T else captured
        raw_list, support_ids, support_logits, final_list = [], [], [], []
        for step in logits_steps:
            lg = step.numpy().reshape(-1)                     # [talker_vocab]
            p_raw, p_final = reconstruct_raw_and_final(
                lg, cfg["temperature"], cfg["top_k"], cfg["top_p"])
            raw_list.append(p_raw.astype(np.float16))
            sup = np.where(p_final > 0)[0]
            support_ids.append(sup.astype(np.int32))
            support_logits.append(lg[sup].astype(np.float32))
            final_list.append(p_final)

        meta_extra = {"n_codec_head_calls": len(captured), "n_pt_steps_used": len(logits_steps),
                      "talker_vocab": self.talker_vocab_size,
                      "codec_semantic_vocab": self.semantic_vocab_size}
        if cfg.get("repetition_penalty") not in (None, 1.0):
            meta_extra["warning_repetition_penalty"] = cfg["repetition_penalty"]

        return GenerationTrace(
            sample_id=sample_id, text=text, voice_id=voice_id, seed=seed,
            waveform=waveform, sample_rate=int(sr), semantic_ids=semantic_ids,
            all_codec_ids=space.hook_metadata.get("all_codec_ids"),
            raw_semantic_probs=np.stack(raw_list) if raw_list else None,
            support_ids=support_ids, support_logits=support_logits,
            final_semantic_probs=final_list, valid_frame_mask=np.ones(T, dtype=bool),
            sampling_metadata={**cfg, "codec_head": "talker.codec_head", **meta_extra},
            model_metadata=self.get_model_metadata())
