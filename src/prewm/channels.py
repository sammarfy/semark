"""E1 derivative channel transforms (spec §7.3). Frame-synchronous families only here.

These are the derivative transforms applied to a generated waveform before re-encoding. The
frame-synchronous families (clean, resample round-trip, AWGN, low/high-pass) preserve timing
up to a fixed delay, so RTD-fit uses fixed-delay + boundary trimming (no DTW). Codec families
(MP3/Opus) need ffmpeg on the host and are provided as named specs to run there. Timing-
changing families (speed, crop, resynthesis) are diagnostic-only and MUST NOT enter the
synchronous RTD estimator (spec §7.3).

numpy-only DSP so the transforms are testable without extra deps.
"""
from __future__ import annotations

import numpy as np

# Families safe for the synchronous RTD-fit estimator (fixed delay, no time warp).
SYNCHRONOUS_FAMILIES = ("clean", "resample_16k", "resample_8k", "awgn_30", "awgn_20",
                        "awgn_10", "lowpass", "highpass")
# Families that CHANGE timing — diagnostic only, never in the synchronous fit.
TIMING_FAMILIES = ("speed_105", "speed_095", "crop", "resynth")


def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sig_p = np.mean(x ** 2) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10))
    return (x + rng.standard_normal(x.shape) * np.sqrt(noise_p)).astype(np.float32)


def _linear_resample(x: np.ndarray, ratio: float) -> np.ndarray:
    n_out = max(1, int(round(len(x) * ratio)))
    xp = np.linspace(0, 1, len(x))
    return np.interp(np.linspace(0, 1, n_out), xp, x).astype(np.float32)


def resample_roundtrip(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Downsample to target_sr and back (linear) — a mild frame-synchronous derivative."""
    down = _linear_resample(x, target_sr / sr)
    return _linear_resample(down, len(x) / len(down))[: len(x)]


def one_pole_lowpass(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float64)
    acc = 0.0
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y.astype(np.float32)


def one_pole_highpass(x: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    prev_x = 0.0
    prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = alpha * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return y.astype(np.float32)


CODEC_FAMILIES = ("mp3_128", "mp3_64", "opus_64", "opus_32")
_CODEC_SPEC = {"mp3_128": ("mp3", "128k", "libmp3lame"), "mp3_64": ("mp3", "64k", "libmp3lame"),
               "opus_64": ("ogg", "64k", "libopus"), "opus_32": ("ogg", "32k", "libopus")}


def codec_roundtrip(family: str, wav: np.ndarray, sr: int) -> np.ndarray:
    """Real lossy-codec round trip via ffmpeg (host). Introduces a FIXED encoder delay, so the
    caller must offset-align the re-encoded latents before differencing (not frame-synchronous)."""
    import os
    import subprocess
    import tempfile
    import soundfile as sf
    ext, br, codec = _CODEC_SPEC[family]
    with tempfile.TemporaryDirectory() as d:
        inp, comp, outp = (os.path.join(d, f) for f in ("in.wav", f"c.{ext}", "out.wav"))
        sf.write(inp, np.asarray(wav, np.float32).ravel(), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp, "-c:a", codec, "-b:a", br, comp],
                       check=False)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", comp, "-ar", str(sr), outp], check=False)
        w, _ = sf.read(outp)
    return np.asarray(w, np.float32).ravel()


def apply_synchronous(family: str, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Apply a named synchronous derivative family to a waveform. Returns same-length audio."""
    if family == "clean":
        return np.asarray(wav, dtype=np.float32)
    if family == "resample_16k":
        return resample_roundtrip(wav, sr, 16000)
    if family == "resample_8k":
        return resample_roundtrip(wav, sr, 8000)
    if family.startswith("awgn_"):
        return awgn(wav, float(family.split("_")[1]), rng)
    if family == "lowpass":
        return one_pole_lowpass(wav)
    if family == "highpass":
        return one_pole_highpass(wav)
    raise ValueError(f"unknown synchronous family: {family}")
