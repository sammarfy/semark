"""Deterministic frame alignment for E0 (Notion §10).

RULE: one-to-one frame correspondence after a FIXED deterministic delay/padding
compensation. NOT per-example DTW. DTW is banned in E0.

The offset is not assumed: it is estimated from calibration examples and then
VERIFIED by cross-correlating the pre- and post-round-trip latent-norm sequences.
The empirical peak lag must equal the estimated offset to within ZERO frames
(a 1-frame error at 12.5 Hz is 80 ms and would corrupt every downstream statistic).

Pure numpy. No DTW dependency is imported anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Explicit sentinel asserted by tests/test_alignment.py: E0 never uses DTW.
USES_DTW = False


def norm_sequence(latents: np.ndarray) -> np.ndarray:
    """Reduce a [T, d] latent sequence to a [T] scalar sequence (per-frame L2 norm)."""
    return np.linalg.norm(np.asarray(latents, dtype=np.float64), axis=-1)


def crosscorr_peak_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Integer lag L (in frames) maximizing correlation of a shifted onto b.

    Positive L means b is delayed by L relative to a (i.e. b[t] ~ a[t - L]).
    Searches L in [-max_lag, +max_lag]. Sequences are mean-removed first.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            av, bv = a[: len(a) - lag], b[lag:]
        else:
            av, bv = a[-lag:], b[: len(b) + lag]
        n = min(len(av), len(bv))
        if n < 2:
            continue
        score = float(np.dot(av[:n], bv[:n]))
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag


@dataclass
class Alignment:
    frame_offset: int
    frame_offset_crosscorr_peak: int   # MUST equal frame_offset (§10)
    frames_trimmed_left: int
    frames_trimmed_right: int
    aligned_length: int
    offset_verified: bool

    def to_dict(self) -> dict:
        return asdict(self)


def align_after_offset(
    original: np.ndarray,
    reencoded: np.ndarray,
    frame_offset: int,
    max_lag: int = 8,
    trim_boundary: int = 0,
) -> tuple[np.ndarray, np.ndarray, Alignment]:
    """Apply a fixed deterministic offset, verify it, and return one-to-one frames.

    original, reencoded: [T, d] and [T', d] latent sequences.
    frame_offset:        deterministic encoder/decoder delay in frames (>=0 means
                         reencoded is delayed relative to original).
    trim_boundary:       extra invalid boundary frames to drop on each side.

    Returns (orig_aligned, reenc_aligned, Alignment). Raises if the verified
    cross-correlation peak lag disagrees with frame_offset (stop condition §20.10).
    """
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(reencoded, dtype=np.float64)

    peak = crosscorr_peak_lag(norm_sequence(o), norm_sequence(r), max_lag=max_lag)
    verified = (peak == frame_offset)

    if frame_offset >= 0:
        o2, r2 = o, r[frame_offset:]
    else:
        o2, r2 = o[-frame_offset:], r

    L = min(len(o2), len(r2))
    lo, hi = trim_boundary, L - trim_boundary
    o_al = o2[lo:hi]
    r_al = r2[lo:hi]

    align = Alignment(
        frame_offset=int(frame_offset),
        frame_offset_crosscorr_peak=int(peak),
        frames_trimmed_left=int(trim_boundary + (0 if frame_offset >= 0 else -frame_offset)),
        frames_trimmed_right=int(trim_boundary + max(0, (len(o2) - L)) + max(0, (len(r2) - L))),
        aligned_length=int(max(0, hi - lo)),
        offset_verified=bool(verified),
    )

    if not verified:
        raise AssertionError(
            f"§10 violation: cross-correlation peak lag {peak} != estimated frame_offset "
            f"{frame_offset}. A 1-frame error at 12.5 Hz is 80 ms. HARD STOP."
        )
    return o_al, r_al, align
