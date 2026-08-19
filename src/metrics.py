"""E0 metrics (Notion §12, §13, §15). Pure numpy; no torch, no model.

All distribution metrics are computed on explicit probability vectors so they are
testable without the model. Generation code supplies p_t^raw (dense over V) and
p_t^final (over the truncated support) separately (§7.1).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Distribution metrics (§13). Natural log throughout.                          #
# --------------------------------------------------------------------------- #
def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy H = -sum p log p (nats). Zeros contribute 0."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=axis)


def n_eff(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Effective vocabulary size exp(H)."""
    return np.exp(entropy(p, axis=axis))


def p_max(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).max(axis=axis)


def support_size(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return (np.asarray(p, dtype=np.float64) > 0).sum(axis=axis)


def vt_directions(u: np.ndarray, x_v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Local injection capacity V_t(u) = Var_{p}[u^T x_v]  (Notion §13).

    u:   [m, d] bank of unit directions (already in the shared/whitened space caller).
    x_v: [V, d] shared-space codebook rows x_v = W(c_v - c_bar).
    p:   [V]    distribution over the vocabulary (use p_t^final for the headline
                truncated capacity, p_t^raw for the secondary full-codebook number).

    Returns [m] variances, one per direction. Var_p[s] = E[s^2] - E[s]^2.
    """
    u = np.asarray(u, dtype=np.float64)
    x_v = np.asarray(x_v, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()  # defensive renormalization over the given support
    s = x_v @ u.T                      # [V, m]  projection of each centroid onto each u
    mean = (p[:, None] * s).sum(axis=0)          # [m]
    mean_sq = (p[:, None] * s * s).sum(axis=0)   # [m]
    return np.clip(mean_sq - mean * mean, 0.0, None)


# --------------------------------------------------------------------------- #
# Round-trip representation metrics (§12).                                     #
# --------------------------------------------------------------------------- #
def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def rowwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)


def exact_match_rate(ids_a: np.ndarray, ids_b: np.ndarray) -> float:
    ids_a = np.asarray(ids_a).ravel()
    ids_b = np.asarray(ids_b).ravel()
    if ids_a.shape != ids_b.shape:
        raise ValueError(f"id length mismatch {ids_a.shape} vs {ids_b.shape}")
    if ids_a.size == 0:
        return float("nan")
    return float((ids_a == ids_b).mean())


def distance_summary(dists: np.ndarray) -> dict:
    d = np.asarray(dists, dtype=np.float64).ravel()
    if d.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
    }


# --------------------------------------------------------------------------- #
# Covariance summaries for the crude E0-b R_trace (§15). NOT the RTD fit.      #
# --------------------------------------------------------------------------- #
def cov_from_diffs(diffs: np.ndarray) -> np.ndarray:
    """Sigma_hat = (1/N) sum d d^T for a stack of difference vectors [N, d]."""
    D = np.asarray(diffs, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError(f"diffs must be [N, d] with N>0, got {D.shape}")
    return (D.T @ D) / float(D.shape[0])


def trace_ratio(sigma_r: np.ndarray, sigma_d: np.ndarray, eps: float = 1e-12) -> float:
    """R_trace = tr(Sigma_R) / (tr(Sigma_D) + eps)  (Notion §15).

    CRUDE DIAGNOSTIC ONLY at E0 (naive index alignment, clean-channel Sigma_D).
    It is a systematic UPPER BOUND on the Stage-1 value (§15.2).
    """
    tr_r = float(np.trace(np.asarray(sigma_r, dtype=np.float64)))
    tr_d = float(np.trace(np.asarray(sigma_d, dtype=np.float64)))
    return tr_r / (tr_d + eps)


def random_unit_directions(n: int, d: int, seed: int) -> np.ndarray:
    """Fixed public bank of unit directions for V_t. NOT a watermark key."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u
