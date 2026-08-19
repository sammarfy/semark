"""Shared coordinate-system transform (Notion §6).

ONE transform function is used for BOTH generation-side codebook vectors and
detector-side re-encoded latents. There must never be two separate mathematical
code paths for the two sides -- that is the whole point of the "shared space".

    E0:     z_tilde = W (z - c_bar)                 (rtd_matrix = None)
    later:  z_tilde = B^{-1/2} W (z - c_bar)        (rtd_matrix = B^{-1/2})

At E0 the centering/whitening pair (c_bar, W) is defined by the FROZEN semantic
codebook, a public object of V vectors (Notion §6.1). It uses no data split and is
exactly reproducible across machines.

Pure numpy. No torch import here so the shared math is testable without a GPU.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


def transform_shared_space(
    x: np.ndarray,
    center: np.ndarray,
    whitening_matrix: np.ndarray,
    rtd_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the shared-space transform to x.

    x:      [..., d] array of row vectors (a single [d] vector is also accepted).
    center: [d]      c_bar.
    whitening_matrix: [d, d]  W.
    rtd_matrix:       [d, d] or None. B^{-1/2} for the later RTD stage; None at E0.

    Returns array of the same leading shape as x.

    Convention: y = W (x - c_bar), then optionally y <- B^{-1/2} y.
    For batched row vectors this is (x - c_bar) @ W.T  [then @ rtd.T].
    """
    x = np.asarray(x, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    W = np.asarray(whitening_matrix, dtype=np.float64)

    single = x.ndim == 1
    if single:
        x = x[None, :]

    y = (x - center) @ W.T
    if rtd_matrix is not None:
        rtd = np.asarray(rtd_matrix, dtype=np.float64)
        y = y @ rtd.T

    return y[0] if single else y


@dataclass
class WhiteningParams:
    """Frozen-codebook whitening params plus provenance metadata (§6.1, §18)."""

    center: np.ndarray          # c_bar, [d]
    whitening_matrix: np.ndarray  # W, [d, d]
    ridge: float
    condition_number: float     # cond number of the codebook covariance (pre-ridge)
    center_sha256: str
    whitening_sha256: str
    n_centroids: int
    dim: int

    def to_metadata(self) -> dict:
        return {
            "source": "frozen_semantic_codebook",
            "ridge": self.ridge,
            "codebook_cov_condition_number": self.condition_number,
            "center_sha256": self.center_sha256,
            "whitening_sha256": self.whitening_sha256,
            "n_centroids": self.n_centroids,
            "dim": self.dim,
        }


def _sha256_array(a: np.ndarray) -> str:
    # Serialize in a canonical, dtype/shape-stable way for reproducible hashing.
    a = np.ascontiguousarray(a, dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def inverse_sqrt_psd(matrix: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Symmetric-PSD inverse square root via eigendecomposition (numpy only).

    matrix must be symmetric PSD. eps is added to eigenvalues for conditioning.
    """
    M = np.asarray(matrix, dtype=np.float64)
    M = 0.5 * (M + M.T)  # symmetrize against numerical drift
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None) + eps
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)


def whitening_from_codebook(codebook: np.ndarray, ridge: float = 1e-3) -> WhiteningParams:
    """Compute (c_bar, W) from the frozen semantic codebook (§6.1).

    codebook: [V, d] semantic centroids.
    c_bar = mean over the V centroids.
    Cov   = (1/V) sum (c - c_bar)(c - c_bar)^T
    W     = (Cov + ridge*I)^{-1/2}

    The ridge, the covariance condition number, and SHA256 of both W and c_bar
    are recorded (Notion §6.1, §18).
    """
    C = np.asarray(codebook, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"codebook must be [V, d], got shape {C.shape}")
    V, d = C.shape
    c_bar = C.mean(axis=0)
    Cc = C - c_bar
    cov = (Cc.T @ Cc) / float(V)

    # Condition number of the *pre-ridge* covariance (a diagnostic on estimability).
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    eig_pos = eig[eig > 0]
    cond = float(eig.max() / eig_pos.min()) if eig_pos.size else float("inf")

    W = inverse_sqrt_psd(cov + ridge * np.eye(d))

    return WhiteningParams(
        center=c_bar,
        whitening_matrix=W,
        ridge=float(ridge),
        condition_number=cond,
        center_sha256=_sha256_array(c_bar),
        whitening_sha256=_sha256_array(W),
        n_centroids=int(V),
        dim=int(d),
    )
