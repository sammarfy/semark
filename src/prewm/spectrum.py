"""Regularized generalized-eigenvalue spectrum for RTD (spec §8, §10; Notion §5.0.2).

NEVER form Sigma_D^{-1} Sigma_R directly. Instead, for a derivative covariance Sigma_D
and re-performance covariance Sigma_R:

    B = Sigma_D + eps*I                    (or a Ledoit-Wolf shrinkage estimator)
    M = B^{-1/2} Sigma_R B^{-1/2}          (symmetric)
    solve the ordinary symmetric eigenproblem M a = eta a

eta > 1 means re-performance variance exceeds derivative variance along that regularized
direction. Pure numpy (float64).
"""
from __future__ import annotations

import numpy as np


def inv_sqrt_psd(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    M = np.asarray(matrix, dtype=np.float64)
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, floor, None)
    inv = V @ np.diag(np.where(w > 0, 1.0 / np.sqrt(w), 0.0)) @ V.T
    return 0.5 * (inv + inv.T)


def ledoit_wolf(cov: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a scaled identity. Returns (shrunk_cov, shrinkage).

    cov: sample covariance [d,d]; n: number of (independent) samples it was estimated from.
    A standard, dependency-free approximation of the LW target/intensity.
    """
    S = np.asarray(cov, dtype=np.float64)
    d = S.shape[0]
    mu = np.trace(S) / d
    target = mu * np.eye(d)
    # shrinkage intensity ~ (dispersion) / (distance to target), clipped to [0,1]
    num = np.sum((S - target) ** 2)
    denom = num + (mu ** 2) * d  # heuristic scale ~ variance of the estimator
    if n and n > 1:
        denom = np.sum(S ** 2) + (np.trace(S) ** 2) / d
    beta = 0.0 if denom <= 0 else min(1.0, max(0.0, (num / max(denom, 1e-30)) / max(n, 1)))
    shrunk = (1 - beta) * S + beta * target
    return 0.5 * (shrunk + shrunk.T), float(beta)


def regularized_M(sigma_d: np.ndarray, sigma_r: np.ndarray, eps: float,
                  shrinkage: bool = False, n_texts: int | None = None) -> np.ndarray:
    """Return the SYMMETRIC M = B^{-1/2} Sigma_R B^{-1/2}, B = Sigma_D + eps I
    (or LW-shrunk Sigma_D when shrinkage=True). No direct inverse of Sigma_D."""
    Sd = np.asarray(sigma_d, dtype=np.float64)
    Sr = np.asarray(sigma_r, dtype=np.float64)
    d = Sd.shape[0]
    if shrinkage:
        B, _ = ledoit_wolf(Sd, n_texts or d)
    else:
        B = Sd + eps * np.eye(d)
    Bi = inv_sqrt_psd(B)
    M = Bi @ Sr @ Bi
    return 0.5 * (M + M.T)


def spectrum(M: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of symmetric M."""
    w = np.linalg.eigvalsh(np.asarray(M, dtype=np.float64))
    return np.sort(w)[::-1]


def top_eigvecs(M: np.ndarray, r: int) -> np.ndarray:
    """Top-r eigenvectors (columns) of symmetric M, by descending eigenvalue."""
    w, V = np.linalg.eigh(np.asarray(M, dtype=np.float64))
    order = np.argsort(w)[::-1][:r]
    return V[:, order]


def effective_rank(cov: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized eigenvalue distribution)."""
    w = np.clip(np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64)), 0, None)
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def condition_number(matrix: np.ndarray, eps: float = 0.0) -> float:
    w = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64)) + eps
    w = w[w > 0]
    return float(w.max() / w.min()) if w.size else float("inf")


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(np.sort(s)[::-1], -1.0, 1.0))
