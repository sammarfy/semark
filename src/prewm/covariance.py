"""Within-frame covariance for the first-order channel response (Notion §6.4, spec §1B/§1C).

THE CENTRAL INVARIANT of this milestone:

    C_raw = E_t[ Cov_{V~p_t, Y~C_t(.|V)}( phi_t(V), psi_t(Y) ) ]
          = mean_t( within-frame covariance )

NOT covariance(pool all (phi, psi) across frames together). Different autoregressive
frames have different contexts, p_t, candidate geometry, lexical content and mean scores;
pooling adds between-frame covariance and can manufacture a false positive.

Everything here is pure numpy (float64). Estimating psi (the post-channel response) needs
controlled BRANCH ROLLOUTS (branch.py / Colab) — a natural trajectory reveals only one
sampled token per frame and cannot identify C_raw (spec §1C).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class FrameMoments:
    """p_t-weighted moments within ONE autoregressive frame."""
    mu_phi: float
    mu_psi: float
    V: float          # Var_{v~p_t}[phi]         (pre-channel candidate-score variance)
    S: float          # Var_{v~p_t}[psi]         (sigma_matched^2 at this frame)
    C: float          # Cov_{v~p_t}[phi, psi]    (within-frame channel covariance)
    mass: float       # total probability mass of the candidates used

    def j_t(self, eps: float = 1e-12) -> float:
        """Per-state matched J_t = C / sqrt(V * S). Correlation coefficient; |J_t| <= 1."""
        d = np.sqrt(max(self.V, 0.0) * max(self.S, 0.0))
        return float(self.C / d) if d > eps else 0.0

    def to_dict(self):
        d = asdict(self)
        d["j_t"] = self.j_t()
        return d


def frame_moments(phi: np.ndarray, psi: np.ndarray, p: np.ndarray) -> FrameMoments:
    """p-weighted within-frame moments for one frame.

    phi: [K] pre-channel candidate scores  (phi_t(v) = u^T z_{m(v)})
    psi: [K] post-channel branch responses (mean over continuation seeds)
    p:   [K] base distribution weights over the SAME K candidates (need not sum to 1;
         renormalized here over the provided candidate set — coverage is tracked as `mass`).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    psi = np.asarray(psi, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if not (phi.shape == psi.shape == p.shape):
        raise ValueError(f"shape mismatch: phi{phi.shape} psi{psi.shape} p{p.shape}")
    mass = float(p.sum())
    if mass <= 0:
        raise ValueError("nonpositive probability mass on candidate set")
    w = p / mass
    mu_phi = float((w * phi).sum())
    mu_psi = float((w * psi).sum())
    dphi = phi - mu_phi
    dpsi = psi - mu_psi
    V = float((w * dphi * dphi).sum())
    S = float((w * dpsi * dpsi).sum())
    C = float((w * dphi * dpsi).sum())
    return FrameMoments(mu_phi, mu_psi, V, S, C, mass)


@dataclass
class Aggregate:
    V: float
    C_raw: float
    sigma_matched: float
    J_clean: float
    n_frames: int
    per_state_j: list

    def to_dict(self):
        return {"V": self.V, "C_raw": self.C_raw, "sigma_matched": self.sigma_matched,
                "J_clean": self.J_clean, "n_frames": self.n_frames}


def aggregate_within_frame(moments: list[FrameMoments], weights: np.ndarray | None = None) -> Aggregate:
    """Aggregate per-frame moments with explicit (equal-by-default) frame/text weights.

    C_raw = mean_t C_t ; V = mean_t V_t ; sigma_matched^2 = mean_t S_t ;
    J_clean = C_raw / (sqrt(V) * sigma_matched).

    This is mean_t(within-frame covariance) — never a pooled covariance.
    """
    if not moments:
        raise ValueError("no frames")
    n = len(moments)
    w = np.ones(n) / n if weights is None else np.asarray(weights, float) / np.sum(weights)
    V = float(np.sum([wi * m.V for wi, m in zip(w, moments)]))
    S = float(np.sum([wi * m.S for wi, m in zip(w, moments)]))
    C = float(np.sum([wi * m.C for wi, m in zip(w, moments)]))
    sigma = float(np.sqrt(max(S, 0.0)))
    denom = np.sqrt(max(V, 0.0)) * sigma
    J = float(C / denom) if denom > 1e-12 else 0.0
    return Aggregate(V=V, C_raw=C, sigma_matched=sigma, J_clean=J, n_frames=n,
                     per_state_j=[m.j_t() for m in moments])


def pooled_covariance(phi_all: np.ndarray, psi_all: np.ndarray) -> float:
    """DELIBERATELY WRONG estimator kept ONLY for the contrast unit test (spec §1B/§12).

    Concatenates observations from different frames and computes one covariance. Never call
    this in production — it adds between-frame covariance. `test_within_frame_covariance`
    asserts the production path does NOT equal this.
    """
    a = np.asarray(phi_all, float).ravel()
    b = np.asarray(psi_all, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean())
