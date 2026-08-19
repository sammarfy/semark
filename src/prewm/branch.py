"""Controlled branch-rollout framework for estimating C_raw (spec §5, Phase D).

A natural trajectory reveals only ONE sampled token per frame, so it cannot identify the
within-frame covariance C_raw (spec §1C). We must, at a frozen prefix state at frame t:
  - force each candidate v (selected by cumulative mass, candidates.py),
  - continue with COMMON RANDOM NUMBERS across candidates,
  - decode -> re-encode -> aligned detector score psi_bar_t(v),
then compute the WITHIN-FRAME covariance C_t = Cov_{v~p_t}[phi_t(v), psi_bar_t(v)] and
aggregate over anchor states (covariance.py). The runtime forcing lives in
run_branch_pilot.py (GPU); this module holds the pure, testable pieces.

Mandatory validation (§5.1): forcing the naturally-sampled token must reproduce the base
path under the same continuation randomness. If it does not, the intervention is untrusted
and C/J must be reported NOT IDENTIFIED (never pooled covariance).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .covariance import frame_moments, FrameMoments, aggregate_within_frame, Aggregate


def continuation_seeds(anchor_id: int, n_seeds: int, base: int = 90_000) -> list[int]:
    """Deterministic common-random-number seeds, SHARED across candidates at one anchor."""
    return [base + anchor_id * 100 + s for s in range(n_seeds)]


@dataclass
class BranchFrame:
    """Everything needed to compute one within-frame moment from branch rollouts."""
    anchor_id: int
    text_id: str
    frame_index: int
    candidate_ids: np.ndarray        # [K] talker candidate ids
    candidate_p: np.ndarray          # [K] p_t weights over these candidates (unnormalized ok)
    phi: np.ndarray                  # [K] u^T z_{m(v)} (0 for special/control candidates)
    psi_bar: np.ndarray              # [K] mean detector score over continuation seeds
    covered_mass: float
    low_coverage: bool = False
    per_seed_psi: np.ndarray | None = None   # [K, n_seeds] optional, for CRN diagnostics

    def moment(self) -> FrameMoments:
        return frame_moments(self.phi, self.psi_bar, self.candidate_p)


def aggregate_frames(frames: list[BranchFrame], equal: str = "state") -> Aggregate:
    """Aggregate within-frame moments with equal-state or equal-text weighting (declared)."""
    moments = [f.moment() for f in frames]
    if equal == "state":
        return aggregate_within_frame(moments)
    if equal == "text":
        # equal weight per text, then per state within text
        by_text: dict[str, list[int]] = {}
        for i, f in enumerate(frames):
            by_text.setdefault(f.text_id, []).append(i)
        w = np.zeros(len(frames))
        for _t, idxs in by_text.items():
            for i in idxs:
                w[i] = 1.0 / (len(by_text) * len(idxs))
        return aggregate_within_frame(moments, weights=w)
    raise ValueError(f"unknown weighting {equal}")


def natural_reproduction_ok(base_tokens: np.ndarray, forced_tokens: np.ndarray,
                            base_latent: np.ndarray | None = None,
                            forced_latent: np.ndarray | None = None,
                            tol: float = 1e-4) -> dict:
    """§5.1 check: forcing the naturally-sampled token reproduces the base path under the
    same continuation randomness. Compares emitted token sequences (exact) and, if provided,
    detector latents (numerical). Returns a dict verdict; the pilot HARD-STOPs if not ok."""
    bt = np.asarray(base_tokens).ravel()
    ft = np.asarray(forced_tokens).ravel()
    n = min(bt.size, ft.size)
    token_match = float((bt[:n] == ft[:n]).mean()) if n else float("nan")
    latent_ok = True
    max_dev = 0.0
    if base_latent is not None and forced_latent is not None:
        a = np.asarray(base_latent, float); b = np.asarray(forced_latent, float)
        m = min(len(a), len(b))
        max_dev = float(np.max(np.abs(a[:m] - b[:m]))) if m else float("inf")
        latent_ok = max_dev <= tol
    ok = token_match > 0.999 and latent_ok
    return {"ok": bool(ok), "token_match": token_match, "max_latent_dev": max_dev,
            "reason": "" if ok else "forced-natural path did not reproduce base (see §5.1)"}
