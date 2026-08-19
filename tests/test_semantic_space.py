"""§16.B: semantic-space hook + the graded-policy guard.

Split into:
  * LOCAL tests (run without a GPU): graded acceptance policy, and a source scan
    proving no experiment/analysis code uses a bare `assert rate > 0.999` (§11);
  * MODEL tests (skipped unless torch + the loaded model are available on the host).
"""
import os
import re

import numpy as np
import pytest

from src.semantic_space import graded_agreement_decision

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------- LOCAL: graded policy --------------------------- #
def test_graded_policy_thresholds():
    assert graded_agreement_decision(0.9999) == "PASS"
    assert graded_agreement_decision(0.999) == "PASS"
    assert graded_agreement_decision(0.98) == "INVESTIGATE"
    assert graded_agreement_decision(0.95) == "INVESTIGATE"
    assert graded_agreement_decision(0.9499) == "HARD_STOP"
    assert graded_agreement_decision(0.5) == "HARD_STOP"


def test_no_bare_centroid_assert_in_e0_paths():
    """§11/§16.B: the E0 path must call the frozen quantizer and use the graded policy,
    never a bare `assert rate > 0.999` (or similar) on an agreement rate.
    """
    scan_dirs = [os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "experiments")]
    # Anchor to a REAL assert statement at the start of the (stripped) line, so prose
    # that merely mentions the forbidden pattern (this rule's own docstrings) is ignored.
    # Also drop any inline comment before matching.
    pattern = re.compile(r"^assert\s+[\w\.]*(rate|agreement|acc\w*)\s*>=?\s*0\.99")
    offenders = []
    for d in scan_dirs:
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(root, fn)
                    with open(p) as f:
                        for i, line in enumerate(f, 1):
                            code_part = line.split("#", 1)[0].strip()
                            if pattern.match(code_part):
                                offenders.append(f"{p}:{i}: {line.strip()}")
    assert not offenders, "bare centroid-agreement asserts found:\n" + "\n".join(offenders)


# --------------------------- MODEL: skipped locally ------------------------- #
torch = pytest.importorskip("torch", reason="semantic-space hook needs the loaded model")

MODEL_AVAILABLE = os.environ.get("SEMARK_MODEL_READY") == "1"
model_only = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="set SEMARK_MODEL_READY=1 on the GPU host after loading"
)


@model_only
def test_captured_latent_finite_and_dim(adapter, sample_waveform):
    space = adapter.encode_semantic_space(sample_waveform)
    assert np.isfinite(space.frame_latents).all()
    assert space.frame_latents.shape[1] == space.codebook.shape[1]


@model_only
def test_real_quantizer_roundtrip_matches_ids(adapter, sample_waveform):
    """Q_semantic(h_t) == tokenizer semantic_id_t over valid frames (the E0 invariant, §5).

    encode_semantic_space computes this agreement internally (srvq.encode(pre) vs the
    tokenizer's column-0 id) and stores it in hook_metadata.
    """
    space = adapter.encode_semantic_space(sample_waveform)
    agreement = float(space.hook_metadata["coordinate_agreement"])
    decision = graded_agreement_decision(agreement)
    assert decision != "HARD_STOP", f"agreement={agreement:.4f} -> {decision}"
