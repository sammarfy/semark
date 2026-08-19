"""§16.C: the shared transform. One formula, used identically on both sides."""
import numpy as np

from src import transforms
from src.transforms import (
    transform_shared_space,
    whitening_from_codebook,
    WhiteningParams,
)


def test_transform_matches_explicit_formula():
    rng = np.random.default_rng(0)
    d = 16
    center = rng.standard_normal(d)
    W = rng.standard_normal((d, d))
    x = rng.standard_normal((5, d))
    got = transform_shared_space(x, center, W, rtd_matrix=None)
    expected = (x - center) @ W.T
    assert np.allclose(got, expected)


def test_transform_single_vector():
    d = 8
    center = np.zeros(d)
    W = np.eye(d)
    x = np.arange(d, dtype=float)
    assert np.allclose(transform_shared_space(x, center, W), x)


def test_rtd_matrix_is_second_stage():
    rng = np.random.default_rng(1)
    d = 6
    center = rng.standard_normal(d)
    W = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    x = rng.standard_normal((4, d))
    got = transform_shared_space(x, center, W, rtd_matrix=B)
    expected = ((x - center) @ W.T) @ B.T
    assert np.allclose(got, expected)


def test_identical_function_used_for_both_sides():
    """The SAME callable transforms codebook rows and detector-side latents.

    There must not be two separately implemented formulas (Notion §6).
    """
    rng = np.random.default_rng(2)
    d = 10
    codebook = rng.standard_normal((40, d))
    wp = whitening_from_codebook(codebook, ridge=1e-3)

    detector_latents = rng.standard_normal((7, d))

    side_a = transform_shared_space(codebook, wp.center, wp.whitening_matrix)
    side_b = transform_shared_space(detector_latents, wp.center, wp.whitening_matrix)

    # Same function object, same params -> concatenating and transforming once must
    # equal transforming each side separately.
    both = np.concatenate([codebook, detector_latents], axis=0)
    joint = transform_shared_space(both, wp.center, wp.whitening_matrix)
    assert np.allclose(joint[: len(codebook)], side_a)
    assert np.allclose(joint[len(codebook):], side_b)


def test_whitening_from_frozen_codebook_records_provenance():
    rng = np.random.default_rng(3)
    codebook = rng.standard_normal((128, 12))
    wp = whitening_from_codebook(codebook, ridge=1e-3)
    assert isinstance(wp, WhiteningParams)
    md = wp.to_metadata()
    # §6.1 / §18: ridge, condition number, and SHA256 of BOTH W and c_bar recorded.
    assert md["ridge"] == 1e-3
    assert md["source"] == "frozen_semantic_codebook"
    assert len(md["center_sha256"]) == 64
    assert len(md["whitening_sha256"]) == 64
    assert md["codebook_cov_condition_number"] > 0
    assert md["n_centroids"] == 128 and md["dim"] == 12


def test_whitening_actually_whitens():
    """W(C - c_bar) should have near-identity covariance (up to the ridge)."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((6, 6))
    codebook = rng.standard_normal((5000, 6)) @ A.T  # correlated
    wp = whitening_from_codebook(codebook, ridge=1e-6)
    z = transform_shared_space(codebook, wp.center, wp.whitening_matrix)
    cov = np.cov(z, rowvar=False)
    assert np.allclose(cov, np.eye(6), atol=0.05)


def test_sha_is_deterministic_across_calls():
    codebook = np.arange(200 * 5, dtype=float).reshape(200, 5)
    a = whitening_from_codebook(codebook, ridge=1e-3)
    b = whitening_from_codebook(codebook, ridge=1e-3)
    assert a.center_sha256 == b.center_sha256
    assert a.whitening_sha256 == b.whitening_sha256
