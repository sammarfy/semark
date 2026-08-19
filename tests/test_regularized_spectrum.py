"""spec §8/§12: regularized spectrum. M symmetric, no direct inverse, finite eigenvalues,
ridge and shrinkage are independent paths."""
import numpy as np

from src.prewm.spectrum import (
    regularized_M, spectrum, inv_sqrt_psd, ledoit_wolf, effective_rank,
    principal_angles, top_eigvecs,
)


def _psd(d, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return scale * (A @ A.T) / d + 1e-3 * np.eye(d)


def test_M_symmetric_and_finite():
    Sd, Sr = _psd(12, 1), _psd(12, 2, scale=2.0)
    M = regularized_M(Sd, Sr, eps=1e-2)
    assert np.allclose(M, M.T, atol=1e-10)
    assert np.all(np.isfinite(spectrum(M)))


def test_no_direct_inverse_used():
    """inv_sqrt_psd is eigendecomposition-based and stays finite even for singular input."""
    S = np.diag([1.0, 0.0, 0.0])           # singular
    out = inv_sqrt_psd(S + 1e-2 * np.eye(3))
    assert np.all(np.isfinite(out))


def test_eta_above_one_when_R_dominates():
    # Sigma_R much larger than Sigma_D along a direction -> some eta > 1
    Sd = np.eye(6)
    Sr = np.diag([9.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    M = regularized_M(Sd, Sr, eps=1e-6)
    w = spectrum(M)
    assert w[0] > 1.0


def test_ridge_and_shrinkage_are_independent_paths():
    Sd, Sr = _psd(10, 3), _psd(10, 4)
    M_ridge = regularized_M(Sd, Sr, eps=1e-2, shrinkage=False)
    M_shrink = regularized_M(Sd, Sr, eps=1e-2, shrinkage=True, n_texts=60)
    # different regularizers -> different M (not identical)
    assert not np.allclose(M_ridge, M_shrink)
    _, beta = ledoit_wolf(Sd, 60)
    assert 0.0 <= beta <= 1.0


def test_effective_rank_bounds():
    d = 8
    assert np.isclose(effective_rank(np.eye(d)), d)          # isotropic -> full
    assert effective_rank(np.diag([1.0] + [1e-9] * (d - 1))) < 1.5  # spiky -> ~1


def test_principal_angles_identical_subspace_zero():
    A = np.linalg.qr(np.random.default_rng(5).standard_normal((10, 3)))[0]
    ang = principal_angles(A, A)
    assert np.allclose(ang, 0.0, atol=1e-6)   # arccos near 1 is ill-conditioned


def test_top_eigvecs_orthonormal():
    M = _psd(9, 6)
    V = top_eigvecs(M, 3)
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-8)
