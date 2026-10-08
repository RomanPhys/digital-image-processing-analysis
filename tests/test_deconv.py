"""Sanity checks of the mathematical properties the methods rely on.  Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deconv import (psf_to_otf, conv, conv_adj, degrade, estimate_sigma,  # noqa: E402
                    wiener_fixed, richardson_lucy, rl_discrepancy)
from run_benchmark import gaussian_psf  # noqa: E402

rng = np.random.default_rng(0)
X = rng.random((64, 64))
PSF = gaussian_psf(9, 1.5)


def test_adjoint_identity():
    """<Hx, y> == <x, H^T y> - the adjoint is implemented correctly."""
    otf = psf_to_otf(PSF, X.shape)
    y = rng.random(X.shape)
    assert np.isclose(np.vdot(conv(X, otf), y), np.vdot(X, conv_adj(y, otf)))


def test_rl_preserves_flux_and_positivity():
    y = degrade(X, PSF, 0.0, rng) + 0.01
    u = richardson_lucy(y, PSF, n_iter=20)
    assert u.min() >= 0
    assert np.isclose(u.sum(), y.sum(), rtol=1e-6)


def test_wiener_inverts_noiseless_blur():
    psf = gaussian_psf(5, 0.5)  # mild blur, |H| bounded away from zero
    y = degrade(X, psf, 0.0, rng)
    assert np.abs(wiener_fixed(y, psf, K=1e-10) - X).max() < 1e-4


def test_noise_estimator():
    y = degrade(X * 0 + 0.5, PSF, 0.05, rng)
    assert abs(estimate_sigma(y) - 0.05) < 0.005


def test_discrepancy_stops_earlier_for_more_noise():
    from skimage import data, img_as_float
    x = img_as_float(data.camera()).astype(float)
    _, k_low = rl_discrepancy(degrade(x, PSF, 0.01, rng), PSF)
    _, k_high = rl_discrepancy(degrade(x, PSF, 0.10, rng), PSF)
    assert k_high < k_low
