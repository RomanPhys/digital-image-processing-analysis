"""
Deconvolution methods implemented from scratch (NumPy FFT only).

Forward model (circular boundary, so that convolution is diagonal in Fourier space):
    y = h * x + n,   n ~ N(0, sigma^2)

Methods
-------
wiener_fixed      : classical Wiener / Tikhonov filter with a hand-picked constant K
wiener_powerlaw   : MMSE Wiener filter with a power-law image prior S_x(w) = A |w|^-beta,
                    A, beta and sigma are estimated from the observed image itself (blind w.r.t. noise)
richardson_lucy   : EM algorithm for the Poisson likelihood, fixed number of iterations
rl_discrepancy    : RL with early stopping by Morozov's discrepancy principle
rl_tv             : RL with total-variation regularisation (Dey et al., 2006)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

EPS = 1e-12


def estimate_sigma(y: np.ndarray) -> float:
    """
    Immerkaer (1996) fast noise estimator. The mask M is the difference of two discrete
    Laplacians, so it annihilates locally-quadratic image content and passes white noise
    with ||M||^2 = 36:  sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |y * M|.
    Works especially well on blurred images, where true high-frequency content is suppressed.
    """
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    h, w = y.shape
    return float(np.sqrt(np.pi / 2) * np.abs(convolve2d(y, M, mode="valid")).sum() / (6 * (w - 2) * (h - 2)))


# --------------------------------------------------------------------------- #
#  Linear operators H and H^T via FFT
# --------------------------------------------------------------------------- #
def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad PSF to image size and roll its centre to (0, 0) -> optical transfer function."""
    pad = np.zeros(shape, dtype=float)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def conv(x: np.ndarray, otf: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.fft2(x) * otf))


def conv_adj(x: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Adjoint operator H^T = correlation with the PSF (conjugate OTF)."""
    return np.real(np.fft.ifft2(np.fft.fft2(x) * np.conj(otf)))


def degrade(x: np.ndarray, psf: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    otf = psf_to_otf(psf, x.shape)
    return conv(x, otf) + sigma * rng.standard_normal(x.shape)


# --------------------------------------------------------------------------- #
#  Wiener filters
# --------------------------------------------------------------------------- #
def wiener_fixed(y: np.ndarray, psf: np.ndarray, K: float = 1e-2) -> np.ndarray:
    """X = conj(H) Y / (|H|^2 + K).  K plays the role of noise-to-signal power ratio."""
    H = psf_to_otf(psf, y.shape)
    X = np.conj(H) * np.fft.fft2(y) / (np.abs(H) ** 2 + K)
    return np.real(np.fft.ifft2(X))


def _radial_freq(shape: tuple[int, int]) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.sqrt(fx ** 2 + fy ** 2)


def fit_powerlaw_prior(y: np.ndarray, sigma: float, otf: np.ndarray, n_bins: int = 64):
    """
    Natural images have a power spectrum close to A / |f|^beta with beta ~ 2.
    Observed:  P_y(f) = |H(f)|^2 P_x(f) + N sigma^2.
    Radially average P_y, then fit log((P_y - N sigma^2) / |H|^2) = log A - beta log f
    only on rings where the estimate is reliable: signal clearly above the noise floor
    and |H|^2 not too small (otherwise we would divide noise by ~0).
    """
    N = y.size
    P = np.abs(np.fft.fft2(y - y.mean())) ** 2
    H2 = np.abs(otf) ** 2
    f = _radial_freq(y.shape)
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(f, edges) - 1
    fc, px = [], []
    for b in range(1, n_bins):
        m = idx == b
        if not m.any():
            continue
        Py, Hb = P[m].mean(), H2[m].mean()
        if Py > 2 * N * sigma ** 2 and Hb > 0.2:
            fc.append(f[m].mean())
            px.append((Py - N * sigma ** 2) / Hb)
    if len(fc) < 3:  # degenerate case: fall back to the classical 1/f^2 law
        return float(np.mean(P[(f > 0) & (f < 0.05)] * f[(f > 0) & (f < 0.05)] ** 2)), 2.0
    slope, intercept = np.polyfit(np.log(fc), np.log(px), 1)
    return float(np.exp(intercept)), float(np.clip(-slope, 1.0, 4.0))


def wiener_powerlaw(y: np.ndarray, psf: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    MMSE (true Wiener) filter:  W = conj(H) S_x / (|H|^2 S_x + S_n),
    with S_n = N sigma^2 (white noise, unnormalised FFT) and S_x = A |f|^-beta.
    sigma is estimated from y if not given (wavelet MAD estimator).
    """
    if sigma is None:
        sigma = float(estimate_sigma(y))
    N = y.size
    H = psf_to_otf(psf, y.shape)
    A, beta = fit_powerlaw_prior(y, sigma, H)
    f = _radial_freq(y.shape)
    f[0, 0] = f[f > 0].min()  # avoid division by zero at DC
    Sx = A * f ** (-beta)
    Sn = N * sigma ** 2
    W = np.conj(H) * Sx / (np.abs(H) ** 2 * Sx + Sn)
    Y = np.fft.fft2(y - y.mean())
    return np.real(np.fft.ifft2(W * Y)) + y.mean()


# --------------------------------------------------------------------------- #
#  Richardson-Lucy family
# --------------------------------------------------------------------------- #
def _rl_step(u: np.ndarray, y_pos: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """
    One EM step for the Poisson likelihood:
        u <- u * H^T( y / (H u) )
    Since sum(psf) = 1, H^T 1 = 1, so the update is a multiplicative gradient step that
    preserves positivity and total flux.
    """
    ratio = y_pos / np.maximum(conv(u, otf), EPS)
    return u * conv_adj(ratio, otf)


def richardson_lucy(y: np.ndarray, psf: np.ndarray, n_iter: int = 30) -> np.ndarray:
    otf = psf_to_otf(psf, y.shape)
    y_pos = np.maximum(y, EPS)
    u = np.full_like(y_pos, y_pos.mean())
    for _ in range(n_iter):
        u = _rl_step(u, y_pos, otf)
    return u


def rl_discrepancy(y: np.ndarray, psf: np.ndarray, sigma: float | None = None,
                   tau: float = 1.0, max_iter: int = 300) -> tuple[np.ndarray, int]:
    """
    RL exhibits semi-convergence: the error first falls, then noise gets amplified.
    Morozov's principle: stop at the first k with ||H u_k - y||^2 <= tau * N * sigma^2,
    i.e. when the model explains the data down to the noise level and not further.
    Returns the estimate and the chosen number of iterations.
    """
    if sigma is None:
        sigma = float(estimate_sigma(y))
    otf = psf_to_otf(psf, y.shape)
    y_pos = np.maximum(y, EPS)
    u = np.full_like(y_pos, y_pos.mean())
    target = tau * y.size * sigma ** 2
    for k in range(1, max_iter + 1):
        u = _rl_step(u, y_pos, otf)
        if np.sum((conv(u, otf) - y) ** 2) <= target:
            return u, k
    return u, max_iter


def _tv_divergence(u: np.ndarray) -> np.ndarray:
    """div( grad u / |grad u| ) with forward differences for grad and backward for div."""
    gx = np.roll(u, -1, axis=1) - u
    gy = np.roll(u, -1, axis=0) - u
    norm = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    nx, ny = gx / norm, gy / norm
    return (nx - np.roll(nx, 1, axis=1)) + (ny - np.roll(ny, 1, axis=0))


def rl_tv(y: np.ndarray, psf: np.ndarray, lam: float = 0.005, n_iter: int = 300,
          sigma: float | None = None, tau: float = 1.0) -> tuple[np.ndarray, int]:
    """
    RL + TV (Dey et al., 2006): minimise KL(y || Hu) + lam * TV(u).
    Fixed-point update:  u <- u / (1 - lam * div(grad u/|grad u|)) * H^T( y / (H u) ).
    TV penalises oscillations (ringing, noise) but keeps sharp edges.
    Iterations are stopped by the same discrepancy principle as in rl_discrepancy.
    """
    if sigma is None:
        sigma = estimate_sigma(y)
    otf = psf_to_otf(psf, y.shape)
    y_pos = np.maximum(y, EPS)
    u = np.full_like(y_pos, y_pos.mean())
    target = tau * y.size * sigma ** 2
    for k in range(1, n_iter + 1):
        denom = np.clip(1.0 - lam * _tv_divergence(u), 0.5, 2.0)
        u = _rl_step(u, y_pos, otf) / denom
        if np.sum((conv(u, otf) - y) ** 2) <= target:
            return u, k
    return u, n_iter
