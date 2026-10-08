"""
Reproducible benchmark: images x PSF x noise level x method -> PSNR / SSIM.

    python src/run_benchmark.py                    # built-in scikit-image test images
    python src/run_benchmark.py --images data/original   # your own images (png/jpg/tif)

Outputs
    results/metrics.csv       one row per (image, psf, noise, method)
    results/summary.md        mean PSNR/SSIM by method and noise level + gain vs blurred input
    figures/psnr_vs_noise.png
    figures/example_*.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data, io, color, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from deconv import (degrade, wiener_fixed, wiener_powerlaw, richardson_lucy,
                    rl_discrepancy, rl_tv, psf_to_otf, _rl_step)

SEED = 42
NOISE_LEVELS = (0.01, 0.05, 0.10)


# --------------------------------------------------------------------------- #
#  PSFs
# --------------------------------------------------------------------------- #
def gaussian_psf(size: int = 15, s: float = 2.0) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * s ** 2))
    return g / g.sum()


def motion_psf(length: int = 15, angle_deg: float = 30.0) -> np.ndarray:
    psf = np.zeros((length, length))
    c = length // 2
    t = np.linspace(-c, c, 4 * length)
    a = np.deg2rad(angle_deg)
    rows = np.clip(np.round(c - t * np.sin(a)).astype(int), 0, length - 1)
    cols = np.clip(np.round(c + t * np.cos(a)).astype(int), 0, length - 1)
    psf[rows, cols] = 1.0
    return psf / psf.sum()


def defocus_psf(radius: int = 5) -> np.ndarray:
    ax = np.arange(2 * radius + 1) - radius
    disk = (ax[:, None] ** 2 + ax[None, :] ** 2 <= radius ** 2).astype(float)
    return disk / disk.sum()


PSFS = {"gaussian": gaussian_psf(), "motion": motion_psf(), "defocus": defocus_psf()}


# --------------------------------------------------------------------------- #
#  Images
# --------------------------------------------------------------------------- #
def load_images(folder: str | None) -> dict[str, np.ndarray]:
    if folder is None:
        imgs = {"camera": data.camera(), "moon": data.moon(),
                "astronaut": color.rgb2gray(data.astronaut()), "coins": data.coins()}
    else:
        imgs = {}
        for p in sorted(Path(folder).glob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                im = io.imread(p)
                if im.ndim == 3:
                    im = color.rgb2gray(im[..., :3])
                imgs[p.stem] = im
    return {k: img_as_float(v).astype(float) for k, v in imgs.items()}


# --------------------------------------------------------------------------- #
#  Methods registry
# --------------------------------------------------------------------------- #
K_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)

CLASSIC = ["Wiener, K=1e-2", "RL, 30 iter"]
ORACLE = ["Wiener, oracle K", "RL, oracle iters"]
OURS = ["Wiener + power-law prior", "RL + discrepancy stop", "RL + TV + discrepancy stop"]


def _timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t


def run_methods(y: np.ndarray, psf: np.ndarray, x: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
    """x (ground truth) is used ONLY by the oracle baselines, which are upper bounds
    for hand-tuned classical methods; the proposed methods are fully blind."""
    out = {"blurred (no restoration)": (y, {})}
    r, t = _timed(wiener_fixed, y, psf, K=1e-2)
    out["Wiener, K=1e-2"] = (r, {"time_s": t})
    cands = [wiener_fixed(y, psf, K) for K in K_GRID]
    out["Wiener, oracle K"] = (max(cands, key=lambda e: psnr(x, np.clip(e, 0, 1), data_range=1)), {})
    r, t = _timed(wiener_powerlaw, y, psf)
    out["Wiener + power-law prior"] = (r, {"time_s": t})

    r, t = _timed(richardson_lucy, y, psf, n_iter=30)
    out["RL, 30 iter"] = (r, {"time_s": t})
    # oracle RL: best iterate along one trajectory
    otf = psf_to_otf(psf, y.shape); y_pos = np.maximum(y, 1e-12)
    u = np.full_like(y_pos, y_pos.mean()); best, best_p, best_k = u, -np.inf, 0
    for k in range(1, 101):
        u = _rl_step(u, y_pos, otf)
        p = psnr(x, np.clip(u, 0, 1), data_range=1)
        if p > best_p:
            best, best_p, best_k = u.copy(), p, k
    out["RL, oracle iters"] = (best, {"iters": best_k})
    (r, k), t = _timed(rl_discrepancy, y, psf)
    out["RL + discrepancy stop"] = (r, {"time_s": t, "iters": k})
    (r, k), t = _timed(rl_tv, y, psf, lam=0.005)
    out["RL + TV + discrepancy stop"] = (r, {"time_s": t, "iters": k})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="folder with images; default: skimage test set")
    ap.add_argument("--out", default=".", help="project root for results/ and figures/")
    args = ap.parse_args()

    root = Path(args.out)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    images = load_images(args.images)
    rows, example = [], None
    for img_name, x in images.items():
        for psf_name, psf in PSFS.items():
            for sigma in NOISE_LEVELS:
                y = degrade(x, psf, sigma, rng)
                for method, (est, info) in run_methods(y, psf, x).items():
                    est_c = np.clip(est, 0, 1)
                    rows.append({
                        "image": img_name, "psf": psf_name, "noise_sigma": sigma, "method": method,
                        "psnr": psnr(x, est_c, data_range=1.0),
                        "ssim": ssim(x, est_c, data_range=1.0),
                        **info,
                    })
                if img_name == next(iter(images)) and psf_name == "motion" and sigma == 0.05:
                    example = (x, y, {m: np.clip(e, 0, 1) for m, (e, _) in run_methods(y, psf, x).items()})

    df = pd.DataFrame(rows)
    df.to_csv(root / "results" / "metrics.csv", index=False)

    # ---------------- summary ----------------
    summ = (df.groupby(["noise_sigma", "method"])[["psnr", "ssim"]].mean().round(3).reset_index())
    base = df[df.method == "blurred (no restoration)"].set_index(["image", "psf", "noise_sigma"])["psnr"]
    df["dpsnr_vs_blurred"] = df.apply(lambda r: r.psnr - base.loc[(r.image, r.psf, r.noise_sigma)], axis=1)

    key = ["image", "psf", "noise_sigma"]
    best_classic = df[df.method.isin(CLASSIC)].groupby(key)["psnr"].max()
    best_oracle = df[df.method.isin(ORACLE)].groupby(key)["psnr"].max()
    best_ours = df[df.method.isin(OURS)].groupby(key)["psnr"].max()
    g_cl, g_or = best_ours - best_classic, best_ours - best_oracle
    lines = [
        "# Benchmark summary\n",
        f"Cases: {len(images)} images x {len(PSFS)} PSF x {len(NOISE_LEVELS)} noise levels "
        f"= {len(images) * len(PSFS) * len(NOISE_LEVELS)}\n",
        "## Mean PSNR / SSIM by noise level\n",
        summ.to_markdown(index=False), "\n",
        "## Best blind modification vs best classical method with default params (PSNR, dB)\n",
        f"- mean {g_cl.mean():+.2f} dB, median {g_cl.median():+.2f} dB, wins {(g_cl > 0).sum()}/{len(g_cl)}",
        "\n## Best blind modification vs ORACLE-tuned classical methods (uses ground truth)\n",
        f"- mean {g_or.mean():+.2f} dB, median {g_or.median():+.2f} dB, wins {(g_or > 0).sum()}/{len(g_or)}",
        "\nBy noise level (vs oracle):\n",
        g_or.groupby(level="noise_sigma").agg(["mean", "median"]).round(2).to_markdown(),
    ]
    if "iters" in df:
        it = df.dropna(subset=["iters"]).groupby("noise_sigma")["iters"].median()
        lines += ["\n## Median RL iterations chosen by discrepancy principle\n", it.to_markdown()]
    (root / "results" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in df.groupby("method"):
        s = g.groupby("noise_sigma")["psnr"].mean()
        ax.plot(s.index, s.values, marker="o", label=m)
    ax.set_xlabel("noise sigma"); ax.set_ylabel("mean PSNR, dB"); ax.grid(alpha=.3)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(root / "figures" / "psnr_vs_noise.png", dpi=150); plt.close(fig)

    if example is not None:
        x, y, ests = example
        panels = [("original", x)] + list(ests.items())
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for a in axes.ravel():
            a.axis("off")
        for a, (title, im) in zip(axes.ravel(), panels):
            a.imshow(im, cmap="gray", vmin=0, vmax=1)
            a.set_title(f"{title}\nPSNR {psnr(x, im, data_range=1):.2f}" if title != "original" else title,
                        fontsize=9)
        fig.tight_layout()
        fig.savefig(root / "figures" / "example_motion_sigma005.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
