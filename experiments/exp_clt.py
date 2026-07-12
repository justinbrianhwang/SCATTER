"""Central-limit validation of honest LIMITED telemetry Gaussianity.

This figure samples honest BB84 telemetry blocks and compares each LIMITED
feature's empirical block distribution with a Gaussian fit. A near-Gaussian
shape validates the multivariate Gaussian block law that SCATTER assumes when
turning telemetry shifts into KL detectability.

Run:  PYTHONPATH=. python experiments/exp_clt.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, norm, skew

from qkd.dataset import build_system, const, generate
from qkd.telemetry import LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_BLOCKS = 3000
N_PULSES = 20000
LENGTH_KM = 25.0


def feature_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0.0:
        sigma = 1e-12
    s = float(skew(x, bias=False))
    k = float(kurtosis(x, fisher=True, bias=False))
    return mu, sigma, s, k


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    sys = build_system(LENGTH_KM)
    rng = np.random.default_rng(2603)
    blocks = generate(sys, const(None), N_BLOCKS, N_PULSES, rng, LIMITED_FEATURES)

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 4.6))
    axes = axes.ravel()
    rows = []

    for i, name in enumerate(LIMITED_FEATURES):
        ax = axes[i]
        x = blocks.X[:, i]
        mu, sigma, s, k = feature_stats(x)
        rows.append((name, s, k))

        ax.hist(x, bins=36, density=True, color="#2c6fbb", alpha=0.35,
                edgecolor="white", linewidth=0.4)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        if np.allclose(grid[0], grid[-1]):
            grid = mu + np.linspace(-4.0, 4.0, 300) * sigma
        ax.plot(grid, norm.pdf(grid, mu, sigma), color="#c0392b", lw=2.0)

        ax.set_title(f"{name}\nskew={s:.2f}, kurt={k:.2f}", fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)

    axes[-1].axis("off")
    fig.suptitle("Block telemetry is Gaussian (CLT): the basis for KL detectability",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    path = os.path.join(OUT, "exp_clt.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")  # vector version for the paper

    for name, s, k in rows:
        print(f"{name:14s} skew={s: .4f} excess_kurtosis={k: .4f}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
