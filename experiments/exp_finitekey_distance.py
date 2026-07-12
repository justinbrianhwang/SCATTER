"""Finite-key secret-key rate vs fiber distance.

This figure compares finite-block composable decoy-state key rates against the
asymptotic decoy reference. As the emitted-pulse block size grows, the
finite-key curve approaches the asymptotic limit; shorter blocks reach zero key
rate at shorter fiber distances.

Run:  PYTHONPATH=. python experiments/exp_finitekey_distance.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.dataset import build_system
from qkd.finitekey import secret_key_rate
from qkd.keyrate import keyrate_decoy

OUT = os.path.join(os.path.dirname(__file__), "figures")
DISTANCES = np.arange(0.0, 200.0 + 1e-9, 5.0)
BLOCK_SIZES = (1e8, 1e10, 1e12)
COLORS = ("#c0392b", "#2c6fbb", "#e67e22")
ASYM_COLOR = "#8e44ad"
FLOOR = 1e-9


def finite_rates(distances: np.ndarray, N: float) -> np.ndarray:
    """Finite-key secret-key rates per emitted pulse over distance."""
    return np.array([
        secret_key_rate(build_system(length_km=float(L)), N)
        for L in distances
    ])


def asymptotic_rates(distances: np.ndarray) -> np.ndarray:
    """Asymptotic decoy-state secret-key rates per emitted pulse."""
    return np.array([
        keyrate_decoy(build_system(length_km=float(L)))["rate"]
        for L in distances
    ])


def zero_cutoff(distances: np.ndarray, rates: np.ndarray) -> float | None:
    """First sampled distance where the finite-key rate reaches zero."""
    zero = np.flatnonzero(rates <= 0.0)
    if zero.size:
        return float(distances[int(zero[0])])
    return None


def cutoff_label(cutoff: float | None) -> str:
    if cutoff is None:
        return f">{DISTANCES[-1]:.0f} km"
    return f"{cutoff:.0f} km"


def block_label(N: float) -> str:
    exp = int(np.log10(N))
    return rf"$N=10^{{{exp}}}$"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    finite = {N: finite_rates(DISTANCES, N) for N in BLOCK_SIZES}
    cutoffs = {N: zero_cutoff(DISTANCES, rates) for N, rates in finite.items()}
    asym = asymptotic_rates(DISTANCES)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for (N, rates), color in zip(finite.items(), COLORS):
        label = f"{block_label(N)}  (zero at {cutoff_label(cutoffs[N])})"
        ax.semilogy(DISTANCES, np.maximum(rates, FLOOR), lw=2.1,
                    color=color, label=label)

        positive = np.flatnonzero(rates > 0.0)
        if positive.size:
            i = int(positive[-1])
            ax.annotate(
                cutoff_label(cutoffs[N]),
                xy=(DISTANCES[i], max(rates[i], FLOOR)),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )

    ax.semilogy(DISTANCES, np.maximum(asym, FLOOR), "--", lw=2.0,
                color=ASYM_COLOR, label="asymptotic decoy")

    ax.set_xlabel("Fiber distance (km)")
    ax.set_ylabel("Secret key rate per pulse")
    ax.set_xlim(float(DISTANCES.min()), float(DISTANCES.max()))
    ax.set_ylim(FLOOR, 2e-2)
    ax.set_title("Finite-key rate approaches the asymptotic limit as N grows;\n"
                 "shorter blocks cut the secure distance")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(OUT, "exp_finitekey_distance.png")
    fig.savefig(path, dpi=150)

    for N in BLOCK_SIZES:
        print(f"N={N:.0e} cutoff distance: {cutoff_label(cutoffs[N])}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
