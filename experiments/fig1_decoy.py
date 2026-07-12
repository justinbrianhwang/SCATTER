"""Fig. 1 (validation anchor): PNS key-rate collapse and decoy-state recovery.

Plots the asymptotic secret key rate vs fiber distance for:
  * GLLP without decoy (PNS-pessimistic) -- collapses at moderate distance
  * Decoy-state BB84                       -- extends the secure range

Both curves come from closed-form formulas (qkd.keyrate), so this figure also
certifies that the analytical layer behaves as the literature predicts.

Run:  PYTHONPATH=. python experiments/fig1_decoy.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.keyrate import keyrate_decoy, keyrate_gllp_nodecoy
from qkd.params import ChannelParams, DetectorParams, SourceParams, SystemParams

OUT = os.path.join(os.path.dirname(__file__), "figures")


def make_system(length_km: float, mu: float, nu: float = 0.1) -> SystemParams:
    return SystemParams(
        source=SourceParams(intensities=(mu, nu, 0.0), probs=(0.7, 0.15, 0.15)),
        channel=ChannelParams(length_km=length_km, attenuation_db_km=0.2,
                              misalignment=0.015),
        detector=DetectorParams(efficiency=0.1, dark_count=1e-6),
    )


# The signal intensity mu is a free protocol parameter; the fair comparison
# optimises it independently for each method at every distance. (No-decoy GLLP
# prefers mu ~ eta to suppress the untrusted multi-photon fraction, whereas
# decoy-state can push mu ~ 0.5.)
_MU_GRID = np.concatenate([np.linspace(0.005, 0.2, 40), np.linspace(0.2, 0.9, 36)])


def _best(fn, L: float, nu: float | None = None) -> float:
    best = 0.0
    for mu in _MU_GRID:
        if nu is not None and mu <= nu:
            continue
        s = make_system(L, mu, nu) if nu is not None else make_system(L, mu)
        best = max(best, fn(s)["rate"])
    return best


def sweep(distances: np.ndarray):
    r_decoy, r_nodecoy = [], []
    for L in distances:
        r_decoy.append(_best(keyrate_decoy, L, nu=0.1))
        r_nodecoy.append(_best(keyrate_gllp_nodecoy, L))
    return np.array(r_decoy), np.array(r_nodecoy)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    distances = np.linspace(0, 200, 201)
    r_decoy, r_nodecoy = sweep(distances)

    def cutoff(r):
        pos = distances[r > 0]
        return pos.max() if pos.size else 0.0

    L_decoy, L_nodecoy = cutoff(r_decoy), cutoff(r_nodecoy)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    floor = 1e-9
    ax.semilogy(distances, np.maximum(r_nodecoy, floor), "--", lw=2,
                color="#c0392b", label=f"GLLP, no decoy (PNS)  [≤{L_nodecoy:.0f} km]")
    ax.semilogy(distances, np.maximum(r_decoy, floor), "-", lw=2,
                color="#2c6fbb", label=f"Decoy-state BB84  [≤{L_decoy:.0f} km]")
    ax.axvspan(L_nodecoy, L_decoy, color="#2c6fbb", alpha=0.07,
               label="recovered range")

    ax.set_xlabel("Fiber distance (km)")
    ax.set_ylabel("Secret key rate per pulse")
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlim(0, 200)
    ax.set_title("Fig. 1  PNS collapse and decoy-state recovery")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, which="major", ls=":", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(OUT, "fig1_decoy.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")  # vector version for the paper
    print(f"no-decoy secure range : {L_nodecoy:.0f} km")
    print(f"decoy    secure range : {L_decoy:.0f} km")
    print(f"recovered             : +{L_decoy - L_nodecoy:.0f} km")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
