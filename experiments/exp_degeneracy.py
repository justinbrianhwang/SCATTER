"""DEGENERACY attack: the adversarial optimum where honest and eavesdropped
telemetry become observationally degenerate under LIMITED telemetry.

Eve runs gain-matched PNS and sweeps her single free knob -- the gain-restore
fraction that tops the count rate back up. As she tunes it, the LIMITED
detectability D_lim traces a sharp "degeneracy valley": at the optimum the
attacked telemetry law collapses onto the honest one (D_lim -> noise floor) even
though Eve still knows a large fraction I of the key. The FULL detectability
D_full stays an order of magnitude higher across the whole sweep (data-processing
gap), so decoy telemetry still catches her -- but cheap telemetry cannot.

Right axis shows the operational consequence via the Stein floor:
    N* = log(1/alpha) / D  = minimum blocks any detector needs.

Run:  PYTHONPATH=. python experiments/exp_degeneracy.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import PNS
from qkd.dataset import build_system, const, generate
from qkd.infometrics import fit_gaussian, kl_gaussian, stein_detection_blocks
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
NLIM = len(LIMITED_FEATURES)
ALPHA = 0.01


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(25.0)
    rng = np.random.default_rng(101)

    Xh = generate(sys, const(None), 900, N_PULSES, rng, FULL_FEATURES).X
    P0f, P0l = fit_gaussian(Xh), fit_gaussian(Xh[:, :NLIM])

    restores = np.linspace(0.12, 0.30, 19)
    Dl, Df, Iv = [], [], []
    for r in restores:
        bs = generate(sys, const(PNS(1.0, restore=r)), 450, N_PULSES, rng,
                     FULL_FEATURES)
        Dl.append(kl_gaussian(fit_gaussian(bs.X[:, :NLIM]), P0l))
        Df.append(kl_gaussian(fit_gaussian(bs.X), P0f))
        Iv.append(float(bs.eve_info.mean()))
    Dl, Df, Iv = map(np.array, (Dl, Df, Iv))

    imin = int(np.argmin(Dl))
    r_opt, D_opt, I_opt = restores[imin], Dl[imin], Iv[imin]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(restores, Dl, "-o", color="#c0392b", lw=2.2, ms=4,
            label="$D_{\\mathrm{LIMITED}}$ (cheap telemetry)")
    ax.plot(restores, Df, "-o", color="#2c6fbb", lw=2.2, ms=4,
            label="$D_{\\mathrm{FULL}}$ (decoy telemetry)")
    ax.axvline(r_opt, color="#555", ls=":", lw=1.2)
    ax.scatter([r_opt], [D_opt], s=160, marker="*", color="#c0392b",
               edgecolor="k", zorder=5)
    ax.annotate(f"degeneracy valley\n$D_{{lim}}$={D_opt:.3f}, "
                f"$I$={I_opt:.2f}\n$N^*_{{lim}}$={stein_detection_blocks(D_opt,ALPHA):.0f} "
                f"vs $N^*_{{full}}$={stein_detection_blocks(Df[imin],ALPHA):.0f} blocks",
                xy=(r_opt, D_opt), xytext=(r_opt + 0.005, D_opt * 8),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_yscale("log")
    ax.set_xlabel("Eve's gain-restore parameter  (her single free knob)")
    ax.set_ylabel("Detectability  $D$  (KL divergence, nats)")
    ax.set_title("DEGENERACY attack:  cheap-telemetry detectability collapses\n"
                 "while decoy telemetry still sees it")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # secondary axis: Stein detection-block floor
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax2.set_ylim(np.log(1 / ALPHA) / hi, np.log(1 / ALPHA) / lo)
    ax2.set_ylabel("Min blocks to detect  $N^* = \\log(1/\\alpha)/D$")

    fig.tight_layout()
    path = os.path.join(OUT, "exp_degeneracy.png")
    fig.savefig(path, dpi=150)
    print(f"optimum restore={r_opt:.3f}  D_lim={D_opt:.4f}  D_full={Df[imin]:.4f}  "
          f"I={I_opt:.3f}")
    print(f"N*_lim={stein_detection_blocks(D_opt,ALPHA):.1f}  "
          f"N*_full={stein_detection_blocks(Df[imin],ALPHA):.1f} blocks  "
          f"(penalty x{Df[imin]/D_opt:.1f})")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
