"""Security consequence: certified-secret key stolen before detection, vs distance.

Combines the SCATTER detection delay N*_T with the finite-key ledger to report
how many *certified-secret* bits the DEGENERACY attack harvests before SCATTER
raises an alarm, under LIMITED vs FULL telemetry, as a function of fiber length.
The cheap-telemetry curve sits an order of magnitude higher: the same attack,
the same protocol, but far more key stolen with a security certificate attached.

Run:  PYTHONPATH=. python experiments/exp_stolen_key.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import PNS
from qkd.dataset import build_system, const, generate
from qkd.degeneracy import gain_match
from qkd.infometrics import fit_gaussian, kl_gaussian, stein_detection_blocks
from qkd.security import ledger
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
NLIM = len(LIMITED_FEATURES)
ALPHA = 0.01


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(303)
    lengths = [10, 25, 40, 55, 70, 85, 100]
    k_lim, k_full, Ns_lim, Ns_full = [], [], [], []

    for L in lengths:
        sys = build_system(length_km=L)
        r, m = gain_match(sys)
        Xh = generate(sys, const(None), 500, N_PULSES, rng, FULL_FEATURES).X
        P0f, P0l = fit_gaussian(Xh), fit_gaussian(Xh[:, :NLIM])
        bs = generate(sys, const(PNS(1.0, restore=r, multi_forward=m)), 500,
                     N_PULSES, rng, FULL_FEATURES)
        Df = kl_gaussian(fit_gaussian(bs.X), P0f)
        Dl = kl_gaussian(fit_gaussian(bs.X[:, :NLIM]), P0l)
        I = float(bs.eve_info.mean())
        Nf = stein_detection_blocks(max(Df, 1e-9), ALPHA)
        Nl = stein_detection_blocks(max(Dl, 1e-9), ALPHA)
        lf = ledger(sys, I, Nf, N_PULSES)
        ll = ledger(sys, I, Nl, N_PULSES)
        k_full.append(lf.k_stolen); k_lim.append(ll.k_stolen)
        Ns_full.append(Nf); Ns_lim.append(Nl)
        print(f"L={L:3d}km  I={I:.2f}  D_lim={Dl:.3f} D_full={Df:.3f}  "
              f"N*_lim={Nl:5.1f} N*_full={Nf:5.1f}  "
              f"K_stolen: lim={ll.k_stolen:8.0f}  full={lf.k_stolen:7.0f} bits")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(lengths, k_lim, "-o", color="#c0392b", lw=2.2,
            label="LIMITED telemetry")
    ax.plot(lengths, k_full, "-o", color="#2c6fbb", lw=2.2,
            label="FULL telemetry")
    ax.fill_between(lengths, k_full, k_lim, color="#c0392b", alpha=0.08)
    ax.set_yscale("log")
    ax.set_xlabel("Fiber distance (km)")
    ax.set_ylabel("Certified-secret bits stolen before detection")
    ax.set_title("Security cost of cheap telemetry under the DEGENERACY attack")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(OUT, "exp_stolen_key.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
