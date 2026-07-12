"""Where the degeneracy is exposed: the analytic decoy residual.

The 1-knob DEGENERACY attack matches the honest signal gain exactly, but the
decoy intensity nu carries an irreducible residual Delta(nu) = Q_E(nu) - Q_H(nu)
-- the only trace left in the telemetry. These curves are closed-form
(qkd/degeneracy.py), so they are exact.

(a) |Delta(nu)| vs fiber distance: the exposure shrinks with distance as the
    channel loss makes the single-photon term dominate.
(b) |Delta(nu)| vs the decoy intensity nu at fixed distance: there is an optimal
    decoy setting that maximises the residual -- a protocol-design lever for the
    defender to make the DEGENERACY attack maximally visible.

Run:  PYTHONPATH=. python experiments/exp_residual_map.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.dataset import build_system
from qkd.degeneracy import decoy_residual, gain_match_restore

OUT = os.path.join(os.path.dirname(__file__), "figures")


def main():
    os.makedirs(OUT, exist_ok=True)

    # (a) residual vs distance, at the protocol decoy intensity nu = 0.1
    dists = np.linspace(5, 150, 30)
    res_d = []
    for L in dists:
        sys = build_system(length_km=L)
        r = gain_match_restore(sys)
        res_d.append(abs(decoy_residual(sys, r, 1.0, nu=0.1)))

    # (b) residual vs decoy intensity nu, at 25 km
    sys25 = build_system(25.0)
    r25 = gain_match_restore(sys25)
    nus = np.linspace(0.02, 0.45, 40)
    res_nu = [abs(decoy_residual(sys25, r25, 1.0, nu=float(nu))) for nu in nus]
    nu_star = nus[int(np.argmax(res_nu))]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axA.plot(dists, res_d, "-o", color="#2c6fbb", lw=2.2, ms=4)
    axA.set_yscale("log")
    axA.set_xlabel("Fiber distance (km)")
    axA.set_ylabel("|decoy residual $\\Delta(\\nu)$|")
    axA.set_title("(a)  Exposure vs distance  ($\\nu$=0.1)")
    axA.grid(True, which="both", ls=":", alpha=0.4)

    axB.plot(nus, res_nu, "-o", color="#c0392b", lw=2.2, ms=4)
    axB.axvline(nu_star, color="#555", ls=":", lw=1.2)
    axB.annotate(f"optimal decoy\n$\\nu^*$={nu_star:.2f}",
                 xy=(nu_star, max(res_nu)), xytext=(nu_star + 0.05, max(res_nu) * 0.8),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    axB.set_xlabel("Decoy intensity  $\\nu$")
    axB.set_ylabel("|decoy residual $\\Delta(\\nu)$|")
    axB.set_title("(b)  A decoy setting maximises exposure  (25 km)")
    axB.grid(True, ls=":", alpha=0.4)

    fig.suptitle("The decoy channel is where the DEGENERACY attack is exposed "
                 "(exact analytic residual)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "exp_residual_map.png")
    fig.savefig(path, dpi=150)
    print(f"residual at 25km, nu=0.1: {abs(decoy_residual(sys25, r25, 1.0, 0.1)):.2e}")
    print(f"optimal decoy intensity nu* = {nu_star:.3f}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
