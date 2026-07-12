"""Validation: sequential detection delay obeys Stein's law, delay ~ log(ARL0)/D.

For several attacks (different KL divergences D) we sweep the CUSUM threshold h,
measuring the false-alarm average-run-length ARL0 and the detection delay ARL1.
Stein/Lorden theory predicts ARL1 ~ log(ARL0)/D, i.e. a line of slope 1/D on
(log ARL0, ARL1) axes. Confirming this certifies that D is the right currency
for the detectability-leakage frontier.

Run:  PYTHONPATH=. python experiments/exp_stein_validation.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import InterceptResend, PNS, TimeShift, calibrate_pns
from qkd.dataset import build_system, const, generate
from qkd.infometrics import fit_gaussian, kl_gaussian
from qkd.sequential import arl0, detection_delay, llr_stream
from qkd.telemetry import FULL_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
FEAT = FULL_FEATURES


def build_laws(sys, attack_factory, rng, n_train=600):
    P0 = fit_gaussian(generate(sys, const(None), n_train, N_PULSES, rng, FEAT).X)
    P1 = fit_gaussian(generate(sys, attack_factory, n_train, N_PULSES, rng, FEAT).X)
    return P0, P1


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(25.0)
    rng = np.random.default_rng(21)
    r = calibrate_pns(sys)

    attacks = [
        ("Intercept f=0.10", lambda: InterceptResend(0.10), "#c0392b"),
        ("Time-shift s=0.5",  lambda: TimeShift(0.5),        "#e67e22"),
        ("PNS duty=0.6",      lambda: PNS(0.6, restore=r),   "#2980b9"),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    hs = np.linspace(2, 22, 12)
    for name, fac, color in attacks:
        P0, P1 = build_laws(sys, fac, rng)
        D = kl_gaussian(P1, P0)
        # honest stream for ARL0, attacked runs for delay
        Xh = generate(sys, const(None), 4000, N_PULSES, rng, FEAT).X
        llr_h = llr_stream(Xh, P1, P0)
        runs = [llr_stream(generate(sys, fac, 200, N_PULSES, rng, FEAT).X, P1, P0)
                for _ in range(8)]
        a0, a1 = [], []
        for h in hs:
            a0.append(arl0(llr_h, h, block_len=1000))
            a1.append(detection_delay(runs, h))
        a0, a1 = np.array(a0), np.array(a1)
        ok = (a0 > 1) & np.isfinite(a1)
        ax.plot(np.log(a0[ok]), a1[ok], "o-", color=color, ms=4,
                label=f"{name}  (D={D:.3f})")
        # Stein/Lorden information floor: achievable delay >= log(ARL0)/D
        xline = np.log(a0[ok])
        ax.plot(xline, xline / D, "--", color=color, alpha=0.5, lw=1)

    ax.set_xlabel("log ARL$_0$  (false-alarm run length)")
    ax.set_ylabel("Detection delay ARL$_1$  (blocks)")
    ax.set_title("Detection delay respects the Stein floor  log(ARL$_0$)/D\n"
                 "(solid = CUSUM Monte-Carlo, dashed = information floor)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(OUT, "exp_stein_validation.png")
    fig.savefig(path, dpi=150)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
