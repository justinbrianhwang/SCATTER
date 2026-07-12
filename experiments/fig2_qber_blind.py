"""Fig. 2 (motivation): QBER is blind to device-imperfection attacks.

For each attack we plot Eve's actual information fraction against the QBER Alice
and Bob observe. Intercept-resend lands past the ~11% BB84 abort threshold and is
caught -- but PNS, time-shift, and stealthy blinding sit at honest-level QBER
while Eve knows most of the key. QBER alone cannot see them; richer telemetry
(Fig. 3) is required.

Run:  PYTHONPATH=. python experiments/fig2_qber_blind.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import (PNS, Blinding, InterceptResend, TimeShift,
                        calibrate_blinding, calibrate_pns)
from qkd.dataset import build_system, const, generate

OUT = os.path.join(os.path.dirname(__file__), "figures")
QBER_THRESHOLD = 0.11        # BB84 one-way abort threshold (~11%)


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(length_km=25.0)
    rng = np.random.default_rng(7)

    r = calibrate_pns(sys)
    cp = calibrate_blinding(sys, dc_match=0.002, timing_jitter=0.043)
    stealth = dict(click_prob=cp, dc_match=0.002, timing_jitter=0.043)

    specs = [
        ("Honest",            None,                         "#7f8c8d", "o"),
        ("Intercept-resend",  InterceptResend(0.5),         "#c0392b", "s"),
        ("Time-shift",        TimeShift(1.0),               "#e67e22", "^"),
        ("PNS (gain-matched)",PNS(1.0, restore=r),          "#2980b9", "D"),
        ("Blinding (stealth)",Blinding(1.0, **stealth),     "#8e44ad", "P"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axvspan(QBER_THRESHOLD, 0.30, color="#c0392b", alpha=0.06)
    ax.axvline(QBER_THRESHOLD, color="#c0392b", ls="--", lw=1.3)
    ax.text(QBER_THRESHOLD + 0.004, 0.04, "QBER abort\nthreshold (11%)",
            color="#c0392b", fontsize=8, va="bottom")

    for name, atk, color, mk in specs:
        bs = generate(sys, const(atk), n_blocks=60, n_pulses=20000, rng=rng)
        q_m, q_s = bs.qber.mean(), bs.qber.std()
        e_m, e_s = bs.eve_info.mean(), bs.eve_info.std()
        ax.errorbar(q_m, e_m, xerr=q_s, yerr=e_s, fmt=mk, ms=9, color=color,
                    capsize=3, label=name, zorder=3)

    ax.annotate("invisible to QBER,\nbut Eve knows the key",
                xy=(0.02, 0.9), xytext=(0.05, 0.62), fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlabel("Observed QBER")
    ax.set_ylabel("Eve information fraction (of sifted key)")
    ax.set_xlim(-0.01, 0.28)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Fig. 2  QBER is blind to device-imperfection attacks")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_qber_blind.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")  # vector version for the paper
    print(f"saved -> {path}")
    for name, atk, *_ in specs:
        pass


if __name__ == "__main__":
    main()
