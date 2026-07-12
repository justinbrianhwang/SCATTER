"""Telemetry budget: how much telemetry must a defender observe to lift the
DEGENERACY attack out of degeneracy?

Greedy feature selection adds, at each step, the telemetry feature that most
increases detectability against the attack. The resulting curve is the cheapest
telemetry that reaches a given detection capability. For the 1-knob attack a
single feature -- the decoy-gain residual -- lifts it; the 2-knob attack, which
mimics the honest single-photon yield, resists even the full feature set,
staying near the noise floor.

Run:  PYTHONPATH=. python experiments/exp_telemetry_budget.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import PNS
from qkd.dataset import build_system, const, generate
from qkd.degeneracy import gain_match, gain_match_restore
from qkd.infometrics import stein_detection_blocks
from qkd.subset import greedy_budget
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
NLIM = len(LIMITED_FEATURES)
ALPHA = 0.01


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(25.0)
    rng = np.random.default_rng(404)

    Xh = generate(sys, const(None), 900, N_PULSES, rng, FULL_FEATURES).X
    r1 = gain_match_restore(sys)
    r2, m2 = gain_match(sys)
    X1 = generate(sys, const(PNS(1.0, restore=r1)), 900, N_PULSES, rng,
                  FULL_FEATURES).X
    X2 = generate(sys, const(PNS(1.0, restore=r2, multi_forward=m2)), 900,
                  N_PULSES, rng, FULL_FEATURES).X

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for X, color, label in [(X1, "#2980b9", "1-knob DEGENERACY"),
                            (X2, "#c0392b", "2-knob DEGENERACY")]:
        order, curve = greedy_budget(Xh, X, len(FULL_FEATURES))
        steps = np.arange(1, len(curve) + 1)
        ax.plot(steps, curve, "-o", color=color, lw=2.2, ms=5, label=label)
        # annotate the feature that produced the biggest single jump
        jumps = np.diff([0] + curve)
        k = int(np.argmax(jumps))
        ax.annotate(FULL_FEATURES[order[k]],
                    xy=(k + 1, curve[k]), xytext=(k + 1.4, curve[k] * 1.6 + 0.05),
                    color=color, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))

    ax.set_yscale("log")
    ax.set_xlabel("Number of telemetry features observed (greedily chosen)")
    ax.set_ylabel("Detectability  $D$  (KL, nats)")
    ax.set_title("Telemetry budget: how much observation lifts the DEGENERACY attack")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax2.set_ylim(np.log(1 / ALPHA) / hi, np.log(1 / ALPHA) / lo)
    ax2.set_ylabel("Min blocks to detect  $N^*=\\log(1/\\alpha)/D$")

    fig.tight_layout()
    path = os.path.join(OUT, "exp_telemetry_budget.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")  # vector version for the paper

    for X, name in [(X1, "1-knob"), (X2, "2-knob")]:
        order, curve = greedy_budget(Xh, X, len(FULL_FEATURES))
        print(f"\n{name}: greedy feature order and running D")
        for i, (c, D) in enumerate(zip(order, curve), 1):
            print(f"  {i:2d}. +{FULL_FEATURES[c]:16s} D={D:8.4f}  "
                  f"N*={stein_detection_blocks(max(D,1e-9),ALPHA):7.1f}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
