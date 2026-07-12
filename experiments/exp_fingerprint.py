"""Degeneracy fingerprint: at the DEGENERACY-attack optimum, every LIMITED
telemetry feature matches honest to within statistical noise, while the decoy
residual is the lone surviving trace -- the analytic proposition, visualised.

Run:  PYTHONPATH=. python experiments/exp_fingerprint.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import PNS
from qkd.dataset import build_system, const, generate
from qkd.degeneracy import decoy_residual, gain_match_restore
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
NLIM = len(LIMITED_FEATURES)


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(25.0)
    rng = np.random.default_rng(202)
    r = gain_match_restore(sys)                       # analytic gain-match

    h = generate(sys, const(None), 1500, 20000, rng, FULL_FEATURES)
    p = generate(sys, const(PNS(1.0, restore=r)), 1500, 20000, rng, FULL_FEATURES)

    devs = []
    for i, f in enumerate(FULL_FEATURES):
        hs = h.X[:, i].std() + 1e-15
        devs.append(abs(p.X[:, i].mean() - h.X[:, i].mean()) / hs)
    devs = np.array(devs)
    colors = ["#c0392b" if i >= NLIM else "#7f8c8d" for i in range(len(FULL_FEATURES))]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(FULL_FEATURES))
    ax.bar(x, devs, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(len(x) - 0.5, 1.05, "1$\\sigma$ (per-block)", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(FULL_FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|mean deviation| / $\\sigma$")
    ax.set_title(f"Degeneracy fingerprint (analytic $r^*$={r:.3f}, "
                 f"$I$={p.eve_info.mean():.2f}):  "
                 "LIMITED matched, decoy residual survives")

    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#7f8c8d", label="LIMITED features"),
                       Patch(color="#c0392b", label="FULL-only (decoy) features")],
              frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # annotate analytic prediction on res_gain_dec
    idc = FULL_FEATURES.index("res_gain_dec")
    Delta = decoy_residual(sys, r)
    ax.annotate(f"analytic $\\Delta(\\nu)$={Delta:+.2e}",
                xy=(idc, devs[idc]), xytext=(idc - 3.2, devs[idc] + 0.25),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))

    fig.tight_layout()
    path = os.path.join(OUT, "exp_fingerprint.png")
    fig.savefig(path, dpi=150)
    print(f"analytic r*={r:.4f}  analytic decoy residual={Delta:+.5f}")
    print(f"max LIMITED deviation = {devs[:NLIM].max():.2f} sigma")
    print(f"decoy   deviation     = {devs[idc]:.2f} sigma")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
