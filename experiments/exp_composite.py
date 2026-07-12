"""Composite imperfections (angle B'): detectability is sub-additive.

A receiver with detector-efficiency mismatch admits a time-shift channel; a
weak-coherent source admits a PNS channel. We measure detectability D and
leakage I for each attack alone and for the composite that exploits both at
once. If D(composite) < D(PNS) + D(time-shift) while I adds up, then a defender
who budgets detection per-imperfection (additive) underestimates the true blind
spot -- combined loopholes are more dangerous than the sum of their parts.

Run:  PYTHONPATH=. python experiments/exp_composite.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import Composite, PNS, TimeShift
from qkd.dataset import build_system, const, generate
from qkd.degeneracy import gain_match
from qkd.infometrics import fit_gaussian, kl_gaussian, stein_detection_blocks
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
NLIM = len(LIMITED_FEATURES)
ALPHA = 0.01


def measure(sys, factory, P0f, P0l, rng, n=700):
    bs = generate(sys, factory, n, N_PULSES, rng, FULL_FEATURES)
    Df = kl_gaussian(fit_gaussian(bs.X), P0f)
    Dl = kl_gaussian(fit_gaussian(bs.X[:, :NLIM]), P0l)
    return Df, Dl, float(bs.eve_info.mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    # A receiver with a real detector-efficiency mismatch (imperfection #2).
    sys = build_system(25.0, eta_mismatch=0.12)
    rng = np.random.default_rng(505)
    r, m = gain_match(sys)

    Xh = generate(sys, const(None), 900, N_PULSES, rng, FULL_FEATURES).X
    P0f, P0l = fit_gaussian(Xh), fit_gaussian(Xh[:, :NLIM])

    # Both sub-attacks individually stealthy, so the composite's sub-additivity
    # (variance-inflation masking) is visible rather than swamped.
    TS = 0.10
    pns = lambda: PNS(1.0, restore=r, multi_forward=m)
    tsh = lambda: TimeShift(TS)
    comp = lambda: Composite([PNS(1.0, restore=r, multi_forward=m), TimeShift(TS)])

    Df_p, Dl_p, I_p = measure(sys, const(pns()), P0f, P0l, rng)
    Df_t, Dl_t, I_t = measure(sys, const(tsh()), P0f, P0l, rng)
    Df_c, Dl_c, I_c = measure(sys, const(comp()), P0f, P0l, rng)

    add_f, add_l = Df_p + Df_t, Dl_p + Dl_t   # additive (naive) budget

    print("attack           I      D_full   D_lim    N*_full  N*_lim")
    for name, I, Df, Dl in [("PNS", I_p, Df_p, Dl_p),
                            ("time-shift", I_t, Df_t, Dl_t),
                            ("composite", I_c, Df_c, Dl_c),
                            ("naive sum", I_p + I_t, add_f, add_l)]:
        print(f"{name:14s} {I:5.2f}  {Df:7.3f}  {Dl:7.3f}  "
              f"{stein_detection_blocks(max(Df,1e-9),ALPHA):7.1f}  "
              f"{stein_detection_blocks(max(Dl,1e-9),ALPHA):7.1f}")

    # ---- plot: detectability bars, composite vs additive budget ----
    fig, (axF, axL) = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=False)
    for ax, (Dp, Dt, Dc, Dadd), title in [
        (axF, (Df_p, Df_t, Df_c, add_f), "FULL telemetry"),
        (axL, (Dl_p, Dl_t, Dl_c, add_l), "LIMITED telemetry"),
    ]:
        bars = ax.bar([0, 1, 2, 3], [Dp, Dt, Dc, Dadd],
                      color=["#2980b9", "#e67e22", "#8e44ad", "#95a5a6"])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.axhline(Dadd, color="#95a5a6", ls="--", lw=1)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["PNS", "time-shift", "composite\n(actual)",
                            "naive sum\n(budget)"], fontsize=8)
        ax.set_ylabel("Detectability $D$ (KL, nats)")
        ax.set_title(title)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    gap = (1 - Df_c / add_f) * 100
    fig.suptitle(f"Composite loophole: detectability is sub-additive  "
                 f"(FULL composite {gap:.0f}% below additive budget; "
                 f"I: {I_p:.2f}+{I_t:.2f}$\\to${I_c:.2f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, "exp_composite.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
