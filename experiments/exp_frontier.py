"""MAIN RESULT: the detectability-leakage frontier and the impossibility region.

For a grid of attacks x duty-cycles rho, each configuration yields
    I   = Eve's per-block information leakage (fraction of sifted key),
    D_T = KL detectability of the induced telemetry law, for T in {LIMITED, FULL}.
Because LIMITED features are a deterministic sub-vector of FULL, the
data-processing inequality forces D_LIMITED <= D_FULL for every configuration.

Panel (a): the (I, D) cloud with the Pareto lower envelope D*_T(I) -- the least
detectable way to leak I bits. The LIMITED envelope lies far below FULL: cheap
telemetry lets Eve leak the same information at a fraction of the detectability.

Panel (b): translating via the Stein floor N* = log(1/alpha)/D*_T(I) gives the
minimum blocks any detector needs. The region under each curve is the
IMPOSSIBILITY REGION: attacks leaking I that no detector can catch within N
blocks. Under LIMITED telemetry the region extends to arbitrarily large N at
finite I (gain-matched PNS drives D*->0) -- an unconditional blind spot.

Run:  PYTHONPATH=. python experiments/exp_frontier.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qkd.attacks import (Blinding, InterceptResend, PNS, TimeShift,
                        calibrate_blinding, calibrate_pns)
from qkd.dataset import build_system, const, generate
from qkd.infometrics import fit_gaussian, kl_gaussian, stein_detection_blocks
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
NLIM = len(LIMITED_FEATURES)          # LIMITED = FULL[:NLIM]
ALPHA = 0.01                          # target false-alarm level


def laws_from(X):
    return fit_gaussian(X), fit_gaussian(X[:, :NLIM])


def lower_envelope(I, D, n_bins=14):
    """Pareto lower envelope: min D in each I-bin (monotone-ised)."""
    I, D = np.asarray(I), np.asarray(D)
    edges = np.linspace(0, I.max() + 1e-9, n_bins + 1)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (I >= a) & (I < b)
        if m.any():
            xs.append(I[m].mean())
            ys.append(D[m].min())
    xs, ys = np.array(xs), np.array(ys)
    # enforce non-increasing D as I grows is NOT physical; keep as-is but sort.
    order = np.argsort(xs)
    return xs[order], ys[order]


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(25.0)
    rng = np.random.default_rng(31)
    r = calibrate_pns(sys)
    cp = calibrate_blinding(sys, dc_match=0.002, timing_jitter=0.043)
    stealth = dict(click_prob=cp, dc_match=0.002, timing_jitter=0.043)

    # Honest reference laws.
    Xh = generate(sys, const(None), 800, N_PULSES, rng, FULL_FEATURES).X
    P0_full, P0_lim = laws_from(Xh)

    duties = np.linspace(0.1, 1.0, 8)
    families = [
        ("PNS",        lambda d: PNS(d, restore=r),        "#2980b9", "D"),
        ("Blinding",   lambda d: Blinding(d, **stealth),   "#8e44ad", "P"),
        ("Time-shift", lambda d: TimeShift(d),             "#e67e22", "^"),
        ("Intercept",  lambda d: InterceptResend(d),       "#c0392b", "s"),
    ]

    rows = []  # (family, rho, I, D_lim, D_full)
    for fname, fac, color, mk in families:
        for d in duties:
            bs = generate(sys, const(fac(d)), 300, N_PULSES, rng, FULL_FEATURES)
            P1_full, P1_lim = laws_from(bs.X)
            I = float(bs.eve_info.mean())
            D_full = kl_gaussian(P1_full, P0_full)
            D_lim = kl_gaussian(P1_lim, P0_lim)
            rows.append((fname, d, I, D_lim, D_full, color, mk))

    # ---------------- plotting ----------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    seen = set()
    for fname, d, I, Dl, Df, color, mk in rows:
        lbl_f = f"{fname}" if (fname, "F") not in seen else None
        seen.add((fname, "F"))
        axA.scatter(I, Df, c=color, marker=mk, s=45, edgecolor="k",
                    linewidth=0.4, zorder=3, label=lbl_f)
        axA.scatter(I, Dl, c=color, marker=mk, s=45, alpha=0.35, zorder=2)

    allI = [x[2] for x in rows]
    Dl_all = [x[3] for x in rows]
    Df_all = [x[4] for x in rows]
    xf, yf = lower_envelope(allI, Df_all)
    xl, yl = lower_envelope(allI, Dl_all)
    axA.plot(xf, yf, "-", color="#2c6fbb", lw=2.4, label="FULL envelope $D^*$")
    axA.plot(xl, yl, "-", color="#c0392b", lw=2.4, label="LIMITED envelope $D^*$")
    axA.set_yscale("log")
    axA.set_xlabel("Information leaked per block  $I$  (fraction of sifted key)")
    axA.set_ylabel("Detectability  $D$  (KL, nats)  —  filled=FULL, faded=LIMITED")
    axA.set_title("(a)  Detectability–leakage cloud & Pareto envelopes")
    axA.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    axA.grid(True, which="both", ls=":", alpha=0.35)

    # Panel B: impossibility region via Stein floor.
    def floor(xs, ys):
        return xs, np.array([stein_detection_blocks(max(D, 1e-6), ALPHA) for D in ys])
    xfN, yfN = floor(xf, yf)
    xlN, ylN = floor(xl, yl)
    axB.fill_between(xlN, ylN, 1e6, color="#c0392b", alpha=0.10)
    axB.plot(xfN, yfN, "-o", color="#2c6fbb", lw=2.2, ms=4,
             label="FULL telemetry  $N^*(I)$")
    axB.plot(xlN, ylN, "-o", color="#c0392b", lw=2.2, ms=4,
             label="LIMITED telemetry  $N^*(I)$")
    axB.set_yscale("log")
    axB.set_xlabel("Information leaked per block  $I$")
    axB.set_ylabel(f"Min blocks to detect  $N^*=\\log(1/\\alpha)/D^*$  ($\\alpha$={ALPHA})")
    axB.set_title("(b)  Impossibility region (shaded: LIMITED undetectable)")
    axB.legend(frameon=False, fontsize=9, loc="upper right")
    axB.grid(True, which="both", ls=":", alpha=0.35)

    fig.suptitle("Detectability–Leakage Frontier:  cheap telemetry enlarges the "
                 "undetectable region (data-processing gap)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "exp_frontier.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")  # vector version for the paper

    print(f"{'family':11s} {'rho':>5s} {'I':>7s} {'D_lim':>9s} {'D_full':>9s} "
          f"{'N*_lim':>8s} {'N*_full':>8s}")
    for fname, d, I, Dl, Df, *_ in rows:
        nl = stein_detection_blocks(max(Dl, 1e-6), ALPHA)
        nf = stein_detection_blocks(max(Df, 1e-6), ALPHA)
        print(f"{fname:11s} {d:5.2f} {I:7.3f} {Dl:9.4f} {Df:9.4f} {nl:8.1f} {nf:8.1f}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
