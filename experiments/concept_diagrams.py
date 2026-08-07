"""Hand-drawn conceptual figures (vector) matching the data-figure style.

Produces two schematics as PNG + PDF:
  concept_pipeline   : threat model + the SCATTER pipeline (physics -> telemetry
                       -> Gaussian laws -> KL detectability -> Stein delay), with
                       the LIMITED vs FULL data-processing gap as the payoff.
  concept_degeneracy : the observational-degeneracy metaphor -- honest and
                       attacked states collapse onto one point under LIMITED
                       telemetry, stay separate under FULL.

Run:  PYTHONPATH=. python experiments/concept_diagrams.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "figures")

BLUE = "#2c6fbb"
RED = "#c0392b"
PURPLE = "#8e44ad"
GRAY = "#5b6670"
INK = "#222831"


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=GRAY, fs=10, tc=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    print(f"saved -> {os.path.join(OUT, name)}.{{png,pdf}}")


# --------------------------------------------------------------------------- #
def pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 27); ax.set_ylim(0, 10); ax.axis("off")

    # ---- Stage 1: physical layer / threat model ----
    ax.text(4.2, 9.4, "physical layer", ha="center", fontsize=9,
            color=GRAY, style="italic")
    rbox(ax, 0.4, 5.2, 2.4, 1.6, "Alice\nWCP source", fc="#eaf1fb", ec=BLUE)
    rbox(ax, 5.9, 5.2, 2.4, 1.6, "Bob\ndetectors", fc="#eaf1fb", ec=BLUE)
    # fiber with pulses
    ax.plot([2.8, 5.9], [6.0, 6.0], color=INK, lw=1.4, zorder=1)
    for xp in np.linspace(3.2, 5.5, 5):
        ax.add_patch(plt.Circle((xp, 6.0), 0.09, color="#f1c40f", ec=INK,
                                lw=0.5, zorder=3))
    # Eve on the fiber
    rbox(ax, 3.55, 7.2, 1.9, 1.0, "Eve\nDEGENERACY", fc="#fdeaea", ec=RED,
         fs=8.5, tc=RED)
    arrow(ax, 4.5, 7.2, 4.5, 6.15, color=RED, lw=1.4, ls=(0, (3, 2)))
    # detector clicks -> telemetry
    arrow(ax, 8.3, 6.0, 9.4, 6.0)

    # ---- Stage 2: telemetry block ----
    ax.text(10.9, 9.4, "telemetry block", ha="center", fontsize=9,
            color=GRAY, style="italic")
    for i in range(7):
        ax.add_patch(FancyBboxPatch((9.5 + i * 0.36, 5.55), 0.32, 0.9,
                                    boxstyle="round,pad=0.005,rounding_size=0.03",
                                    fc="#f4f6f8", ec=GRAY, lw=1.0, zorder=2))
    ax.text(10.75, 5.15, r"$x\in\mathbb{R}^d$  (block of $\sim\!10^5$ pulses)",
            ha="center", fontsize=8.5, color=INK)

    # split into two telemetry budgets
    arrow(ax, 12.3, 6.4, 13.4, 8.1)      # to FULL (top)
    arrow(ax, 12.3, 5.6, 13.4, 3.9)      # to LIMITED (bottom)

    # ---- Stage 3: two branches with Gaussian pairs ----
    def branch(y0, label, sep, color, tag):
        rbox(ax, 13.4, y0 - 0.05, 2.0, 1.0, label, fc="#ffffff", ec=color,
             fs=8.5, tc=color)
        # gaussian pair inset
        gx = np.linspace(-3.2, 3.2, 200)
        base = y0 + 0.02
        x0 = 15.9
        scale = 0.9
        g0 = gauss(gx, -sep, 1.0); g1 = gauss(gx, sep, 1.0)
        ax.plot(x0 + (gx + 3.2) * 0.28, base + g0 * scale, color=GRAY, lw=1.4)
        ax.plot(x0 + (gx + 3.2) * 0.28, base + g1 * scale, color=color, lw=1.6)
        ax.text(x0 + 0.9, base + 1.15, tag, fontsize=8, color=color, ha="center")
        return x0 + 1.8

    xend_f = branch(7.7, "FULL\n(14 features)", 1.7, BLUE, r"$P_0,\,P_1$ separated")
    xend_l = branch(3.3, "LIMITED\n(7 features)", 0.35, RED, r"$P_0,\,P_1$ overlap")

    # ---- Stage 4: detectability + Stein delay ----
    arrow(ax, xend_f, 8.2, 20.6, 8.2, color=BLUE)
    arrow(ax, xend_l, 3.8, 20.6, 3.8, color=RED)
    rbox(ax, 20.6, 7.6, 2.7, 1.2, r"$D_{\rm FULL}$" + "\nlarge", fc="#eaf1fb",
         ec=BLUE, fs=9, tc=BLUE)
    rbox(ax, 20.6, 3.2, 2.7, 1.2, r"$D_{\rm LIMITED}$" + "\nsmall", fc="#fdeaea",
         ec=RED, fs=9, tc=RED)
    arrow(ax, 23.3, 8.2, 24.2, 8.2, color=BLUE)
    arrow(ax, 23.3, 3.8, 24.2, 3.8, color=RED)
    rbox(ax, 24.2, 7.6, 2.5, 1.2, r"$N^*$ small" + "\nfast detect", fc="#ffffff",
         ec=BLUE, fs=8.5, tc=BLUE)
    rbox(ax, 24.2, 3.2, 2.5, 1.2, r"$N^*$ large" + "\nblind spot", fc="#ffffff",
         ec=RED, fs=8.5, tc=RED)

    # DPI note between branches
    ax.annotate("", xy=(21.9, 7.4), xytext=(21.9, 4.5),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.2))
    ax.text(22.15, 6.0, "data-processing\ngap:  " + r"$D_{\rm LIM}\leq D_{\rm FULL}$",
            fontsize=8, color=GRAY, va="center")

    # key relation
    ax.text(13.5, 1.2, r"detectability $D=\mathrm{KL}(P_1\Vert P_0)$   "
            r"$\Rightarrow$   detection delay $N^*=\log(1/\alpha)/D$   (Stein floor)",
            fontsize=10, color=INK)

    ax.set_title("SCATTER: from device telemetry to a detection-delay bound",
                 fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "concept_pipeline")


# --------------------------------------------------------------------------- #
def degeneracy():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 22); ax.set_ylim(0, 11); ax.axis("off")

    # ---- left: microscopic reality ----
    rbox(ax, 0.6, 2.6, 6.2, 6.0, "", fc="#fbfbfc", ec=GRAY)
    ax.text(3.7, 8.15, "microscopic reality", ha="center", fontsize=10, color=GRAY,
            style="italic")
    ax.add_patch(plt.Circle((2.6, 6.2), 0.30, color=BLUE, ec=INK, lw=0.6, zorder=3))
    ax.text(2.6, 6.9, "honest  $P_0$", ha="center", fontsize=9, color=BLUE)
    ax.add_patch(plt.Circle((5.0, 4.3), 0.30, color=RED, ec=INK, lw=0.6, zorder=3))
    ax.text(5.0, 3.6, "eavesdropped  $P_1$", ha="center", fontsize=9, color=RED)

    # ---- projection cones ----
    arrow(ax, 7.0, 6.4, 11.2, 8.4, color=BLUE, lw=1.6)
    ax.text(8.7, 8.05, "FULL telemetry", fontsize=9, color=BLUE, rotation=22)
    arrow(ax, 7.0, 4.6, 11.2, 2.9, color=RED, lw=1.6)
    ax.text(8.5, 3.1, "LIMITED telemetry", fontsize=9, color=RED, rotation=-19)

    # ---- top-right: FULL -> separable ----
    rbox(ax, 11.4, 6.7, 9.8, 3.6, "", fc="#f4f8fd", ec=BLUE)
    ax.text(16.3, 9.9, "FULL observation:  distinguishable", ha="center",
            fontsize=10, color=BLUE)
    ax.add_patch(plt.Circle((14.2, 8.3), 0.30, color=BLUE, ec=INK, lw=0.6, zorder=3))
    ax.add_patch(plt.Circle((18.2, 8.3), 0.30, color=RED, ec=INK, lw=0.6, zorder=3))
    ax.annotate("", xy=(17.9, 8.3), xytext=(14.5, 8.3),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.1))
    ax.text(16.2, 7.55, r"$D_{\rm FULL}>0$  $\to$  detectable", ha="center",
            fontsize=9, color=GRAY)

    # ---- bottom-right: LIMITED -> degenerate ----
    rbox(ax, 11.4, 1.0, 9.8, 3.6, "", fc="#fdf3f3", ec=RED)
    ax.text(16.3, 4.2, "LIMITED observation:  degenerate", ha="center",
            fontsize=10, color=RED)
    # two dots collapsed onto one point (draw overlapping)
    ax.add_patch(plt.Circle((16.25, 2.7), 0.34, color=BLUE, ec=INK, lw=0.6, zorder=3))
    ax.add_patch(plt.Circle((16.35, 2.7), 0.30, color=RED, ec=INK, lw=0.6,
                            alpha=0.75, zorder=4))
    ax.text(16.3, 1.85, r"$P_0\approx P_1$:  $D_{\rm LIMITED}\to 0$  "
            r"$\to$  undetectable", ha="center", fontsize=9, color=GRAY)

    ax.set_title("The DEGENERACY attack: distinct realities collapse to one "
                 "observation under cheap telemetry", fontsize=12.5, color=INK)
    fig.tight_layout()
    save(fig, "concept_degeneracy")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    degeneracy()
