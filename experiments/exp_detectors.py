"""Why attack-blind ML detectors miss the DEGENERACY attack.

On the 1-knob DEGENERACY attack (which carries a real decoy-residual signal) we
compare three single-block detectors -- the Gaussian log-likelihood-ratio test
(Neyman-Pearson optimal, attack-aware), a one-class SVM, and an isolation forest
(both attack-blind, trained on honest data only) -- under LIMITED and FULL
telemetry. Even under FULL telemetry the attack-blind detectors are near chance:
the attack's signature is a sub-sigma shift in a single decoy feature that a
one-class boundary cannot isolate among fourteen. Only the attack-aware LLR test
extracts it (and only over many blocks does the Stein floor make detection
reliable). Under LIMITED telemetry every detector, LLR included, collapses to
chance -- the information simply is not there.

Run:  PYTHONPATH=. python experiments/exp_detectors.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from qkd.attacks import PNS
from qkd.dataset import build_system, const, generate
from qkd.degeneracy import gain_match_restore
from qkd.infometrics import fit_gaussian, kl_gaussian
from qkd.mldetect import AnomalyDetector
from qkd.sequential import llr_stream
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000
NLIM = len(LIMITED_FEATURES)


def scores(kind, cols, Xtr, Xte_h, Xte_a):
    if kind == "llr":                       # Neyman-Pearson optimal
        P0 = fit_gaussian(Xtr[:, cols])
        P1 = fit_gaussian(Xte_a[:, cols])   # attack law (best-case knowledge)
        s_h = llr_stream(Xte_h[:, cols], P1, P0)
        s_a = llr_stream(Xte_a[:, cols], P1, P0)
    else:
        det = AnomalyDetector(kind, nu=0.05).fit(Xtr[:, cols], far=0.01,
                                                 X_cal=Xtr[:, cols])
        s_h = det.anomaly_score(Xte_h[:, cols])
        s_a = det.anomaly_score(Xte_a[:, cols])
    y = np.r_[np.zeros(len(s_h)), np.ones(len(s_a))]
    s = np.r_[s_h, s_a]
    return roc_curve(y, s), roc_auc_score(y, s)


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(25.0)
    rng = np.random.default_rng(818)
    r = gain_match_restore(sys)              # 1-knob attack: real signal

    Xtr = generate(sys, const(None), 800, N_PULSES, rng, FULL_FEATURES).X
    Xte_h = generate(sys, const(None), 600, N_PULSES, rng, FULL_FEATURES).X
    Xte_a = generate(sys, const(PNS(1.0, restore=r)), 600, N_PULSES, rng,
                    FULL_FEATURES).X

    lim, full = list(range(NLIM)), list(range(len(FULL_FEATURES)))
    D_lim = kl_gaussian(fit_gaussian(Xte_a[:, lim]), fit_gaussian(Xte_h[:, lim]))
    D_full = kl_gaussian(fit_gaussian(Xte_a), fit_gaussian(Xte_h))

    fig, (axF, axL) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    panels = [(axF, full, f"FULL telemetry  ($D$={D_full:.2f})"),
              (axL, lim, f"LIMITED telemetry  ($D$={D_lim:.2f})")]
    styles = [("llr", "#111111", "-", "LLR (optimal)"),
              ("ocsvm", "#2c6fbb", "-", "one-class SVM"),
              ("iforest", "#e67e22", "--", "isolation forest")]
    for ax, cols, title in panels:
        for kind, color, ls, label in styles:
            (fpr, tpr, _), auc = scores(kind, cols, Xtr, Xte_h, Xte_a)
            ax.plot(fpr, tpr, ls, color=color, lw=2.0,
                    label=f"{label} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
        ax.set_xlabel("False-alarm rate")
        ax.set_ylabel("Detection rate")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(True, ls=":", alpha=0.4)

    fig.suptitle("Attack-blind ML detectors are near chance even under FULL "
                 "telemetry; only the attack-aware LLR test sees the signal",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "exp_detectors.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")  # vector version for the paper
    print(f"D_lim={D_lim:.3f}  D_full={D_full:.3f}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
