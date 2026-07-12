"""Fig. 3: ML anomaly detection, LIMITED vs FULL telemetry ablation.

A one-class detector is trained on honest blocks only, once with cheap LIMITED
telemetry and once with FULL (decoy-augmented) telemetry. We evaluate on each
attack:

  Left  : ROC curves for the PNS attack -- FULL separates it perfectly, LIMITED
          is near chance (the decoy residual is the only trace, and LIMITED
          discards it).
  Right : detection rate at a calibrated 1% false-alarm rate, per attack, for
          both telemetry sets. LIMITED catches intercept/time-shift/naive
          blinding but is blind to gain-matched PNS and stealthy blinding.

Run:  PYTHONPATH=. python experiments/fig3_ml_ablation.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from qkd.attacks import (PNS, Blinding, InterceptResend, TimeShift,
                        calibrate_blinding, calibrate_pns)
from qkd.dataset import build_system, const, generate
from qkd.mldetect import AnomalyDetector
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "figures")
N_PULSES = 20000


def make_attacks(sys):
    r = calibrate_pns(sys)
    cp = calibrate_blinding(sys, dc_match=0.002, timing_jitter=0.043)
    stealth = dict(click_prob=cp, dc_match=0.002, timing_jitter=0.043)
    return [
        ("Intercept\n(f=0.5)",  lambda: InterceptResend(0.5)),
        ("Time-shift",          lambda: TimeShift(1.0)),
        ("Blinding\n(naive)",   lambda: Blinding(1.0)),
        ("Blinding\n(stealth)", lambda: Blinding(1.0, **stealth)),
        ("PNS\n(gain-matched)", lambda: PNS(1.0, restore=r)),
    ]


def build_detectors(sys, rng, feature_set):
    train = generate(sys, const(None), 400, N_PULSES, rng, feature_set)
    cal = generate(sys, const(None), 400, N_PULSES, rng, feature_set)
    det = AnomalyDetector("ocsvm", nu=0.05).fit(train.X, far=0.01, X_cal=cal.X)
    honest_test = generate(sys, const(None), 300, N_PULSES, rng, feature_set)
    return det, honest_test


def main():
    os.makedirs(OUT, exist_ok=True)
    sys = build_system(length_km=25.0)
    rng = np.random.default_rng(11)
    attacks = make_attacks(sys)

    results = {}          # (feat, attack_name) -> (auc, detection_rate)
    roc_pns = {}
    for feat_name, feat in [("LIMITED", LIMITED_FEATURES), ("FULL", FULL_FEATURES)]:
        det, honest_test = build_detectors(sys, rng, feat)
        s_honest = det.anomaly_score(honest_test.X)
        for aname, afac in attacks:
            atk = generate(sys, afac, 300, N_PULSES, rng, feat)
            s_atk = det.anomaly_score(atk.X)
            y = np.r_[np.zeros(len(s_honest)), np.ones(len(s_atk))]
            s = np.r_[s_honest, s_atk]
            auc = roc_auc_score(y, s)
            det_rate = float((s_atk > det.threshold_).mean())     # at 1% FAR
            results[(feat_name, aname)] = (auc, det_rate)
            if "PNS" in aname:
                roc_pns[feat_name] = roc_curve(y, s)

    # ---------------- plotting ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for feat_name, color in [("FULL", "#2c6fbb"), ("LIMITED", "#c0392b")]:
        fpr, tpr, _ = roc_pns[feat_name]
        auc = results[(feat_name, "PNS\n(gain-matched)")][0]
        axL.plot(fpr, tpr, lw=2.2, color=color,
                 label=f"{feat_name} telemetry (AUC={auc:.3f})")
    axL.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
    axL.set_xlabel("False-alarm rate")
    axL.set_ylabel("Detection rate")
    axL.set_title("(a)  Detecting gain-matched PNS")
    axL.legend(frameon=False, fontsize=9, loc="lower right")
    axL.grid(True, ls=":", alpha=0.4)

    names = [a[0] for a in attacks]
    x = np.arange(len(names))
    w = 0.38
    dr_full = [results[("FULL", n)][1] for n in names]
    dr_lim = [results[("LIMITED", n)][1] for n in names]
    axR.bar(x - w / 2, dr_full, w, color="#2c6fbb", label="FULL")
    axR.bar(x + w / 2, dr_lim, w, color="#c0392b", label="LIMITED")
    axR.axhline(0.01, color="k", ls="--", lw=1, alpha=0.5)
    axR.text(len(names) - 0.5, 0.03, "1% FAR", fontsize=8, ha="right")
    axR.set_xticks(x)
    axR.set_xticklabels(names, fontsize=8)
    axR.set_ylabel("Detection rate @ 1% FAR")
    axR.set_ylim(0, 1.05)
    axR.set_title("(b)  Detection by attack and telemetry set")
    axR.legend(frameon=False, fontsize=9, loc="center right")
    axR.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle("Fig. 3  Cheap telemetry is blind to gain-matched attacks",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "fig3_ml_ablation.png")
    fig.savefig(path, dpi=150)

    print(f"{'attack':22s} {'FULL auc':>9s} {'FULL det':>9s} "
          f"{'LIM auc':>9s} {'LIM det':>9s}")
    for n in names:
        fa, fd = results[("FULL", n)]
        la, ld = results[("LIMITED", n)]
        print(f"{n.replace(chr(10),' '):22s} {fa:9.3f} {fd:9.3f} {la:9.3f} {ld:9.3f}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
