"""Smoke test: verify each attack's telemetry signature and Eve information.

Runs many blocks under honest operation and each attack, printing the mean of
key telemetry features + Eve's info fraction. Confirms the qualitative story:
  honest       : moderate QBER (~e_d), some double clicks, ~0 Eve info
  intercept    : QBER ~ 0.25, high Eve info      -> caught by QBER (LIMITED)
  pns          : QBER ~ e_d, decoy residuals move -> LIMITED-blind
  blinding     : QBER ~ 0, double_rate ~ 0        -> caught by doubles (LIMITED)
  time_shift   : QBER low, timing/imbalance move  -> caught by timing (LIMITED)
"""
from __future__ import annotations

import numpy as np

from qkd.attacks import (PNS, Blinding, InterceptResend, TimeShift,
                        calibrate_blinding, calibrate_pns)
from qkd.params import ChannelParams, DetectorParams, SourceParams, SystemParams
from qkd.session import Session
from qkd.telemetry import Calibration, extract, set_intensity_probs


def build_sys():
    return SystemParams(
        source=SourceParams(intensities=(0.5, 0.1, 0.0), probs=(0.7, 0.15, 0.15)),
        channel=ChannelParams(length_km=25.0, misalignment=0.015),
        detector=DetectorParams(efficiency=0.15, dark_count=2e-5),
    )


def run(attack, sys, n_blocks=200, n_pulses=20000, seed=0):
    rng = np.random.default_rng(seed)
    calib = Calibration(sys)
    sess = Session(sys, rng)
    rows, eve = [], []
    for _ in range(n_blocks):
        blk = sess.run(n_pulses, attack=attack)
        rows.append(extract(blk, calib))
        eve.append(blk.eve_info)
    keys = ["gain", "qber", "double_rate", "t_std",
            "res_gain_dec", "res_gain_vac"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    mean["eve_info"] = float(np.mean(eve))
    return mean


def main():
    sys = build_sys()
    set_intensity_probs(sys.source.probs)
    # Stealthy Eve matches the honest double-rate and timing spread too.
    stealth_kw = dict(dc_match=0.002, timing_jitter=0.043)
    r = calibrate_pns(sys)
    cp = calibrate_blinding(sys, **stealth_kw)
    print(f"[calibrated] PNS restore={r:.3f}  blinding click_prob={cp:.4f}\n")
    attacks = {
        "honest": None,
        "intercept(0.5)": InterceptResend(0.5),
        "pns-matched": PNS(1.0, restore=r),
        "blinding-naive": Blinding(1.0),
        "blinding-stealth": Blinding(1.0, click_prob=cp, **stealth_kw),
        "time_shift(1.0)": TimeShift(1.0),
    }
    cols = ["gain", "qber", "double_rate", "t_std",
            "res_gain_dec", "res_gain_vac", "eve_info"]
    print(f"{'attack':16s} " + " ".join(f"{c:>11s}" for c in cols))
    for name, atk in attacks.items():
        m = run(atk, sys)
        print(f"{name:16s} " + " ".join(f"{m[c]:11.5f}" for c in cols))


if __name__ == "__main__":
    main()
