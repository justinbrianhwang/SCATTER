"""Regression checks locking in each attack's qualitative telemetry signature."""
from __future__ import annotations

import numpy as np

from qkd.attacks import (PNS, Blinding, InterceptResend, TimeShift,
                        calibrate_blinding, calibrate_pns)
from qkd.params import ChannelParams, DetectorParams, SourceParams, SystemParams
from qkd.session import Session
from qkd.telemetry import Calibration, extract, set_intensity_probs


def _sys():
    return SystemParams(
        source=SourceParams(intensities=(0.5, 0.1, 0.0), probs=(0.7, 0.15, 0.15)),
        channel=ChannelParams(length_km=25.0, misalignment=0.015),
        detector=DetectorParams(efficiency=0.15, dark_count=2e-5),
    )


def _mean_feats(attack, n_blocks=120, n_pulses=20000, seed=0):
    s = _sys()
    set_intensity_probs(s.source.probs)
    rng = np.random.default_rng(seed)
    calib = Calibration(s)
    sess = Session(s, rng)
    rows, eve = [], []
    for _ in range(n_blocks):
        blk = sess.run(n_pulses, attack=attack)
        rows.append(extract(blk, calib))
        eve.append(blk.eve_info)
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, float(np.mean(eve))


def test_honest_baseline():
    f, eve = _mean_feats(None)
    assert eve == 0.0
    assert 0.008 < f["qber"] < 0.03          # ~ misalignment
    assert f["double_rate"] > 0              # crosstalk produces real doubles


def test_intercept_raises_qber_and_leaks():
    f, eve = _mean_feats(InterceptResend(0.5))
    assert f["qber"] > 0.10                   # LIMITED-visible
    assert eve > 0.15


def test_pns_matched_is_limited_invisible_but_shifts_decoy():
    s = _sys()
    r = calibrate_pns(s)
    fh, _ = _mean_feats(None)
    fp, eve = _mean_feats(PNS(1.0, restore=r))
    assert abs(fp["gain"] - fh["gain"]) < 0.1 * fh["gain"]     # gain matched
    assert abs(fp["qber"] - fh["qber"]) < 0.005               # QBER matched
    assert fp["res_gain_dec"] < fh["res_gain_dec"] - 5e-4      # decoy residual shifts
    assert eve > 0.3


def test_blinding_naive_caught_by_limited():
    f, eve = _mean_feats(Blinding(1.0))
    assert f["gain"] > 0.4 or f["double_rate"] < 1e-4          # gross anomaly
    assert eve > 0.9


def test_time_shift_broadens_timing():
    fh, _ = _mean_feats(None)
    ft, eve = _mean_feats(TimeShift(1.0))
    assert ft["t_std"] > 1.4 * fh["t_std"]                     # LIMITED-visible
    assert eve > 0.1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
