"""Block-level telemetry feature extraction.

Two feature sets, central to the impossibility-frontier study:

LIMITED  -- obtainable from raw detector clicks with NO extra hardware and NO
            decoy-state analysis: overall gain, QBER, double-click rate,
            detector click imbalance, Z/X basis detection asymmetry, and the
            first two moments of the detection-time histogram.

FULL     -- LIMITED plus per-intensity decoy residuals (signal/decoy/vacuum gain
            and QBER deviations from the honest calibration) and higher timing
            moments (skew, kurtosis). This is the "expensive" telemetry the
            2603.03502 defender implicitly assumes.

A detector trained on LIMITED is structurally blind to attacks whose only
signature lives in the decoy residuals (e.g. PNS) -- that blindness is the
frontier we map.
"""
from __future__ import annotations

import numpy as np

from .session import BlockRecord

LIMITED_FEATURES = [
    "gain", "qber", "double_rate", "det_imbalance",
    "basis_asym", "t_mean", "t_std",
]
FULL_EXTRA_FEATURES = [
    "res_gain_sig", "res_gain_dec", "res_gain_vac",
    "res_qber_sig", "res_qber_dec", "t_skew", "t_kurt",
]
FULL_FEATURES = LIMITED_FEATURES + FULL_EXTRA_FEATURES


def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    if x.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return float(m), 0.0, 0.0, 0.0
    z = (x - m) / s
    return float(m), float(s), float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


class Calibration:
    """Honest-operation reference values used to form decoy residuals.

    Built from the analytic model so residuals are ~0 under honest operation
    regardless of Monte-Carlo noise scale.
    """

    def __init__(self, sys):
        from .keyrate import gain_qber
        self.gain = {}
        self.qber = {}
        for i, mu in enumerate(sys.source.intensities):
            Q, E = gain_qber(sys, mu)
            self.gain[i] = Q
            self.qber[i] = E


def extract(block: BlockRecord, calib: Calibration) -> dict:
    """Compute the full telemetry dict for one block."""
    n = block.n_pulses
    det = block.intensity_idx.size            # number of detected events
    feats: dict[str, float] = {}

    # --- overall gain & QBER (sifted) ---
    feats["gain"] = det / n
    sift = block.basis_match
    n_sift = int(sift.sum())
    feats["qber"] = float(block.error[sift].mean()) if n_sift else 0.0

    # --- double-click rate (per detected event) ---
    feats["double_rate"] = float(block.double.mean()) if det else 0.0

    # --- detector click imbalance: (N0 - N1)/(N0 + N1) over sifted key ---
    if n_sift:
        b = block.bit[sift]
        n0, n1 = int((b == 0).sum()), int((b == 1).sum())
        feats["det_imbalance"] = (n0 - n1) / max(n0 + n1, 1)
    else:
        feats["det_imbalance"] = 0.0

    # --- Z vs X basis detection asymmetry ---
    nz = int((block.basis == 0).sum())
    nx = int((block.basis == 1).sum())
    feats["basis_asym"] = (nz - nx) / max(nz + nx, 1)

    # --- timing moments ---
    tm, ts, tsk, tk = _moments(block.arrival_t)
    feats["t_mean"], feats["t_std"], feats["t_skew"], feats["t_kurt"] = tm, ts, tsk, tk

    # --- per-intensity decoy residuals (FULL only) ---
    names = ["sig", "dec", "vac"]
    for i, nm in enumerate(names):
        sel = block.intensity_idx == i
        n_sent_i = n * _prob_of_intensity(block, i)  # expected sends
        g_obs = int(sel.sum()) / max(n_sent_i, 1e-9)
        feats[f"res_gain_{nm}"] = g_obs - calib.gain.get(i, 0.0)
        if i < 2:
            sel_sift = sel & block.basis_match
            e_obs = float(block.error[sel_sift].mean()) if int(sel_sift.sum()) else 0.0
            feats[f"res_qber_{nm}"] = e_obs - calib.qber.get(i, 0.0)
    return feats


# Intensity send-probabilities are needed to normalise per-intensity gains.
# They are a fixed protocol constant; store on the block via a module cache.
_PROBS: tuple[float, ...] = (0.7, 0.15, 0.15)


def set_intensity_probs(probs) -> None:
    global _PROBS
    _PROBS = tuple(probs)


def _prob_of_intensity(block: BlockRecord, i: int) -> float:
    return _PROBS[i] if i < len(_PROBS) else 0.0


def vectorise(feats: dict, feature_set: list[str]) -> np.ndarray:
    return np.array([feats[k] for k in feature_set], dtype=float)
