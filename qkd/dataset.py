"""Dataset generation and security metrics shared across experiments.

Produces feature matrices of telemetry blocks under honest operation and under
attacks, plus the per-block Eve-information used for the secret-fraction metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ChannelParams, DetectorParams, SourceParams, SystemParams
from .session import Session
from .telemetry import (FULL_FEATURES, LIMITED_FEATURES, Calibration, extract,
                        set_intensity_probs, vectorise)


def build_system(length_km: float = 25.0, efficiency: float = 0.15,
                 dark_count: float = 2e-5, misalignment: float = 0.015,
                 eta_mismatch: float = 0.0) -> SystemParams:
    return SystemParams(
        source=SourceParams(intensities=(0.5, 0.1, 0.0), probs=(0.7, 0.15, 0.15)),
        channel=ChannelParams(length_km=length_km, misalignment=misalignment),
        detector=DetectorParams(efficiency=efficiency, dark_count=dark_count,
                               eta_mismatch=eta_mismatch),
    )


@dataclass
class BlockSet:
    X: np.ndarray            # (n_blocks, n_features)
    eve_info: np.ndarray     # (n_blocks,) fraction of sifted key Eve knows
    qber: np.ndarray         # (n_blocks,) observed QBER
    gain: np.ndarray         # (n_blocks,) observed gain
    feature_names: list


def generate(sys: SystemParams, attack_factory, n_blocks: int, n_pulses: int,
             rng: np.random.Generator, feature_set=FULL_FEATURES) -> BlockSet:
    """Generate a set of telemetry blocks.

    ``attack_factory`` is a callable returning a fresh attack object (or None)
    per block, so stochastic/adaptive attacks can vary block to block.
    """
    set_intensity_probs(sys.source.probs)
    calib = Calibration(sys)
    sess = Session(sys, rng)
    X, eve, qber, gain = [], [], [], []
    for _ in range(n_blocks):
        atk = attack_factory()
        blk = sess.run(n_pulses, attack=atk)
        feats = extract(blk, calib)
        X.append(vectorise(feats, feature_set))
        eve.append(blk.eve_info)
        qber.append(feats["qber"])
        gain.append(feats["gain"])
    return BlockSet(np.array(X), np.array(eve), np.array(qber), np.array(gain),
                   list(feature_set))


def const(attack):
    """Factory that returns the same attack object every block."""
    return lambda: attack
