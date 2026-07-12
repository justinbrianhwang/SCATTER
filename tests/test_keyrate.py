"""Sanity checks for the analytical key-rate layer against known limits.

Run: python -m pytest tests/ -q   (or just python tests/test_keyrate.py)
"""
from __future__ import annotations

import math

from qkd.params import (ChannelParams, DetectorParams, SourceParams,
                        SystemParams)
from qkd.keyrate import (binary_entropy, decoy_estimate, gain_qber,
                        keyrate_decoy, keyrate_gllp_nodecoy, photon_gain_qber)


def _sys(length_km=50.0, mu=0.5, nu=0.1):
    return SystemParams(
        source=SourceParams(intensities=(mu, nu, 0.0), probs=(0.7, 0.15, 0.15)),
        channel=ChannelParams(length_km=length_km),
        detector=DetectorParams(),
    )


def test_binary_entropy_bounds():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert abs(binary_entropy(0.5) - 1.0) < 1e-12
    assert abs(binary_entropy(0.11) - 0.4999) < 1e-3


def test_gain_monotonic_in_transmittance():
    # Shorter fiber -> higher transmittance -> higher gain.
    q_near, _ = gain_qber(_sys(length_km=10), mu=0.5)
    q_far, _ = gain_qber(_sys(length_km=100), mu=0.5)
    assert q_near > q_far > 0


def test_decoy_recovers_true_single_photon_yield():
    # With the analytic channel, the decoy estimator's Y1 lower bound must not
    # exceed and should be close to the true single-photon yield Y_1.
    s = _sys(length_km=50)
    mu, nu = s.source.intensities[0], s.source.intensities[1]
    Q_mu, E_mu = gain_qber(s, mu)
    Q_nu, E_nu = gain_qber(s, nu)
    Y1_true, e1_true = photon_gain_qber(s, 1, mu)
    Y1_L, Q1_L, e1_U = decoy_estimate(mu, nu, Q_mu, E_mu, Q_nu, E_nu, s.Y0)
    assert 0.0 < Y1_L <= Y1_true * 1.001          # valid lower bound
    assert Y1_L > 0.8 * Y1_true                    # and reasonably tight
    assert e1_U >= e1_true - 1e-9                   # valid upper bound on error


def test_decoy_beats_nodecoy_at_distance():
    # The whole point of Fig 1: at long distance decoy-state yields a positive
    # rate where the PNS-pessimistic GLLP rate has collapsed to zero.
    s = _sys(length_km=100)
    r_decoy = keyrate_decoy(s)["rate"]
    r_nodecoy = keyrate_gllp_nodecoy(s)["rate"]
    assert r_decoy > 0.0
    assert r_decoy > r_nodecoy


def test_nodecoy_collapses_before_decoy():
    # Find the crossover: no-decoy hits zero at a shorter distance than decoy.
    def max_range(fn):
        last = 0.0
        for L in range(0, 260, 5):
            if fn(_sys(length_km=L))["rate"] > 0:
                last = L
        return last
    assert max_range(keyrate_gllp_nodecoy) < max_range(keyrate_decoy)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
