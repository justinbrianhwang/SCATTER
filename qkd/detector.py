"""Threshold single-photon detector pair with device imperfections.

Models a passive-basis-choice BB84 receiver with two threshold detectors (one
per bit value). Imperfections: finite efficiency, efficiency mismatch between
the two detectors (time-shift loophole), dark counts, and optional afterpulsing.

A detection outcome per gate is one of:
    -1  no click
     0  detector-0 click (bit 0)
     1  detector-1 click (bit 1)
     2  double click (both fired) -> randomly assigned at sifting
"""
from __future__ import annotations

import numpy as np

from .params import DetectorParams

NO_CLICK = -1
DOUBLE = 2


class DetectorPair:
    def __init__(self, params: DetectorParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self._prev_click = None  # for afterpulsing

    def detect(self, arrived_photons: np.ndarray, encoded_bit: np.ndarray,
               eta_scale: np.ndarray | None = None) -> np.ndarray:
        """Produce a click outcome per gate.

        Parameters
        ----------
        arrived_photons : photons reaching the receiver this gate.
        encoded_bit     : the bit value (0/1) Bob's basis-matched measurement
                          would map the photon to (already basis-sifted upstream
                          for the matched-basis subset; mismatched bases produce
                          a random bit). Passed in as 0/1.
        eta_scale       : optional per-gate multiplicative efficiency factor in
                          [0,1] (e.g. time-shift attack shifting arrival time).
                          Broadcasts against the detector efficiencies.
        """
        n = arrived_photons.shape[0]
        eta0, eta1 = self.p.eta_pair()
        if eta_scale is not None:
            eta0 = eta0 * eta_scale
            eta1 = eta1 * eta_scale

        # Probability the correct detector fires from photons: 1-(1-eta)^k.
        # The photon maps to detector = encoded_bit; the other detector only
        # sees dark counts (and afterpulse).
        eta_correct = np.where(encoded_bit == 0, eta0, eta1)
        p_signal = 1.0 - (1.0 - eta_correct) ** arrived_photons  # 0 if k==0

        signal_click = self.rng.random(n) < p_signal
        dc = self.p.dark_count
        dark0 = self.rng.random(n) < dc
        dark1 = self.rng.random(n) < dc

        # Afterpulse: a detector that clicked last gate may spuriously fire.
        if self.p.afterpulse > 0.0 and self._prev_click is not None:
            ap = self.p.afterpulse
            ap0 = (self._prev_click == 0) & (self.rng.random(n) < ap)
            ap1 = (self._prev_click == 1) & (self.rng.random(n) < ap)
            dark0 = dark0 | ap0
            dark1 = dark1 | ap1

        # Combine into per-detector fired flags.
        fired0 = (signal_click & (encoded_bit == 0)) | dark0
        fired1 = (signal_click & (encoded_bit == 1)) | dark1

        outcome = np.full(n, NO_CLICK, dtype=np.int8)
        outcome[fired0 & ~fired1] = 0
        outcome[fired1 & ~fired0] = 1
        outcome[fired0 & fired1] = DOUBLE

        if self.p.afterpulse > 0.0:
            self._prev_click = np.where(
                outcome == DOUBLE, self.rng.integers(0, 2, n), outcome
            )
        return outcome
