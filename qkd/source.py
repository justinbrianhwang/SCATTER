"""Weak coherent pulse source with Poisson photon-number statistics."""
from __future__ import annotations

import numpy as np

from .params import SourceParams


class WCPSource:
    """Samples pulse intensities and photon numbers for a WCP BB84 transmitter."""

    def __init__(self, params: SourceParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng

    def emit(self, n_pulses: int) -> tuple[np.ndarray, np.ndarray]:
        """Emit ``n_pulses``.

        Returns
        -------
        intensity_idx : int array, which (signal/decoy/vacuum) intensity was chosen
        photons       : int array, Poisson photon number n for each pulse
        """
        idx = self.rng.choice(
            len(self.p.intensities), size=n_pulses, p=self.p.probs
        )
        mus = np.asarray(self.p.intensities)[idx]
        photons = self.rng.poisson(mus)
        return idx, photons
