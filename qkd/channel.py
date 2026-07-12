"""Fiber channel: independent per-photon transmission (Bernoulli thinning)."""
from __future__ import annotations

import numpy as np

from .params import ChannelParams


class FiberChannel:
    """Lossy channel. Each of the n photons survives with prob = transmittance."""

    def __init__(self, params: ChannelParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng

    @property
    def transmittance(self) -> float:
        return self.p.transmittance

    def transmit(self, photons: np.ndarray) -> np.ndarray:
        """Binomial thinning: photons arriving at the receiver per pulse."""
        return self.rng.binomial(photons, self.transmittance)
