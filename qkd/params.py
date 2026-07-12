"""Physical and protocol parameter containers.

All optical/detector defaults are phenomenological values consistent with the
practical decoy-state BB84 literature (e.g. Ma et al., PRA 72, 012326 (2005);
GYS-type fiber setups). They are *simulation* parameters, not a specific device.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceParams:
    """Weak coherent pulse (WCP) source.

    A WCP has a Poisson photon-number distribution P(n) = mu^n e^-mu / n!.
    ``intensities`` are the signal + decoy mean photon numbers (mu, nu1, nu2),
    with ``probs`` their sending probabilities (must sum to 1).
    """
    intensities: tuple[float, ...] = (0.5, 0.1, 0.0)  # signal, weak decoy, vacuum
    probs: tuple[float, ...] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        if len(self.intensities) != len(self.probs):
            raise ValueError("intensities and probs must have equal length")
        if abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError(f"probs must sum to 1, got {sum(self.probs)}")

    @property
    def signal(self) -> float:
        return self.intensities[0]


@dataclass(frozen=True)
class ChannelParams:
    """Fiber channel with distance-dependent loss."""
    length_km: float = 50.0
    attenuation_db_km: float = 0.2      # standard telecom fiber @1550 nm
    misalignment: float = 0.015         # intrinsic optical error probability e_d

    @property
    def transmittance(self) -> float:
        """Channel transmittance t = 10^(-alpha L / 10)."""
        return 10.0 ** (-self.attenuation_db_km * self.length_km / 10.0)


@dataclass(frozen=True)
class DetectorParams:
    """Threshold single-photon detectors (one per bit value, per basis).

    ``efficiency`` is the baseline detection efficiency eta_det. ``eta_mismatch``
    is the fractional efficiency imbalance between the two detectors (bit 0 vs
    bit 1); this is the loophole exploited by time-shift / efficiency-mismatch
    attacks and is nominally 0 for an ideal receiver.
    """
    efficiency: float = 0.1             # eta_det (detector quantum efficiency)
    dark_count: float = 1e-6            # per-gate dark count probability Y_0/2 per detector
    afterpulse: float = 0.0             # afterpulse probability (0 = off for now)
    eta_mismatch: float = 0.0           # (eta_0 - eta_1)/(eta_0 + eta_1), signed
    error_correction_eff: float = 1.16  # f_EC, Shannon-limit inefficiency factor

    def eta_pair(self) -> tuple[float, float]:
        """Return (eta_0, eta_1) given baseline efficiency and mismatch."""
        m = self.eta_mismatch
        eta0 = self.efficiency * (1.0 + m)
        eta1 = self.efficiency * (1.0 - m)
        return eta0, eta1


@dataclass(frozen=True)
class SystemParams:
    """Full system: source + channel + detector, plus derived optics."""
    source: SourceParams = field(default_factory=SourceParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    detector: DetectorParams = field(default_factory=DetectorParams)

    @property
    def eta(self) -> float:
        """Overall transmittance eta = channel * detector efficiency."""
        return self.channel.transmittance * self.detector.efficiency

    @property
    def Y0(self) -> float:
        """Background yield Y_0 (dark counts from the two detectors per gate)."""
        # Two threshold detectors, each firing with prob p_dc -> Y0 ~= 2 p_dc.
        return 2.0 * self.detector.dark_count
