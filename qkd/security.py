"""From telemetry detectability to the finite-key security deficit.

The security-relevant consequence of the DEGENERACY attack: because the attacked
statistics look honest, Alice and Bob run standard decoy-state privacy
amplification and *certify* a secret key at the honest rate -- yet Eve actually
knows a fraction I of those bits through the photon-number side channel that the
honest-looking statistics never expose. Every block until SCATTER raises an
alarm therefore yields certified-but-compromised key.

Quantities (per block of ``n_pulses``):
  r_cert  : certified secret fraction per sifted bit (decoy asymptotic rate).
  n_sift  : sifted detections per block.
  I       : fraction of sifted key Eve knows (ground-truth leakage).
  N*_T    : Stein detection delay in blocks under telemetry T.

Certified-secret key stolen before detection:
        K_stolen(T) = N*_T * I * n_sift * r_cert     [bits]
This diverges as the telemetry-limited detectability D_T -> 0, which is exactly
the regime the degeneracy attack engineers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .keyrate import keyrate_decoy
from .params import SystemParams


@dataclass
class SecurityLedger:
    r_cert: float          # certified secret fraction per sifted bit
    n_sift: float          # sifted detections per block
    I: float               # Eve information fraction
    N_star: float          # detection delay (blocks)
    k_stolen: float        # certified-secret bits stolen before detection


def sifted_rate(sys: SystemParams) -> float:
    """Sifted detections per pulse: gain x basis-sift (1/2)."""
    from .keyrate import gain_qber
    Q = gain_qber(sys, sys.source.intensities[0])[0]
    # weight by signal-state probability; sifting keeps matched bases (1/2).
    return 0.5 * Q * sys.source.probs[0]


def certified_secret_fraction(sys: SystemParams) -> float:
    """Certified secret bits per sifted bit under the honest-looking statistics.

    Alice/Bob see honest gain & QBER, so they apply the standard decoy key rate;
    r_cert = R_decoy / sifted_rate.
    """
    R = keyrate_decoy(sys)["rate"]        # secret bits per pulse
    s = sifted_rate(sys)
    return R / s if s > 0 else 0.0


def ledger(sys: SystemParams, I: float, N_star: float,
           n_pulses: int) -> SecurityLedger:
    r_cert = certified_secret_fraction(sys)
    n_sift = sifted_rate(sys) * n_pulses
    k_stolen = N_star * I * n_sift * r_cert
    return SecurityLedger(r_cert, n_sift, I, N_star, k_stolen)
