"""Analytical key-rate formulas: GLLP and decoy-state (Ma et al. 2005).

References
----------
GLLP:   Gottesman, Lo, Lutkenhaus, Preskill, Quant. Inf. Comput. 4, 325 (2004).
Decoy:  Ma, Qi, Zhao, Lo, "Practical decoy state for QKD", PRA 72, 012326 (2005).

These are closed-form asymptotic (infinite-key) rates. Because they are exact
given the channel model in :func:`gain_qber`, they serve as the ground-truth
validation anchor (Fig. 1) against which the Monte-Carlo simulator is checked.

Convention: rates are *per emitted pulse*, with sifting factor q = 1/2 for
standard BB84. e0 = 1/2 (dark counts are unbiased).
"""
from __future__ import annotations

import math

from .params import SystemParams

E0 = 0.5  # error rate of background/dark counts (random -> 1/2)


def binary_entropy(x: float) -> float:
    """Binary Shannon entropy H2(x) in bits."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def gain_qber(sys: SystemParams, mu: float) -> tuple[float, float]:
    """Overall gain Q_mu and QBER E_mu for a WCP of intensity ``mu``.

    Canonical practical-BB84 model (Ma et al. 2005, Eqs. 6-7):
        Q_mu = Y0 + 1 - exp(-eta*mu)
        E_mu = (e0*Y0 + e_d*(1 - exp(-eta*mu))) / Q_mu
    where eta is the overall transmittance and e_d the optical misalignment.
    """
    eta = sys.eta
    Y0 = sys.Y0
    e_d = sys.channel.misalignment
    Q = Y0 + 1.0 - math.exp(-eta * mu)
    if Q <= 0.0:
        return 0.0, 0.0
    EQ = E0 * Y0 + e_d * (1.0 - math.exp(-eta * mu))
    return Q, EQ / Q


def photon_gain_qber(sys: SystemParams, n: int, mu: float) -> tuple[float, float]:
    """True n-photon yield Y_n and error e_n (for cross-checking estimators).

    Y_n = Y0 + eta_n - Y0*eta_n, with eta_n = 1 - (1-eta)^n.
    e_n = (e0*Y0 + e_d*eta_n) / Y_n.
    """
    eta = sys.eta
    Y0 = sys.Y0
    e_d = sys.channel.misalignment
    eta_n = 1.0 - (1.0 - eta) ** n
    Yn = Y0 + eta_n - Y0 * eta_n
    if Yn <= 0.0:
        return 0.0, 0.0
    en = (E0 * Y0 + e_d * eta_n) / Yn
    return Yn, en


# --------------------------------------------------------------------------- #
# Decoy-state estimation: recover single-photon contribution from observables
# --------------------------------------------------------------------------- #
def decoy_estimate(mu: float, nu: float,
                   Q_mu: float, E_mu: float,
                   Q_nu: float, E_nu: float,
                   Y0: float) -> tuple[float, float, float]:
    """One-weak-decoy + vacuum estimation of (Y1_L, Q1_L, e1_U).

    Ma et al. 2005, Eqs. (34)-(37). Vacuum decoy fixes Y0; the weak decoy nu
    lower-bounds the single-photon yield Y1 and upper-bounds its error e1.

    Returns
    -------
    (Y1_L, Q1_L, e1_U) : lower bound on single-photon yield, single-photon gain,
                         and upper bound on single-photon QBER.
    """
    # Lower bound on single-photon yield (Ma Eq. 34):
    #   Y1_L = mu/(mu*nu - nu^2) * ( Q_nu e^nu - Q_mu e^mu * nu^2/mu^2
    #                                - (mu^2 - nu^2)/mu^2 * Y0 )
    coeff = mu / (mu * nu - nu * nu)
    Y1_L = coeff * (
        Q_nu * math.exp(nu)
        - Q_mu * math.exp(mu) * (nu * nu) / (mu * mu)
        - (mu * mu - nu * nu) / (mu * mu) * Y0
    )
    Y1_L = max(Y1_L, 0.0)

    # Single-photon gain of the signal state: Q1 = Y1 * mu * e^-mu
    Q1_L = Y1_L * mu * math.exp(-mu)

    # Upper bound on single-photon error (Ma Eq. 37):
    #   e1_U = (E_nu Q_nu e^nu - e0 Y0) / (Y1_L * nu)
    if Y1_L > 0.0:
        e1_U = (E_nu * Q_nu * math.exp(nu) - E0 * Y0) / (Y1_L * nu)
        e1_U = min(max(e1_U, 0.0), 0.5)
    else:
        e1_U = 0.5
    return Y1_L, Q1_L, e1_U


def keyrate_decoy(sys: SystemParams, q: float = 0.5) -> dict:
    """Asymptotic GLLP key rate *with* decoy-state estimation.

    R = q { -Q_mu f_EC H2(E_mu) + Q1_L [1 - H2(e1_U)] }.

    Uses the system's (signal, decoy, vacuum) intensities. Returns a dict with
    the rate plus intermediate quantities for inspection/plotting.
    """
    src = sys.source
    mu = src.intensities[0]
    nu = src.intensities[1]
    f = sys.detector.error_correction_eff

    Q_mu, E_mu = gain_qber(sys, mu)
    Q_nu, E_nu = gain_qber(sys, nu)
    Y0 = sys.Y0

    Y1_L, Q1_L, e1_U = decoy_estimate(mu, nu, Q_mu, E_mu, Q_nu, E_nu, Y0)

    R = q * (-Q_mu * f * binary_entropy(E_mu) + Q1_L * (1.0 - binary_entropy(e1_U)))
    return {
        "rate": max(R, 0.0),
        "Q_mu": Q_mu, "E_mu": E_mu,
        "Y1_L": Y1_L, "Q1_L": Q1_L, "e1_U": e1_U,
    }


def keyrate_gllp_nodecoy(sys: SystemParams, q: float = 0.5) -> dict:
    """Asymptotic GLLP key rate *without* decoy states (PNS-pessimistic).

    Without decoy, Eve may perform photon-number splitting: assume every
    multi-photon pulse leaks full information (is "tagged"). Only the untagged
    single-photon detections are secure.

        Delta = P_multi / Q_mu        (tagged fraction of detected events)
        Q_mu  = Y0 + 1 - e^{-eta mu}
        P_multi = 1 - e^{-mu} - mu e^{-mu}
        R = q { -Q_mu f_EC H2(E_mu) + Q_mu (1-Delta)[1 - H2(E_mu/(1-Delta))] }

    This is the rate that *collapses* with distance and that decoy-state
    recovers (Fig. 1).
    """
    mu = sys.source.intensities[0]
    f = sys.detector.error_correction_eff
    Q_mu, E_mu = gain_qber(sys, mu)

    P_multi = 1.0 - math.exp(-mu) - mu * math.exp(-mu)
    Delta = min(P_multi / Q_mu, 1.0) if Q_mu > 0 else 1.0
    untagged = 1.0 - Delta

    if untagged <= 0.0:
        return {"rate": 0.0, "Q_mu": Q_mu, "E_mu": E_mu, "Delta": Delta}

    e_untagged = min(E_mu / untagged, 0.5)
    R = q * (
        -Q_mu * f * binary_entropy(E_mu)
        + Q_mu * untagged * (1.0 - binary_entropy(e_untagged))
    )
    return {"rate": max(R, 0.0), "Q_mu": Q_mu, "E_mu": E_mu, "Delta": Delta}
