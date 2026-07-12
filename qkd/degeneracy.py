"""Analytic backbone of the DEGENERACY attack.

Closed-form account of why gain-matched PNS is observationally degenerate with
honest operation under LIMITED telemetry, yet irreducibly visible in the decoy
channel (FULL telemetry). This turns the numerical "degeneracy valley" into a
proposition with a predictive formula.

Setup (stealth PNS, single_forward): Eve applies one photon-number policy to
every pulse -- she cannot distinguish signal from decoy:
    n = 0   -> only dark counts,            yield  y0 = Y0
    n = 1   -> forwarded loss-free w.p. r,  yield  y1 = r*eta_det + Y0
    n >= 2  -> one photon forwarded,        yield  y>=2 = eta_det + Y0   (she keeps the rest)
The observed gain at intensity mu is  Q_E(mu) = sum_n Poisson(n;mu) y_n.

Proposition. Eve has one free parameter r; she sets it so Q_E(mu_signal) equals
the honest gain Q_H(mu_signal) -> the LIMITED gain feature is matched exactly.
But the SAME policy at the decoy intensity gives Q_E(nu) != Q_H(nu): the
photon-number statistics differ, so a single r cannot match both. The decoy
residual
    Delta(nu) = Q_E(nu) - Q_H(nu)
is therefore nonzero and is the unavoidable FULL-telemetry signature. Because
LIMITED telemetry omits the decoy residual, D_LIMITED can be driven to the noise
floor while D_FULL >= (Delta(nu))^2 / (2 Var) > 0 -- the data-processing gap made
explicit.
"""
from __future__ import annotations

import math

from .keyrate import gain_qber
from .params import SystemParams


def _poisson_split(mu: float):
    p0 = math.exp(-mu)
    p1 = mu * math.exp(-mu)
    pm = 1.0 - p0 - p1              # P(n >= 2)
    return p0, p1, pm


def pns_gain(sys: SystemParams, mu: float, r: float, m: float = 1.0) -> float:
    """Analytic observed gain at intensity ``mu`` under stealth PNS with
    single-photon forward prob ``r`` and multi-photon forward prob ``m``."""
    eta_det = sys.detector.efficiency
    Y0 = sys.Y0
    p0, p1, pm = _poisson_split(mu)
    return p0 * Y0 + p1 * (r * eta_det + Y0) + pm * (m * eta_det + Y0)


def gain_match(sys: SystemParams, mu_signal: float | None = None) -> tuple[float, float]:
    """Two-knob gain match (r, m).

    Eve mimics the honest single-photon yield exactly by forwarding singles with
    r = channel transmittance (so y_1 = eta_ch*eta_det = honest y_1, gaining no
    single-photon information), then throttles the multi-photon forward prob m to
    match the signal gain at any distance.
    """
    if mu_signal is None:
        mu_signal = sys.source.intensities[0]
    eta_det = sys.detector.efficiency
    Y0 = sys.Y0
    r = sys.channel.transmittance
    p0, p1, pm = _poisson_split(mu_signal)
    Q_h = gain_qber(sys, mu_signal)[0]
    # Q_h = p0*Y0 + p1*(r*eta_det+Y0) + pm*(m*eta_det+Y0) -> solve for m
    m = (Q_h - p0 * Y0 - p1 * (r * eta_det + Y0) - pm * Y0) / (pm * eta_det)
    return r, float(min(max(m, 0.0), 1.0))


def gain_match_restore(sys: SystemParams, mu_signal: float | None = None) -> float:
    """Backward-compatible single-knob match (multi fully forwarded, m=1)."""
    if mu_signal is None:
        mu_signal = sys.source.intensities[0]
    eta_det = sys.detector.efficiency
    Y0 = sys.Y0
    p0, p1, pm = _poisson_split(mu_signal)
    Q_h = gain_qber(sys, mu_signal)[0]
    return (Q_h - (p0 * Y0 + p1 * Y0 + pm * (eta_det + Y0))) / (p1 * eta_det)


def decoy_residual(sys: SystemParams, r: float | None = None,
                   m: float = 1.0, nu: float | None = None) -> float:
    """Predicted decoy-intensity gain residual Delta(nu) = Q_E(nu) - Q_H(nu)
    at the signal-gain-matched (r, m). The analytic degeneracy signature."""
    if r is None:
        r, m = gain_match(sys)
    if nu is None:
        nu = sys.source.intensities[1]
    return pns_gain(sys, nu, r, m) - gain_qber(sys, nu)[0]
