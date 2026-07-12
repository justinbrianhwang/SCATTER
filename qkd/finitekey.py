"""Finite-key decoy-state bounds for practical BB84.

Implements Lim, Curty, Walenta, Xu, and Zbinden, Phys. Rev. A 89,
022307 (2014), Eqs. (1)-(5): Hoeffding decoy bounds, lower bounds on
vacuum and single-photon events, the single-photon phase-error bound, and
the composable secret-key length. The existing analytic gain/QBER model is
used for deterministic expected observed counts.
"""
from __future__ import annotations

import math

from .keyrate import binary_entropy, gain_qber
from .params import SystemParams

_BB84_SIFTING = 0.5
_SECURITY_SPLIT = 19.0


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _tau(intensities: tuple[float, ...], probs: tuple[float, ...],
         photon_number: int) -> float:
    factorial = math.factorial(photon_number)
    return sum(
        p * math.exp(-mu) * mu ** photon_number / factorial
        for mu, p in zip(intensities, probs)
    )


def _hoeffding_delta(total: float, eps_sec: float) -> float:
    if total <= 0.0:
        return 0.0
    return math.sqrt(0.5 * total * math.log(_SECURITY_SPLIT / eps_sec))


def _scaled_bounds(values: list[float], total: float,
                   intensities: tuple[float, ...],
                   probs: tuple[float, ...],
                   eps_sec: float) -> tuple[list[float], list[float]]:
    delta = _hoeffding_delta(total, eps_sec)
    lower = []
    upper = []
    for value, mu, prob in zip(values, intensities, probs):
        scale = math.exp(mu) / prob
        lower.append(scale * max(value - delta, 0.0))
        upper.append(scale * (value + delta))
    return lower, upper


def _gamma(eps_sec: float, error_rate: float, sample_bits: float,
           key_bits: float) -> float:
    if sample_bits <= 0.0 or key_bits <= 0.0:
        return 1.0
    b = _clip(error_rate, 1e-15, 1.0 - 1e-15)
    prefactor = (
        (sample_bits + key_bits) * (1.0 - b) * b
        / (sample_bits * key_bits * math.log(2.0))
    )
    argument = (
        (sample_bits + key_bits)
        / (sample_bits * key_bits * (1.0 - b) * b)
        * (_SECURITY_SPLIT / eps_sec) ** 2
    )
    if argument <= 1.0:
        return 0.0
    return math.sqrt(max(0.0, prefactor * math.log2(argument)))


def secret_key_length(sys: SystemParams, N: float, eps_sec: float = 1e-9,
                      eps_cor: float = 1e-15,
                      f_EC: float | None = None) -> float:
    """Return the Lim-2014 epsilon-secure finite-key length in bits.

    Counts use Eqs. (2)-(4) with Hoeffding fluctuations, Eq. (5) for the
    phase-error rate, and Eq. (1) with ``6 log2(19/eps_sec)`` privacy
    overhead. ``N`` is the total emitted-pulse count; a standard BB84
    same-basis sifting factor of 1/2 is included to match the repository's
    per-emitted-pulse asymptotic convention.
    """
    if N <= 0.0:
        return 0.0
    if eps_sec <= 0.0 or eps_cor <= 0.0:
        raise ValueError("eps_sec and eps_cor must be positive")

    intensities = tuple(float(mu) for mu in sys.source.intensities)
    probs = tuple(float(p) for p in sys.source.probs)
    if len(intensities) != 3 or len(probs) != 3:
        raise ValueError("finite-key decoy bound requires three intensities")

    mu1, mu2, mu3 = intensities
    if not (mu1 > mu2 + mu3 and mu2 > mu3 >= 0.0):
        raise ValueError("intensities must satisfy mu1 > mu2 + mu3 and mu2 > mu3 >= 0")
    if any(p <= 0.0 for p in probs):
        raise ValueError("all intensity probabilities must be positive")

    f_ec = sys.detector.error_correction_eff if f_EC is None else float(f_EC)

    counts = []
    errors = []
    for mu, prob in zip(intensities, probs):
        gain, qber = gain_qber(sys, mu)
        counts.append(_BB84_SIFTING * N * prob * gain)
        errors.append(_BB84_SIFTING * N * prob * qber * gain)

    raw_bits = sum(counts)
    bit_errors = sum(errors)
    if raw_bits <= 0.0:
        return 0.0

    n_minus, n_plus = _scaled_bounds(counts, raw_bits, intensities, probs, eps_sec)
    m_minus, m_plus = _scaled_bounds(errors, bit_errors, intensities, probs, eps_sec)

    tau0 = _tau(intensities, probs, 0)
    tau1 = _tau(intensities, probs, 1)

    # Eq. (2): lower bound on detected vacuum-origin events in the key basis.
    s0 = tau0 * (mu2 * n_minus[2] - mu3 * n_plus[1]) / (mu2 - mu3)
    s0 = max(s0, 0.0)

    # Eq. (3): lower bound on detected single-photon events in the key basis.
    denom = mu1 * (mu2 - mu3) - mu2 * mu2 + mu3 * mu3
    s1 = tau1 * mu1 * (
        n_minus[1]
        - n_plus[2]
        - ((mu2 * mu2 - mu3 * mu3) / (mu1 * mu1)) * (n_plus[0] - s0 / tau0)
    ) / denom
    s1 = max(s1, 0.0)
    if s1 <= 0.0:
        return 0.0

    # Eq. (4): upper bound on single-photon bit errors in the test statistics.
    v1 = tau1 * (m_plus[1] - m_minus[2]) / (mu2 - mu3)
    v1 = max(v1, 0.0)

    # Eq. (5): phase-error upper bound, reusing the same decoy statistics.
    single_error_rate = _clip(v1 / s1, 0.0, 0.5)
    phi1 = single_error_rate + _gamma(eps_sec, single_error_rate, s1, s1)
    phi1 = _clip(phi1, 0.0, 0.5)

    observed_qber = _clip(bit_errors / raw_bits, 0.0, 0.5)
    lambda_ec = f_ec * raw_bits * binary_entropy(observed_qber)
    security_overhead = (
        6.0 * math.log2(_SECURITY_SPLIT / eps_sec)
        + math.log2(2.0 / eps_cor)
    )

    ell = s0 + s1 * (1.0 - binary_entropy(phi1)) - lambda_ec - security_overhead
    return max(float(math.floor(ell)), 0.0)


def secret_key_rate(sys: SystemParams, N: float, eps_sec: float = 1e-9,
                    eps_cor: float = 1e-15,
                    f_EC: float | None = None) -> float:
    """Return the finite-key secret-key rate per emitted pulse."""
    if N <= 0.0:
        return 0.0
    return secret_key_length(sys, N, eps_sec, eps_cor, f_EC) / float(N)
