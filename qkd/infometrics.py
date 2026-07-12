"""Information-theoretic detectability metrics for telemetry-based QKD IDS.

The core of the new method. Each block's telemetry vector is a sum over ~10^4-10^6
pulses, so by the CLT its law is well-approximated by a multivariate Gaussian.
We therefore estimate the honest law P0 = N(m0, S0) and the attacked law
P1 = N(m1, S1) from Monte-Carlo blocks and compute closed-form quantities:

  * KL divergence D(P1 || P0)  -- governs sequential detection (Stein's lemma):
        expected blocks to detect at false-alarm level alpha  ~  log(1/alpha) / D.
  * Symmetric / Chernoff information -- alternative detectability measures.

Because these are functions of the *observed features only*, restricting to a
cheaper telemetry set is a deterministic map of the data; by the data-processing
inequality D_LIMITED <= D_FULL. That inequality is the backbone of the
impossibility-frontier result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussLaw:
    mean: np.ndarray
    cov: np.ndarray

    @property
    def dim(self) -> int:
        return self.mean.shape[0]


def fit_gaussian(X: np.ndarray, reg: float = 1e-12) -> GaussLaw:
    """Fit a multivariate Gaussian to block feature rows, with covariance
    regularisation to stay positive-definite for near-constant features."""
    m = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    # Regularise: add a floor proportional to each feature's own variance.
    d = S.shape[0]
    diag_floor = np.maximum(np.diag(S), 0) * 1e-6 + reg
    S = S + np.diag(diag_floor)
    return GaussLaw(m, S)


def kl_gaussian(p1: GaussLaw, p0: GaussLaw) -> float:
    """D_KL( N(m1,S1) || N(m0,S0) ), in nats.

    D = 0.5 [ tr(S0^-1 S1) - k + (m0-m1)^T S0^-1 (m0-m1) + ln(det S0/det S1) ].
    """
    k = p1.dim
    S0inv = np.linalg.pinv(p0.cov, rcond=1e-15)
    dm = p0.mean - p1.mean
    tr = np.trace(S0inv @ p1.cov)
    maha = dm @ S0inv @ dm
    sign0, logdet0 = np.linalg.slogdet(p0.cov)
    sign1, logdet1 = np.linalg.slogdet(p1.cov)
    logdet_ratio = logdet0 - logdet1
    d = 0.5 * (tr - k + maha + logdet_ratio)
    return float(max(d, 0.0))


def kl_symmetric(p1: GaussLaw, p0: GaussLaw) -> float:
    return 0.5 * (kl_gaussian(p1, p0) + kl_gaussian(p0, p1))


def bhattacharyya_gaussian(p1: GaussLaw, p0: GaussLaw) -> float:
    """Bhattacharyya distance (Chernoff-1/2). Bounds the two-sided test error
    exponent; robust when only one distribution's mean shifts."""
    S = 0.5 * (p0.cov + p1.cov)
    Sinv = np.linalg.pinv(S, rcond=1e-15)
    dm = p1.mean - p0.mean
    term1 = 0.125 * dm @ Sinv @ dm
    sign, logdetS = np.linalg.slogdet(S)
    _, ld0 = np.linalg.slogdet(p0.cov)
    _, ld1 = np.linalg.slogdet(p1.cov)
    term2 = 0.5 * (logdetS - 0.5 * (ld0 + ld1))
    return float(max(term1 + term2, 0.0))


def stein_detection_blocks(kl: float, alpha: float = 0.01) -> float:
    """Expected blocks to detect at false-alarm level alpha (Stein/CUSUM):
    N* ~ log(1/alpha) / D.  Returns inf when D -> 0 (undetectable)."""
    if kl <= 0:
        return np.inf
    return np.log(1.0 / alpha) / kl
