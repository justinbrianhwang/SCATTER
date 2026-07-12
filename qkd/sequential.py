"""Sequential change detection (CUSUM) on the telemetry stream.

We use the Lorden-optimal CUSUM test with a *known* post-change law P1. Giving
the detector full knowledge of the attack yields the smallest possible detection
delay -- the fundamental limit. Any impossibility shown here ("even this
omniscient detector cannot catch Eve within N blocks") is therefore
unconditional over all detectors.

CUSUM statistic on per-block Gaussian log-likelihood ratios:
    L_t = log p1(x_t) - log p0(x_t)
    S_t = max(0, S_{t-1} + L_t),   alarm when S_t > h.
The false-alarm average-run-length ARL0(h) rises ~ e^h; the detection delay
ARL1(h) ~ h / D(P1||P0). Sweeping h traces the operating curve.
"""
from __future__ import annotations

import numpy as np

from .infometrics import GaussLaw


def gaussian_logpdf(X: np.ndarray, law: GaussLaw) -> np.ndarray:
    k = law.dim
    Sinv = np.linalg.pinv(law.cov, rcond=1e-15)
    sign, logdet = np.linalg.slogdet(law.cov)
    dm = X - law.mean
    maha = np.einsum("ij,jk,ik->i", dm, Sinv, dm)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + maha)


def llr_stream(X: np.ndarray, p1: GaussLaw, p0: GaussLaw) -> np.ndarray:
    """Per-block log-likelihood ratio log p1/p0."""
    return gaussian_logpdf(X, p1) - gaussian_logpdf(X, p0)


def cusum_first_alarm(llr: np.ndarray, h: float) -> int | None:
    """Return index of first CUSUM alarm, or None if none within the stream."""
    S = 0.0
    for t, l in enumerate(llr):
        S = max(0.0, S + l)
        if S > h:
            return t
    return None


def arl0(p0_stream_llr: np.ndarray, h: float, block_len: int) -> float:
    """Average run length to false alarm on an honest stream, in blocks.

    ``p0_stream_llr`` is a long LLR sequence computed on honest blocks. We slide
    non-overlapping CUSUM runs and average the time to (false) alarm; runs that
    never alarm contribute the full window (a lower bound on ARL0)."""
    times = []
    i, n = 0, len(p0_stream_llr)
    while i < n:
        seg = p0_stream_llr[i:i + block_len]
        t = cusum_first_alarm(seg, h)
        if t is None:
            i += block_len
        else:
            times.append(t + 1)
            i += t + 1
    return float(np.mean(times)) if times else float(block_len)


def detection_delay(attack_llr_runs: list[np.ndarray], h: float) -> float:
    """Mean detection delay (blocks) over independent attacked streams."""
    delays = []
    for llr in attack_llr_runs:
        t = cusum_first_alarm(llr, h)
        delays.append(t + 1 if t is not None else len(llr) + 1)
    return float(np.mean(delays))
