"""KL detectability on arbitrary telemetry feature subsets, and greedy
telemetry-budget selection: which features lift an attack out of degeneracy."""
from __future__ import annotations

import numpy as np

from .infometrics import fit_gaussian, kl_gaussian


def kl_on_subset(Xh: np.ndarray, Xa: np.ndarray, cols: list[int]) -> float:
    """Symmetric-free KL D(P1||P0) restricted to the given feature columns."""
    P0 = fit_gaussian(Xh[:, cols])
    P1 = fit_gaussian(Xa[:, cols])
    return kl_gaussian(P1, P0)


def greedy_budget(Xh: np.ndarray, Xa: np.ndarray, n_feat: int):
    """Greedily add the feature that maximises detectability at each step.

    Returns the ordered list of added column indices and the running D after
    each addition -- the cheapest telemetry that reaches a given detectability.
    """
    remaining = list(range(n_feat))
    chosen: list[int] = []
    curve: list[float] = []
    while remaining:
        best_c, best_D = None, -1.0
        for c in remaining:
            D = kl_on_subset(Xh, Xa, chosen + [c])
            if D > best_D:
                best_D, best_c = D, c
        chosen.append(best_c)
        remaining.remove(best_c)
        curve.append(best_D)
    return chosen, curve
