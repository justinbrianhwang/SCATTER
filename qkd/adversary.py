"""Adversarial Eve: numerically minimise LIMITED-telemetry detectability D_lim
while holding the information leakage I above a target.

This closes the min-max: rather than a fixed attack, Eve searches her free
parameters (gain-restore, double-click injection) for the *least detectable* way
to steal the key under a given telemetry budget. The residual D_lim she achieves
is the operative frontier value; when it approaches the honest noise floor the
attack is unconditionally undetectable (N* -> infinity).
"""
from __future__ import annotations

import numpy as np

from .attacks import PNS, honest_gain
from .dataset import const, generate
from .infometrics import fit_gaussian, kl_gaussian
from .telemetry import LIMITED_FEATURES


def _eval(sys, attack_factory, rng, P0_lim, n_blocks, n_pulses):
    bs = generate(sys, attack_factory, n_blocks, n_pulses, rng, LIMITED_FEATURES)
    P1 = fit_gaussian(bs.X)
    return kl_gaussian(P1, P0_lim), float(bs.eve_info.mean())


def minimise_dlim_pns(sys, rng, P0_lim, n_blocks=250, n_pulses=20000,
                      restore_grid=None, dinj_grid=None):
    """Grid-search PNS (restore, double_inject) minimising D_lim.

    Returns (best_params, D_lim, I) and the full evaluated grid for plotting.
    """
    if restore_grid is None:
        restore_grid = np.linspace(0.05, 0.45, 9)
    if dinj_grid is None:
        dinj_grid = np.linspace(0.0, 0.010, 6)
    best = None
    grid = []
    for r in restore_grid:
        for di in dinj_grid:
            D, I = _eval(sys, const(PNS(1.0, restore=r, double_inject=di)),
                         rng, P0_lim, n_blocks, n_pulses)
            grid.append((r, di, D, I))
            if best is None or D < best[2]:
                best = (r, di, D, I)
    return best, grid
