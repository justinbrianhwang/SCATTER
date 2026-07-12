"""Regression checks for the SCATTER framework and the DEGENERACY attack."""
from __future__ import annotations

import numpy as np

from qkd.attacks import PNS
from qkd.dataset import build_system, const, generate
from qkd.degeneracy import (decoy_residual, gain_match, gain_match_restore,
                          pns_gain)
from qkd.infometrics import (fit_gaussian, kl_gaussian, kl_symmetric,
                            stein_detection_blocks)
from qkd.keyrate import gain_qber
from qkd.security import ledger
from qkd.telemetry import FULL_FEATURES, LIMITED_FEATURES

NLIM = len(LIMITED_FEATURES)


def test_kl_nonneg_and_zero_on_identity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    P = fit_gaussian(X)
    assert kl_gaussian(P, P) < 1e-6
    Q = fit_gaussian(X + 3.0)
    assert kl_gaussian(Q, P) > 0.5


def test_dpi_limited_le_full():
    # Data-processing inequality: dropping features cannot increase KL.
    sys = build_system(25.0)
    rng = np.random.default_rng(1)
    r, m = gain_match(sys)
    Xh = generate(sys, const(None), 400, 20000, rng, FULL_FEATURES).X
    Xa = generate(sys, const(PNS(1.0, restore=r, multi_forward=m)),
                  400, 20000, rng, FULL_FEATURES).X
    P0f, P1f = fit_gaussian(Xh), fit_gaussian(Xa)
    P0l, P1l = fit_gaussian(Xh[:, :NLIM]), fit_gaussian(Xa[:, :NLIM])
    Dfull = kl_symmetric(P1f, P0f)
    Dlim = kl_symmetric(P1l, P0l)
    assert Dlim <= Dfull + 1e-9              # DPI


def test_stein_monotone():
    assert stein_detection_blocks(0.1) > stein_detection_blocks(1.0) > 0
    assert np.isinf(stein_detection_blocks(0.0))


def test_analytic_gain_match_and_residual():
    # Analytic (r, m) should match the signal gain and predict a nonzero,
    # small decoy residual (the degeneracy signature).
    sys = build_system(25.0)
    mu, nu = sys.source.intensities[0], sys.source.intensities[1]
    r, m = gain_match(sys)
    assert abs(pns_gain(sys, mu, r, m) - gain_qber(sys, mu)[0]) < 1e-9  # matched
    Delta = decoy_residual(sys, r, m)
    assert 0 < abs(Delta) < 0.02              # nonzero but small


def test_degeneracy_fingerprint_single_knob():
    # 1-knob attack (multi fully forwarded): LIMITED features matched, but the
    # decoy residual is the lone surviving fingerprint.
    sys = build_system(25.0)
    rng = np.random.default_rng(2)
    r = gain_match_restore(sys)
    h = generate(sys, const(None), 600, 20000, rng, FULL_FEATURES)
    p = generate(sys, const(PNS(1.0, restore=r)), 600, 20000, rng, FULL_FEATURES)
    devs = np.abs(p.X.mean(0) - h.X.mean(0)) / (h.X.std(0) + 1e-15)
    idc = FULL_FEATURES.index("res_gain_dec")
    assert devs[:NLIM].max() < 0.4            # LIMITED all matched
    assert devs[idc] > 2 * devs[:NLIM].max()  # decoy residual dominates
    assert p.eve_info.mean() > 0.2            # and Eve still leaks


def test_degeneracy_two_knob_stealthier_in_full():
    # 2-knob attack mimics the honest single-photon yield -> lower FULL
    # detectability than the 1-knob attack, while DPI keeps LIMITED even lower.
    sys = build_system(25.0)
    rng = np.random.default_rng(2)
    Xh = generate(sys, const(None), 500, 20000, rng, FULL_FEATURES).X
    P0f, P0l = fit_gaussian(Xh), fit_gaussian(Xh[:, :NLIM])
    r1 = gain_match_restore(sys)
    r2, m2 = gain_match(sys)
    X1 = generate(sys, const(PNS(1.0, restore=r1)), 500, 20000, rng, FULL_FEATURES).X
    X2 = generate(sys, const(PNS(1.0, restore=r2, multi_forward=m2)),
                  500, 20000, rng, FULL_FEATURES).X
    D1 = kl_gaussian(fit_gaussian(X1), P0f)
    D2 = kl_gaussian(fit_gaussian(X2), P0f)
    D2l = kl_gaussian(fit_gaussian(X2[:, :NLIM]), P0l)
    assert D2 < D1                            # 2-knob stealthier in FULL
    assert D2l <= D2 + 1e-9                    # DPI


def test_subset_kl_monotone():
    # Adding a feature cannot decrease KL (KL is monotone under refinement).
    from qkd.subset import kl_on_subset
    sys = build_system(25.0)
    rng = np.random.default_rng(4)
    r, m = gain_match(sys)
    Xh = generate(sys, const(None), 400, 20000, rng, FULL_FEATURES).X
    Xa = generate(sys, const(PNS(1.0, restore=r, multi_forward=m)),
                  400, 20000, rng, FULL_FEATURES).X
    d1 = kl_on_subset(Xh, Xa, [0, 1])
    d2 = kl_on_subset(Xh, Xa, [0, 1, 8])      # add decoy-gain residual
    assert d2 >= d1 - 1e-9


def test_composite_subadditive():
    # Composite detectability is below the additive (naive) budget.
    from qkd.attacks import Composite, TimeShift
    sys = build_system(25.0, eta_mismatch=0.12)
    rng = np.random.default_rng(5)
    r, m = gain_match(sys)
    Xh = generate(sys, const(None), 500, 20000, rng, FULL_FEATURES).X
    P0 = fit_gaussian(Xh)

    def D(fac):
        return kl_gaussian(
            fit_gaussian(generate(sys, fac, 500, 20000, rng, FULL_FEATURES).X), P0)

    pns = PNS(1.0, restore=r, multi_forward=m)
    Dp = D(const(PNS(1.0, restore=r, multi_forward=m)))
    Dt = D(const(TimeShift(0.10)))
    Dc = D(const(Composite([PNS(1.0, restore=r, multi_forward=m), TimeShift(0.10)])))
    assert Dc < Dp + Dt                       # sub-additive


def test_security_ledger_scales_with_delay():
    sys = build_system(25.0)
    a = ledger(sys, I=0.3, N_star=10, n_pulses=20000)
    b = ledger(sys, I=0.3, N_star=100, n_pulses=20000)
    assert b.k_stolen > 9 * a.k_stolen        # ~ linear in N*


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
