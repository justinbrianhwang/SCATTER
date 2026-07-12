"""Regression checks for the Lim-2014 finite-key decoy-state bound."""
from __future__ import annotations

from qkd.dataset import build_system
from qkd.finitekey import secret_key_length, secret_key_rate
from qkd.keyrate import keyrate_decoy


def test_secret_key_length_nonnegative_and_monotone():
    sys = build_system(25.0)
    lengths = [secret_key_length(sys, N) for N in (1e6, 1e8, 1e10)]
    assert all(length >= 0.0 for length in lengths)
    assert lengths == sorted(lengths)


def test_finite_rate_approaches_asymptotic_from_below():
    sys = build_system(25.0)
    asymptotic = keyrate_decoy(sys)["rate"]
    finite = secret_key_rate(sys, 1e12)
    assert finite > 0.0
    assert 0.5 * asymptotic < finite < asymptotic


def test_long_distance_small_block_aborts():
    sys = build_system(200.0)
    assert secret_key_length(sys, 1e6) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
