"""Per-pulse BB84 session: the Monte-Carlo core that produces raw detection
records for one block of pulses, under honest operation or an attack.

The output ``BlockRecord`` holds per-detected-event arrays (bit, basis, which
detector, arrival time, intensity, double-click flag) plus block-level
bookkeeping (pulses sent, Eve's information fraction). Telemetry features
(qkd.telemetry) are computed from these records; the ML detector never sees the
ground-truth attack label, only the features.

Timing model
------------
Each accepted photon has an arrival time t ~ N(0, sigma_t). The two threshold
detectors have time-dependent efficiency eta_i(t) = eta_i * G(t - t_i; w),
where a small offset t_0 != t_1 encodes detector *efficiency mismatch in time*
-- the loophole the time-shift attack exploits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SystemParams

NO_CLICK = -1
DOUBLE = 2


@dataclass
class BlockRecord:
    """Raw per-event detection data for one block (only detected gates kept)."""
    intensity_idx: np.ndarray   # 0=signal,1=decoy,2=vacuum
    basis_match: np.ndarray     # bool: Alice/Bob bases agreed (sifted)
    basis: np.ndarray           # Bob's basis, 0=Z 1=X
    bit: np.ndarray             # sifted key bit (detector that fired)
    error: np.ndarray           # bool: bit != Alice's bit (only meaningful if matched)
    arrival_t: np.ndarray       # ns, detection time
    double: np.ndarray          # bool: double click at this gate
    n_pulses: int               # total pulses sent in block
    eve_info: float             # fraction of sifted key bits Eve knows (ground truth)
    label: str                  # 'honest' or attack name (ground truth, for eval only)


def _time_efficiency(t: np.ndarray, t_center: float, w: float) -> np.ndarray:
    return np.exp(-((t - t_center) ** 2) / (2.0 * w * w))


class Session:
    """Runs one block of BB84 pulses through the physical pipeline."""

    def __init__(self, sys: SystemParams, rng: np.random.Generator,
                 sigma_t: float = 0.05, gate_w: float = 0.08,
                 t_offset: float = 0.04):
        self.sys = sys
        self.rng = rng
        self.sigma_t = sigma_t      # source timing jitter (ns)
        self.gate_w = gate_w        # detector temporal acceptance width (ns)
        self.t_offset = t_offset    # +/- offset of the two detectors' peaks (ns)

    # ------------------------------------------------------------------ #
    def run(self, n_pulses: int, attack=None) -> BlockRecord:
        """Simulate a block. ``attack`` is an object from qkd.attacks or None."""
        rng = self.rng
        src = self.sys.source
        eta_ch = self.sys.channel.transmittance
        e_d = self.sys.channel.misalignment
        eta0_base, eta1_base = self.sys.detector.eta_pair()
        dc = self.sys.detector.dark_count

        # --- Alice: intensities, photon numbers, bases, bits ---
        idx = rng.choice(len(src.intensities), size=n_pulses, p=src.probs)
        mus = np.asarray(src.intensities)[idx]
        photons = rng.poisson(mus)
        a_basis = rng.integers(0, 2, n_pulses)
        a_bit = rng.integers(0, 2, n_pulses)
        b_basis = rng.integers(0, 2, n_pulses)

        # --- Channel: photons arriving at Bob ---
        arrived = rng.binomial(photons, eta_ch)

        # --- Per-pulse detection modifiers (attacks write here) ---
        # eta_scale multiplies detector efficiency; time_shift shifts arrival;
        # forced holds a forced outcome (>=0) for faked-state/blinding, else -1.
        ctx = AttackContext(
            idx=idx, photons=photons, arrived=arrived,
            a_basis=a_basis, a_bit=a_bit, b_basis=b_basis,
            eta_scale=np.ones(n_pulses), time_shift=np.zeros(n_pulses),
            t_override=np.full(n_pulses, np.nan),
            forced=np.full(n_pulses, -1, dtype=np.int8),
            extra_error=np.zeros(n_pulses),
            eve_knows=np.zeros(n_pulses, dtype=bool),
        )
        if attack is not None:
            attack.apply(ctx, self, rng)

        # --- Bob: basis match, intended detector ---
        match = ctx.a_basis == ctx.b_basis
        rand_bits = rng.integers(0, 2, n_pulses)
        # The photon's "intended" detector: encodes Alice's bit on matched bases,
        # random on mismatched bases. IR conjugate-basis resends flip it (0.5).
        d_int = np.where(match, ctx.a_bit, rand_bits).astype(np.int8)
        flip = rng.random(n_pulses) < ctx.extra_error
        d_int = np.where(flip, 1 - d_int, d_int).astype(np.int8)

        # --- Arrival times, time-dependent detector efficiencies ---
        t = rng.normal(0.0, self.sigma_t, n_pulses) + ctx.time_shift
        override = ~np.isnan(ctx.t_override)
        t = np.where(override, ctx.t_override, t)
        g0 = _time_efficiency(t, +self.t_offset, self.gate_w)
        g1 = _time_efficiency(t, -self.t_offset, self.gate_w)
        eta0 = np.clip(eta0_base * ctx.eta_scale * g0, 0, 1)
        eta1 = np.clip(eta1_base * ctx.eta_scale * g1, 0, 1)

        # Intended detector sees full efficiency; the other sees crosstalk
        # suppressed by the misalignment e_d. Crosstalk is what produces both
        # the baseline QBER (~e_d) and physical double clicks.
        eta_int = np.where(d_int == 0, eta0, eta1)
        eta_cross = np.where(d_int == 0, eta1, eta0) * e_d
        k = ctx.arrived
        sig_int = rng.random(n_pulses) < (1.0 - (1.0 - eta_int) ** k)
        sig_cross = rng.random(n_pulses) < (1.0 - (1.0 - eta_cross) ** k)
        dark0 = rng.random(n_pulses) < dc
        dark1 = rng.random(n_pulses) < dc

        fired0 = ((d_int == 0) & sig_int) | ((d_int == 1) & sig_cross) | dark0
        fired1 = ((d_int == 1) & sig_int) | ((d_int == 0) & sig_cross) | dark1

        # --- Forced outcomes (blinding/faked-state) override physics ---
        # forced: 0 -> det0 only, 1 -> det1 only, 3 -> both (fake double), -1 none.
        single_force = (ctx.forced == 0) | (ctx.forced == 1)
        if single_force.any():
            fired0 = np.where(single_force, ctx.forced == 0, fired0)
            fired1 = np.where(single_force, ctx.forced == 1, fired1)
        dbl_force = ctx.forced == 3
        if dbl_force.any():
            fired0 = fired0 | dbl_force
            fired1 = fired1 | dbl_force

        outcome = np.full(n_pulses, NO_CLICK, dtype=np.int8)
        outcome[fired0 & ~fired1] = 0
        outcome[fired1 & ~fired0] = 1
        double = fired0 & fired1
        # double clicks: assign a random bit (standard sifting rule)
        outcome[double] = rng.integers(0, 2, int(double.sum()))

        detected = outcome != NO_CLICK
        bit = outcome.copy()
        # error flag: only meaningful for matched-basis events (bit vs Alice's)
        err = (bit != ctx.a_bit) & match

        d = detected
        sifted_det = detected & match
        n_sifted = max(int(sifted_det.sum()), 1)
        eve_info = float((ctx.eve_knows & sifted_det).sum()) / n_sifted
        return BlockRecord(
            intensity_idx=ctx.idx[d], basis_match=match[d], basis=ctx.b_basis[d],
            bit=bit[d], error=err[d], arrival_t=t[d], double=double[d],
            n_pulses=n_pulses,
            eve_info=float(np.clip(eve_info, 0, 1)),
            label="honest" if attack is None else attack.name,
        )


@dataclass
class AttackContext:
    """Mutable per-pulse state that attacks modify in place."""
    idx: np.ndarray
    photons: np.ndarray
    arrived: np.ndarray
    a_basis: np.ndarray
    a_bit: np.ndarray
    b_basis: np.ndarray
    eta_scale: np.ndarray
    time_shift: np.ndarray
    t_override: np.ndarray
    forced: np.ndarray
    extra_error: np.ndarray
    eve_knows: np.ndarray
