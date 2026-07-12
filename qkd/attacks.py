"""Eavesdropping attacks that act on the per-pulse AttackContext.

Each attack exposes ``name``, a ``strength`` in [0,1] (fraction of pulses
attacked / attack intensity), and ``apply(ctx, session, rng)`` which mutates
the context in place and marks ``ctx.eve_knows`` for bits Eve learns.

Modelling is at the detection-statistics level (standard for telemetry-based
QKD security studies), not full quantum-state simulation. Each attack's docstring
states its physical signature and which telemetry it perturbs.
"""
from __future__ import annotations

import numpy as np


class Attack:
    name = "attack"

    def __init__(self, strength: float = 1.0):
        self.strength = float(strength)

    def apply(self, ctx, session, rng) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Composite(Attack):
    """Apply several attacks to the same block, exploiting multiple device
    imperfections at once. Because each sub-attack perturbs a different part of
    the telemetry, the composite detectability is generally sub-additive:
    D(composite) < sum of the individual D's -- a defender who budgets detection
    per-imperfection underestimates the blind spot (the "combined loophole"
    effect). Sub-attacks mutate the shared context in sequence.
    """
    name = "composite"

    def __init__(self, attacks: list):
        super().__init__(1.0)
        self.attacks = attacks

    def apply(self, ctx, session, rng) -> None:
        for a in self.attacks:
            a.apply(ctx, session, rng)


class InterceptResend(Attack):
    """Intercept-resend on a fraction ``strength`` of pulses.

    Signature: raises QBER (~0.25 * strength on sifted key). Fully caught by
    LIMITED telemetry via QBER. Eve learns bits where her basis matched Alice's.
    """
    name = "intercept_resend"

    def apply(self, ctx, session, rng) -> None:
        n = ctx.a_bit.shape[0]
        intercepted = rng.random(n) < self.strength
        eve_basis = rng.integers(0, 2, n)
        eve_right = eve_basis == ctx.a_basis          # Eve measured in Alice's basis
        # Resent photon is in eve_basis; when Bob's basis (=Alice's, on sifted)
        # differs from eve_basis, Bob's outcome is random -> 0.5 flip prob.
        wrong = intercepted & ~eve_right
        ctx.extra_error[wrong] = 0.5                    # pushes error to ~0.25 avg
        ctx.eve_knows[intercepted & eve_right] = True


class PNS(Attack):
    """Photon-number-splitting with signal-gain matching.

    Eve applies ONE photon-number policy to every pulse (she cannot tell decoy
    from signal): block vacuum, forward each single loss-free (unmeasured) with
    probability ``restore``, forward every multi-photon pulse loss-free while
    keeping a copy (she learns those bits). ``restore`` is tuned (see
    :func:`calibrate_pns`) so the *signal* gain matches honest calibration --
    making PNS invisible to overall-gain (LIMITED) telemetry. Because decoy and
    vacuum intensities have different photon statistics, their yields no longer
    match honest: the decoy residuals (FULL telemetry) are the only trace.

    ``strength`` interpolates honest(0) -> full PNS(1).
    """
    name = "pns"

    def __init__(self, strength: float = 1.0, restore: float = 1.0,
                 multi_forward: float = 1.0,
                 single_forward: bool = True, double_inject: float = 0.0):
        super().__init__(strength)
        self.restore = float(restore)          # single-photon forward prob r
        self.multi_forward = float(multi_forward)  # multi-photon forward prob m
        self.single_forward = single_forward   # forward exactly 1 photon (stealth)
        self.double_inject = double_inject      # extra doubles to match honest rate

    def apply(self, ctx, session, rng) -> None:
        n = ctx.photons.shape[0]
        act = rng.random(n) < self.strength
        single = (ctx.photons == 1) & act
        multi = (ctx.photons >= 2) & act
        # Single photons: forwarded loss-free with prob ``restore``. Setting
        # restore = channel transmittance mimics the honest single-photon yield
        # exactly (Eve does not measure singles -> no info from them).
        keep_single = single & (rng.random(n) < self.restore)
        drop_single = single & ~keep_single
        ctx.arrived[drop_single] = 0
        ctx.arrived[keep_single] = 1            # one photon reaches Bob, loss-free
        # Multi-photon: forwarded with prob ``multi_forward`` (Eve throttles to
        # match the honest gain at long distance); she keeps a copy and knows
        # the bit of every forwarded multi-photon pulse.
        keep_multi = multi & (rng.random(n) < self.multi_forward)
        drop_multi = multi & ~keep_multi
        ctx.arrived[drop_multi] = 0
        if self.single_forward:
            ctx.arrived[keep_multi] = 1
        else:
            ctx.arrived[keep_multi] = np.maximum(ctx.arrived[keep_multi],
                                                 ctx.photons[keep_multi] - 1)
        ctx.eve_knows[keep_multi] = True
        # Optionally inject a few doubles (Eve fires the idle detector) to match
        # the honest double-click rate -- the last LIMITED trace of stealth PNS.
        if self.double_inject > 0:
            fwd = keep_single | multi
            inj = fwd & (rng.random(n) < self.double_inject)
            ctx.forced[inj] = 3


class Blinding(Attack):
    """Detector-blinding / faked-state intercept-resend.

    Eve blinds Bob's detectors and injects bright faked states that force a
    deterministic click *only* when Bob's basis matches Eve's. Result:
    QBER ~ 0, Eve knows the entire key, and **no double clicks ever occur**;
    detection timing is tightly controlled (narrow, offset).

    Signature: QBER collapses to ~0, double-click rate -> 0, timing distribution
    narrows/shifts. The zero double-click rate is the LIMITED-telemetry tell for
    a *naive* blinder; ``dc_match`` lets a stealthier Eve inject fake doubles.
    ``strength`` = fraction of pulses Eve controls.
    """
    name = "blinding"

    def __init__(self, strength: float = 1.0, dc_match: float = 0.0,
                 timing_jitter: float = 0.02, timing_offset: float = 0.0,
                 click_prob: float = 1.0):
        super().__init__(strength)
        self.dc_match = dc_match          # prob of injecting a fake double click
        self.timing_jitter = timing_jitter
        self.timing_offset = timing_offset
        self.click_prob = click_prob      # throttle forced clicks to match gain

    def apply(self, ctx, session, rng) -> None:
        n = ctx.a_bit.shape[0]
        controlled = rng.random(n) < self.strength
        eve_basis = rng.integers(0, 2, n)
        eve_bit = np.where(eve_basis == ctx.a_basis, ctx.a_bit,
                          rng.integers(0, 2, n))
        # Bob clicks (forced) only if his basis matches Eve's faked-state basis,
        # throttled by click_prob so a stealthy Eve can match the honest gain.
        click = (controlled & (ctx.b_basis == eve_basis)
                 & (rng.random(n) < self.click_prob))
        ctx.forced[click] = eve_bit[click].astype(np.int8)
        # Every controlled pulse that is not a forced click is blinded (no click);
        # detectors are held below threshold, so honest physics cannot fire them.
        suppress = controlled & ~click
        ctx.arrived[suppress] = 0
        ctx.forced[suppress] = -1                       # ensure no click
        # Zero intrinsic error on forced clicks; Eve knows them all.
        ctx.eve_knows[click] = True
        # Tight controlled timing on forced clicks (Eve's own resend clock),
        # overriding the physical arrival time -> narrower t_std, a signature.
        ctx.t_override[click] = (self.timing_offset
                                 + rng.normal(0, self.timing_jitter, n)[click])
        # Optional fake double clicks to mimic honest double-click statistics.
        if self.dc_match > 0:
            fake_dbl = click & (rng.random(n) < self.dc_match)
            # represent as forcing detector then flagging via force_error-free
            # double: we emulate by leaving both detectors to fire -> handled in
            # session only for physical clicks, so mark via a sentinel: set
            # forced to 3 meaning "double". session treats forced>=2 specially.
            ctx.forced[fake_dbl] = 3


class TimeShift(Attack):
    """Time-shift attack exploiting detector efficiency mismatch in time.

    Eve shifts each pulse's arrival time so that the detector whose temporal
    acceptance peak it lands nearer to fires preferentially; the winning
    detector (hence the sifted bit) becomes correlated with the (Eve-chosen)
    shift, leaking partial information with little QBER increase.

    Signature: timing-histogram mean shifts, detector click imbalance grows,
    QBER stays low. Caught by LIMITED telemetry (timing mean + imbalance).
    ``strength`` scales the shift magnitude (in units of the detector offset).
    """
    name = "time_shift"

    def apply(self, ctx, session, rng) -> None:
        n = ctx.a_bit.shape[0]
        act = rng.random(n) < 1.0                        # applies to all pulses
        # Eve randomly shifts early/late; she records the sign, which biases
        # which detector clicks and thus the bit she can guess.
        sign = rng.choice([-1.0, 1.0], size=n)
        shift = sign * self.strength * session.t_offset * 2.0
        ctx.time_shift[act] += shift[act]
        # A shift toward a detector's temporal peak biases which detector clicks;
        # Eve, who chose the sign, can guess the resulting bit with excess
        # probability set by that bias. The bimodal shift also broadens the
        # detection-time histogram (t_std / kurtosis) -- the observable trace.
        # p_know grows with shift but stays modest (partial information leak).
        p_know = 0.5 * (1.0 - np.exp(-1.2 * self.strength))
        ctx.eve_knows[act & (rng.random(n) < p_know)] = True


# --------------------------------------------------------------------------- #
# Calibration: tune attack parameters so the attack matches the honest gain,
# i.e. is invisible to overall-gain (LIMITED) telemetry.
# --------------------------------------------------------------------------- #
def _mean_gain(sys, make_attack, rng, n_blocks=6, n_pulses=40000) -> float:
    from .session import Session
    sess = Session(sys, rng)
    g = []
    for _ in range(n_blocks):
        blk = sess.run(n_pulses, attack=make_attack())
        g.append(blk.intensity_idx.size / blk.n_pulses)
    return float(np.mean(g))


def honest_gain(sys, rng, n_blocks=6, n_pulses=40000) -> float:
    return _mean_gain(sys, lambda: None, rng, n_blocks, n_pulses)


def calibrate_pns(sys, strength: float = 1.0, seed: int = 1) -> float:
    """Binary-search the ``restore`` fraction so PNS matches the honest gain."""
    rng = np.random.default_rng(seed)
    target = honest_gain(sys, rng)
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        g = _mean_gain(sys, lambda: PNS(strength, restore=mid), rng)
        if g < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_blinding(sys, strength: float = 1.0, seed: int = 1, **kw) -> float:
    """Binary-search ``click_prob`` so blinding matches the honest gain."""
    rng = np.random.default_rng(seed)
    target = honest_gain(sys, rng)
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        g = _mean_gain(sys, lambda: Blinding(strength, click_prob=mid, **kw), rng)
        if g < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
