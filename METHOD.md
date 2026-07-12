# Method: SCATTER & the DEGENERACY attack

## SCATTER — the framework

**S**equential **C**USUM **A**nalysis of **T**elemetry for **T**hreat-**E**xposure **R**egions.

SCATTER treats telemetry-based QKD intrusion detection as an information-theoretic
sequential-testing problem in which every step is closed-form or theorem-backed:

```
attack parameters θ
     │  ① Monte-Carlo QKD simulator  (source → channel → attack → detector)
     ▼
per-block telemetry vector x ∈ ℝ^d   (sum of ~10^4–10^6 pulses ⇒ CLT ⇒ Gaussian)
     │  ② fit honest law P₀ = N(m₀,S₀),  attacked law P₁(θ) = N(m₁,S₁)
     ▼
detectability  D_T(θ) = KL( P₁ ‖ P₀ )   [closed form; T = telemetry set]
     │  ③ Stein / Lorden bound
     ▼
min blocks to detect  N*_T(θ) = log(1/α) / D_T(θ)   ← operational detection delay
```

Two theorems carry the analysis:

- **Data-processing inequality (DPI).** The LIMITED telemetry set is a
  deterministic sub-vector of FULL, so `D_LIMITED(θ) ≤ D_FULL(θ)` for **every** θ.
  Cheaper telemetry can only lower detectability, hence raise the detection delay.

- **Stein / Lorden floor.** Even an omniscient CUSUM detector that *knows* the
  attack needs `N* ≳ log(1/α)/D` blocks. So `D → 0 ⟹ N* → ∞`: undetectable by any
  detector at any latency. (Validated in `experiments/exp_stein_validation.py`.)

### Telemetry sets (`qkd/telemetry.py`)

- **LIMITED** (7 features, raw clicks only, no decoy analysis): gain, QBER,
  double-click rate, detector imbalance, Z/X basis asymmetry, timing mean & std.
- **FULL** (13): LIMITED + per-intensity decoy residuals + timing skew/kurtosis.

## DEGENERACY attack — the adversary

Eve does not merely evade a fixed detector; she **engineers an observational
degeneracy**. Under the LIMITED telemetry map she tunes her free parameters so
the attacked law P₁(θ) collapses onto the honest law P₀ — the two hypotheses
become statistically degenerate in the observed feature space
(`D_LIMITED → noise floor`) — while she still learns a fraction `I` of the key
from the photon-number side channel.

**Analytic backbone (`qkd/degeneracy.py`).** For gain-matched PNS, Eve applies one
photon-number policy to every pulse (she cannot distinguish signal from decoy):
block vacuum; forward each single-photon pulse loss-free with probability `r`;
forward each multi-photon pulse with probability `m`, keeping a copy she measures.
The observed gain at intensity μ is `Q_E(μ) = Σ_n Poisson(n;μ) y_n`. Matching the
honest signal gain fixes her knob(s); the decoy intensity, having different photon
statistics, then satisfies `Q_E(ν) ≠ Q_H(ν)`. The residual `Δ(ν) = Q_E(ν) − Q_H(ν)`
is the unavoidable decoy signature — present only in FULL telemetry. This turns the
numerical "degeneracy valley" into a proposition with a predictive formula
(analytic `r* = 0.218` matches the Monte-Carlo optimum `0.22`).

### Two attack variants

- **1-knob** (`gain_match_restore`, multi fully forwarded): concentrates its
  signature in a single decoy residual (`res_gain_dec`, ≈0.8σ/block) — the
  pedagogical *degeneracy fingerprint*.
- **2-knob** (`gain_match`, `r = channel transmittance` + throttled multi):
  mimics the honest single-photon yield exactly, matching signal *and* decoy gains
  to leading order — stealthy even in FULL telemetry, the stronger adversary. DPI
  still gives `D_LIMITED ≤ D_FULL`, so cheap telemetry remains strictly worse.

## Security consequence (`qkd/security.py`)

Because the attacked statistics look honest, Alice and Bob run standard
decoy-state privacy amplification and *certify* a secret key at the honest rate,
yet Eve knows a fraction `I` of those bits. Every block until SCATTER alarms
yields certified-but-compromised key:

```
K_stolen(T) = N*_T · I · n_sift · r_cert     [bits]
```

which grows without bound as the telemetry-limited detectability `D_T → 0`.
