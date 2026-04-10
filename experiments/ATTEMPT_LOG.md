# MC-WM Attempt Log

## Evidence Base (from diagnostics)

### What works:
- SINDy gap signal rank correlation ρ=0.80 (reliable detector of WHERE gap is)
- SINDy correction on training distribution: +32% improvement
- GravityCheetah has real gap: c1=880 vs c3=1301 (32% improvement space)

### What doesn't work:
- SINDy magnitude in OOD: explodes (pred/true ratio up to 13x)
- Ensemble disagreement: doesn't correlate with harm (ρ≈0)
- Gate: direction reversed on trained policy
- QΔ Bellman (γ=0.99): becomes constant (CoV=7%)
- Additive Q-target penalty: too small vs reward signal (<20%)
- MixedBuffer with random data: hurts (low quality)

### Key insight:
Penalty must be MULTIPLICATIVE (scale Q-loss weight), not ADDITIVE (subtract from Q-target).
High gap → Q-loss weight ↓ → policy ignores unreliable sim transitions.
This is what H2O+ does with importance weighting.

---

## Pre-attempt results

| Attempt | Method | Env | Result | vs Baseline |
|---------|--------|-----|--------|-------------|
| pre-1 | Single SINDy correction | WindHopper | N/A | gap≈0, wrong env |
| pre-2 | Single SINDy correction | CarpetAnt | 891 | -6.4% |
| pre-3 | Ensemble gate | CarpetAnt | 891 | -6.4% |
| pre-4 | QΔ Bellman | CarpetAnt | 939 | +0.3% |
| pre-5 | Direct gap (CarpetAnt) | CarpetAnt | N/A | gap too small |
| pre-6 | Direct gap (raw) | GravityCheetah | -70 at 15k | Q collapse |
| pre-7 | Direct gap (norm, additive) | GravityCheetah | 115 | -87% |

Baseline: GravityCheetah Raw Sim = 879.7
Upper bound: GravityCheetah Real Online = 1301.1

---

## Attempt 1/8: [PENDING]

### Plan:
TBD — multiplicative importance weighting using SINDy rank signal

### Hypothesis:
TBD

### Result:
TBD
