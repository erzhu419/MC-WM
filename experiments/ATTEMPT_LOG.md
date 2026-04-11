# MC-WM Attempt Log

## Evidence Base

### What works:
- SINDy gap signal rank correlation ρ=0.80
- Multiplicative importance weighting (attempt 1/8)

### What doesn't work:
- SINDy magnitude in OOD (explodes)
- Ensemble disagreement (doesn't correlate with harm)
- Gate (direction reversed)
- QΔ Bellman γ=0.99 (becomes constant)
- Additive Q-target penalty (too small vs reward)
- MixedBuffer with random data (hurts)

---

## Attempt 1/8: Multiplicative Importance Weighting

**Method**: critic_loss = (w * TD_error²).mean()
where w = w_min + (1-w_min)*(1-gap_normalized), w_min=0.1

**Env**: GravityCheetah (2x gravity sim, 1x real)

**Results**:
| Condition | Last 3 avg (real) |
|-----------|-------------------|
| c1 Raw Sim | 702.5 |
| c2 IW (w_min=0.1) | **2129.1 (+203%)** |
| c3 Real Online | 1301.1 |

**Diagnostics**:
- Weight decreased over training: 0.607 → 0.168 (83% reduction at end)
- Q exploded: 1 → 120
- Sim reward exploded: -216 → 4099
- Variance very high (±765 to ±1079)

**Concern**: Is this IW or just reduced learning rate effect?
Need ablation: uniform w=0.17 (same avg weight, no gap signal) vs gap-dependent w.

**Status**: FALSE POSITIVE. Ablation shows improvement is from reduced LR, not gap signal.

### Ablation (same run, different seed):
| Condition | Last 3 avg (real) |
|-----------|-------------------|
| c1 Raw Sim (w=1.0) | 933.0 |
| c4 Uniform w=0.25 | **1002.6 (+7.5%)** |
| c2 Gap-dependent w | 960.7 (+3.0%) |

c4 > c2. Gap signal provides no additional value beyond uniform LR reduction.
c2=2129 in first run was lucky seed (second run: 960.7).

### Root cause:
Gap signal has insufficient spatial variation at training time.
gap std=0.20 in range [0.7, 1.0] — all transitions look "high gap".
Need gap signal with wider dynamic range, or fundamentally different approach.

---

## Attempt 2/8: MLP Ensemble Gap Detector

**Hypothesis**: SINDy poly2 explodes OOD. MLP with tanh has bounded output → better gap signal.

**Result**:
| Condition | Last 3 avg (real) |
|-----------|-------------------|
| c1 Raw Sim | 809.8 |
| c4 Uniform w=0.25 | **950.8 (+17.4%)** |
| c5 MLP gap IW | 766.0 (-5.4%) |

**Status**: FAILED. MLP gap signal also lacks spatial variation (mean=0.64, std=0.15).
MLP weight stabilized at 0.42 (vs SINDy dropping to 0.17) — tanh works for stability.
But spatial discrimination is still zero. c4 uniform wins again.

**Consistent finding across attempts**: uniform LR reduction gives +7-17%. Gap-dependent methods add nothing or hurt.

**Root cause confirmed**: 3000 steps of random policy paired data cannot support ANY gap detector.
All states visited by trained policy are OOD for the gap detector. Gap signal → constant → uniform weight.

**Remaining 6 attempts should address data coverage, not gap detector architecture.**
