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
