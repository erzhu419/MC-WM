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

---

## Attempt 3/8: Confident Residual + Online Refit + Actor Constraint

**Method**: MLP ensemble (K=5, τ=0.91) + confidence-based IW + actor constraint + online refit every 5k steps

**Result**:

| Condition | Last 3 avg (real) |
|-----------|-------------------|
| c1 Raw Sim | 864.2 |
| c2 Confident+Refit+Constraint | 716.3 (-17.1%) |

**Status**: FAILED.

**Diagnostics**:
- Weight stable at 0.85 [min=0.72], 15% reduction — mild, no blowup
- Confidence declining: 0.62 → 0.53 → 0.52 → 0.51 → 0.50 (refit doesn't help)
- Actor constraint (penalty_scale=0.5) restricts exploration → lower returns
- Sim suppressed (1489 vs 3199) but real doesn't benefit

**Why it failed**:
1. Online refit doesn't improve confidence — policy outruns the gap detector
2. Actor constraint punishes exploration of necessary high-return regions
3. 15% weight reduction is weaker than the proven 25% uniform (which gives +17%)

**Accumulated evidence (3 attempts)**:
- Uniform LR reduction: +7-17% consistently
- Gap-dependent IW: 0% or negative (SINDy, MLP, or confidence — all fail)
- Online refit: doesn't close the coverage gap fast enough
- Actor constraint: hurts more than helps

---

## Step 2: Model-Based RL with Residual World Model

### c3: M_sim + residual, frozen model, pure model training → 7.0
### c4 (frozen): Direct M_real, frozen, pure model → 140.2
### c4 (MBPO): Direct M_real, online refit, real env → **6792** ✓
### c5: M_sim + residual, sim env, online δ refit → **-357** ✗

Root cause of c5 failure: env_buf has sim transitions (sim reward/dynamics),
model_buf has M_real transitions (corrected reward/dynamics).
Q-function receives contradictory signals → cannot converge.

c4 works because env_buf = real transitions → consistent with model_buf.

**Core unsolved problem**: how to train in sim env but learn real-dynamics policy,
when env_buf and model_buf have different reward/dynamics distributions.
