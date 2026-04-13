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

---

## c6: MBPO with M_sim + residual δ (real env, online δ refit) → **3738** ✓

First successful residual architecture result!
- M_sim frozen (pretrained on 50k sim), δ refitted online on real data
- M_real = M_sim + δ, RMSE = 0.575 (vs c4 direct M_real = 0.553)
- Residual improvement: 84.6% (3.74 → 0.575)
- Real return: 3738 (+327% vs c1 baseline 875)
- c4 (direct M_real): 6792 → c6 is 55% of c4

Gap c6 vs c4: m_s=0.98 (c6) vs 0.29 (c4) — residual refit quality lower.
Possible fixes: keep δ weights across refits, larger δ net, less frequent full retrain.

---

## c6v2: Warm-start δ + 128×2 + 50 epochs → **4840** ✓

Fixes vs c6: keep δ weights across refits, larger net, more epochs.
- m_s: 0.43→0.65 (vs c6旧 0.53→0.98, vs c4 0.37→0.29)
- Last 3: 4840 (c4的71%, vs c6旧的55%)
- Gap c6v2 vs c4: 1952 (from 3054)

Remaining gap likely from:
1. m_s still higher than c4 (0.55-0.65 vs 0.29-0.44)
2. δ capacity: 128×2 (20k params) vs c4's full model 200×3 (120k params)
3. model_buf=400 too small for meaningful augmentation

---

## c6v3: δ 200-dim + rollout_freq=50 → **572** ✗ (worse than c6v2!)

Overfitting: larger δ (200×2) + frequent warm-start refit = overfit to each batch.
m_s is lower (0.2-0.7) but performance wildly unstable (-433 to 2492).
c6v2 (128×2) at 4840 remains the best residual configuration.

**Conclusion**: δ capacity sweet spot is ~128 hidden. Larger overfits, smaller underfits.
c6v2 = 4840 (71% of c4=6792) is the validated result for residual architecture.

---

## c6 rerun (memory leak fix): **4674** ✓ (confirms c6v2=4840)

Two independent runs: 4840 and 4674. Stable within ~3% seed variance.
Memory leak fix (deque maxlen, batch actions) does not affect convergence.

**Validated residual world model result:**
- M_sim(frozen) + δ(128×2, warm-start, online refit) = ~4750 avg
- vs c4 Direct M_real = 6792 (70% of upper bound)
- vs c1 Raw Sim baseline = 875 (+440% improvement)

---

## c7: SINDy+NAU δ in MBPO pipeline → **5147** ✓ (beats c6 MLP!)

SINDy discovers symbolic structure, NAU/NMU provides OOD bounds.
- SINDy+NAU RMSE: 0.600 (vs MLP 0.575 — close)
- Real return: 5147 (vs c6 MLP=4750, +8%)
- c7/c4 = 76% (vs c6/c4 = 70%)
- L_eff: 136→296 (NAU/NMU Lipschitz constant, live OOD monitoring)
- Discovered terms: x0²:-0.113, x0·x1:-0.606, x0²:1.777 (gravity correction)

**SINDy+NAU is a viable (and better!) replacement for MLP δ.**
Adds interpretability + formal OOD bound at no performance cost.

---

## c7 + self-hypothesis loop: IN PROGRESS

Self-hypothesis loop working correctly:
- Round 1: poly2 (300 features) → diagnosed 17/17 dims with structure
  Findings: autocorr, heteroscedastic, non_normal (kurtosis 14-40)
- Round 2: expanded (+49 features: x²,x³,x·|x|,x×a) → 349 features
  Still 17/17 diagnosed
- Round 3: expanded (+48 more) → 397 features → max rounds reached

**Key discovery: x0_cube (cubic term) found by auto-expansion!**
This is a term that poly2 library couldn't capture.
Δs_2 = -0.547 * x0³, Δs_4 = +0.426 * x0³

Performance at step 15k: real=-65 (vs c7 without loop: 2277 at 15k)
Slower due to 3-round hypothesis loop on every refit (every 1000 steps).
Model accuracy good: m_s=0.267 at step 15k.

---

## c7+loop (fixed): Loop once → warm-start refit → **3999**

- Initial loop: 3 rounds, discovered x1_sq, x1_cube, x1_signmag etc. (386 features)
- NAU improvement: 14.9% over SINDy-only
- Refit: warm-start only (no re-running loop), fast
- Performance: 3999 (vs c7 no loop=5147, vs c6v2 MLP=4674)

Self-hypothesis loop discovery is working (found cubic terms, cross-action terms).
But 386 features makes warm-start refit harder → L_eff grows to 1032 (unstable NMU).
Expansion adds value for interpretability but slightly hurts online performance.

**Summary table:**
| Config | Return | Notes |
|--------|--------|-------|
| c1 Raw Sim | 875 | baseline |
| c6v2 MLP δ (128×2) | 4674 | best MLP |
| c7 SINDy+NAU (no loop) | 5147 | best overall |
| c7+loop (fixed) | 3999 | loop adds features but hurts refit stability |
| c4 Direct M_real | 6792 | upper bound |

---

## c8: LLM Oracle physics features → **4419**

10 physics-informed features (grav_bias, cos_theta, sin_theta, vz, z_vz, etc.)
added on top of 300 poly2 → 310 total features.

Result: 4419 (vs c7 poly2-only=5147, -14%).

Physics features don't help for GravityCheetah: poly2 already covers the relevant
terms (z, vz, z*vz, vz² are all in poly2). cos(θ)≈1 for small angles.
LLM oracle may be more valuable for complex nonlinear gaps (e.g., real robots).

**Final scoreboard:**
| Config | Return | Notes |
|--------|--------|-------|
| c1 Raw Sim | 875 | baseline |
| c6v2 MLP δ | 4674 | best MLP |
| c7 SINDy+NAU | **5147** | **BEST** — poly2 + NAU/NMU |
| c7+loop | 3999 | auto-expand hurts stability |
| c8 LLM oracle | 4419 | physics features redundant with poly2 |
| c4 Direct M_real | 6792 | upper bound |
