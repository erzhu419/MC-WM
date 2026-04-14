# MC-WM: Meta-Cognitive World Model with Full-Tuple Residual Dynamics

## Development Manual v4 — Final

---

## 1. Project Identity

**Title:** *Extrapolatable Residual World Models for Sim-to-Real Transfer via Symbolic Dynamics Discovery and Residual Bellman Equations*

**One-sentence pitch:** We correct sim dynamics with a full-tuple symbolic residual model that extrapolates via NAU/NMU, gate corrections using a Residual Bellman Equation that accounts for multi-step compounding, and employ an LLM as a meta-cognitive module that both discovers hidden structure in residuals and maintains a monotonically growing set of physical and semantic constraints.

**Foundational principle:** *Every new hypothesis about the sim-real gap must satisfy all previously established constraints.* The constraint set only grows, never shrinks — just as in science, new theories must explain everything old theories explained, plus more.

---

## 2. Architecture

```
Phase 0 — Offline Setup (before training)
  LLM generates initial constraint set
  Human reviews → freeze initial constraints

Training Loop:
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  Layer 1: Full-Tuple Residual Extraction                   │
  │    Δ_s, Δ_r, Δ_d from paired (s,a) queries                │
  │                                                            │
  │  Layer 2: Self-Hypothesizing Loop                          │
  │    SINDy + auto-expand (4 mechanisms)                      │
  │    IF stuck → LLM proposes new features   ← LLM role #2   │
  │    NAU/NMU enforces extrapolation                          │
  │                                                            │
  │  Layer 3: Correction + Gating                              │
  │    Q_Δ (Residual Bellman) estimates trajectory-level risk   │
  │    Gate based on Q_Δ                                       │
  │    Constraint filter (monotonically growing set)            │
  │                                                            │
  │  Layer 4: Constraint Augmentation (periodic)               │
  │    Collect suspicious corrections (high Δ, all pass)        │
  │    LLM audits: "is this reasonable?"       ← LLM role #3   │
  │    New constraints added (set only grows)                   │
  │                                                            │
  │  Layer 5: Robust Policy on Augmented Buffer                │
  │    Real data + gated corrected sim data                    │
  │    Pessimistic on high-Q_Δ regions                         │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
```

### 2.1 The Three Roles of LLM

| Role | When | What | Frequency |
|---|---|---|---|
| #1 Constraint initialization | Phase 0, before training | Generate physical + semantic constraints from env description | Once |
| #2 Feature hypothesis | Training, inside self-hypothesizing loop | Propose new features when 4 auto-mechanisms fail | Rare (cumulative 20-60 calls) |
| #3 Constraint augmentation | Training, every N epochs | Audit suspicious corrections, generate new constraints | Periodic (cumulative 10-30 calls) |

Role #1 answers: "What is impossible in this world?"
Role #2 answers: "What hidden pattern explains this residual?"
Role #3 answers: "This passed all my rules but looks wrong — what rule am I missing?"

### 2.2 Component Count: 7

| # | Component | Remove it and... |
|---|---|---|
| 1 | Full-tuple residual extraction | No correction at all |
| 2 | SINDy + auto-expansion + diagnosis | No symbolic structure, can't use NAU/NMU |
| 3 | NAU/NMU output head | OOD bound doesn't hold (Thm 4.12) |
| 4 | Residual Bellman Q_Δ | Gate ignores trajectory-level compounding |
| 5 | Constraint set (monotonically growing) | Physically impossible corrections pass through |
| 6 | LLM (3 roles) | Initial constraints hand-coded only; no feature hypothesis fallback; no constraint augmentation |
| 7 | Robust policy on augmented buffer | No policy |

---

## 3. Full-Tuple Residual

### 3.1 Definition

$$\Delta(s,a) = \begin{pmatrix} s'_{\text{real}} - s'_{\text{sim}} \\ r_{\text{real}} - r_{\text{sim}} \\ d_{\text{real}} - d_{\text{sim}} \end{pmatrix}$$

Each element gets its own SINDy model, its own NAU/NMU head, its own Q_Δ component.

### 3.2 Why Every Element

State-only correction misses: wrong rewards (→ wrong value estimates), wrong termination (→ wrong horizon), wrong noise level (→ false confidence). Example: Ice-Walker with soft carpet — state dynamics change AND fall detection threshold changes AND motor energy cost changes. Correcting only state still learns a wrong policy.

### 3.3 Dimension Mismatch

If sim has 17 dims and real has 15 dims: LLM Role #1 generates dimension mapping rules in Phase 0. Either drop sim-only dims or infer missing real dims from physical coupling. Mapping frozen before training.

---

## 4. Self-Hypothesizing Loop

### 4.1 The Loop

```
FOR round = 1 to max_rounds:

    HYPOTHESIZE: Fit SINDy on current basis library
    TEST: Quality gate on holdout (derivative error < ε_threshold)
    IF PASS → accept symbolic model, done

    FALSIFY: Run diagnosis battery on remainder
    IF all tests negative → remainder is genuine aleatoric, done

    EXPAND (automated, 4 mechanisms):
        Autocorrelation positive → add time-delay features (Takens)
        Heteroscedasticity positive → add nonlinear terms of culprit variable
        Heavy tails → add piecewise contact masks
        Non-stationarity → add time features

    IF auto-expand added new features → CONTINUE to next round

    EXPAND (LLM Role #2):                          ← only if auto-expand exhausted
        Send diagnosis report to LLM
        LLM proposes candidate features
        Safe-evaluate via ASTEval
        Add valid features to SINDy library
        IF new SINDy fit passes quality gate → CONTINUE
        ELSE → accept current best model, done
```

### 4.2 Four Automated Expansion Mechanisms

**Mechanism 1 — Takens Time-Delay Embedding**
- Trigger: Autocorrelation positive at lag k
- Action: Add $s_{t-1}, s_{t-2}, (s_t - s_{t-1})/dt$
- Handles: Hidden wind, motor delay, load variation

**Mechanism 2 — Algebraic Feature Crossing**
- Trigger: Heteroscedasticity positive, culprit = variable $j$
- Action: Add $s_j^2, s_j|s_j|, s_j^3, s_j \cdot a_k$
- Handles: Quadratic drag, nonlinear friction

**Mechanism 3 — Piecewise Logical Masking**
- Trigger: Heavy-tailed residuals (kurtosis > 4)
- Action: k-means(k=2) on |residual|, add $\mathbb{1}(s_j < \text{threshold})$
- Handles: Contact/flight mode switches

**Mechanism 4 — Temporal Features**
- Trigger: Stationarity test positive
- Action: Add normalized step index, cumulative quantities
- Handles: Time-varying gap (wear, temperature drift)

### 4.3 LLM Feature Hypothesis (Role #2)

Activated only after all 4 automated mechanisms fail to improve quality gate.

```
System: You are a physics feature engineer.
Observables: {obs_keys}. History: obs_prev. Actions: {action_keys}.
RULES: Only use existing variables + numpy math. No invented sensors.

User: Residual dim {name}. Diagnosis: {report}.
Already tried: {list of failed auto-expanded features}.
What feature am I missing?
```

LLM-proposed features go through the same quality gate as auto-expanded features. No special treatment.

### 4.4 Output

1. **Symbolic residual model** per tuple element → NAU/NMU head
2. **Aleatoric remainder** (confirmed structureless by diagnosis) → robust RL uncertainty set

---

## 5. NAU/NMU Extrapolation Head

SINDy discovers the symbolic form. NAU/NMU enforces it in a differentiable network with guaranteed extrapolation:

$$\|\hat{\Delta}(s_{\text{ood}},a) - \Delta_{\text{true}}(s_{\text{ood}},a)\| \leq \epsilon + \varepsilon\|d\| + \frac{L}{2}\|d\|^2$$

- NAU: $L = 0$ (linear extrapolation, tightest bound)
- NMU: $L = 2|c|$ (quadratic extrapolation)
- ReLU: $L = \infty$ (no bound exists — Theorem 4.12)

---

## 6. Residual Bellman Equation

### 6.1 The Problem

Per-transition correction ignores compounding. Error at step $t$ shifts state at step $t+1$, which shifts correction at $t+1$, accumulating across the trajectory. This is the credit assignment problem applied to correction quality.

### 6.2 Definition

$$Q_\Delta(s,a) = \underbrace{\|\hat{\Delta}(s,a) - \Delta_{\text{true}}(s,a)\|^2}_{\text{current correction error}} + \gamma \cdot \mathbb{E}_{a' \sim \pi}\left[Q_\Delta(s'_{\text{corrected}}, a')\right]$$

"Reward" = correction error. Bellman recursion propagates future compounding backward to current state.

### 6.3 Gate

$$g(s,a) = \sigma\left(\frac{\tau - Q_\Delta(s,a)}{\text{temperature}}\right)$$

Low Q_Δ → corrections reliable over trajectory → gate open.
High Q_Δ → corrections will compound badly → gate closed → fall back to raw sim.

### 6.4 Training

On offline paired data, ground-truth correction error is known. Use fitted Q iteration with pessimistic estimation when Q_Δ at corrected next state is uncertain.

---

## 7. Constraint System

### 7.1 The Monotonic Growth Principle

The constraint set $\mathcal{C}$ only grows:

$$\mathcal{C}_0 \subseteq \mathcal{C}_1 \subseteq \mathcal{C}_2 \subseteq \ldots$$

New constraints are added. Old constraints are never removed. This mirrors the structure of scientific knowledge: new theories must satisfy all previously established laws.

Any correction that violates any constraint in $\mathcal{C}$ is rejected. As $\mathcal{C}$ grows, the feasible space of corrections shrinks, becoming more precise.

### 7.2 Phase 0: Initial Constraint Generation (LLM Role #1)

LLM generates constraints from environment description. Three types:

**Physical possibility:**
```json
{"name": "joint_limit", "expr": "abs(obs_next['3']) > 3.14"}
{"name": "underground", "expr": "obs_next['1'] < -0.3"}
```

**Semantic reasonableness:**
```json
{"name": "robot_flying", "expr": "obs_next['1'] > 2.0"}
{"name": "impossible_acceleration", "expr": "abs(obs_next['9'] - obs['9']) / dt > 100"}
```

**Dimension mapping (if sim ≠ real dims):**
```json
{"type": "drop", "dims": [15, 16], "reason": "sim-internal variables"}
```

Human reviews. False rejection on real data < 1%. Freeze as $\mathcal{C}_0$.

### 7.3 Training-Time: Constraint Augmentation (LLM Role #3)

**Why Phase 0 is not enough:** Phase 0 constraints are generated from the environment description alone, without seeing actual data. LLM cannot anticipate every edge case. When training starts and sim explores diverse states, novel situations arise that Phase 0 didn't cover.

**Trigger:** Every N epochs, collect "suspicious corrections":
- All existing constraints $\mathcal{C}_k$ pass (not caught by current rules)
- AND correction magnitude > 2× median (unusually large)
- AND Q_Δ in uncertain zone (not clearly safe, not clearly dangerous)

**Process:**
```
Batch suspicious corrections → LLM:
  "These 10 corrected transitions passed all my constraints.
   Are any of them physically impossible or semantically unreasonable?
   For each problematic one, give me a new constraint rule."

LLM responds:
  "Transition 3: robot is inverted (rootangle = 2.8 rad).
   New constraint: obs_next['rootangle'] > 2.5 or obs_next['rootangle'] < -2.5"

  "Transition 7: forward velocity 45 m/s is unreasonable for this robot.
   New constraint: abs(obs_next['rootx_vel']) > 30"

New constraints → human can review (optional, async) → add to C_{k+1}
```

**Key property:** Existing corrections already in the augmented buffer are NOT retroactively removed. New constraints only apply to future corrections. This prevents the buffer from shrinking unpredictably during training.

### 7.4 Runtime Application

```python
def passes_constraints(s, a, s_corrected, constraint_set):
    for c in constraint_set:
        if eval_constraint(c, s, a, s_corrected):
            return False  # violated → reject
    return True  # all passed → accept
```

### 7.5 Why Not Discriminator

| | Discriminator | LLM constraints |
|---|---|---|
| Judges | "statistically normal?" | "physically possible? semantically reasonable?" |
| Inverted robot with valid joint angles | Might pass (statistically plausible) | Fails ("robot shouldn't be inverted") |
| Needs | Training data (OOD = unreliable) | World knowledge (OOD = still reliable) |
| Interpretable | No | Yes (boolean rules, auditable) |
| Grows over time | No (fixed after training) | Yes (monotonically augmented) |

---

## 8. Policy Learning

### 8.1 Augmented Buffer Construction

```
For each sim transition (s, a):
    1. Compute symbolic correction: Δ̂_s, Δ̂_r, Δ̂_d
    2. Apply correction: s' = s'_sim + Δ̂_s, r' = r_sim + Δ̂_r
    3. Check constraints: if any violated → reject, skip
    4. Compute Q_Δ gate: g = σ((τ - Q_Δ(s,a)) / temp)
    5. Final: s' = s'_sim + g·Δ̂_s, r' = r_sim + g·Δ̂_r
    6. If g > threshold → add to buffer with confidence = g
```

All real data added with confidence = 1.0.

### 8.2 Robust Policy

IQL/CalQL with confidence-weighted critic loss:
- High confidence (g ≈ 1): standard loss
- Low confidence (g ≈ 0): pessimistic penalty proportional to aleatoric variance

---

## 9. Theoretical Guarantees

| Theorem | Statement | Lean 4 est. |
|---|---|---|
| Per-element OOD bound | $\|\hat{\Delta}_e - \Delta_e^{\text{true}}\| \leq \epsilon + \varepsilon\|d\| + \frac{L}{2}\|d\|^2$ | Inherited from CS-BAPR |
| Q_Δ contraction | Under frozen residual model, Q_Δ operator is γ-contraction | ~80 lines, novel |
| Gate safety | g → 0 when Q_Δ large → fallback to raw sim, never worse | ~40 lines |
| Monotonic improvement | Each quality-gate-passing round of hypothesis loop tightens ε | ~100 lines |
| Constraint tightening | diam(C) < OOD_bound → constraint wins | ~40 lines |
| Constraint monotonicity | C_k ⊆ C_{k+1} → feasible correction set only shrinks → safety only improves | ~20 lines, trivial |

---

## 10. Experiments

### 10.1 HP-MuJoCo Benchmark

| Environment | State Gap | Reward Gap | Termination Gap |
|---|---|---|---|
| Aero-Cheetah | Quadratic drag $-kv^2$ | Energy cost | — |
| Ice-Walker | Friction drop at $x>5$ | Velocity scaling | Softer fall threshold |
| Wind-Hopper | Sinusoidal side wind | — | Wind-induced falls |
| Carpet-Ant | Damped contacts | Motor current | Soft falls |

### 10.2 Baselines

H2O+ (IS), DARC (reward augmentation), IGDF (contrastive filtering), ReDRAW (NN residual).

### 10.3 Key Experiments

1. **Full-tuple vs state-only:** Ice-Walker — wrong termination kills state-only agent
2. **OOD extrapolation:** 1x→8x speed — ours polynomial, ReDRAW explodes
3. **Q_Δ vs distance gate:** Q_Δ opens "far but well-modeled," closes "near but dangerous"
4. **Self-hypothesizing ablation:** per-round improvement, auto-expand vs +LLM
5. **Constraint augmentation:** show new constraints catch cases Phase 0 missed
6. **LLM feature hypothesis:** show LLM discovers cross-dimensional features auto-expand can't

---

## 11. Implementation Roadmap

| Week | Phase | Milestone |
|---|---|---|
| 0 | Install + LLM initial constraints | constraints_approved.json, <1% false rejection |
| 1 | Full-tuple residual extraction | Paired residuals computed, debug plots confirm |
| 2 | SINDy + quality gate | Finds $v^2$, gate passes |
| 3 | Diagnosis + auto-expansion | Self-hypothesizing resolves without LLM |
| 4 | NAU/NMU + OOD test | Polynomial error at 2x/4x/8x |
| 5 | Q_Δ training | Q_Δ far > Q_Δ near, gate behavior correct |
| 6 | Augmented buffer + policy | Corrected > raw sim |
| 7 | Constraint augmentation during training | New constraints caught ≥1 case Phase 0 missed |
| 8-9 | All envs + baselines + ablations | Tables and figures |
| 10 | Lean 4 proofs + paper | Submission draft |

---

## 12. Contributions

1. **Full-Tuple Residual World Model.** Corrects state, reward, termination — not just state.
2. **Self-Hypothesizing Symbolic Discovery.** Automated diagnosis → expansion → SINDy, with LLM fallback for features beyond automated mechanisms' reach.
3. **Residual Bellman Equation.** Trajectory-aware correction quality. Replaces ad-hoc gates.
4. **Extrapolatable Residuals (NAU/NMU).** Formal OOD polynomial bounds. Addresses ReDRAW's limitation.
5. **Monotonically Growing Constraint Set.** LLM generates initial constraints and augments them during training. New hypotheses satisfy all past constraints.

---

## 13. Limitations

- **set_state() assumption:** Sim must be queryable at arbitrary (s,a).
- **SINDy expressiveness:** Gap must be sparse in some basis. Covers parameter mismatches; not entire engine differences.
- **Q_Δ bootstrapping:** Pessimistic estimation mitigates but doesn't eliminate compounding in far-OOD regions.
- **LLM constraint quality:** Depends on LLM's world knowledge of the specific domain. Mitigated by human review.

---

## 14. Code Structure

```
mc-wm/
├── envs/hp_mujoco/
├── residual/
│   ├── extractor.py             # Full-tuple paired residual
│   ├── sindy_track.py           # SINDy + auto-expansion
│   ├── nau_nmu_head.py          # NAU/NMU output layer
│   └── residual_bellman.py      # Q_Δ training and inference
├── self_audit/
│   ├── diagnosis.py             # Statistical diagnosis battery
│   ├── auto_expand.py           # 4 automated expansion mechanisms
│   ├── hypothesis_loop.py       # Core loop (includes LLM Role #2 fallback)
│   └── constraint_augmentor.py  # LLM Role #3: periodic constraint audit
├── constraints/
│   ├── llm_generator.py         # LLM Role #1: Phase 0 constraint generation
│   ├── constraint_filter.py     # Runtime evaluation of constraint set
│   └── constraints.json         # Monotonically growing constraint set
├── policy/
│   ├── augmented_buffer.py
│   └── iql_robust.py
├── theory/lean4/
│   ├── ResidualBellman.lean
│   ├── GateSafety.lean
│   ├── FullTupleOOD.lean
│   └── ConstraintMonotonicity.lean
└── experiments/
```

---

## 15. Dependencies

| Library | Purpose | Required |
|---|---|---|
| CORL | Offline RL | Yes |
| PySINDy | Symbolic regression | Yes |
| stable-nalu | NAU/NMU | Yes |
| statsmodels | Diagnosis | Yes |
| asteval | Safe eval | Yes |
| anthropic/openai | LLM (Phase 0 + training-time roles #2 and #3) | Yes |
