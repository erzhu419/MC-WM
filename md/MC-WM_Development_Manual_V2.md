# MC-WM: Meta-Cognitive World Model with Full-Tuple Residual Dynamics

## Complete Development Manual (v2 — Occam's Razor Edition)

---

## 1. Project Identity

**Working Title:** *Extrapolatable Residual World Models for Sim-to-Real Transfer via Symbolic Dynamics Discovery*

**One-sentence pitch:** We correct sim dynamics with a full-tuple residual model whose symbolic structure is automatically discovered via SINDy, extrapolated via NAU/NMU with formal OOD bounds, and trusted via uncertainty-gated correction — forming a self-auditing agent that iteratively hypothesizes, tests, and falsifies its own understanding of the sim-real gap.

**Core philosophical claim:** Existing sim-to-real methods ask "how different is this data?" We ask "why is it different, in every dimension of the transition tuple?" — and once you know why, you can extrapolate.

---

## 2. Architecture Overview: The Self-Hypothesizing Agent

### 2.1 Design Principle

The system is an **autonomous hypothesis-falsification loop**:

```
┌──────────────────────────────────────────────────────────┐
│                  SELF-AUDITING AGENT                      │
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐           │
│  │ Hypothe-│───→│ SINDy    │───→│ Quality   │──→ PASS   │
│  │ size    │    │ Fit      │    │ Gate      │   → Accept │
│  └────┬────┘    └──────────┘    └─────┬─────┘           │
│       │                               │ FAIL             │
│       │         ┌──────────┐          │                  │
│       │←────────│ Diagnose │←─────────┘                  │
│       │         │ Residual │   (what structure            │
│       │         └──────────┘    was missed?)              │
│       │                                                  │
│       ▼ expand basis library automatically               │
│  ┌─────────┐                                            │
│  │ Auto    │  algebraic crossings, time-delay,           │
│  │ Expand  │  piecewise masks, higher-order terms        │
│  └────┬────┘                                            │
│       │ still stuck after K rounds?                      │
│       ▼                                                  │
│  ┌─────────────────────┐                                │
│  │ 🆘 Consult LLM     │  ← optional, last resort       │
│  │ (external oracle)   │                                │
│  └─────────────────────┘                                │
└──────────────────────────────────────────────────────────┘
```

**The agent does NOT need LLM to function.** The LLM is a fallback oracle consulted only when the automated expansion mechanisms are exhausted. The core loop — hypothesize, fit, test, expand — runs entirely without external intelligence.

### 2.2 Minimal Necessary Architecture (6 modules, 0 optional)

```
Full-Tuple Residual Extraction
    → SINDy with Auto-Expanding Basis Library
        → NAU/NMU Output Head
            → Uncertainty Gate
                → Augmented Buffer
                    → Robust Policy Learning
```

Every module is load-bearing. Remove any one and a specific capability breaks:

| Module | Remove it and... |
|---|---|
| Full-tuple residual | No sim-real correction at all |
| SINDy | No symbolic structure, can't use NAU/NMU |
| NAU/NMU | No OOD extrapolation guarantee (Thm 4.12) |
| Gate | Wrong corrections worse than no correction |
| Augmented buffer | Can't combine real + corrected sim data |
| Robust policy | No handling of irreducible uncertainty |

### 2.3 Optional Extension: LLM as External Oracle

**When activated:** Only after the automated basis expansion loop has run K rounds (default K=3) without passing the quality gate.

**What it does:** Reads the statistical diagnosis report, proposes candidate features that require conceptual leaps the automated system cannot make (e.g., "this looks like Coulomb friction, try a sign function" or "use Takens delay embedding").

**What it does NOT do:** It does not participate in the core training loop, does not touch the Q-function, does not influence the gate, and does not need to be called for the system to work.

---

## 3. The Full-Tuple Residual

### 3.1 Why Not Just States

A complete MDP transition is $(s, a, r, s', d)$. The sim-real gap exists in **every element**:

| Tuple Element | Gap Type | Example |
|---|---|---|
| $s' - s'_{\text{sim}}$ | Dynamics gap | Friction, gravity, contact model |
| $r - r_{\text{sim}}$ | Reward gap | Sensor-dependent rewards, energy costs |
| $d - d_{\text{sim}}$ | Termination gap | Different failure thresholds |
| $\text{Var}(s') - \text{Var}(s'_{\text{sim}})$ | Stochasticity gap | Deterministic sim vs noisy real |

### 3.2 Formal Definition

$$\Delta(s,a) = \begin{pmatrix} \Delta_s(s,a) \\ \Delta_r(s,a) \\ \Delta_d(s,a) \end{pmatrix} = \begin{pmatrix} s'_{\text{real}} - s'_{\text{sim}} \\ r_{\text{real}} - r_{\text{sim}} \\ d_{\text{real}} - d_{\text{sim}} \end{pmatrix}$$

Each element gets its own SINDy model, its own NAU/NMU head, its own gate. They are independent pipelines sharing the same (s,a) input.

---

## 4. The Self-Hypothesizing Loop (Core Contribution)

### 4.1 Loop Structure

This is the heart of the system. It is a **scientific method implemented as an algorithm**:

```
Input: raw residuals Δ(s,a) for each tuple element
Output: structured residual model Δ̂(s,a) + aleatoric remainder

Initialize: basis_library = PolynomialLibrary(degree=2)

FOR round = 1 to max_rounds:

    STEP 1 — HYPOTHESIZE
    Fit SINDy on current basis_library:
        Δ̂ = SINDy(X, Δ, library=basis_library)

    STEP 2 — TEST (Quality Gate)
    Compute holdout derivative error ε₁
    IF ε₁ < ε_threshold:
        ACCEPT: this round's SINDy model explains the residual
        BREAK

    STEP 3 — FALSIFY (Statistical Diagnosis)
    Compute remainder: Δ_remainder = Δ - Δ̂
    Run diagnosis battery on Δ_remainder:
        - Autocorrelation test → temporal structure?
        - Heteroscedasticity test → state-dependent variance?
        - Normality test → heavy tails / mode switches?
        - Stationarity test → drifting dynamics?

    IF all tests pass (remainder is pure white noise):
        ACCEPT: SINDy captured all learnable structure
        Classify remainder as true aleatoric
        BREAK

    STEP 4 — EXPAND (Automated, no LLM needed)
    Based on which diagnosis fired:
        Autocorrelation positive →
            Add time-delay features: s(t-1), s(t-2), Δs/Δt
        Heteroscedasticity positive (culprit = variable j) →
            Add nonlinear crossings: s_j², s_j³, s_j*|s_j|, s_j*a
        Heavy-tailed / non-normal →
            Add piecewise masks: 1(s_j < threshold) for likely contact
        Non-stationary →
            Add trajectory-position features: cumulative step count

    Append new features to basis_library
    CONTINUE to next round

IF round == max_rounds AND quality gate still fails:
    OPTIONAL: consult LLM oracle with diagnosis report
    IF LLM proposes valid features (pass ASTEval sandbox):
        Add to basis_library, run one more SINDy round
    ELSE:
        Accept current best model, classify remainder as aleatoric
```

### 4.2 Why This Is Not Just "Bigger Basis Library"

A naive approach would be: start with degree=5 polynomial + sin + cos + exp + everything. This fails because:

1. **Combinatorial explosion:** With 17 state dims + 6 action dims, degree=5 polynomials alone create ~300,000 candidate terms. SINDy's L1 regression chokes.
2. **Time-delay and piecewise features are not polynomials.** No polynomial degree captures $s(t) - s(t-1)$ or $\mathbb{1}(z < 0.05)$.
3. **The diagnosis tells you WHERE to look.** Instead of blindly adding everything, autocorrelation tells you "add time delays," heteroscedasticity tells you "add nonlinear terms of variable j specifically."

The diagnosis-guided expansion is **targeted search**, not exhaustive search. This is what makes it tractable.

### 4.3 The Four Automated Expansion Mechanisms

These are the agent's built-in "hypotheses" that do not require LLM:

**Mechanism 1: Time-Delay Embedding (Takens' Theorem)**
- Trigger: autocorrelation test positive
- Action: add $s_{t-1}, s_{t-2}, (s_t - s_{t-1})/\Delta t$ to basis library
- Justification: any coupled hidden variable (wind, latency) is encoded in observed history
- Implementation: requires storing 2-step history in residual buffer

**Mechanism 2: Algebraic Feature Crossing**
- Trigger: heteroscedasticity test positive, culprit = variable $j$
- Action: add $s_j^2, s_j^3, s_j \cdot |s_j|, s_j \cdot a_k$ to basis library
- Justification: if variance depends on $s_j$, the mean likely has a nonlinear $s_j$ dependence too
- Implementation: pure algebraic computation on existing features

**Mechanism 3: Piecewise Logical Masking**
- Trigger: normality test positive (heavy tails, kurtosis > 4)
- Action: find threshold via residual magnitude clustering, add $\mathbb{1}(s_j < \text{threshold})$ to basis library
- Justification: bimodal residuals suggest discrete physical mode switches (contact/flight)
- Implementation: k-means (k=2) on |residual|, use cluster boundary as threshold

**Mechanism 4: Trajectory Position Features**
- Trigger: stationarity test positive (mean drift)
- Action: add normalized step index $t/T$, cumulative features
- Justification: sim-real gap may depend on episode progress (e.g., temperature drift)
- Implementation: store step index in residual buffer

### 4.4 When and How LLM Enters (Optional)

**Precondition:** All 4 automated mechanisms tried, quality gate still fails after K=3 rounds.

**What LLM receives:**
```
System: You are a physics feature engineer. [constraints...]
User: After 3 rounds of automated expansion, residual on dim vx still
shows structure. Tests: autocorrelation NEGATIVE, heteroscedasticity
POSITIVE (culprit: vx, but adding vx² and vx³ didn't help),
normality NEGATIVE. Current library includes: [1, vx, vx², vx³,
vx*|vx|, a0, a0*vx, vx(t-1)].
What feature am I missing?
```

**What LLM might return:**
```json
["obs['vx'] * obs['vz']", "obs['theta'] * obs['vx']**2"]
```

(Cross-dimensional interactions that automated mechanisms don't try, because they require physical intuition about which dimensions are coupled.)

**Safety:** ASTEval sandbox, max 3 features per query, SINDy quality gate still must pass.

---

## 5. Residual Model Architecture

### 5.1 Two-Track Design

```
Δ(s,a) ──┬── Track A: SINDy + NAU/NMU (symbolic, extrapolatable)
          │   Fed by self-hypothesizing loop
          │   → Δ_symbolic(s,a)
          │   → has formal OOD bound
          │
          └── Track B: Ensemble NN (flexible, interpolation only)
              Catches what SINDy misses
              → Δ_neural(s,a)
              → no OOD guarantee, gated aggressively
```

Track A is the primary model. Track B is the safety net for whatever Track A's expanded library still cannot capture. Track B uses a more aggressive (lower) gate threshold than Track A, because its extrapolation is unreliable.

### 5.2 NAU/NMU Output Head

SINDy discovers the symbolic form. NAU/NMU enforces it in a differentiable network:

```python
class SymbolicResidualHead(nn.Module):
    def __init__(self, feature_dim, output_dim, alpha=0.5):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.LeakyReLU(),
            nn.Linear(64, 32)
        )
        self.nau = NeuralAdditionUnit(32, output_dim)    # L=0
        self.nmu = NeuralMultiplicationUnit(32, output_dim)  # L=2|c|
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        h = self.feature_net(x)
        return self.alpha * self.nau(h) + (1 - self.alpha) * self.nmu(h)
```

**OOD guarantee (CS-BAPR Theorem 4.35):**
$$\|\hat{\Delta}(s_{\text{ood}},a) - \Delta_{\text{true}}(s_{\text{ood}},a)\| \leq \epsilon + \varepsilon\|d\| + \frac{L}{2}\|d\|^2$$

ReLU provably cannot satisfy this (Theorem 4.12). This is the hard architectural advantage.

---

## 6. Uncertainty-Gated Correction

### 6.1 Per-Element Gating

Each tuple element has its own gate:

$$s'_{\text{corrected}} = s'_{\text{sim}} + g_s(s,a) \cdot \hat{\Delta}_s(s,a)$$
$$r_{\text{corrected}} = r_{\text{sim}} + g_r(s,a) \cdot \hat{\Delta}_r(s,a)$$
$$d_{\text{corrected}} = \text{clip}(d_{\text{sim}} + g_d(s,a) \cdot \hat{\Delta}_d(s,a),\; 0,\; 1)$$

### 6.2 Gate Design

**Track A gate (symbolic, has OOD bound):**
$$g_A(s,a) = \max\left(0,\; 1 - \frac{\epsilon + \varepsilon\|d\| + \frac{L}{2}\|d\|^2}{\tau}\right)$$

**Track B gate (ensemble, no OOD bound):**
$$g_B(s,a) = \sigma\left(\frac{\tau_B - \text{ensemble\_disagreement}(s,a)}{\text{temperature}}\right)$$

Track B gate decays faster (lower $\tau_B$) because ensemble extrapolation is untrustworthy.

### 6.3 Safety Guarantee

**Proposition:** With calibrated $\tau$, gated correction is never worse than uncorrected sim:
$$\|T^{\text{gated}} - T^{\text{real}}\| \leq \|T^{\text{sim}} - T^{\text{real}}\|$$

This holds because $g \to 0$ as uncertainty grows, gracefully falling back to raw sim.

---

## 7. Policy Learning

### 7.1 Augmented Buffer Construction

```python
buffer = []

# All real data: full trust
for (s, a, r, s_next, d) in D_real:
    buffer.append((s, a, r, s_next, d, confidence=1.0))

# Sim data: gated correction
for (s, a) in sim_exploration:
    s_next_sim, r_sim, d_sim = sim.step(s, a)
    s_corr, r_corr, d_corr, gates = gated_correct(s, a, s_next_sim, r_sim, d_sim)
    conf = min(gates)
    if conf > min_threshold:
        buffer.append((s, a, r_corr, s_corr, d_corr, confidence=conf))
```

### 7.2 Robust Offline RL

Use IQL or CalQL from CORL, with confidence-weighted critic loss:

```python
critic_loss = confidence * expectile_loss(Q_target - Q_pred, tau) \
            + (1 - confidence) * lambda_robust * max_penalty
```

High-confidence transitions train normally. Low-confidence transitions push the value function toward pessimism, exactly as robust RL prescribes.

---

## 8. Theoretical Guarantees

### 8.1 Full-Tuple OOD Bound

For each tuple element $e \in \{s, r, d\}$ with SINDy accuracy $\epsilon_e$, Jacobian consistency $\varepsilon_e$, and NAU/NMU Lipschitz constant $L_e$:

$$\|\hat{\Delta}_e(s_{\text{ood}},a) - \Delta_e^{\text{true}}(s_{\text{ood}},a)\| \leq \epsilon_e + \varepsilon_e\|d\| + \frac{L_e}{2}\|d\|^2$$

### 8.2 Value Function Bound

$$|V^{\text{corrected}}(s) - V^{\text{real}}(s)| \leq \frac{1}{1-\gamma}\left(\Delta_r^{\text{bound}} + \gamma\|\nabla_s V\|\Delta_s^{\text{bound}} + \gamma V_{\max}\Delta_d^{\text{bound}}\right)$$

### 8.3 Gate Safety

Gated correction is Pareto-safe: it never degrades performance below raw sim baseline.

### 8.4 Monotonic Improvement of Self-Hypothesizing Loop

Each round that passes the quality gate strictly shrinks the aleatoric uncertainty set, which weakly improves the learned policy.

---

## 9. Experimental Plan

### 9.1 HP-MuJoCo Benchmark (Multi-Element Gaps)

| Environment | State Gap | Reward Gap | Termination Gap |
|---|---|---|---|
| Aero-Cheetah | Quadratic drag $-kv^2$ | Energy cost differs | — |
| Ice-Walker | Friction drop at $x>5$ | Velocity reward scaling | Softer fall threshold |
| Wind-Hopper | Sinusoidal side wind | — | Wind-induced falls |
| Carpet-Ant | Damped contacts | Motor current penalty | Soft falls don't terminate |

### 9.2 Baselines

| Method | Corrects | How | OOD guarantee |
|---|---|---|---|
| H2O+ | Nothing | IS reweighting | No |
| DARC | Reward only | Reward augmentation | No |
| IGDF | Nothing | Contrastive filtering | No |
| ReDRAW | State (latent) | NN residual | No |
| **Ours (state-only)** | State | SINDy+NAU/NMU | Yes |
| **Ours (full-tuple)** | All elements | SINDy+NAU/NMU | Yes |

### 9.3 Key Experiments

**Exp 1: Full-tuple vs state-only.** On Ice-Walker (friction + termination gap): state-only corrects dynamics but agent still dies from wrong termination. Full-tuple fixes both.

**Exp 2: OOD extrapolation.** Train at 1x speed, test at 2x/4x/8x. ReDRAW explodes, ours grows polynomially.

**Exp 3: Self-hypothesizing loop ablation.** Round-by-round: base SINDy → +auto-expansion → (+LLM optional). Show monotonic improvement per round.

**Exp 4: Gate ablation.** No gate vs binary gate vs continuous gate. Show continuous gate is strictly best.

---

## 10. Implementation Roadmap (10 weeks)

| Week | Phase | Milestone |
|---|---|---|
| 1 | Env + paired residual extraction | `debug_residuals.png` shows expected structure |
| 2 | SINDy on single dim | SINDy finds $v^2$ term, quality gate passes |
| 3 | Statistical diagnosis module | Dim 8 reports heteroscedasticity, dim 0 clean |
| 4 | Self-hypothesizing loop (no LLM) | Auto-expansion finds $v^2$ from diagnosis alone |
| 5 | Gate + augmented buffer | Gated policy > raw sim policy on 1 env |
| 6 | Full-tuple extension | Reward + termination correction working |
| 7-8 | All 4 envs + all baselines | Complete comparison tables |
| 9 | OOD + ablation experiments | Extrapolation curves match theory |
| 10 | Paper writing | Submission-ready draft |

---

## 11. Paper Contribution Summary

1. **Full-Tuple Residual World Model.** First to model sim-real gap across all transition elements, not just state dynamics.

2. **Extrapolatable Residuals.** First residual world model with formal OOD polynomial bounds, via SINDy + NAU/NMU inheriting CS-BAPR guarantees. Addresses ReDRAW's fundamental OOD limitation.

3. **Self-Hypothesizing Agent.** Automated diagnosis-guided basis expansion that iteratively discovers residual structure without external intelligence. The agent proposes, tests, and falsifies its own hypotheses about the sim-real gap.

4. **Uncertainty-Gated Correction.** Per-element gating that provably never makes corrections worse than raw sim.

5. **(Extension) LLM as External Oracle.** When automated expansion is insufficient, LLM provides cross-dimensional physical intuition. Presented as optional extension, not core contribution.

---

## 12. Code Structure

```
mc-wm/
├── envs/hp_mujoco/              # 4 benchmark environments
├── residual/
│   ├── extractor.py             # Full-tuple paired residual
│   ├── sindy_track.py           # Track A: SINDy + NAU/NMU
│   ├── ensemble_track.py        # Track B: Ensemble fallback
│   └── gate.py                  # Per-element uncertainty gate
├── self_audit/
│   ├── diagnosis.py             # Statistical diagnosis battery
│   ├── auto_expand.py           # 4 automated expansion mechanisms
│   ├── hypothesis_loop.py       # The core hypothesize-test-expand loop
│   └── llm_oracle.py            # Optional LLM fallback (not in core path)
├── policy/
│   ├── augmented_buffer.py      # Confidence-weighted buffer
│   └── iql_robust.py            # IQL with robust penalties
├── theory/lean4/                # Formal proofs
└── experiments/                 # All experiment scripts
```

Note: `llm_oracle.py` is the ONLY file that imports an LLM client. Delete it and the entire system still works.

---

## 13. Key Dependencies

| Library | Purpose | Required? |
|---|---|---|
| CORL | Base offline RL | Yes |
| PySINDy | Symbolic regression | Yes |
| stable-nalu | NAU/NMU layers | Yes |
| statsmodels | Statistical diagnosis | Yes |
| asteval | Safe expression eval (for auto-expand + optional LLM) | Yes |
| anthropic/openai | LLM API | **No** (optional extension) |


