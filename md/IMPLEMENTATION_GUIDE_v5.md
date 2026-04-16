# MC-WM Implementation Guide v5

## 从 v2/v4 到 v5：所有模块的最终状态

本文档记录 MC-WM 的完整工程状态（截至 2026-04-16），包括每个模块从 v2/v4 设想到最终实现的演化、碍于障碍采用的替代方案、以及最终性能数据。

---

## 1. 整体架构演化

### v2 设想
```
SINDy → correction → 修改 sim obs → 在 corrected sim 中训练 policy
```
**失败原因**: SINDy 在 trained policy 分布上 OOD 爆炸 (-837%)。

### v4 设想
```
SINDy → NAU/NMU → QΔ gate (γ=0.99 Bellman) → Q-target penalty
LLM 提出约束 + 假设（hardcoded 物理 as placeholder）
```
**部分失败**: QΔ γ=0.99 坍缩为常数；Q-target penalty 太弱（<20% of reward）。

### v5 最终实现
```
Phase 1a: M_sim 预训练（5-ensemble world model on sim data）
Phase 1b: 自假设循环 + LLM Role #2 → SINDy+NAU residual δ
          + Δr (reward MLP) + Δd (termination MLP) ← FULL TUPLE
Phase 2:  MBPO pipeline
          M_real = M_sim(frozen) + δ(SGD warm-start)
          Bellman QΔ (γ=0.5, log-confidence, target net) ← v4 的 γ=0.99 修复版
          ConstraintSystem (20 physics + LLM Role #1/#3/#4 增删)
          LLM Role #5 HP orchestrator (Bayesian opt)
          ICRL 已证等价 → 默认 OFF
```

### 从 v4 到 v5 的关键变化

| v4 设想 | v5 最终 | 变化原因 |
|---------|---------|---------|
| γ=0.99 Bellman QΔ | **γ=0.5 log-confidence Bellman** + target net + Polyak | γ=0.99 坍缩常数；γ=0 太 trivial；γ=0.5 甜点 (5388→5915) |
| LLM hardcoded physics | **Claude API (Haiku 4.5) × 5 roles** + async + cache | hardcoded 不可扩展；API 实测 12/12 feature acceptance |
| ICRL 独立模块 | **理论证明 ≡ QΔ → 默认 OFF** | Lean §VIII 证 rank-equivalence；A/B 实测 -10% (5314 vs 5915) |
| State + Reward 残差 | **+ Termination (Δd) = Full-tuple** | 论文声称 full-tuple 但原来没实现 |
| 20 hardcoded constraints | **LLM Role #1 生成 + Role #3 async 审计 + Role #4 prune** | hardcoded 不可扩展且过于粗糙 |
| 无 HP 自动调参 | **LLM Role #5 (BO-style)** + step clamp + cooldown | LLM-as-acquisition-function |
| SINDy+NAU 唯一 | **KAN 作为 drop-in 替代** (5763 > SINDy 5388) | KAN ICLR 2025 后自然升级 |

---

## 2. 各组件详解

### 2.1 世界模型 M_sim
**文件**: `mc_wm/residual/world_model.py` → `WorldModelEnsemble`

| 项 | v4 | v5 | 变化 |
|---|---|---|---|
| 架构 | K=5 prob MLP, SiLU | 同左 | 无 |
| 训练 | 50k sim, bootstrap | 同左 | 无 |
| 冻结 | Phase 2 后 freeze | 同左 | 无 |

**未改动**。M_sim 是 ResNet-style skip connection 的 "highway"，δ 是 "residual branch"。

### 2.2 残差适配器 δ
**文件**: `mc_wm/residual/sindy_nau_adapter.py` + `mc_wm/residual/kan_adapter.py`

#### v4 → v5 差异

| 项 | v4 | v5 |
|---|---|---|
| State δ | SINDy+NAU 唯一 | SINDy+NAU 或 **KAN** (--use_kan_residual) |
| Reward δ | 独立 MLP | 同左 |
| **Termination δ** | **不存在** | **新增 MLP: P(done\|s,a,s')** via BCE |
| 假设循环 | 正交展开 only | 正交展开 + **LLM Role #2** features |
| Feature eval | 无安全检查 | **Sandboxed eval** (禁 `__`/`import`/`os`) |
| 特征 predict-time 重建 | 只认 `orthogonal` spec | + `llm_features` spec (存 expr+std) |
| Refit | SGD warm-start | 同左 + **传 real_dones** 更新 Δd head |

#### Full-tuple 输出

```python
corrected.predict(s, a)              → (s', r)        # 向后兼容
corrected.predict_full_tuple(s, a)   → (s', r, d)     # v5 新增
```

`d = P(done|s, a, s')` 经阈值 0.5 → 二值化写入 model_buf，替代旧的 `done=0`。

### 2.3 QΔ（残差 Bellman 信用分配）

**v4 的最大未解决问题**，v5 彻底解决。

| 项 | v4 | v5 |
|---|---|---|
| 定义 | per-step MSE weight (γ=0) | **Bellman log-confidence critic** |
| γ | 0 (trivial) | **0.5** (甜点) |
| 网络 | 无（直接计算） | **QDeltaNet MLP** + target net + Polyak τ=0.005 |
| 训练 | 无 | **TD update** 每 1000 env steps, 20 iters |
| 数学基础 | 无 | **Lean bellman_contraction 定理** |

**文件**: `mc_wm/policy/qdelta_bellman.py` → `QDeltaBellman`

**关键**: `log QΔ(s,a) = log c_step(s,a) + γ·E[log QΔ(s',a')]`，其中 `c_step = 1/(1+MSE/τ)`。
Weight = `exp(QΔ) ∈ (0, 1]`。不会坍缩（log-变换使平均 ≠ log(平均)）。

**γ ablation** (gravity_soft_ceiling):

| γ | Reward | Viol/ep |
|:-:|-------:|--------:|
| 0 | 4618 | 2.03 |
| **0.5** | **5388** | **0.03** |
| 0.9 | 4347 | 2.17 |

### 2.4 ICRL（已证等价 → 默认 OFF）

**v4 引入 ICRL 作为 Type 2 OOD 检测器**。v5 经理论+实验**双重证伪后移除**。

#### 理论依据 (Lean §VIII)

**Theorem (Uniform Rank Equivalence)**: 在 Gaussian 残差 + 均匀拟合假设下：
```
rank(QΔ) ≡ rank(φ_ICRL)    (严格 IFF)
```
**Theorem (Bounded-Variation)**: `|e_resid - mean| ≤ δ` 时：
```
|log QΔ gap| > 2δ/τ  ⟹  rank(QΔ) = rank(φ)
```
**结论**: QΔ 已覆盖 ICRL 信号。

#### 实验依据 (A/B test, gravity_soft_ceiling)

| 配置 | Reward | Viol |
|------|-------:|-----:|
| QΔ only | **5915** | **1.63** |
| QΔ + ICRL | 5314 | 4.13 |

ICRL φ_e=0.51, φ_n=0.43, sep=0.093 — **基本未学到有效判别**。

**保留方式**: `--enable_icrl` flag (legacy/ablation)。默认 OFF。

### 2.5 约束系统 (5 roles)
**文件**: `mc_wm/self_audit/constraint_system.py` + `mc_wm/self_audit/claude_cli_oracle.py`

#### v4 vs v5

| 项 | v4 | v5 |
|---|---|---|
| Role #1 | 20 hardcoded physics | + **LLM 生成 ~8 条** (async-safe sandbox eval) |
| Role #3 | 统计异常检测 | + **LLM Type 1/2/artifact 三分类审计** (async) |
| Role #4 | **不存在** | **LLM prune** (删低效/冗余 LLM 约束) |
| Role #5 | **不存在** | **LLM HP orchestrator** (Bayesian opt) |
| 方向 | 单调增长 only | **增+删闭环**（Role #1/#3 加，Role #4 删） |
| 执行 | 同步阻塞 | **ThreadPoolExecutor async** (0 阻塞训练) |
| 信息 | 约束名 only | + per-constraint reject_rate + decision_history + training_metrics |

#### Role #3 的三分类

| 类型 | 信号 | LLM 判决 | 动作 |
|------|------|---------|------|
| **Type 2 OOD** | state 本身物理不可能 (z<0, |θ|>π) | "infeasible" | 加约束 |
| **Type 1 OOD** | correction 大但 state 合理 | "valid_large_correction" | 不动 |
| **Artifact** | QΔ weight 分布坍缩 / reward=0 ceiling hit | "valid_large_correction + reasoning" | 不动 |

### 2.6 LLM Oracle (Claude API)
**文件**: `mc_wm/self_audit/claude_cli_oracle.py` → `ClaudeCLIOracle`

#### v4 vs v5

| 项 | v4 | v5 |
|---|---|---|
| 实现 | `llm_oracle.py` hardcoded 10 features | **Claude API SDK** (Haiku 4.5) |
| Transport | 无 | SDK (primary) + CLI (fallback) + SHA256 cache |
| Roles | 无 | **5 roles** (init, features, audit, prune, HP tune) |
| 安全 | 无 | sandbox eval + retry + max_tokens 2048 |
| 成本 | $0 | **~$0.06/run** |
| 异步 | 无 | Role #3/#4 via ThreadPoolExecutor |

#### LLM 输入完整度

每个 Role prompt 包含：
- env_description
- decision_history (过往 add/drop + outcome)
- training_metrics (reward_trend, violation_trend, buffer_size, QΔ weight stats)
- per-constraint reject_rate (Role #3/#4)
- residual_per_dim_std + val_MSE + L_eff (Role #2)
- feature_history with outcomes (Role #2)
- Type 1/2/artifact 三分类框架 (Role #2/#3)

### 2.7 HP Orchestrator (Role #5)
**文件**: `mc_wm/self_audit/hp_orchestrator.py` → `HPOrchestrator`

**v4 完全不存在**。v5 新增。

可调 HP（通过 `runtime_cfg` dict）：
| HP | 范围 | 说明 |
|---|---|---|
| qdelta_gamma | [0.0, 0.95] | Bellman QΔ discount |
| rollout_batch | [100, 1000] | 每周期 rollout 数 |
| rollout_freq | [100, 1000] | rollout 生成间隔 |
| model_train_freq | [500, 5000] | δ refit 频率 |
| audit_percentile | [80, 99] | Role #3 阈值 |
| icrl_combine | {top_k, soft} | φ 组合方式 |
| icrl_top_k_frac | [0.3, 0.95] | top-K 保留比 |

**安全**: step-clamp (≤10%/call) + cooldown (2 evals after change) + schema validation。

### 2.8 KAN Alternative
**文件**: `mc_wm/residual/kan_adapter.py` → `KANResidualAdapter`

**v4 不存在**。v5 新增为 drop-in 替代。

| 对比 | SINDy+NAU | KAN |
|------|:-:|:-:|
| c9 soft_ceiling reward | 5388 | **5763** |
| c9 soft_ceiling viol | 0.03 | **0.00** |
| 需要预定义 basis | 是 (poly2) | **否** |
| 需要 NAU head | 是 | 否 |
| L_eff formal bound | 有 | **无**（近似） |

### 2.9 Lean 形式化
**文件**: `proof/MC-WM/ResidualMDP.lean` (~1000 lines)

| Part | 定理 | Sorry |
|------|------|:-----:|
| III | Bellman γ-contraction | 0 |
| IV | Residual Simulation Lemma | 0 |
| V | Policy-conditional convergence | 0 |
| VI | Universal convergence FALSE | 0 (placeholder) |
| VII | Residual refit contraction | 0 |
| VIII-a | QΔ-ICRL uniform rank equiv | **0** |
| VIII-b | QΔ-ICRL bounded rank equiv | **0** |
| **Total** | **7 theorems** | **0 sorry** |

5 个建模公理 (Gaussian assumption, Bayes optimality, QΔ log-form, Banach, BCE optimality)。

---

## 3. 失败尝试完整记录

### v2 → v4 阶段（前文已有）

| 尝试 | 结果 | 根因 |
|------|------|------|
| SINDy correction on sim obs | -837% OOD | trained policy 分布外推失败 |
| Additive Q-target penalty | <20% | penalty 太小 |
| QΔ Bellman (γ=0.99) | 坍缩常数 | 累积抹平空间变化 |
| STLSQ refit | -22% | 系数跳变 |
| auto-expand (x³) | -22% | 非正交冗余 |

### v4 → v5 阶段（新增）

| 尝试 | 结果 | 根因 |
|------|------|------|
| ICRL v1 (IS-weighted + conf) | φ_e→0.19 坍缩 | IS loss + conf 信号弱 |
| ICRL v2 (BCE + Δs + soft mod) | 61.7 viol (崩坏) | soft mod 配 transition 不合适 |
| ICRL v3 (BCE + trans + soft) | 4442 | 冗余 with QΔ |
| ICRL v4 (BCE + trans + top_k) | 4268 | 过滤过度 |
| ICRL v2 (conf + soft) | **4397/0.20** | 最好的 ICRL，但仍 < QΔ-only |
| LLM prompt 语义反转 | 100% reject | check 表达式 True=feasible 而非 violation |
| LLM f-string bug `{x:.3f if...}` | 崩溃 | Python 不支持 |
| LLM predict_theta 缺 llm_features spec | 崩溃 | Θ 重建时 LLM 列丢失 |
| LLM history outcome 内联更新 | 崩溃 | feature name 对齐问题 |
| Role #5 太激进 (γ 0.5→0.7→0.75) | 5326/4.17 | thrashing → violation spike |

---

## 4. 性能演化（完整时间线）

| 阶段 | Config | Return | Viol | vs c1 |
|------|--------|-------:|-----:|------:|
| v2 | c1 Raw Sim | 875 | — | baseline |
| v2 | c6 MLP δ | 4,674 | — | +434% |
| v4 | c7 SINDy+NAU no-QΔ | 5,179 | — | +492% |
| v4 | c9 QΔ γ=0 + constraints | 5,264 | — | +502% |
| v5 | c9 QΔ **γ=0.5** | 5,388 | 0.03 | +516% |
| v5 | c9 + **KAN** residual | 5,763 | 0.00 | +559% |
| v5 | c9 γ=0.5 + **5 LLM roles** | **5,915** | 1.63 | **+576%** |
| v5 | c9 γ=0.5 + LLM + ICRL (A/B) | 5,314 | 4.13 | +507% |
| — | c4 Direct M_real (upper bound) | 6,792 | — | +676% |

**最优配置**: c9 + Bellman QΔ γ=0.5 + LLM 5 roles (无 ICRL) = **5915 (87% of oracle)**。
**单步最高**: step 50k = **6355** (93% of oracle)。

---

## 5. 文件结构（v5 完整）

```
mc_wm/
├── envs/hp_mujoco/
│   ├── gravity_cheetah.py          # GravityCheetahEnv + CeilingEnv + SoftCeilingEnv
│   ├── friction_walker.py          # FrictionWalkerEnv + SoftCeilingEnv
│   ├── ant_wall_broken.py          # AntWallBrokenEnv (ICRL paper benchmark)   ← v5 新增
│   ├── carpet_ant.py               # CarpetAntEnv
│   └── wind_hopper.py              # WindHopperEnv
├── networks/
│   └── nau_nmu.py                  # NAU/NMU layers + SymbolicResidualHead
├── policy/
│   ├── resac_agent.py              # RE-SAC (ensemble critic + LCB)
│   └── qdelta_bellman.py           # Bellman QΔ (γ>0, target net, Polyak)     ← v5 新增
├── residual/
│   ├── world_model.py              # WorldModelEnsemble + CorrectedWorldModel
│   │                                 (+ predict_full_tuple: s', r, d)         ← v5 新增
│   ├── sindy_nau_adapter.py        # SINDy+NAU δ (state+reward+done heads)
│   │                                 + LLM Role #2 hook + sandbox eval        ← v5 升级
│   └── kan_adapter.py              # KAN drop-in alternative                  ← v5 新增
├── self_audit/
│   ├── diagnosis.py                # 4-test statistical diagnosis battery
│   ├── orthogonal_expand.py        # Orthogonal feature discovery
│   ├── constraint_system.py        # ConstraintSystem (Role #1/#3/#4, async)  ← v5 大升级
│   ├── claude_cli_oracle.py        # Claude API SDK oracle (5 roles)          ← v5 新增
│   ├── hp_orchestrator.py          # Role #5 HP tuner (LLM-as-BO)            ← v5 新增
│   └── icrl_constraint.py          # ICRL φ (legacy, --enable_icrl)          ← v5 证等价后降级
│   └── llm_oracle.py               # hardcoded physics (legacy)

experiments/
├── step2_mbrl_residual.py          # 完整 pipeline (c1-c10)
│                                     + --seed, --qdelta_gamma, --use_claude_llm,
│                                     + --use_role5_hp, --enable_icrl,
│                                     + --use_kan_residual, --use_kan_phi,
│                                     + --save_phi, --load_phi, --claude_model
├── llm_smoke_test.py               # Claude oracle 独立测试                   ← v5 新增
└── ATTEMPT_LOG.md                  # 所有尝试详细记录

proof/MC-WM/
├── ResidualMDP.lean                # ~1000 行, 0 sorry, 7 theorems            ← v5 新增
└── MC-WM_proof_doc.md              # 证明工程文档                              ← v5 新增

md/
├── IMPLEMENTATION_GUIDE_v5.md      # 本文件
├── ICRL_Type2_OOD_Report.md        # ICRL 26 实验完整报告
├── Assumption_upgrade.md           # KAN 升级路径分析
├── LLM_mode.md                     # 论文拆分建议
└── MC-WM_Constraint_Subsystem_Report.md  # ICRL + 约束理论报告
```

---

## 6. 关键超参数（v5）

| 参数 | 默认值 | v4 值 | CLI flag |
|------|:------:|:-----:|---------|
| SEED | 42 | 42 | `--seed` |
| M_sim pretrain | 50k | 50k | — |
| Real data | 50k | 50k | — |
| SINDy library | poly2 | poly2 | — |
| NAU alpha_init | 0.7 | 0.7 | — |
| **QΔ γ** | **0.5** | 0 | `--qdelta_gamma` |
| Model rollout batch | 400 | 400 | `runtime_cfg` |
| Model rollout freq | 250 | 250 | `runtime_cfg` |
| Model refit freq | 1000 | 1000 | `runtime_cfg` |
| Constraint audit percentile | 95 | 95 | `runtime_cfg` |
| **ICRL** | **OFF** | ON (c10) | `--enable_icrl` |
| **LLM model** | **haiku-4.5** | N/A | `--claude_model` |
| **LLM Role #5 interval** | 2 evals | N/A | `--role5_every_n_evals` |
| **Role #5 step clamp** | 10% | N/A | HPOrchestrator init |
| **Role #5 cooldown** | 2 evals | N/A | HPOrchestrator init |
| RE-SAC n_critics | 3 | 3 | — |
| RE-SAC β (LCB) | -2.0 | -2.0 | — |

---

## 7. 推荐训练命令

### Paper A（技术论文，无 LLM）
```bash
conda run -n LSTM-RL python3 -u experiments/step2_mbrl_residual.py \
    --mode c9 --env gravity_soft_ceiling --qdelta_gamma 0.5 --seed 42
```

### Paper B（系统论文，全模块）
```bash
source ~/.config/mcwm/api_key.env && \
conda run -n LSTM-RL python3 -u experiments/step2_mbrl_residual.py \
    --mode c9 --env gravity_soft_ceiling --qdelta_gamma 0.5 \
    --use_claude_llm --use_role5_hp --claude_model claude-haiku-4-5-20251001
```

### KAN 替代版本
```bash
conda run -n LSTM-RL python3 -u experiments/step2_mbrl_residual.py \
    --mode c9 --env gravity_soft_ceiling --qdelta_gamma 0.5 --use_kan_residual
```
