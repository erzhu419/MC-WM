# MC-WM Implementation Guide

## 最终实现 vs 初始设想的对比

本文档记录从 v2/v4 开发手册到最终实现的演化过程，包括每个组件的设计决策、失败尝试、以及最终方案。

---

## 1. 整体架构

### 初始设想（v2）
```
SINDy → correction → 修改 sim obs → 在 corrected sim 中训练 policy
```

### 初始设想（v4）
```
SINDy → NAU/NMU → QΔ gate → 在 sim 中训练但 Q-target 被 gate 调节
```

### 最终实现
```
Phase 1: M_sim 预训练（世界模型 ensemble on sim data）
Phase 2: 自假设循环发现残差结构 → SINDy+NAU residual δ
Phase 3: MBPO（在 real env 交互 + M_real rollouts augmentation）
         M_real = M_sim(frozen) + δ(SGD warm-start)
         QΔ per-step confidence weight on model rollouts
         约束系统过滤不合理的 model predictions
```

### 为什么变了
- **correction 方案失败**：SINDy 50% 预测精度，OOD 爆炸（-837%），correction 在 trained policy 分布上有害
- **Q-target penalty 方案失败**：additive penalty 太小 vs reward signal（<20%），multiplicative weight 的 gap signal 无空间变异
- **model-based RL 方案成功**：ReDRAW 启发，world model + residual adapter，policy 在 model rollouts 上训练

---

## 2. 各组件详解

### 2.1 世界模型 M_sim
**文件**: `mc_wm/residual/world_model.py` → `WorldModelEnsemble`

| 设计 | 说明 |
|------|------|
| 架构 | K=5 probabilistic MLP ensemble, hidden=200×3, SiLU |
| 输出 | (mean, logvar) per transition → Gaussian NLL loss |
| 预测 | delta format: s' = s + Δs_predicted |
| 训练 | Bootstrap subsampling + early stopping |
| 冻结 | Phase 2 后 freeze，不再更新 |

**关键经验**：
- M_sim 在 sim data 上 RMSE=0.55（很准）
- M_sim 对 real data RMSE=1.91（sim-real gap）
- Frozen M_sim 提供稳定的 structural prior（类似 ResNet 的 skip connection）

### 2.2 残差适配器 δ（SINDy + NAU/NMU）
**文件**: `mc_wm/residual/sindy_nau_adapter.py` → `SINDyNAUAdapter`

| 阶段 | 说明 |
|------|------|
| SINDy 特征提取 | poly2 library (300 features) + STLSQ 稀疏化 |
| 自假设循环 | 3 轮：fit → diagnose → orthogonal expand → re-fit |
| NAU/NMU 输出头 | SymbolicResidualHead: α·NAU + (1-α)·NMU |
| State correction | SINDy+NAU: obs_dim outputs |
| Reward correction | 独立 MLP (64-dim): 1 output |
| Online refit | **SGD warm-start**（不是 STLSQ re-solve） |

**初始设想 vs 最终**：

| 设想 | 最终 | 原因 |
|------|------|------|
| SINDy 直接修正 sim obs | SINDy 作为 world model 的 residual | correction 在 OOD 有害 |
| STLSQ 每次 refit | SGD warm-start（锁定 sparsity） | STLSQ 系数跳变破坏 NAU 稳定性 |
| State + reward 共享 SINDy | State: SINDy+NAU, Reward: 独立 MLP | 共享时 reward correction -7% |
| auto-expand (x³, masks) | 正交展开 (sin, cos, 3-way cross) | 非正交特征冗余导致过拟合 |

**自假设循环发现的特征**：
- `sin(3·x₁)`: correlation 0.41 — gravity torque 的周期性（poly2 无法表达）
- `cos(3·x₁)`: correlation 0.28
- `x₀·x₁·x₅`: 三维交叉项（poly2 只有二维）

### 2.3 QΔ（残差 Bellman 信用分配）
**文件**: 集成在 `experiments/step2_mbrl_residual.py` c9 mode

| 设想 | 最终 | 原因 |
|------|------|------|
| QΔ(s,a) = r_Δ + γ·QΔ(s') | per-step error weight (γ=0) | γ=0.99 → 常数（空间变化被抹平） |
| additive Q-target penalty | multiplicative reward weight | additive 太小 vs reward signal |
| 独立 Q-network | 直接用 model prediction error | 不需要额外网络 |

**实现**：
```python
# 每个 model rollout (s, a):
error = ||M_real(s,a) - real_next_from_env_buf||²
w = 1 / (1 + error / τ)       # τ = median(error)
r_weighted = r_predicted * w    # 不可靠 rollout 的 reward 被压低
```

**效果**：+9.6% performance，L_eff 从 770→303（更稳定）

### 2.4 约束系统
**文件**: `mc_wm/self_audit/constraint_system.py` → `ConstraintSystem`

| Role | 时机 | 功能 |
|------|------|------|
| #1 初始约束 | 训练前 | 20 条物理约束（关节限制、速度上限等） |
| #3 运行时审计 | 每 10k 步 | 检测大幅 correction → 自动添加新约束 |

**单调增长**：20 → 26 → 32 → 44 constraints（只加不减）

**约束类型**：
- Physical: `|rootz| < 2`, `|body_angle| < 3`, `|velocity| < 15`
- Joint: `|angle| < 1.5`, `|angular_vel| < 40`
- Transition: `|s' - s| < 30` per dim
- Reward: `r ∈ [-10, 10]`

**关键经验**：
- 初版 `|Δs| < 5` 拒绝了 99.6%（因为检查的是 state transition 不是 correction）
- 修正为 `|transition| < 30` 后 reject rate = 25-42%（合理）

### 2.5 正交特征展开
**文件**: `mc_wm/self_audit/orthogonal_expand.py` → `OrthogonalExpander`

**替代了 v2 的 auto-expand (4 种 mechanism)**。

| 步骤 | 说明 |
|------|------|
| 1. 生成候选 | 83 个：time, trig, 3-way cross, state-action, cumulative |
| 2. QR 正交化 | 对 Θ 做 QR，投影每个候选到正交补 |
| 3. 过滤 | 保留 >10% 正交分量 且 与 remainder correlation > 0.05 |
| 4. 选择 | 按 correlation 排序，取 top 10 |

**为什么比 auto-expand 好**：
- auto-expand 加 x³（和 x 的 correlation 0.97 → 冗余）
- 正交展开加 sin(3θ)（和 {1, θ, θ²} 正交 → 真正新信息）

### 2.6 诊断电池
**文件**: `mc_wm/self_audit/diagnosis.py` → `DiagnosisBattery`

4 个统计检验 on remainder:
1. **Autocorrelation** (Ljung-Box): 时序结构？
2. **Heteroscedasticity** (Breusch-Pagan): state-dependent 方差？
3. **Non-normality** (D'Agostino): 重尾/多模？
4. **Non-stationarity** (KPSS): 均值漂移？

**关键经验**：在 50k 数据上 O(N²)，需要 subsample 到 2000

### 2.7 RE-SAC Policy
**文件**: `mc_wm/policy/resac_agent.py` → `RESACAgent`

| 组件 | 说明 |
|------|------|
| Ensemble critic | K=3 vectorized Q-networks |
| LCB policy loss | -(Q_mean + β·Q_std - α·log_π) |
| OOD penalty | β_ood · Q_std.mean() |
| BC regularization | β_bc · MSE(π(s), a_behavior) |

### 2.8 MBPO Pipeline
**文件**: `experiments/step2_mbrl_residual.py` → mode c9

```
每 1 步: real env interaction → env_buf (ground truth)
每 250 步: M_real rollouts → QΔ weight → constraint filter → model_buf
每 1000 步: SGD warm-start refit δ (SINDy coefs + NAU)
每 5000 步: constraint audit (Role #3)
每步训练: agent.update(env_buf) + agent.update(model_buf)
```

---

## 3. 失败尝试记录

| 尝试 | 结果 | 根因 |
|------|------|------|
| SINDy correction on sim obs | -837% OOD | SINDy 在 trained policy 分布上不准 |
| Additive Q-target penalty | <20% of reward | penalty 太小 |
| Multiplicative IW (SINDy gap) | gap signal 无空间变异 | SINDy 在 OOD 全部预测"高 gap" |
| MLP gap detector | 同上 | 3000 步数据覆盖不够 |
| QΔ Bellman (γ=0.99) | 变常数 | 累积抹平空间变化 |
| Online confidence + actor constraint | -17% | 约束限制探索 |
| Frozen world model | 崩溃 | policy drift → model 失效 |
| Sim env + M_real rollouts | reward 矛盾 | env_buf(sim reward) vs model_buf(real reward) |
| STLSQ refit + hypothesis loop | -22% vs no-loop | 系数跳变破坏 NAU |
| L_eff over-regularization | -25% | 限制 NAU 表达能力 |
| auto-expand (x³, masks) | -22% | 非正交特征，STLSQ 不稳定 |

---

## 4. 性能演化

| 阶段 | Config | Return | vs Baseline |
|------|--------|--------|-------------|
| 0 | c1 Raw Sim (model-free) | 875 | baseline |
| 1 | c3 Real Online (model-free) | 1,301 | +49% |
| 2 | c6v2 MLP δ (MBPO) | 4,674 | +434% |
| 3 | c7 SINDy+NAU no-loop | 5,147 | +488% |
| 4 | c7 SINDy+NAU + SGD refit + loop | 5,179 | +492% |
| 5 | c9 + QΔ weighted rollouts | 5,264 | +502% |
| 6 | c9 + QΔ + constraints (full v4) | 5,015 | +473% |
| — | c4 Direct M_real (upper bound) | 6,792 | +676% |

---

## 5. 文件结构

```
mc_wm/
├── envs/hp_mujoco/
│   ├── gravity_cheetah.py        # HalfCheetah 2x gravity (H2O benchmark)
│   ├── carpet_ant.py             # Ant 30% vel damping (小 gap，已弃用)
│   ├── wind_hopper.py            # Hopper sinusoidal wind (gap≈0，已弃用)
│   ├── ice_walker.py             # Walker 0.8x friction
│   └── friction_walker.py        # Walker 0.3x friction
├── networks/
│   └── nau_nmu.py                # NAU/NMU layers + SymbolicResidualHead
├── policy/
│   ├── resac_agent.py            # RE-SAC (ensemble critic + LCB)
│   └── q_delta.py                # QΔ module (ensemble, pre-trainable)
├── residual/
│   ├── world_model.py            # WorldModelEnsemble + ResidualAdapter + CorrectedWorldModel
│   ├── sindy_nau_adapter.py      # SINDy+NAU residual δ (core)
│   ├── sindy_ensemble.py         # SINDy ensemble (弃用，replaced by sindy_nau_adapter)
│   ├── mlp_gap_detector.py       # MLP gap detector (弃用)
│   └── confident_residual.py     # Confident residual (弃用)
├── self_audit/
│   ├── diagnosis.py              # 4-test statistical diagnosis battery
│   ├── orthogonal_expand.py      # Orthogonal feature discovery (替代 auto_expand)
│   ├── auto_expand.py            # 4-mechanism auto-expand (弃用)
│   ├── hypothesis_loop.py        # Original hypothesis loop (弃用，integrated into adapter)
│   ├── constraint_system.py      # Constraint system (Role #1 + #3)
│   └── llm_oracle.py             # LLM oracle (hardcoded physics features)
experiments/
├── step1_residual_model.py       # 残差模型验证 (MLP ensemble, data scaling)
├── step2_mbrl_residual.py        # 完整 MBPO pipeline (c1-c9)
├── step3_annealed_mbpo.py        # Annealed real→model (未用于最终)
├── ATTEMPT_LOG.md                # 所有尝试的详细记录
└── diag_data/                    # 诊断数据 (npz)
```

---

## 6. 关键超参数

| 参数 | 值 | 说明 |
|------|---|------|
| M_sim pretrain data | 50k sim transitions | random policy |
| Real data (online) | 50k real transitions | policy-guided |
| SINDy library | poly2 (degree=2, bias=True) | 300 features |
| SINDy threshold | 0.05 | STLSQ sparsification |
| NAU alpha_init | 0.7 | favor linearity |
| δ hidden | 128 (MLP) / poly2+NAU (SINDy) | |
| SGD refit epochs | 30 | per refit cycle |
| SGD refit freq | every 1000 env steps | |
| Model rollout freq | every 250 env steps | |
| Model rollout batch | 400 | |
| QΔ τ | median(error) | auto-calibrated |
| Constraint reject threshold | 30 per-dim | extreme transitions |
| RE-SAC n_critics | 3 | |
| RE-SAC β (LCB) | -2.0 | pessimistic |
| RE-SAC hidden | 256 | |
| Learning rate | 3e-4 | all components |
