# MC-WM Status Report (2026-04-13)

## 一、项目目标

在 sim-to-real RL 场景中，用**残差贝叶斯世界模型**修正 sim 和 real 之间的 dynamics gap，使得在 sim 中训练的 policy 能接近在 real 中直接训练的效果。

**测试环境**：GravityCheetah（HalfCheetah，sim=2x gravity，real=1x gravity），来自 H2O 论文的标准 benchmark。

## 二、当前 Baselines

| 条件 | Real Env Return | 说明 |
|------|----------------|------|
| c1: Raw Sim (model-free) | **875** | 在 2x gravity sim 中训练 50k 步，eval in real |
| c3: Real Online (model-free) | **1301** | 在 1x gravity real 中训练 50k 步，eval in real |
| c4: MBPO Real (model-based) | **6792** | 在 real env 中交互 + 在线 refit world model + model rollouts，50k 步 + 50k pretrain |

c1→c3 的 gap 是 **32%**（875 vs 1301），这是 residual correction 需要 close 的空间。

c4 证明了 MBPO pipeline 在 real env 上完全 work（6792 远超 model-free 的 1301）。

## 三、已验证可工作的组件

### 3.1 残差模型（MLP ensemble）本身能学会 sim-real gap

**实验**：用不同量级数据（3k/10k/30k/100k）训练 MLP ensemble（K=5，64×2 层，weight decay + early stopping）预测 Δs = s_real - s_sim。

| 数据量 | RMSE Reduction | Rank Correlation |
|--------|---------------|-----------------|
| 3,000 | 10.7% | 0.181 |
| 10,000 | 16.5% | 0.370 |
| 30,000 | 16.8% | 0.342 |
| 100,000 | 21.5% | 0.421 |

**关键发现**：
- 小网络 + 正则化 + early stopping 后，泛化到 trained policy 分布不崩溃（10.0% reduction 维持）
- 之前用大网络（256×3层）无正则化 → 严重过拟合（val loss 上升）
- 数据越多越好但收益递减

### 3.2 MBPO pipeline 在 real env 上 work

c4 证明了：
- **Online model refit** 是必须的（frozen model 导致 policy drift → model 失效）
- **混合训练（env_buf + model_buf）** 是必须的（纯 model-based 训练不稳定）
- Model error (m_s) 从 0.87 → 0.29 持续下降（online refit 有效）
- Real return: -90 → 2944 → 3724 → 5278 → 6206 → 6714 → 7028（持续上升）

## 四、当前核心问题

### 4.1 M_real = M_sim + δ 的精度不够

| 模型 | 预测目标 | RMSE |
|------|---------|------|
| M_sim → sim | sim next state | **0.55** (很好) |
| Raw sim → real | real next state | **2.02** (gap) |
| M_sim → real (无修正) | real next state | **1.91** |
| M_real → real (3k paired) | real next state | **1.59** (修正 17%) |
| M_real → real (100k直接训练) | real next state | **0.53** (很好) |

**M_real 的 RMSE=1.59 不够支撑 MBRL**。对比 c4 中直接训练的 M_real（RMSE=0.53），差了 3 倍。

**原因**：
1. M_sim 对 sim 很准（0.55），但对 real 的初始误差就有 1.91
2. Residual δ 需要修正 1.91 的误差，但只有 3k paired data，只能修正 17%
3. 即使 δ 完美，M_real 的 RMSE = M_sim 自身误差 + δ 残余误差

### 4.2 Sim env 交互 + M_real rollouts 的 reward 不一致

**已尝试三种方案，均失败**：

1. **混合 env_buf(sim reward) + model_buf(M_real reward)**：Q-function 收到矛盾信号（sim reward 是 2x gravity dynamics 下的 reward，M_real reward 是 1x gravity dynamics 下的 reward），policy 崩溃。

2. **Relabel env_buf reward 用 M_real**：reward 一致了但 next_state 不一致（env_buf 的 s2 是 sim env 给的，Q(s,a)=r_real + γQ(s2_sim) 仍然是混合的）。

3. **纯 model_buf 训练（不用 env_buf 做 Q-learning）**：没有 real data anchor，policy 在 model 的弱点上 exploit，compounding error。

### 4.3 Policy-Model 耦合（灰犀牛）

Model 在 random policy 数据上训练 → policy 训练后访问不同的 state → model 在新 state 上不准 → policy 学到错误信号 → 进一步 drift → model 更不准。

c4 用 **online model refit**（每 1000 步在 growing env_buf 上重训 world model）解决了这个问题。但 c4 是在 real env 中交互的——model 重训用的是 real transitions，所以 model 始终准确。

对于 c5（sim env 交互），online refit 只能重训 M_sim（用 sim transitions），不能重训 residual δ（需要 paired data）。δ 的重训需要额外的 real env 交互，而 real env 交互预算有限。

## 五、正在运行的实验

**c5**：Sim env + residual-corrected MBPO
- Sim env 交互，world model = M_sim + δ
- 每 5k 步收集 500 paired samples（用当前 policy），重训 δ
- 混合训练 env_buf + model_buf
- 初始结果：5k 步 real=-133（baseline 875）
- 还在跑，等最终结果

## 六、关键未解决问题

### Q1：如何让 M_real 更准？

当前 M_real（M_sim + δ）的 RMSE=1.59，需要降到 <0.5 才能支撑 MBRL。

**可能的方向**：
- 更多 paired data（10k-100k），但 real data 预算有限
- 更大的 residual 网络（但会过拟合小数据）
- 在 latent space 做 residual（ReDRAW 的做法），可能更紧凑更易学
- Online 重训 residual（c5 在做，每 5k 步 500 新 paired samples）

### Q2：如何解决 sim env_buf 和 M_real rollouts 的不一致？

核心矛盾：policy 在 sim env 里走，但需要按 real dynamics 学习。

**可能的方向**：
- 用 M_real 做完整的 reward + next_state relabel（把 env_buf 完全转成 "virtual real" transitions）
- 不用 env_buf 做 Q-learning，只做 start-state sampling（c3 的做法，但需要 model 非常准）
- H2O+ 的做法：用 discriminator 做 importance weighting，而不是 correction

### Q3：如何解决 policy-model 耦合？

Model 是静态的（或 refit 频率低），policy 在变。

**可能的方向**：
- 更频繁的 refit（需要更多 paired data）
- ReDRAW 的做法：freeze base model (M_sim)，只 refit residual δ（更少参数，更快收敛）
- 在 latent space 做 residual（更紧凑，更少数据就能 refit）

## 七、代码结构

```
MC-WM/
├── mc_wm/
│   ├── envs/hp_mujoco/
│   │   ├── gravity_cheetah.py    # 2x gravity sim / 1x real
│   │   ├── carpet_ant.py         # 30% vel damping (gap 太小，已弃用)
│   │   └── ...
│   ├── policy/
│   │   ├── resac_agent.py        # RE-SAC (ensemble critic + LCB)
│   │   └── q_delta.py            # QΔ (已弃用，Bellman 累积抹平空间变化)
│   └── residual/
│       ├── world_model.py        # WorldModelEnsemble + ResidualAdapter + CorrectedWorldModel
│       ├── sindy_ensemble.py     # SINDy ensemble (已弃用，poly2 OOD 爆炸)
│       ├── mlp_gap_detector.py   # MLP gap detector (已弃用，gap signal 无空间变异)
│       └── confident_residual.py # Confident residual (已弃用)
├── experiments/
│   ├── step1_residual_model.py   # 残差模型验证（通过）
│   ├── step2_mbrl_residual.py    # MBRL pipeline（c1-c5）
│   ├── ATTEMPT_LOG.md            # 前 3 次尝试的记录
│   └── ...
└── reference/
    ├── Adapting World Models with Latent-State Dynamics Residuals.pdf  # ReDRAW
    └── RESWORLD TEMPORAL RESIDUAL WORLD MODEL...pdf                    # ResWorld
```

## 八、关键数值总结

### GravityCheetah Baselines
- Raw Sim (model-free, 50k): **875**
- Real Online (model-free, 50k): **1301**
- MBPO Real (model-based, 50k+50k pretrain): **6792**

### 残差模型精度
- M_sim on sim: RMSE = **0.55**
- M_real (3k paired): RMSE = **1.59** (修正 17%)
- M_real (100k 直接训练): RMSE = **0.53**
- 需要 RMSE < 0.5 才能支撑 MBRL

### 参考：H2O+ SIM 公交项目为何 work

在用户的 H2O+ SUMO 项目中，SIM + SUMO offline data 能表现更好，关键设计：
1. **50/50 固定 batch ratio**：每个 batch 50% real data + 50% sim data，real data 永远不被稀释
2. **Discriminator importance weighting**：用 contrastive discriminator 区分 sim/real transitions，给 sim Q-target 做 importance reweighting（不是 penalty，是 weight）
3. **Cal-QL**：防止 sim Q-target 低于 offline Q baseline（防止 Q 崩溃）
4. **Online discriminator 训练**：discriminator 和 policy 一起训练，始终适应当前分布
5. **Snapshot reset**：50% 的 episode 从 offline data 的 state 重启（coverage）

**对比 MC-WM 的差异**：
- H2O+ 不修正 dynamics（只 reweight），MC-WM 试图修正 world model 的 dynamics
- H2O+ 的 discriminator 是 online 更新的，MC-WM 的 residual 是 offline 或低频更新的
- H2O+ 有大量 offline real data（数千条 SUMO trajectories），MC-WM 只有 3k paired steps

**可能的方向**：像 H2O+ 一样，放弃 "修正 world model dynamics"，改为 "在 sim env 训练但用 importance weighting 修正 Q-learning"，同时用残差模型作为 discriminator 的替代品。

### 关键失败模式
1. SINDy poly2 OOD 爆炸 → 改用 MLP
2. Additive Q-penalty 太小 vs reward signal → 改用 multiplicative IW → 但 gap signal 无空间变异
3. Frozen model → policy drift → 改用 online refit
4. Sim reward / real reward 混合 → Q-function 矛盾 → 需要一致的 data pipeline
5. 纯 model-based（无 env anchor）→ compounding error

### 有效的发现
- Uniform LR reduction (+17%) — 降低学习率本身是一种 regularization
- Online model refit — 模型跟随 policy 更新，保持准确
- 小网络 + 正则化 — 防止残差模型过拟合
- MBPO mixed training — env data 做 anchor，model data 做 augmentation
