# MC-WM 下一步改进方向

## 当前状态 (c7 = 5147, +488% vs baseline)

已实现并验证：
- ✅ World model ensemble (M_sim, pretrained)
- ✅ Residual adapter δ (SINDy poly2 + NAU/NMU)
- ✅ MBPO pipeline (real env interaction + online δ refit + mixed training)
- ✅ Warm-start δ refit
- ✅ OOD bound (L_eff from NAU/NMU)
- ✅ Self-hypothesis loop (但扩展特征反而降低了性能)
- ✅ LLM oracle (hardcoded physics, 和 poly2 重叠)

未实现（v4 manual 中的）：
- ❌ 全元组残差 (Δr, Δd) — 目前只做 Δs
- ❌ QΔ 作为 gate（之前测过，γ=0.99 变常数）
- ❌ 约束系统（单调增长）
- ❌ LLM 角色 #3（约束增强）
- ❌ 真正的 LLM API 调用（目前 hardcoded）
- ❌ 多环境验证

## 改进空间分析

### 1. 最大增量：缩小 c7 (5147) vs c4 (6792) 的 gap

gap = 1645 (24%)。根据 step-by-step log：
- c7 的 m_s 在 0.4-0.9 波动（c4 在 0.3-0.6）
- c7 的 L_eff 从 136→296 增长（NAU/NMU 不稳定）
- c4 每次 refit 整个 200×3 model，c7 只 refit 小的 δ

**潜在改进**：
- δ 的 NAU/NMU 在 refit 时加更强的 regularization（限制 L_eff 增长）
- 增大 δ 容量（但 c6v3 证明 200-dim 过拟合 — 需要更好的正则化平衡）
- 用 ensemble δ（K=3 bootstrap δ，用 mean 预测，std 做 uncertainty）

### 2. Aleatoric 桶分析（v4 manual 的核心贡献）

**当前问题**：SINDy fit 后的 remainder（aleatoric 余量）没有被利用。
v4 设计：remainder 进入"aleatoric 桶" → 用于 robust RL 的不确定性集合。

**具体做法**：
1. SINDy fit 后计算 remainder = Δ_true - Δ_predicted
2. 对 remainder 做 diagnosis（autocorrelation, heteroscedasticity 等）
3. 如果 diagnosis 全阴：remainder 是真正的 aleatoric noise → 用方差估计做 robust RL
4. 如果 diagnosis 阳性：remainder 有未发现的结构 → 触发 LLM 假设

当前 c7 的 remainder RMSE ≈ eps_max ≈ 1.88。这说明大量结构未被捕获。

### 3. 全元组残差 (Δr, Δd)

GravityCheetah 的 reward gap = 0.49（显著）。目前只修正 Δs，reward 没修正。
如果同时修正 reward → 模型 rollout 的 reward 更准 → policy 学到更好的行为。

### 4. QΔ 的正确实现

之前 QΔ 失败因为：
- γ=0.99 让 QΔ 变常数（spatial info 被 Bellman 抹平）
- 用 SINDy gap signal 作为 reward（不准）

v4 的设计更好：
- QΔ 的 reward = ||Δ̂ - Δ_true||²（在 paired data 上已知）
- 用 corrected next state 做 TD target
- Pessimistic estimation

但在当前 MBPO pipeline 里，QΔ 的角色不是 gate（c7 不需要 gate），而是 **决定什么时候可以增加 model rollout 的比例**。类似退火 schedule 但数据驱动。

### 5. 约束系统

不是性能改进，而是安全改进。确保 corrected transitions 不违反物理。
对 GravityCheetah 可能没有明显效果（dynamics gap 是 smooth 的），但对 contact-rich 环境（Ice-Walker, Carpet-Ant）可能重要。

## 推荐优先级

1. **全元组残差 (Δr)** — 最容易加，收益明确（reward gap 0.49 未修正）
2. **δ regularization 改进** — 限制 L_eff 增长，缩小 c7 vs c4 gap
3. **多环境验证** — 在 IceWalker, FrictionWalker 上跑 c7
4. **Aleatoric 桶 → robust RL** — 把 remainder 方差用于 policy 训练
5. **真正的 LLM API 调用** — 替换 hardcoded oracle
6. **约束系统** — 安全性改进
