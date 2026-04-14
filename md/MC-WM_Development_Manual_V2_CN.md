# MC-WM：元认知世界模型与全元组残差动力学

## 完整开发手册（v2 — 奥卡姆剃刀版）

---

## 1. 项目定位

**工作标题：** *通过符号动力学发现实现外推式残差世界模型的仿真到现实迁移*

**一句话概括：** 我们用全元组残差模型修正仿真动力学，该模型的符号结构由 SINDy 自动发现，通过 NAU/NMU 外推并具备正式的分布外（OOD）误差界，并通过不确定性门控实现可信纠正——构成一个自我审计的智能体，迭代地假设、测试并证伪自身对仿真-现实差距的理解。

**核心哲学主张：** 现有的仿真到现实方法问的是"这些数据有多不同？"我们问的是"为什么不同，体现在状态迁移元组的每一个维度？"——一旦知道了原因，就可以外推。

---

## 2. 架构概览：自我假设智能体

### 2.1 设计原则

系统是一个**自主假设-证伪循环**：

```
┌──────────────────────────────────────────────────────────┐
│                  自我审计智能体                            │
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐           │
│  │ 假设    │───→│ SINDy    │───→│ 质量      │──→ 通过   │
│  │         │    │ 拟合     │    │ 门控      │   → 接受  │
│  └────┬────┘    └──────────┘    └─────┬─────┘           │
│       │                               │ 失败             │
│       │         ┌──────────┐          │                  │
│       │←────────│ 诊断     │←─────────┘                  │
│       │         │ 残差     │   （遗漏了什么结构？）        │
│       │         └──────────┘                             │
│       │                                                  │
│       ▼ 自动扩展基函数库                                   │
│  ┌─────────┐                                            │
│  │ 自动    │  代数交叉项、时间延迟、                        │
│  │ 扩展    │  分段掩码、高阶项                              │
│  └────┬────┘                                            │
│       │ 经过 K 轮仍然卡住？                                │
│       ▼                                                  │
│  ┌─────────────────────┐                                │
│  │ 🆘 咨询 LLM        │  ← 可选，最后手段               │
│  │ （外部先知）         │                                │
│  └─────────────────────┘                                │
└──────────────────────────────────────────────────────────┘
```

**智能体无需 LLM 也能正常工作。** LLM 是仅在自动扩展机制耗尽时才咨询的备用先知。核心循环——假设、拟合、测试、扩展——完全不依赖外部智能。

### 2.2 最小必要架构（6 个模块，0 个可选）

```
全元组残差提取
    → SINDy + 自动扩展基函数库
        → NAU/NMU 输出头
            → 不确定性门控
                → 增强缓冲区
                    → 鲁棒策略学习
```

每个模块都是承重结构，移除任何一个都会破坏特定能力：

| 模块 | 移除后… |
|---|---|
| 全元组残差 | 完全没有仿真-现实纠正 |
| SINDy | 没有符号结构，无法使用 NAU/NMU |
| NAU/NMU | 没有 OOD 外推保证（定理 4.12） |
| 门控 | 错误纠正比不纠正更糟 |
| 增强缓冲区 | 无法合并真实数据 + 纠正后的仿真数据 |
| 鲁棒策略 | 无法处理不可约不确定性 |

### 2.3 可选扩展：LLM 作为外部先知

**激活条件：** 仅在自动化基函数扩展循环运行 K 轮（默认 K=3）后仍未通过质量门控时。

**功能：** 读取统计诊断报告，提出自动系统无法完成的概念飞跃所需的候选特征（例如"这看起来像库仑摩擦，试试符号函数"或"使用 Takens 延迟嵌入"）。

**不做的事：** 不参与核心训练循环，不接触 Q 函数，不影响门控，无需调用系统也能正常工作。

---

## 3. 全元组残差

### 3.1 为何不只看状态

完整的 MDP 转换是 $(s, a, r, s', d)$。仿真-现实差距存在于**每个元素**中：

| 元组元素 | 差距类型 | 示例 |
|---|---|---|
| $s' - s'_{\text{sim}}$ | 动力学差距 | 摩擦、重力、接触模型 |
| $r - r_{\text{sim}}$ | 奖励差距 | 传感器相关奖励、能量消耗 |
| $d - d_{\text{sim}}$ | 终止差距 | 不同的失败阈值 |
| $\text{Var}(s') - \text{Var}(s'_{\text{sim}})$ | 随机性差距 | 确定性仿真 vs 噪声真实环境 |

### 3.2 形式化定义

$$\Delta(s,a) = \begin{pmatrix} \Delta_s(s,a) \\ \Delta_r(s,a) \\ \Delta_d(s,a) \end{pmatrix} = \begin{pmatrix} s'_{\text{real}} - s'_{\text{sim}} \\ r_{\text{real}} - r_{\text{sim}} \\ d_{\text{real}} - d_{\text{sim}} \end{pmatrix}$$

每个元素都有自己的 SINDy 模型、NAU/NMU 头和门控。它们是独立的流水线，共享相同的 $(s,a)$ 输入。

---

## 4. 自我假设循环（核心贡献）

### 4.1 循环结构

这是系统的核心，是**以算法形式实现的科学方法**：

```
输入：每个元组元素的原始残差 Δ(s,a)
输出：结构化残差模型 Δ̂(s,a) + 偶然性余项

初始化：basis_library = PolynomialLibrary(degree=2)

FOR round = 1 to max_rounds:

    步骤 1 — 假设
    在当前 basis_library 上拟合 SINDy：
        Δ̂ = SINDy(X, Δ, library=basis_library)

    步骤 2 — 测试（质量门控）
    计算留出集导数误差 ε₁
    IF ε₁ < ε_threshold：
        接受：本轮 SINDy 模型可以解释残差
        BREAK

    步骤 3 — 证伪（统计诊断）
    计算余项：Δ_remainder = Δ - Δ̂
    对 Δ_remainder 进行诊断：
        - 自相关检验 → 时间结构？
        - 异方差检验 → 状态相关方差？
        - 正态性检验 → 重尾 / 模式切换？
        - 平稳性检验 → 漂移动力学？

    IF 所有检验通过（余项为纯白噪声）：
        接受：SINDy 捕获了所有可学习结构
        将余项归类为真正的偶然不确定性
        BREAK

    步骤 4 — 扩展（自动化，无需 LLM）
    根据触发的诊断：
        自相关阳性 →
            添加时间延迟特征：s(t-1), s(t-2), Δs/Δt
        异方差阳性（罪魁变量 = j）→
            添加非线性交叉项：s_j², s_j³, s_j*|s_j|, s_j*a
        重尾 / 非正态 →
            添加分段掩码：1(s_j < threshold)（针对可能的接触点）
        非平稳 →
            添加轨迹位置特征：累积步数

    将新特征追加到 basis_library
    CONTINUE 进入下一轮

IF round == max_rounds 且质量门控仍然失败：
    可选：将诊断报告发给 LLM 先知
    IF LLM 提出有效特征（通过 ASTEval 沙箱）：
        添加到 basis_library，再运行一轮 SINDy
    ELSE：
        接受当前最优模型，将余项归类为偶然不确定性
```

### 4.2 为何这不只是"更大的基函数库"

朴素方法：从 degree=5 多项式 + sin + cos + exp + 所有特征开始。这会失败，因为：

1. **组合爆炸：** 17 个状态维度 + 6 个动作维度，仅 degree=5 的多项式就产生约 300,000 个候选项。SINDy 的 L1 回归无法处理。
2. **时间延迟和分段特征不是多项式。** 任何多项式次数都无法捕获 $s(t) - s(t-1)$ 或 $\mathbb{1}(z < 0.05)$。
3. **诊断告诉你在哪里找。** 自相关告诉你"添加时间延迟"，异方差告诉你"专门添加变量 j 的非线性项"。

诊断引导的扩展是**有针对性的搜索**，而非穷举搜索，这使其具备实用性。

### 4.3 四种自动扩展机制

这些是智能体内置的"假设"，无需 LLM：

**机制 1：时间延迟嵌入（Takens 定理）**
- 触发条件：自相关检验阳性
- 动作：添加 $s_{t-1}, s_{t-2}, (s_t - s_{t-1})/\Delta t$ 到基函数库
- 理由：任何耦合的隐变量（风、延迟）都编码在观测历史中
- 实现：需要在残差缓冲区中存储 2 步历史

**机制 2：代数特征交叉**
- 触发条件：异方差检验阳性，罪魁变量 = $j$
- 动作：添加 $s_j^2, s_j^3, s_j \cdot |s_j|, s_j \cdot a_k$ 到基函数库
- 理由：若方差依赖于 $s_j$，均值很可能也有非线性 $s_j$ 依赖
- 实现：对现有特征的纯代数计算

**机制 3：分段逻辑掩码**
- 触发条件：正态性检验阳性（重尾，峰度 > 4）
- 动作：通过残差幅度聚类找到阈值，添加 $\mathbb{1}(s_j < \text{threshold})$ 到基函数库
- 理由：双峰残差表明存在离散物理模式切换（接触/飞行）
- 实现：对 |残差| 做 k-means（k=2），以聚类边界作为阈值

**机制 4：轨迹位置特征**
- 触发条件：平稳性检验阳性（均值漂移）
- 动作：添加归一化步数索引 $t/T$、累积特征
- 理由：仿真-现实差距可能依赖于回合进度（如温度漂移）
- 实现：在残差缓冲区中存储步数索引

### 4.4 LLM 何时以及如何介入（可选）

**前提条件：** 所有 4 种自动机制均已尝试，经过 K=3 轮后质量门控仍然失败。

**LLM 接收到的内容：**
```
系统：你是一名物理特征工程师。[约束条件...]
用户：经过 3 轮自动扩展后，维度 vx 的残差仍然显示出结构。
检验：自相关 负，异方差 正（罪魁：vx，但添加 vx² 和 vx³ 没有帮助），
正态性 负。当前库包含：[1, vx, vx², vx³, vx*|vx|, a0, a0*vx, vx(t-1)]。
我遗漏了什么特征？
```

**LLM 可能返回：**
```json
["obs['vx'] * obs['vz']", "obs['theta'] * obs['vx']**2"]
```

（跨维度的交互特征，自动机制不会尝试，因为它们需要关于哪些维度耦合的物理直觉。）

**安全性：** ASTEval 沙箱，每次查询最多 3 个特征，SINDy 质量门控仍须通过。

---

## 5. 残差模型架构

### 5.1 双轨设计

```
Δ(s,a) ──┬── 轨道 A：SINDy + NAU/NMU（符号化、可外推）
          │   由自我假设循环提供
          │   → Δ_symbolic(s,a)
          │   → 具有正式 OOD 误差界
          │
          └── 轨道 B：集成 NN（灵活，仅插值）
              捕获 SINDy 遗漏的内容
              → Δ_neural(s,a)
              → 无 OOD 保证，门控更严格
```

轨道 A 是主要模型。轨道 B 是轨道 A 扩展库仍无法捕获内容的安全网。轨道 B 使用比轨道 A 更严格（更低）的门控阈值，因为其外推不可靠。

### 5.2 NAU/NMU 输出头

SINDy 发现符号形式，NAU/NMU 在可微网络中强制执行：

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

**OOD 保证（CS-BAPR 定理 4.35）：**
$$\|\hat{\Delta}(s_{\text{ood}},a) - \Delta_{\text{true}}(s_{\text{ood}},a)\| \leq \epsilon + \varepsilon\|d\| + \frac{L}{2}\|d\|^2$$

ReLU 可证明无法满足此条件（定理 4.12）。这是硬性的架构优势。

---

## 6. 不确定性门控纠正

### 6.1 逐元素门控

每个元组元素都有自己的门控：

$$s'_{\text{corrected}} = s'_{\text{sim}} + g_s(s,a) \cdot \hat{\Delta}_s(s,a)$$
$$r_{\text{corrected}} = r_{\text{sim}} + g_r(s,a) \cdot \hat{\Delta}_r(s,a)$$
$$d_{\text{corrected}} = \text{clip}(d_{\text{sim}} + g_d(s,a) \cdot \hat{\Delta}_d(s,a),\; 0,\; 1)$$

### 6.2 门控设计

**轨道 A 门控（符号化，有 OOD 误差界）：**
$$g_A(s,a) = \max\left(0,\; 1 - \frac{\epsilon + \varepsilon\|d\| + \frac{L}{2}\|d\|^2}{\tau}\right)$$

**轨道 B 门控（集成，无 OOD 误差界）：**
$$g_B(s,a) = \sigma\left(\frac{\tau_B - \text{ensemble\_disagreement}(s,a)}{\text{temperature}}\right)$$

轨道 B 门控衰减更快（较低的 $\tau_B$），因为集成模型的外推不可信。

### 6.3 安全保证

**命题：** 在经过校准的 $\tau$ 下，门控纠正永远不会比未纠正的仿真更差：
$$\|T^{\text{gated}} - T^{\text{real}}\| \leq \|T^{\text{sim}} - T^{\text{real}}\|$$

这成立是因为当不确定性增大时 $g \to 0$，优雅地回退到原始仿真。

---

## 7. 策略学习

### 7.1 增强缓冲区构建

```python
buffer = []

# 所有真实数据：完全信任
for (s, a, r, s_next, d) in D_real:
    buffer.append((s, a, r, s_next, d, confidence=1.0))

# 仿真数据：门控纠正
for (s, a) in sim_exploration:
    s_next_sim, r_sim, d_sim = sim.step(s, a)
    s_corr, r_corr, d_corr, gates = gated_correct(s, a, s_next_sim, r_sim, d_sim)
    conf = min(gates)
    if conf > min_threshold:
        buffer.append((s, a, r_corr, s_corr, d_corr, confidence=conf))
```

### 7.2 鲁棒离线强化学习

使用 CORL 中的 IQL 或 CalQL，配合置信度加权的 Critic 损失：

```python
critic_loss = confidence * expectile_loss(Q_target - Q_pred, tau) \
            + (1 - confidence) * lambda_robust * max_penalty
```

高置信度转换正常训练。低置信度转换将价值函数推向悲观方向，完全符合鲁棒强化学习的要求。

---

## 8. 理论保证

### 8.1 全元组 OOD 误差界

对于每个元组元素 $e \in \{s, r, d\}$，SINDy 精度 $\epsilon_e$，Jacobian 一致性 $\varepsilon_e$，NAU/NMU Lipschitz 常数 $L_e$：

$$\|\hat{\Delta}_e(s_{\text{ood}},a) - \Delta_e^{\text{true}}(s_{\text{ood}},a)\| \leq \epsilon_e + \varepsilon_e\|d\| + \frac{L_e}{2}\|d\|^2$$

### 8.2 价值函数误差界

$$|V^{\text{corrected}}(s) - V^{\text{real}}(s)| \leq \frac{1}{1-\gamma}\left(\Delta_r^{\text{bound}} + \gamma\|\nabla_s V\|\Delta_s^{\text{bound}} + \gamma V_{\max}\Delta_d^{\text{bound}}\right)$$

### 8.3 门控安全性

门控纠正是 Pareto 安全的：其性能永远不会低于原始仿真基线。

### 8.4 自我假设循环的单调改进性

每一轮通过质量门控的迭代都会严格缩小偶然不确定性集合，从而弱改进学到的策略。

---

## 9. 实验计划

### 9.1 HP-MuJoCo 基准（多元素差距）

| 环境 | 状态差距 | 奖励差距 | 终止差距 |
|---|---|---|---|
| Aero-Cheetah | 二次阻力 $-kv^2$ | 能量成本不同 | — |
| Ice-Walker | $x>5$ 处摩擦下降 | 速度奖励缩放 | 更软的跌倒阈值 |
| Wind-Hopper | 正弦侧风 | — | 风致跌倒 |
| Carpet-Ant | 阻尼接触 | 电机电流惩罚 | 软跌倒不终止 |

### 9.2 基线方法

| 方法 | 纠正内容 | 方式 | OOD 保证 |
|---|---|---|---|
| H2O+ | 无 | 重要性采样重加权 | 无 |
| DARC | 仅奖励 | 奖励增强 | 无 |
| IGDF | 无 | 对比滤波 | 无 |
| ReDRAW | 状态（潜在空间） | NN 残差 | 无 |
| **我们（仅状态）** | 状态 | SINDy+NAU/NMU | 有 |
| **我们（全元组）** | 所有元素 | SINDy+NAU/NMU | 有 |

### 9.3 关键实验

**实验 1：全元组 vs 仅状态。** 在 Ice-Walker（摩擦 + 终止差距）上：仅状态纠正修正了动力学，但智能体仍因错误的终止而死亡。全元组两者都修正。

**实验 2：OOD 外推。** 在 1x 速度下训练，在 2x/4x/8x 速度下测试。ReDRAW 崩溃，我们的方法呈多项式增长。

**实验 3：自我假设循环消融。** 逐轮展示：基础 SINDy → +自动扩展 → （+LLM 可选）。展示每轮的单调改进。

**实验 4：门控消融。** 无门控 vs 二值门控 vs 连续门控。展示连续门控严格最优。

---

## 10. 实现路线图（10 周）

| 周次 | 阶段 | 里程碑 |
|---|---|---|
| 1 | 环境 + 配对残差提取 | `debug_residuals.png` 显示预期结构 |
| 2 | 单维 SINDy | SINDy 发现 $v^2$ 项，质量门控通过 |
| 3 | 统计诊断模块 | 维度 8 报告异方差，维度 0 干净 |
| 4 | 自我假设循环（无 LLM） | 自动扩展仅从诊断中找到 $v^2$ |
| 5 | 门控 + 增强缓冲区 | 门控策略 > 原始仿真策略（1 个环境） |
| 6 | 全元组扩展 | 奖励 + 终止纠正正常工作 |
| 7-8 | 所有 4 个环境 + 所有基线 | 完整对比表格 |
| 9 | OOD + 消融实验 | 外推曲线与理论匹配 |
| 10 | 论文写作 | 可提交草稿 |

---

## 11. 论文贡献总结

1. **全元组残差世界模型。** 首次对所有状态迁移元素（而非仅状态动力学）建模仿真-现实差距。

2. **可外推残差。** 首个具有正式 OOD 多项式误差界的残差世界模型，通过 SINDy + NAU/NMU 继承 CS-BAPR 保证。解决了 ReDRAW 的根本性 OOD 局限。

3. **自我假设智能体。** 自动诊断引导的基函数扩展，无需外部智能迭代发现残差结构。智能体对仿真-现实差距提出、测试并证伪自身假设。

4. **不确定性门控纠正。** 逐元素门控，可证明纠正效果永远不会差于原始仿真。

5. **（扩展）LLM 作为外部先知。** 当自动扩展不足时，LLM 提供跨维度的物理直觉。作为可选扩展呈现，而非核心贡献。

---

## 12. 代码结构

```
mc-wm/
├── envs/hp_mujoco/              # 4 个基准环境
├── residual/
│   ├── extractor.py             # 全元组配对残差
│   ├── sindy_track.py           # 轨道 A：SINDy + NAU/NMU
│   ├── ensemble_track.py        # 轨道 B：集成备选
│   └── gate.py                  # 逐元素不确定性门控
├── self_audit/
│   ├── diagnosis.py             # 统计诊断电池
│   ├── auto_expand.py           # 4 种自动扩展机制
│   ├── hypothesis_loop.py       # 核心假设-测试-扩展循环
│   └── llm_oracle.py            # 可选 LLM 备选（不在核心路径中）
├── policy/
│   ├── augmented_buffer.py      # 置信度加权缓冲区
│   └── iql_robust.py            # 带鲁棒惩罚的 IQL
├── theory/lean4/                # 形式化证明
└── experiments/                 # 所有实验脚本
```

注意：`llm_oracle.py` 是**唯一**导入 LLM 客户端的文件。删除它，整个系统仍然正常工作。

---

## 13. 关键依赖

| 库 | 用途 | 是否必需？ |
|---|---|---|
| CORL | 基础离线强化学习 | 是 |
| PySINDy | 符号回归 | 是 |
| stable-nalu | NAU/NMU 层 | 是 |
| statsmodels | 统计诊断 | 是 |
| asteval | 安全表达式求值（用于自动扩展 + 可选 LLM） | 是 |
| anthropic/openai | LLM API | **否**（可选扩展） |
