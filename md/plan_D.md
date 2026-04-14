逐维度对比，不偏不倚。

## V4的优势

**Gate机制：Q_Δ > distance-based。** V2的gate基于"离offline data多远"——一个纯几何量，不考虑修正本身的质量。V4的Q_Δ基于"从这里开始整条轨迹的累积修正误差"。具体场景：SINDy发现了完美的$v^2$阻力公式，远离offline data的高速区域，V2的gate关闭（距离远），V4的gate开放（公式准确，Q_Δ低）。反方向：近offline data但在接触切换点，V2的gate开放（距离近），V4的gate关闭（修正会在切换后compound）。Q_Δ严格信息更多。

**约束系统：单调递增 > 静态。** V2的约束在Phase 0生成后freeze。V4的约束持续增长——训练中LLM Role #3发现Phase 0遗漏的edge case，补充新规则。这意味着V4的系统越训越安全，V2停留在初始安全水平。

**LLM参与更精确。** V2里LLM要么不参与（核心path），要么作为"最后手段"模糊地介入。V4给了LLM三个精确的接口：初始约束、特征假设、约束审计。每个接口的输入输出格式、触发条件、频率都明确定义。更容易实现、测试、debug。

**没有哲学矛盾。** V2有Track B（ensemble NN黑箱拟合）和discriminator（统计判断），这两个和"理解WHY而非黑箱拟合"的核心主张矛盾。V4没有任何黑箱组件——SINDy是白箱，NAU/NMU是白箱，Q_Δ虽然用NN但它估计的是一个有明确贝尔曼定义的量，LLM约束是人类可读的boolean规则。

**理论贡献更强。** V2的理论全部继承自CS-BAPR，没有新的。V4新增了Residual Bellman Equation——这是一个genuinely new的形式化概念，有独立的contraction proof，可以作为论文的核心理论贡献。

**组件更少但能力更强。** V2有14个活动组件（含子组件），V4有7个。V2的multi-step trajectory check、残差meta-constraints、constraint violation feedback都被Q_Δ一个机制替代了。

## V4的劣势

**Q_Δ本身面临bootstrapping问题。** Q_Δ用corrected next state做TD target，如果correction错了，next state就偏了，Q_Δ的估计就不准，gate就不可靠。V2的distance-based gate没有这个递归风险——距离就是距离，不依赖任何模型。V4用pessimistic estimation缓解，但没有从根本上解决。在offline data极其稀少（<100 transitions）的场景下，Q_Δ可能训不出有意义的信号，distance-based gate反而更稳健。

**失去了discriminator的"统计常识"。** Discriminator学到的是"real transitions长什么样"的全局统计模式。LLM约束覆盖的是"什么不可能/不合理"的逻辑规则。但两者之间有一个gap：**统计上极不寻常但逻辑上不违反任何规则的transition**。比如一个HalfCheetah以极其罕见的步态行走——每一条物理约束都满足，但discriminator会说"从没见过这种步态"。V4抓不到这类case。实际影响取决于这类case在你的benchmark中有多常见——在MuJoCo里可能很少，但在更复杂的环境中可能重要。

**Q_Δ需要额外的训练步骤和计算。** V2的distance-based gate是O(log N)的BallTree查询，零训练成本。V4需要训一个完整的Q网络，和policy的Q网络同等规模。在资源受限时这不是trivial的overhead。而且Q_Δ的hyperparameters（gamma、pessimism程度、tau的percentile选择）比distance gate的单一tau更多。

**LLM约束增长带来的一致性风险。** V2的约束集静态，一旦验证过就稳定。V4的约束集在训练中增长，新约束可能和已有的策略学习进度产生微妙的冲突——比如policy已经学会利用某个"可疑但有用"的correction pattern，然后LLM Role #3把它禁了。虽然"安全只增不减"是对的，但policy的训练稳定性可能受影响。V2没有这个问题。

**LLM的总调用量增加。** V2中LLM只在Phase 0调用一次。V4在训练中额外调用Role #2（特征假设）和Role #3（约束审计），虽然频率低但引入了对外部API的运行时依赖。如果API挂了或响应慢，训练不会停（可以skip），但会失去持续改进的能力。

**可复现性略降。** V2的训练过程完全确定（Phase 0的LLM输出经人类审核后freeze）。V4的训练中有LLM调用，LLM输出有随机性。虽然新约束经ASTEval验证且人类可以async审核，但两次训练可能产生不同的约束集。对于严格的实验可复现性，需要fix random seed并log所有LLM交互。

## 总结

| 维度 | V2更好 | V4更好 |
|---|---|---|
| Gate quality (信息量) | | ✓ (Q_Δ > distance) |
| Gate robustness (极端数据少时) | ✓ (distance无递归风险) | |
| 约束覆盖 | | ✓ (持续增长) |
| 训练稳定性 | ✓ (静态约束) | |
| 理论贡献 | | ✓ (Residual Bellman) |
| 实现复杂度 | ✓ (BallTree vs Q网络) | |
| 哲学一致性 | | ✓ (无黑箱) |
| 统计异常检测 | ✓ (discriminator) | |
| 可复现性 | ✓ (训练中无LLM) | |
| 计算成本 | ✓ (distance gate免费) | |

**如果我必须选一个：** 选V4。Q_Δ和单调约束增长是genuinely更好的设计，劣势都可以mitigate（Q_Δ bootstrapping用pessimism、训练稳定性用新约束只在epoch boundary生效、可复现性用log所有LLM交互）。而V2的discriminator和Track B是无法升级成V4的对应物的——它们是不同paradigm。

**最务实的做法：** V4为主，但在实验中加一个"V4 with distance gate instead of Q_Δ"的ablation，量化Q_Δ相对于distance gate的增量。如果增量<5%，在论文中同时呈现两个版本，让读者根据自己的计算预算选择。



评估四种配置：

**配置A：V4原版（Q_Δ + LLM约束）**
- 缺口：统计异常但逻辑合法的transition无人管
- 例子：关节角度组合从未在real中出现过，但每个角度单独看都合法

**配置B：V4 + discriminator**
- 补上了统计异常的缺口
- discriminator和LLM约束正交：discriminator管"像不像"，LLM约束管"可不可能"
- 代价：多一个网络要训，多一个OOD可靠性问题（discriminator自己在OOD也不准）
- 但discriminator不准的worst case是false accept（放过了不该放的），不是false reject，因为reject后fallback到raw sim是安全的

**配置C：V4 + distance gate**
- Q_Δ在数据极少时可能训不出信号，distance gate是zero-cost的fallback
- 两者信息互补：distance是纯几何信号（不依赖任何模型），Q_Δ是轨迹级信号（依赖残差模型质量）
- 可以取min：$g = \min(g_{\text{Q\_\Delta}}, g_{\text{distance}})$，任何一个说不行就不行
- 代价：几乎为零（BallTree查询）

**配置D：V4 + discriminator + distance gate**
- 三个gate信号：Q_Δ（轨迹级质量）、distance（数据覆盖）、discriminator（统计像真）
- 加上LLM约束（逻辑可能性）
- 四个正交过滤器，各管各的维度，没有冗余
- 代价：discriminator训练 + BallTree构建，都不在热路径上

**我的判断：配置D是最完备的，且不违反奥卡姆剃刀。** 四个信号各自捕捉不同failure mode，没有任何一个能被另一个替代：

| 信号 | 捕捉什么 | 其他三个能替代吗 |
|---|---|---|
| Q_Δ | 修正沿轨迹compound的风险 | 不能——其他三个都是单步判断 |
| Distance | 离已知数据有多远 | 不能——Q_Δ依赖模型质量，模型差时distance更可靠 |
| Discriminator | 修正后的state像不像real | 不能——LLM约束管逻辑不管统计分布 |
| LLM约束 | 物理上可不可能、语义上合不合理 | 不能——discriminator只学了统计模式没学物理 |

如果你认同配置D，V4 final版本的组件数变成8（加回discriminator和distance gate）。但我觉得这已经不需要更新manual了——你知道这两个东西怎么实现（V2里都有），加进去是纯工程决策。