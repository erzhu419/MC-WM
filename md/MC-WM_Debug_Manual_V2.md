# MC-WM Debug & Integration Manual (v2 — Occam's Razor Edition)

## 实战调试手册：自假设Agent的每一步

---

## 核心变更 vs v1

LLM 从核心模块降级为可选 oracle。核心循环变为：

```
诊断 → 自动扩展basis库 → SINDy重拟合 → 质量门
  ↑                                        │
  └──── FAIL ──────────────────────────────┘
  
  连续K轮FAIL → (可选) 请示LLM oracle
```

**调试时，先不接LLM。只有Phase 5（可选）才涉及LLM。**

---

## Phase 0：环境准备（Day 1-2）

```bash
conda create -n mcwm python=3.10 && conda activate mcwm
pip install gymnasium mujoco torch tensorboard pysindy statsmodels scipy asteval
git clone https://github.com/tinkoff-ai/CORL.git && cd CORL && pip install -e . && cd ..
git clone https://github.com/AndreasMadsen/stable-nalu.git && cd stable-nalu && pip install -e . && cd ..
git clone https://github.com/t6-thu/H2O.git  # 环境修改脚本
```

验证：
```python
import gymnasium, pysindy, statsmodels, asteval, torch
print("All OK")
```

**Milestone 0：** 无报错。

---

## Phase 1：制造病人 + 全元组残差提取（Week 1）

### 1.1 制造 Aero-Cheetah（已知病灶：$-0.5 v^2$ 阻力）

```python
# envs/aero_cheetah.py
import gymnasium as gym
import numpy as np

class AeroCheetahWrapper(gym.Wrapper):
    def __init__(self, env, drag_coeff=0.5):
        super().__init__(env)
        self.drag_coeff = drag_coeff

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vx = obs[8]
        drag = -self.drag_coeff * vx * abs(vx)
        obs[8] += drag * self.env.dt          # state gap
        reward += drag * self.env.dt           # reward gap
        return obs, reward, terminated, truncated, info
```

### 1.2 收集全元组残差

```python
# collect_residuals.py
def collect_full_tuple_residuals(real_env, sim_env, n_episodes=50, max_steps=200):
    data = {'states':[], 'actions':[], 'delta_s':[], 'delta_r':[], 'delta_d':[]}

    for ep in range(n_episodes):
        obs_r, _ = real_env.reset(seed=ep)
        obs_s, _ = sim_env.reset(seed=ep)
        for _ in range(max_steps):
            a = real_env.action_space.sample()
            obs_r2, r_r, d_r, tr_r, _ = real_env.step(a)
            obs_s2, r_s, d_s, tr_s, _ = sim_env.step(a)
            data['states'].append(obs_r)
            data['actions'].append(a)
            data['delta_s'].append(obs_r2 - obs_s2)
            data['delta_r'].append(r_r - r_s)
            data['delta_d'].append(float(d_r) - float(d_s))
            obs_r, obs_s = obs_r2, obs_s2
            if d_r or tr_r: break

    return {k: np.array(v) for k, v in data.items()}
```

### 1.3 Debug 检查

```python
# debug_phase1.py
data = np.load('paired_residuals.npz')
# 检查1：dim 8 残差最大
print("Per-dim |Δs|:", np.mean(np.abs(data['delta_s']), axis=0).round(4))
# 检查2：reward残差非零
print("Mean |Δr|:", np.mean(np.abs(data['delta_r'])).round(4))
# 检查3：Δs[8] vs vx² 线性
import matplotlib.pyplot as plt
vx = data['states'][:, 8]
plt.scatter(vx**2 * np.sign(vx), data['delta_s'][:, 8], alpha=0.1, s=1)
plt.savefig('debug_phase1.png')
```

**Milestone 1：**
- dim 8 的 mean |Δs| 明显大于其他维度
- Δr 非零
- `debug_phase1.png` 第三张图显示线性关系

---

## Phase 2：SINDy 提取符号残差（Week 2）

### 2.1 基础 SINDy（degree=2 多项式库）

```python
import pysindy as ps

X = np.hstack([data['states'], data['actions']])
names = [f's{i}' for i in range(17)] + [f'a{i}' for i in range(6)]

# State dim 8
model_s8 = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.01),
    feature_names=names
)
model_s8.fit(X, x_dot=data['delta_s'][:, 8:9], t=1.0)
model_s8.print()

# Reward
model_r = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.01),
    feature_names=names
)
model_r.fit(X, x_dot=data['delta_r'].reshape(-1, 1), t=1.0)
model_r.print()
```

### 2.2 Quality Gate

```python
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, data['delta_s'][:, 8], test_size=0.2)
model_s8.fit(X_tr, x_dot=y_tr.reshape(-1,1), t=1.0)
y_pred = model_s8.predict(X_te).flatten()
error = np.mean(np.abs(y_pred - y_te))
print(f"Quality gate: error={error:.4f}, {'PASS ✓' if error < 0.1 else 'FAIL ✗'}")
```

**Milestone 2：**
- SINDy 输出包含 $s_8^2$ 项
- 系数接近 $-0.5 \times dt$
- Quality gate PASS

**Debug 诊断树（如果 FAIL）：**
```
系数全0？→ 降 threshold 到 0.001
太多非零项？→ 升 threshold 到 0.05
没有 s8² 项？→ print(library.fit_transform(X[:3])) 检查库内容
R² 低？→ 试 degree=3
```

---

## Phase 3：统计诊断模块（Week 3）

### 3.1 诊断电池实现

```python
# diagnosis.py
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from scipy.stats import shapiro
import statsmodels.api as sm

def diagnose(residuals, conditioning_vars, var_names=None, alpha=0.05):
    report = {}

    # 自相关
    lb = acorr_ljungbox(residuals, lags=[1,3,5,10], return_df=True)
    sig_lags = lb[lb['lb_pvalue'] < alpha].index.tolist()
    report['autocorr'] = {'positive': len(sig_lags) > 0, 'lags': sig_lags}

    # 异方差
    exog = sm.add_constant(conditioning_vars)
    _, bp_p, _, _ = het_breuschpagan(residuals, exog)
    culprit = None
    if bp_p < alpha and var_names:
        corrs = [abs(np.corrcoef(residuals**2, conditioning_vars[:,j])[0,1])
                 for j in range(conditioning_vars.shape[1])]
        culprit = var_names[np.argmax(corrs)]
    report['hetero'] = {'positive': bp_p < alpha, 'p': bp_p, 'culprit': culprit}

    # 正态性
    _, sh_p = shapiro(residuals[:5000])
    kurt = float(np.mean((residuals - residuals.mean())**4) / residuals.std()**4)
    report['normality'] = {'positive': sh_p < alpha, 'kurtosis': kurt}

    # 平稳性
    w = len(residuals) // 5
    wmeans = [residuals[i*w:(i+1)*w].mean() for i in range(5)]
    drift = np.std(wmeans) / (np.std(residuals) / np.sqrt(w))
    report['stationarity'] = {'positive': drift > 2.0, 'drift': drift}

    report['any_positive'] = any(v['positive'] for v in report.values() if isinstance(v, dict))
    return report
```

### 3.2 验证：有结构的维度 vs 无结构的维度

```python
# test_diagnosis.py
r8 = diagnose(data['delta_s'][:,8], data['states'],
              [f's{i}' for i in range(17)])
r0 = diagnose(data['delta_s'][:,0], data['states'],
              [f's{i}' for i in range(17)])

print("Dim 8 (有阻力):", {k:v.get('positive','') for k,v in r8.items() if isinstance(v,dict)})
print("Dim 0 (无结构):", {k:v.get('positive','') for k,v in r0.items() if isinstance(v,dict)})
```

**Milestone 3：**
- Dim 8: hetero=True, culprit='s8'
- Dim 0: 全 False

---

## Phase 4：自假设循环——无需LLM（Week 4）⭐ 核心

### 4.1 自动基库扩展器

```python
# auto_expand.py
import numpy as np

def auto_expand_library(diagnosis_report, current_features, states, actions, dt=0.05):
    """基于诊断报告自动生成新特征。不需要LLM。"""
    new_features = []
    new_names = []
    n_state = states.shape[1]

    # 机制1：自相关 → 时间延迟
    if diagnosis_report['autocorr']['positive']:
        # 需要 states_prev 已存在于 buffer 中
        # 这里假设 collect 时已保存
        pass  # 在实际实现中加入 s(t-1), ds/dt

    # 机制2：异方差 → 非线性项
    if diagnosis_report['hetero']['positive']:
        culprit = diagnosis_report['hetero']['culprit']
        if culprit:
            j = int(culprit.replace('s', ''))  # 's8' → 8
            sj = states[:, j]
            new_features.append(sj**2 * np.sign(sj))  # s_j * |s_j|
            new_names.append(f'{culprit}_abs_sq')
            new_features.append(sj**2)
            new_names.append(f'{culprit}_sq')
            new_features.append(sj**3)
            new_names.append(f'{culprit}_cube')
            # 和动作的交叉项
            for k in range(actions.shape[1]):
                new_features.append(sj * actions[:, k])
                new_names.append(f'{culprit}_x_a{k}')

    # 机制3：非正态/重尾 → 分段掩码
    if diagnosis_report['normality']['positive'] and diagnosis_report['normality']['kurtosis'] > 4:
        # 用残差大小做 k-means(k=2) 找分割点
        from sklearn.cluster import KMeans
        for j in range(n_state):
            km = KMeans(n_clusters=2, n_init=5).fit(states[:, j:j+1])
            threshold = km.cluster_centers_.mean()
            mask = (states[:, j] < threshold).astype(float)
            if mask.mean() > 0.05 and mask.mean() < 0.95:  # 非退化
                new_features.append(mask)
                new_names.append(f's{j}_mask_{threshold:.2f}')
                break  # 只加一个最可能的

    # 机制4：非平稳 → 轨迹位置
    if diagnosis_report['stationarity']['positive']:
        # 简单加 normalized step index（需要在buffer中记录）
        pass

    if new_features:
        return np.column_stack(new_features), new_names
    return None, []
```

### 4.2 完整自假设循环

```python
# hypothesis_loop.py
import pysindy as ps
import numpy as np
from diagnosis import diagnose
from auto_expand import auto_expand_library

def self_hypothesizing_loop(X_base, y_target, states, actions, var_names,
                            max_rounds=3, epsilon_threshold=0.1):
    """
    核心循环：假设 → SINDy → 质量门 → 诊断 → 自动扩展 → 重复
    全程不需要LLM。
    """
    X = X_base.copy()
    feature_names = var_names.copy()
    best_model = None
    best_error = float('inf')

    for round_idx in range(max_rounds):
        print(f"\n=== Round {round_idx + 1} ===")
        print(f"Library size: {X.shape[1]} features")

        # STEP 1: HYPOTHESIZE — fit SINDy
        # 只用一次项（因为非线性已经在特征里了）
        lib = ps.PolynomialLibrary(degree=1 if round_idx > 0 else 2)
        model = ps.SINDy(feature_library=lib,
                         optimizer=ps.STLSQ(threshold=0.01),
                         feature_names=feature_names)

        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_target, test_size=0.2,
                                                    random_state=42)
        model.fit(X_tr, x_dot=y_tr.reshape(-1,1), t=1.0)
        model.print()

        # STEP 2: TEST — quality gate
        y_pred = model.predict(X_te).flatten()
        error = np.mean(np.abs(y_pred - y_te))
        print(f"Quality gate: error={error:.4f} (threshold={epsilon_threshold})")

        if error < best_error:
            best_error = error
            best_model = model

        if error < epsilon_threshold:
            print("✓ PASSED — structure fully captured")
            return best_model, 'epistemic_resolved', round_idx + 1

        # STEP 3: FALSIFY — diagnose remainder
        remainder = y_target - model.predict(X).flatten()
        report = diagnose(remainder, states, [f's{i}' for i in range(states.shape[1])])
        print(f"Diagnosis: {', '.join(k for k,v in report.items() if isinstance(v,dict) and v.get('positive'))}")

        if not report['any_positive']:
            print("✓ All tests negative — remainder is genuine aleatoric noise")
            return best_model, 'aleatoric_confirmed', round_idx + 1

        # STEP 4: EXPAND — automatic, no LLM
        new_feats, new_names = auto_expand_library(report, feature_names,
                                                    states, X_tr[:, -actions.shape[1]:])
        if new_feats is not None:
            X = np.hstack([X, new_feats])
            feature_names.extend(new_names)
            print(f"Added features: {new_names}")
        else:
            print("⚠ No automatic expansion available")
            break

    print(f"✗ Max rounds reached. Best error: {best_error:.4f}")
    return best_model, 'partially_resolved', max_rounds
```

### 4.3 端到端测试

```python
# test_self_hypothesis.py
data = np.load('paired_residuals.npz')
X = np.hstack([data['states'], data['actions']])
names = [f's{i}' for i in range(17)] + [f'a{i}' for i in range(6)]

model, status, rounds = self_hypothesizing_loop(
    X_base=X,
    y_target=data['delta_s'][:, 8],
    states=data['states'],
    actions=data['actions'],
    var_names=names,
    max_rounds=3,
    epsilon_threshold=0.1
)

print(f"\nResult: {status} in {rounds} rounds")
assert status in ['epistemic_resolved', 'aleatoric_confirmed'], \
    f"Expected resolution, got {status}"
```

**Milestone 4（最关键的milestone）：**
- Round 1: degree=2 多项式库，可能直接找到 $s_8^2$ → PASS
- 或 Round 1 FAIL → 诊断报 heteroscedasticity(culprit=s8) → 自动加 $s_8^2, s_8|s_8|$ → Round 2 PASS
- **不需要任何LLM参与**
- 打印 `epistemic_resolved in N rounds`

**如果 Milestone 4 没过：**
```
还是 FAIL？
├── 检查 auto_expand 是否真的加了新特征
│   print(X.shape) 每轮应该变宽
├── 新特征加了但 SINDy 没选中？
│   降低 STLSQ threshold
├── 诊断全阴性但 quality gate 没过？
│   → 这说明残差确实是 aleatoric，status 应为 'aleatoric_confirmed'
│   → 如果 error 仍然很大，说明你的病灶不是简单的二次关系
└── 诊断阳性但 auto_expand 返回 None？
    → expand 逻辑的 if 分支没覆盖该诊断类型
```

---

## Phase 5（可选）：LLM Oracle 接入（Week 5 如果需要）

**只有在 Phase 4 的自动扩展无法解决某些维度时才需要。**

### 5.1 判断是否需要 LLM

```python
if status == 'partially_resolved' and best_error > epsilon_threshold:
    print("Automated expansion insufficient. Consider LLM oracle.")
    # 此时把 diagnosis report 格式化发给 LLM
else:
    print("No LLM needed. Self-hypothesizing loop sufficient.")
```

### 5.2 LLM 接口（和 v1 相同，但定位变了）

```python
# llm_oracle.py — OPTIONAL, system works without this file
class LLMOracle:
    """当自动扩展机制用尽时的最后手段。"""
    def __init__(self, client, obs_keys, action_keys):
        self.client = client
        self.obs_keys = obs_keys
        self.action_keys = action_keys

    def consult(self, diagnosis_report, dim_name, attempted_features):
        """
        attempted_features: 自动扩展已经尝试过的特征列表
        LLM 需要知道什么已经试过了，避免重复
        """
        prompt = f"""Residual dimension: {dim_name}
Diagnosis: {diagnosis_report}
Already tried features (didn't help): {attempted_features}
What cross-dimensional or physics-based feature might I be missing?
Output JSON list of 1-3 expressions using only obs dict keys."""

        response = self.client.chat(system=SYSTEM_PROMPT, user=prompt)
        return self._parse_and_validate(response)
```

### 5.3 安全集成

```python
# 在 hypothesis_loop.py 的末尾追加：
if status == 'partially_resolved' and use_llm:
    oracle = LLMOracle(client, obs_keys, action_keys)
    llm_features = oracle.consult(last_diagnosis, dim_name, tried_features)
    # 用 ASTEval 安全执行
    # 加入 SINDy 库再跑一轮
    # 如果 quality gate 过了 → status 变为 'epistemic_resolved_with_oracle'
```

**Milestone 5（仅在需要时）：** LLM 提出的特征让 quality gate 通过。

---

## Phase 6：Gate + Policy + 全实验（Week 5-9）

### 6.1 Gate 实现（同 v1）

```python
# gate.py
from sklearn.neighbors import BallTree

class UncertaintyGate:
    def __init__(self, real_sa, tau=0.5):
        self.tree = BallTree(real_sa)
        self.tau = tau

    def __call__(self, query_sa, epsilon=0.01, varepsilon=0.1, L=0.0):
        d = self.tree.query(query_sa.reshape(1,-1), k=1)[0][0,0]
        bound = epsilon + varepsilon * d + (L/2) * d**2
        return max(0.0, 1.0 - bound / self.tau)
```

### 6.2 全元组修正

```python
# 对 state、reward、termination 各自独立跑 self_hypothesizing_loop
# 各自独立有 SINDy model + gate
models = {}
for element, target in [('state_8', delta_s[:,8]), ('reward', delta_r)]:
    model, status, rounds = self_hypothesizing_loop(X, target, states, actions, names)
    models[element] = (model, status)
    print(f"{element}: {status} in {rounds} rounds")
```

### 6.3 Policy 训练 + OOD 测试（同 v1）

省略——逻辑不变，参见 Development Manual。

---

## 常见 Bug 速查表 (v2)

| 症状 | 原因 | 修复 |
|---|---|---|
| 自假设循环第一轮就 PASS | degree=2 库已经够了 | 正常！说明病灶简单 |
| 循环卡在 expand=None | 诊断阳性但 expand 逻辑没覆盖 | 补充对应的 expand 分支 |
| 每轮加特征但 error 不降 | 新特征和目标不相关 | 检查 culprit 变量是否正确 |
| SINDy 在扩展库上反而更差 | 特征太多导致过拟合 | 提高 STLSQ threshold |
| 诊断在大样本下全部阳性 | 统计检验过敏 | 加 effect size 阈值 |
| LLM 返回无法执行的表达式 | prompt 约束不够严格 | 加入更多 few-shot 示例 |
| Gate 全为 0 | tau 太小 | auto-calibrate: tau = median(sim-real gap) |

---

## 文件检查点

```
Phase 1:
  □ paired_residuals.npz
  □ debug_phase1.png

Phase 2:
  □ sindy_state_model (printed to console)
  □ sindy_reward_model (printed to console)
  □ quality_gate_result.txt

Phase 3:
  □ diagnosis_dim8.json (hetero=True)
  □ diagnosis_dim0.json (all False)

Phase 4 ⭐:
  □ self_hypothesis_log.txt (shows round-by-round)
  □ status = 'epistemic_resolved' 或 'aleatoric_confirmed'
  □ NO LLM CALLS MADE

Phase 5 (optional):
  □ llm_oracle_log.txt (if needed)

Phase 6:
  □ augmented_buffer.npz
  □ gate_visualization.png
  □ baseline_comparison.csv
  □ ood_extrapolation.png
```

最大的系统性风险
残差模型的递归OOD问题（你之前提出的灰犀牛）。 你在sim中探索的大部分(s,a)离offline real data很远，残差预测在那里不可靠。gate理论上能解决这个问题（远处g→0），但这意味着远处的sim数据基本没被修正，和不用残差模型差别不大。
换句话说：残差模型真正有效的区域可能只是offline data coverage附近的一个壳。如果offline data很少（你的设定），这个壳很薄，残差模型的增量价值有限。
这是整个方案最需要实验验证的点。 如果实验发现gate把95%的sim数据都退化为raw sim（g≈0），那这篇论文的practical impact就很弱——理论上有OOD bound，但实际上大部分数据用不上。

建议的最小可行性验证： 在Aero-Cheetah单环境上，只做state残差（不做full-tuple），不做自假设循环（手动用degree=2 SINDy），看corrected sim训练的policy是否显著优于raw sim训练的policy。这2周就能出结果，能回答"这条路值不值得走"。