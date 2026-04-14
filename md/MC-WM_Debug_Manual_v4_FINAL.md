# MC-WM Debug Manual v4 — Final

---

## System: 7 Components

```
Residual Extraction → SINDy+AutoExpand(+LLM#2) → NAU/NMU
    → Q_Δ Gate → Constraints(growing, LLM#1+#3) → Robust Policy
```

LLM has 3 roles: #1 initial constraints, #2 feature hypotheses, #3 constraint augmentation.

---

## Phase 0a：安装（Day 1）

```bash
conda create -n mcwm python=3.10 && conda activate mcwm
pip install gymnasium mujoco torch tensorboard pysindy statsmodels scipy asteval
pip install anthropic  # or openai — needed for LLM roles
git clone https://github.com/tinkoff-ai/CORL.git && cd CORL && pip install -e . && cd ..
git clone https://github.com/AndreasMadsen/stable-nalu.git && cd stable-nalu && pip install -e . && cd ..
git clone https://github.com/t6-thu/H2O.git
```

**Milestone 0a：** `import gymnasium, pysindy, statsmodels, asteval, torch, anthropic` 无报错。

---

## Phase 0b：LLM 初始约束（Week 0, LLM Role #1）

```python
# constraints/llm_generator.py
PROMPT = """You are a physical constraint engineer.
Environment: HalfCheetah — 2D planar robot, 9 rigid links.
State dims: [rootx, rootz, rootangle, bthigh_angle, bshin_angle,
             bfoot_angle, fthigh_angle, fshin_angle, ffoot_angle,
             vx, vz, angular_vel, bthigh_vel, bshin_vel,
             bfoot_vel, fthigh_vel, fshin_vel, ffoot_vel]
dt = 0.05s

List:
1. Physically IMPOSSIBLE transitions (joint limits, energy, kinematics)
2. Semantically UNREASONABLE transitions (flying ground robot, impossible speed)
3. Any dimension mapping rules if sim/real have different state dims

Format: JSON list, each {"name": "...", "expr": "python bool, True=violation"}
Use: obs (dict by dim index), obs_next (dict), action (dict), dt."""

response = llm.chat(system=PROMPT, user="Generate 20-30 constraints.")
constraints = json.loads(response)
json.dump(constraints, open('constraints/constraints.json','w'), indent=2)
print(f"Generated {len(constraints)} constraints. REVIEW BEFORE CONTINUING.")
```

**人类审核**（30 min）：打开 JSON，逐条检查，删错的，补漏的。

**验证 false rejection：**
```python
# 需要 Phase 1 的数据，所以实际执行顺序：
# Phase 0b.生成 → Phase 1.收集数据 → Phase 0b.验证 → Phase 1.继续

from asteval import Interpreter
aeval = Interpreter()
false_rej = 0
total = 0
for i in range(len(states)-1):
    obs = {str(j): float(states[i,j]) for j in range(17)}
    obs_next = {str(j): float(states_next[i,j]) for j in range(17)}
    aeval.symtable.update({'obs':obs, 'obs_next':obs_next, 'dt':0.05})
    for c in constraints:
        total += 1
        if aeval(c['expr']):
            false_rej += 1
            print(f"⚠ '{c['name']}' rejected real transition {i}")
rate = false_rej / total
print(f"False rejection: {rate:.4%}")
assert rate < 0.01
```

**Milestone 0b：** `constraints.json` 存在，reviewed，false rejection < 1%。

---

## Phase 1：全元组残差提取（Week 1）

### 1.1 病灶环境

```python
# envs/aero_cheetah.py
class AeroCheetahWrapper(gym.Wrapper):
    def __init__(self, env, drag_coeff=0.5):
        super().__init__(env)
        self.drag_coeff = drag_coeff
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        vx = obs[8]
        drag = -self.drag_coeff * vx * abs(vx)
        obs[8] += drag * self.env.dt
        reward += drag * self.env.dt
        return obs, reward, term, trunc, info
```

### 1.2 收集

```python
def collect(real_env, sim_env, n_ep=50, max_steps=200):
    data = {'states':[],'actions':[],'states_next_real':[],'states_next_sim':[],
            'delta_s':[],'delta_r':[],'delta_d':[]}
    for ep in range(n_ep):
        o_r,_ = real_env.reset(seed=ep)
        o_s,_ = sim_env.reset(seed=ep)
        for _ in range(max_steps):
            a = real_env.action_space.sample()
            o_r2,r_r,d_r,tr_r,_ = real_env.step(a)
            o_s2,r_s,d_s,tr_s,_ = sim_env.step(a)
            data['states'].append(o_r); data['actions'].append(a)
            data['states_next_real'].append(o_r2)
            data['states_next_sim'].append(o_s2)
            data['delta_s'].append(o_r2-o_s2)
            data['delta_r'].append(r_r-r_s)
            data['delta_d'].append(float(d_r)-float(d_s))
            o_r,o_s = o_r2,o_s2
            if d_r or tr_r: break
    return {k:np.array(v) for k,v in data.items()}
```

### 1.3 Debug

```python
print("Per-dim |Δs|:", np.mean(np.abs(data['delta_s']),axis=0).round(4))
print("Mean |Δr|:", np.mean(np.abs(data['delta_r'])).round(4))
```

**Milestone 1：** Dim 8 最大。Δr 非零。

---

## Phase 2：SINDy（Week 2）

```python
import pysindy as ps
X = np.hstack([data['states'], data['actions']])
names = [f's{i}' for i in range(17)] + [f'a{i}' for i in range(6)]

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                 optimizer=ps.STLSQ(threshold=0.01), feature_names=names)
model.fit(X, x_dot=data['delta_s'][:,8:9], t=1.0)
model.print()

# Quality gate
X_tr,X_te,y_tr,y_te = train_test_split(X, data['delta_s'][:,8], test_size=0.2)
model.fit(X_tr, x_dot=y_tr.reshape(-1,1), t=1.0)
err = np.mean(np.abs(model.predict(X_te).flatten() - y_te))
print(f"Quality gate: {err:.4f} {'PASS' if err<0.1 else 'FAIL'}")
```

**Milestone 2：** SINDy 包含 $s_8^2$，quality gate PASS。

---

## Phase 3：统计诊断（Week 3）

```python
def diagnose(resid, exog, alpha=0.05):
    from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
    from scipy.stats import shapiro
    import statsmodels.api as sm
    r = {}
    lb = acorr_ljungbox(resid, lags=[1,3,5,10], return_df=True)
    r['autocorr'] = any(lb['lb_pvalue']<alpha)
    _,p,_,_ = het_breuschpagan(resid, sm.add_constant(exog))
    r['hetero'] = p < alpha
    r['hetero_p'] = p
    _,sp = shapiro(resid[:5000])
    r['normality'] = sp < alpha
    w=len(resid)//5
    wm=[resid[i*w:(i+1)*w].mean() for i in range(5)]
    r['stationary'] = np.std(wm) > 2*np.std(resid)/np.sqrt(w)
    r['any'] = any([r['autocorr'],r['hetero'],r['normality'],r['stationary']])
    return r
```

**Milestone 3：** Dim 8 hetero=True。Dim 0 全 False。

---

## Phase 4：自假设循环（Week 4）⭐

```python
def hypothesis_loop(X, y, states, actions, names, llm_client=None,
                    max_rounds=3, eps_threshold=0.1):
    fnames = names.copy()
    for rnd in range(max_rounds):
        print(f"\n=== Round {rnd+1}, {X.shape[1]} features ===")
        lib = ps.PolynomialLibrary(degree=1 if rnd>0 else 2)
        m = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=0.01),
                     feature_names=fnames)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        m.fit(Xtr, x_dot=ytr.reshape(-1,1), t=1.0); m.print()
        err = np.mean(np.abs(m.predict(Xte).flatten()-yte))
        print(f"Error: {err:.4f}")
        if err < eps_threshold:
            print("✓ PASS"); return m,'resolved',rnd+1

        rem = y - m.predict(X).flatten()
        diag = diagnose(rem, states)
        if not diag['any']:
            print("✓ Aleatoric"); return m,'aleatoric',rnd+1

        # Auto-expand (4 mechanisms)
        expanded = False
        if diag['hetero']:
            corrs=[abs(np.corrcoef(rem**2,states[:,j])[0,1]) for j in range(states.shape[1])]
            j=np.argmax(corrs)
            X=np.hstack([X, states[:,j:j+1]**2, (states[:,j]*np.abs(states[:,j])).reshape(-1,1)])
            fnames.extend([f's{j}_sq',f's{j}_abssq'])
            expanded = True
        if diag['autocorr'] and len(states)>1:
            # Mechanism 1: time delay (requires prev states in buffer)
            # Add finite difference as proxy
            ds = np.diff(states, axis=0, prepend=states[:1])
            X = np.hstack([X, ds])
            fnames.extend([f'ds{j}' for j in range(states.shape[1])])
            expanded = True
        if diag['normality'] and not diag['hetero']:
            # Mechanism 3: piecewise mask
            from sklearn.cluster import KMeans
            for j in range(states.shape[1]):
                km = KMeans(2, n_init=5).fit(states[:,j:j+1])
                th = km.cluster_centers_.mean()
                mask = (states[:,j]<th).astype(float)
                if 0.05 < mask.mean() < 0.95:
                    X = np.hstack([X, mask.reshape(-1,1)])
                    fnames.append(f's{j}_mask')
                    expanded = True; break
        if diag['stationary']:
            # Mechanism 4: step index
            t_idx = np.linspace(0,1,len(y)).reshape(-1,1)
            X = np.hstack([X, t_idx])
            fnames.append('t_norm')
            expanded = True

        if expanded:
            print(f"Auto-expanded to {X.shape[1]} features")
            continue

        # LLM Role #2: feature hypothesis (only if auto-expand exhausted)
        if llm_client is not None:
            print("Auto-expand exhausted. Consulting LLM...")
            prompt = f"Residual dim analysis. Diagnosis: {diag}. "
            prompt += f"Already tried features: {fnames[-10:]}. "
            prompt += "Propose 1-3 new features using only obs dict + numpy."
            resp = llm_client.chat(system=FEATURE_SYSTEM_PROMPT, user=prompt)
            # Parse + ASTEval + add to X
            # (Implementation same as v2, omitted for brevity)
            print(f"LLM proposed features, re-running SINDy...")
            continue
        else:
            print("No LLM available, accepting current best.")
            break

    return m, 'partial', max_rounds
```

**Milestone 4：**
- Aero-Cheetah: `resolved` 或 `aleatoric`
- 首选通过自动机制（无LLM调用）
- 如果自动机制不够，LLM Role #2 介入后 resolve

---

## Phase 5：Q_Δ 训练（Week 5）⭐

```python
class QDelta(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim+adim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,1))
    def forward(self, s, a):
        return self.net(torch.cat([s,a],dim=-1))

def train_qdelta(res_model, data, policy, gamma=0.99, epochs=100):
    Q = QDelta(17,6); Qt = copy.deepcopy(Q)
    opt = torch.optim.Adam(Q.parameters(), lr=1e-3)
    s = torch.tensor(data['states'],dtype=torch.float32)
    a = torch.tensor(data['actions'],dtype=torch.float32)
    snr = torch.tensor(data['states_next_real'],dtype=torch.float32)
    sns = torch.tensor(data['states_next_sim'],dtype=torch.float32)

    for ep in range(epochs):
        dhat = res_model(s,a)
        dtrue = snr - sns
        err = ((dhat-dtrue)**2).sum(-1,keepdim=True)
        sc = sns + dhat.detach()
        an = policy(sc).detach()
        with torch.no_grad():
            target = err + gamma * Qt(sc, an)
        loss = ((Q(s,a)-target)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        for p,pt in zip(Q.parameters(),Qt.parameters()):
            pt.data.copy_(0.995*pt.data + 0.005*p.data)
        if ep%20==0:
            print(f"Ep {ep}: loss={loss:.4f}, meanQ={Q(s,a).mean():.4f}")
    return Q
```

**Debug 检查：**
```python
q_near = Q(s[:100],a[:100]).detach().numpy()
s_far = s[:100].clone(); s_far[:,8] *= 5
q_far = Q(s_far,a[:100]).detach().numpy()
print(f"Near: {q_near.mean():.4f}, Far: {q_far.mean():.4f}")
assert q_far.mean() > q_near.mean()
```

**Milestone 5：** Q_Δ far > Q_Δ near。

---

## Phase 6：Augmented Buffer + Policy（Week 6-7）

```python
def build_buffer(D_real, sim_env, res_model, Q_delta, constraints, policy, n_sim=50000):
    buf = []
    for t in D_real: buf.append((*t, 1.0))  # real: full trust

    # Auto-calibrate tau
    q_vals = Q_delta(s_real, a_real).detach().numpy()
    tau = float(np.percentile(q_vals, 90))

    obs,_ = sim_env.reset()
    for _ in range(n_sim):
        act = policy(obs)
        osn,rs,ds,_,_ = sim_env.step(act)
        ds_hat = res_model_s(obs,act); dr_hat = res_model_r(obs,act)
        sc = osn + ds_hat; rc = rs + dr_hat

        # Constraint filter
        if not passes_constraints(obs, act, sc, constraints):
            obs = osn if not ds else sim_env.reset()[0]; continue

        # Q_Δ gate
        q = Q_delta(obs, act).item()
        g = float(1/(1+np.exp((q-tau)/0.1)))
        sf = osn + g*ds_hat; rf = rs + g*dr_hat

        if g > 0.1: buf.append((obs,act,rf,sf,ds,g))
        obs = osn if not ds else sim_env.reset()[0]

    print(f"Buffer: {len(buf)} ({sum(1 for b in buf if b[-1]==1.0)} real)")
    return buf
```

**Milestone 6：** Corrected policy return > raw sim policy return on real env.

---

## Phase 7：约束增强（Week 7, LLM Role #3）

```python
# constraint_augmentor.py
def augment_constraints(buffer, constraints, Q_delta, llm_client, tau):
    """每N个epoch调用一次。收集可疑修正，请LLM审计。"""

    suspicious = []
    for (s,a,r,sn,d,g) in buffer:
        if g < 1.0 and g > 0.3:  # uncertain zone
            delta_mag = np.linalg.norm(sn - query_sim(s,a))
            if delta_mag > 2 * median_correction:
                suspicious.append({'s':s.tolist(),'a':a.tolist(),
                                   'sn':sn.tolist(),'g':float(g)})

    if len(suspicious) < 3:
        print("Not enough suspicious corrections to audit.")
        return constraints

    # LLM Role #3
    prompt = f"""These {len(suspicious)} corrected transitions passed all existing constraints
but have unusually large corrections. For each, state whether it's
physically impossible or semantically unreasonable. If so, provide a new
constraint rule.

Transitions: {json.dumps(suspicious[:10])}
State meaning: [rootx, rootz, rootangle, ...]

Existing constraints: {[c['name'] for c in constraints]}"""

    response = llm_client.chat(system=CONSTRAINT_AUDIT_PROMPT, user=prompt)
    new_constraints = parse_new_constraints(response)

    # Monotonic growth: only add, never remove
    for nc in new_constraints:
        if nc not in constraints:  # dedup by name
            constraints.append(nc)
            print(f"+ New constraint: {nc['name']}")

    json.dump(constraints, open('constraints/constraints.json','w'), indent=2)
    return constraints
```

**Milestone 7：** 至少 1 条新约束被 LLM 发现并添加。Phase 0 未覆盖的 edge case 被捕获。

---

## Phase 8-9：全实验（Week 8-10）

4 envs × 5 baselines × 10 seeds + ablations + OOD curves + Lean proofs。

---

## Bug 速查表

| 症状 | 原因 | 修复 |
|---|---|---|
| 残差全 0 | real/sim 同一个 env | 检查 wrapper |
| SINDy 系数全 0 | threshold 高 | 降到 0.001 |
| 自假设循环不收敛 | auto_expand 没加对特征 | 检查 culprit |
| Q_Δ near ≈ Q_Δ far | Q 没训练好 | 增加 epoch 或检查 target |
| Q_Δ = inf | gamma 太大 + error 大 | 降 gamma 或 clip error |
| Gate 全 0 | tau 太小 | auto-cal: percentile 90 |
| Gate 全 1 | tau 太大或 Q_Δ 没区分力 | 检查 near vs far |
| 约束误杀 real | 约束太严 | 放宽或删除 |
| LLM Role #2 返回废话 | prompt 约束不够 | 加 few-shot 示例 |
| LLM Role #3 不生成新约束 | 可疑样本不够多或不够离谱 | 降低 suspicious 阈值 |
| Policy 不收敛 | buffer 中 g 分布极端 | 打印 g 的 histogram |

---

## 文件检查点

```
Phase 0b:
  □ constraints/constraints.json (reviewed, <1% false rej)

Phase 1:
  □ paired_residuals.npz
  □ debug_residuals.png

Phase 2:
  □ SINDy: s8^2 found, quality gate PASS

Phase 3:
  □ dim 8: hetero=True, dim 0: all False

Phase 4 ⭐:
  □ hypothesis_loop: 'resolved' or 'aleatoric'
  □ Log: which mechanism(s) fired, whether LLM #2 was called

Phase 5 ⭐:
  □ Q_delta_trained.pt
  □ Q_Δ far > Q_Δ near

Phase 6:
  □ augmented_buffer built
  □ corrected > raw sim

Phase 7:
  □ constraints.json grew (new entries added by LLM #3)
  □ At least 1 edge case caught that Phase 0 missed

Phase 8-9:
  □ All baselines run
  □ All figures generated
  □ Lean proofs complete
```
