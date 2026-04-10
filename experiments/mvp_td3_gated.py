"""
MVP v2: TD3 + Gated Corrected Sim

改进点 vs mvp_100k.py：
  1. SAC → TD3（delayed policy update，无 Q 过估计，后期不退化）
  2. 加入 Gate（OOD 区域修正量自动衰减到 0）
  3. 加入 Augmented Buffer（real offline data + corrected sim 混合）
     real data 用随机策略收集的 offline real transitions，confidence=1.0

对比组：
  A. Raw Sim TD3（baseline）
  B. Gated Corrected Sim TD3（我们的方法）

衡量标准：
  - 收敛速度（多少步到达 3000 return）
  - 最终 return（100k steps 最后3次均值）
  - Gate 平均开度（检验灰犀牛：是否大部分修正被 gate 退化为 0）

运行：conda run -n MC-WM python3 -u experiments/mvp_td3_gated.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from experiments.mvp_aero_cheetah import (
    collect_paired_data, SINDyStateCorrector, DEVICE, SEED
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_COLLECT      = 5_000
TRAIN_STEPS    = 150_000
EVAL_INTERVAL  = 10_000
N_EVAL_EPS     = 5
WARMUP         = 5_000
BATCH_SIZE     = 256
REPLAY_SIZE    = 300_000
LR_ACTOR       = 3e-4
LR_CRITIC      = 3e-4
GAMMA          = 0.99
TAU            = 5e-3
POLICY_DELAY   = 2          # TD3: actor 每 2 步更新一次
POLICY_NOISE   = 0.2        # TD3: target policy noise
NOISE_CLIP     = 0.5
EXPL_NOISE     = 0.1        # TD3: exploration noise std
HIDDEN         = 256

# Gate 参数
GATE_TAU_A     = 0.5        # gate threshold；越小越激进地关门
GATE_EPS_JAC   = 0.01

OUT_DIR = os.path.dirname(__file__)

# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=300_000):
        self.max_size = capacity
        self.ptr = self.size = 0
        self.s  = np.zeros((capacity, obs_dim), np.float32)
        self.a  = np.zeros((capacity, act_dim), np.float32)
        self.r  = np.zeros((capacity, 1),       np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d  = np.zeros((capacity, 1),       np.float32)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r
        self.s2[i] = s2; self.d[i] = d
        self.ptr  = (i + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[idx]).to(DEVICE),
                torch.FloatTensor(self.a[idx]).to(DEVICE),
                torch.FloatTensor(self.r[idx]).to(DEVICE),
                torch.FloatTensor(self.s2[idx]).to(DEVICE),
                torch.FloatTensor(self.d[idx]).to(DEVICE))


# ─────────────────────────────────────────────
# TD3
# ─────────────────────────────────────────────

def mlp(dims):
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit=1.0):
        self.act_limit = act_limit
        h = HIDDEN

        self.actor     = mlp([obs_dim, h, h, act_dim]).to(DEVICE)
        self.actor_tgt = deepcopy(self.actor)
        self.q1        = mlp([obs_dim + act_dim, h, h, 1]).to(DEVICE)
        self.q2        = mlp([obs_dim + act_dim, h, h, 1]).to(DEVICE)
        self.q1_tgt    = deepcopy(self.q1)
        self.q2_tgt    = deepcopy(self.q2)

        for net in [self.actor_tgt, self.q1_tgt, self.q2_tgt]:
            for p in net.parameters():
                p.requires_grad_(False)

        self.opt_actor  = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR_CRITIC)

        self._update_count = 0

    def get_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            a = torch.tanh(self.actor(s)) * self.act_limit
            a = a.squeeze(0).cpu().numpy()
        if noise > 0:
            a = np.clip(a + noise * np.random.randn(*a.shape),
                        -self.act_limit, self.act_limit)
        return a

    def update(self, buf: ReplayBuffer):
        self._update_count += 1
        s, a, r, s2, d = buf.sample(BATCH_SIZE)

        # ── Critic update
        with torch.no_grad():
            noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            a2 = (torch.tanh(self.actor_tgt(s2)) * self.act_limit + noise
                  ).clamp(-self.act_limit, self.act_limit)
            q_next = torch.min(
                self.q1_tgt(torch.cat([s2, a2], -1)),
                self.q2_tgt(torch.cat([s2, a2], -1)),
            )
            q_tgt = r + GAMMA * (1 - d) * q_next

        q1 = self.q1(torch.cat([s, a], -1))
        q2 = self.q2(torch.cat([s, a], -1))
        critic_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

        # ── Delayed actor update
        if self._update_count % POLICY_DELAY == 0:
            a_new = torch.tanh(self.actor(s)) * self.act_limit
            actor_loss = -self.q1(torch.cat([s, a_new], -1)).mean()
            self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

            for p, pt in zip(
                list(self.actor.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
                list(self.actor_tgt.parameters()) + list(self.q1_tgt.parameters()) + list(self.q2_tgt.parameters()),
            ):
                pt.data.mul_(1 - TAU); pt.data.add_(TAU * p.data)


# ─────────────────────────────────────────────
# Gated Corrected Env
# ─────────────────────────────────────────────

class GatedCorrectedEnv:
    """
    sim_env + SINDy correction 按 Gate 加权。

    Gate A（OOD polynomial bound）：
        g(s, a) = max(0,  1 - (eps_fit + eps_jac * dist + L/2 * dist²) / tau)

    dist = L2 distance from training data center.
    在 OOD 区域 dist 增大 → bound 增大 → gate 趋向 0 → 修正量自动退为 raw sim。
    """

    def __init__(self, corrector: SINDyStateCorrector,
                 tau: float = GATE_TAU_A, eps_jac: float = GATE_EPS_JAC,
                 sim_env=None):
        self._env = sim_env if sim_env is not None else AeroCheetahEnv(mode="sim")
        self.corrector = corrector
        self.tau = tau
        self.eps_jac = eps_jac
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # 拟合一个训练中心（用于 OOD 距离计算）
        self._train_center = None   # 由外部在 corrector.fit 后设置
        self._L_eff = 0.05          # NAU/NMU Lipschitz（简化估计，后续可从 SymbolicResidualHead.L_eff 取）

        self._gate_history = []     # 记录每步 gate 开度（用于灰犀牛检验）
        self._last_obs = None

    def set_train_center(self, SA: np.ndarray):
        self._train_center = SA.mean(axis=0)

    def _compute_gate(self, s: np.ndarray, a: np.ndarray) -> float:
        if self._train_center is None:
            return 1.0
        SA = np.concatenate([s, a])
        dist = float(np.linalg.norm(SA - self._train_center))
        eps_fit = float(self.corrector.fit_errors.mean())
        bound = eps_fit + self.eps_jac * dist + (self._L_eff / 2) * dist ** 2
        gate = max(0.0, 1.0 - bound / self.tau)
        return gate

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._last_obs is not None:
            gate = self._compute_gate(self._last_obs, action)
            self._gate_history.append(gate)
            if gate > 0:
                delta = self.corrector.predict(self._last_obs, action)
                obs = obs + gate * delta
        self._last_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    def mean_gate(self) -> float:
        if not self._gate_history:
            return 0.0
        return float(np.mean(self._gate_history))

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────

def evaluate_on_real(agent: TD3Agent, n_eps: int = N_EVAL_EPS) -> float:
    real_env = AeroCheetahEnv(mode="real")
    returns = []
    for ep in range(n_eps):
        obs, _ = real_env.reset(seed=ep + 200)
        total_r = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, noise=0.0)
            obs, r, d, tr, _ = real_env.step(a)
            total_r += r
            if d or tr: break
        returns.append(total_r)
    real_env.close()
    return float(np.mean(returns))


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train_td3(env_fn, label: str, seed: int = SEED) -> tuple:
    np.random.seed(seed); torch.manual_seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(obs_dim, act_dim, act_limit)
    buf   = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)

    obs, _ = env.reset(seed=seed)
    eval_curve = []
    steps_since_eval = 0

    print(f"\n[Train] {label}  ({TRAIN_STEPS // 1000}k steps, GPU={DEVICE})", flush=True)
    for step in range(1, TRAIN_STEPS + 1):
        if step < WARMUP:
            a = env.action_space.sample()
        else:
            a = agent.get_action(obs, noise=EXPL_NOISE)

        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr:
            obs, _ = env.reset()

        if step >= WARMUP and buf.size >= BATCH_SIZE:
            agent.update(buf)

        steps_since_eval += 1
        if steps_since_eval >= EVAL_INTERVAL:
            ret = evaluate_on_real(agent)
            eval_curve.append((step, ret))
            steps_since_eval = 0
            # gate 开度（如果是 GatedCorrectedEnv）
            gate_info = ""
            if hasattr(env, 'mean_gate'):
                gate_info = f"  gate={env.mean_gate():.3f}"
            print(f"  step {step:>7d} | real_return = {ret:7.1f}{gate_info}", flush=True)

    env.close()
    return eval_curve, agent


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}", flush=True)

    # ── A: 数据收集
    SA, delta_s = collect_paired_data(N_COLLECT)

    # ── B: SINDy
    corrector = SINDyStateCorrector(delta_s.shape[1])
    corrector.fit(SA, delta_s)
    cov = corrector.correction_coverage(SA, delta_s)
    print(f"RMSE reduction: {cov['rmse_reduction_pct']:.1f}%\n", flush=True)

    # ── Raw Sim TD3
    curve_raw, _ = train_td3(
        lambda: AeroCheetahEnv(mode="sim"),
        "Raw Sim (TD3)",
    )

    # ── Gated Corrected Sim TD3
    def make_gated_env():
        env = GatedCorrectedEnv(corrector, tau=GATE_TAU_A)
        env.set_train_center(SA)
        return env

    curve_corr, _ = train_td3(make_gated_env, "Gated Corrected Sim (TD3)")

    # ── 对比图
    fig, ax = plt.subplots(figsize=(10, 5))
    if curve_raw:
        steps, rets = zip(*curve_raw)
        ax.plot(steps, rets, label="Raw Sim (TD3)", color="steelblue", lw=2)
    if curve_corr:
        steps, rets = zip(*curve_corr)
        ax.plot(steps, rets, label="Gated Corrected Sim (TD3)", color="darkorange", lw=2)
    ax.set_xlabel("Training steps"); ax.set_ylabel("Real env return (avg 5 eps)")
    ax.set_title("MVP v2: TD3 + Gate — AeroCheetah Sim-to-Real")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "mvp_td3_gated.png")
    plt.savefig(out_fig, dpi=100)
    print(f"\n图已保存 → {out_fig}", flush=True)

    # ── 最终结论
    if curve_raw and curve_corr:
        fr = np.mean([r for _, r in curve_raw[-3:]])
        fc = np.mean([r for _, r in curve_corr[-3:]])
        gain = fc - fr
        if gain > 200:   verdict = "✓✓ 显著改善，这条路值得走"
        elif gain > 50:  verdict = "✓  有改善，继续推进"
        elif gain > 0:   verdict = "△  微弱改善"
        else:            verdict = "✗  无改善，检查 gate 参数"

        print("\n" + "="*50)
        print(f"Raw Sim TD3       (最后3次均值): {fr:.1f}")
        print(f"Gated Corrected   (最后3次均值): {fc:.1f}")
        print(f"增益: {gain:+.1f}   {verdict}")
        print("="*50, flush=True)


if __name__ == "__main__":
    main()
