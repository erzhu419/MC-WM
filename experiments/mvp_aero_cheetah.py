"""
MVP: Aero-Cheetah State-Only SINDy Correction

验证核心问题：corrected-sim 训练的 policy 是否显著优于 raw-sim 训练的 policy？

Pipeline（无自假设循环、无 full-tuple、无 gate）：
  A. 用随机策略收集配对数据 (sim, real) 共 N_COLLECT 步
  B. 用 degree=2 SINDy 拟合 Δs(s, a)，打印发现的符号结构
  C. 建立 CorrectedAeroCheetahEnv（sim + SINDy Δs 修正每一步）
  D. 分别在 raw_sim / corrected_sim 上训练 SAC，相同 budget
  E. 每隔 EVAL_INTERVAL 步在 real_env 上评估，记录 return
  F. 输出对比曲线 mvp_comparison.png + gate_coverage.txt

灰犀牛风险检验（Debug Manual 末尾）：
  额外打印在 corrected sim 上 SINDy 预测幅度与 raw delta 的比值
  → 如果修正量普遍很小，说明数据覆盖范围太窄，实际增益有限

运行：conda run -n MC-WM python3 experiments/mvp_aero_cheetah.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import deque

import pysindy as ps
import gymnasium as gym

from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_COLLECT      = 5_000    # 步数，用于 SINDy 拟合
TRAIN_STEPS    = 100_000  # 每组 SAC 训练步数（正式）
EVAL_INTERVAL  = 5_000    # 每隔多少步评估一次
N_EVAL_EPS     = 5        # 每次评估的 episode 数
WARMUP         = 2_000    # warmup 步数（模块级常量，可从外部覆盖）
BATCH_SIZE     = 256
REPLAY_SIZE    = 200_000
LR             = 3e-4
GAMMA          = 0.99
TAU            = 5e-3
ALPHA_INIT     = 0.2
HIDDEN         = 256
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.dirname(__file__)

# ─────────────────────────────────────────────
# Phase A: 收集配对数据
# ─────────────────────────────────────────────

def collect_paired_data(n_steps: int, seed: int = 0):
    """同步 step sim 和 real，收集残差数据。"""
    sim_env  = AeroCheetahEnv(mode="sim",  seed=seed)
    real_env = AeroCheetahEnv(mode="real", seed=seed)

    states, actions, delta_s_list = [], [], []
    obs_s, _ = sim_env.reset(seed=seed)
    obs_r, _ = real_env.reset(seed=seed)
    ep = 0

    for _ in range(n_steps):
        a = sim_env.action_space.sample()
        obs_s2, _, d_s, tr_s, _ = sim_env.step(a)
        obs_r2, _, d_r, tr_r, _ = real_env.step(a)

        states.append(obs_s.copy())
        actions.append(a.copy())
        delta_s_list.append(obs_r2 - obs_s2)

        obs_s, obs_r = obs_s2, obs_r2
        if d_s or tr_s or d_r or tr_r:
            ep += 1
            obs_s, _ = sim_env.reset(seed=ep + seed)
            obs_r, _ = real_env.reset(seed=ep + seed)

    sim_env.close(); real_env.close()
    SA      = np.concatenate([np.array(states), np.array(actions)], axis=1).astype(np.float32)
    delta_s = np.array(delta_s_list, dtype=np.float32)
    print(f"[A] 收集了 {len(SA)} 步配对数据，{ep} 个 episode")
    return SA, delta_s


# ─────────────────────────────────────────────
# Phase B: SINDy 拟合
# ─────────────────────────────────────────────

class SINDyStateCorrector:
    """
    用 degree=2 SINDy 拟合每个 obs dim 的残差 Δs_i(s, a)。
    predict(s, a) → Δs 的 numpy 向量。
    """

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self.models = []
        self.library = ps.PolynomialLibrary(degree=2, include_bias=True)
        self.fit_errors = np.zeros(obs_dim)

    def fit(self, SA: np.ndarray, delta_s: np.ndarray):
        """对每个 obs dim 单独拟合。"""
        print("\n[B] SINDy 拟合（degree=2）...")
        self.library.fit(SA)
        Theta = np.asarray(self.library.transform(SA))   # 转为普通 ndarray
        feature_names = self.library.get_feature_names()

        self.models = []
        for i in range(self.obs_dim):
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=0.01, fit_intercept=False)
            reg.fit(Theta, delta_s[:, i])
            # STLSQ 一轮阈值化
            mask = np.abs(reg.coef_) > 0.005
            if mask.any():
                reg2 = Ridge(alpha=0.01, fit_intercept=False)
                reg2.fit(Theta[:, mask], delta_s[:, i])
                coef_full = np.zeros(len(feature_names))
                coef_full[mask] = reg2.coef_
            else:
                coef_full = reg.coef_

            self.models.append(coef_full)
            y_pred = Theta @ coef_full
            self.fit_errors[i] = float(np.mean((delta_s[:, i] - y_pred)**2))

            # 打印非零项（只打印有结构的 dim）
            nonzero = [(feature_names[j], coef_full[j])
                       for j in range(len(feature_names)) if abs(coef_full[j]) > 0.005]
            if nonzero:
                terms = ", ".join(f"{n}:{c:.4f}" for n, c in nonzero[:5])
                print(f"  dim {i:2d}: [{terms}]  MSE={self.fit_errors[i]:.5f}")

        # 保存 transform，并包一层 np.asarray 避免 AxesArray 问题
        self._raw_transform = self.library.transform
        self.Theta_transform = lambda x: np.asarray(self._raw_transform(x))
        print(f"  平均 MSE: {self.fit_errors.mean():.5f}")
        print(f"  最大 MSE: {self.fit_errors.max():.5f}  (dim {self.fit_errors.argmax()})")

    def predict(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """单步预测 Δs，shape (obs_dim,)。"""
        SA = np.concatenate([s, a])[None]  # (1, obs_dim + act_dim)
        Theta = self.Theta_transform(SA)   # (1, n_features)
        delta = np.array([Theta[0] @ coef for coef in self.models])
        return delta

    def predict_batch(self, SA: np.ndarray) -> np.ndarray:
        """批量预测 Δs，shape (N, obs_dim)。"""
        Theta = self.Theta_transform(SA)
        return np.array([Theta @ coef for coef in self.models]).T  # (N, obs_dim)

    def correction_coverage(self, SA: np.ndarray, delta_s_true: np.ndarray) -> dict:
        """
        灰犀牛风险检验：
        correction_ratio = mean |Δ̂| / mean |Δ_true|
        如果接近 0 → SINDy 没学到东西
        如果接近 1 → 修正量合理
        """
        delta_pred = self.predict_batch(SA)
        ratio = (np.abs(delta_pred).mean() /
                 (np.abs(delta_s_true).mean() + 1e-8))
        rmse_uncorrected = np.sqrt(np.mean(delta_s_true**2))
        rmse_corrected   = np.sqrt(np.mean((delta_s_true - delta_pred)**2))
        return {
            "correction_ratio": float(ratio),
            "rmse_raw_sim": float(rmse_uncorrected),
            "rmse_corrected_sim": float(rmse_corrected),
            "rmse_reduction_pct": float(100 * (1 - rmse_corrected / (rmse_uncorrected + 1e-8))),
        }


# ─────────────────────────────────────────────
# Phase C: Corrected Sim Env
# ─────────────────────────────────────────────

class CorrectedAeroCheetahEnv(gym.Env):
    """
    sim_env + 在每步 step 后加上 SINDy 预测的 Δs。
    reward 和 termination 不修正（MVP 只做 state-only）。
    """

    def __init__(self, corrector: SINDyStateCorrector, seed: int = 0):
        self._env = AeroCheetahEnv(mode="sim", seed=seed)
        self.corrector = corrector
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._last_obs is not None:
            delta = self.corrector.predict(self._last_obs, action)
            obs = obs + delta
        self._last_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────
# SAC: 极简实现（~150 行）
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=200_000):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0
        self.s  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r  = np.zeros((capacity, 1),       dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d  = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return (torch.FloatTensor(self.s[idx]).to(DEVICE),
                torch.FloatTensor(self.a[idx]).to(DEVICE),
                torch.FloatTensor(self.r[idx]).to(DEVICE),
                torch.FloatTensor(self.s2[idx]).to(DEVICE),
                torch.FloatTensor(self.d[idx]).to(DEVICE))


def mlp(dims, act=nn.ReLU):
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i+1]), act()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit=1.0):
        self.act_limit = act_limit
        h = HIDDEN

        self.actor    = mlp([obs_dim, h, h, act_dim * 2]).to(DEVICE)
        self.q1       = mlp([obs_dim + act_dim, h, h, 1]).to(DEVICE)
        self.q2       = mlp([obs_dim + act_dim, h, h, 1]).to(DEVICE)
        self.q1_tgt   = deepcopy(self.q1)
        self.q2_tgt   = deepcopy(self.q2)
        for p in list(self.q1_tgt.parameters()) + list(self.q2_tgt.parameters()):
            p.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(ALPHA_INIT), dtype=torch.float32,
                                                    device=DEVICE))
        self.target_entropy = -act_dim

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LR)
        self.opt_q     = optim.Adam(list(self.q1.parameters()) +
                                    list(self.q2.parameters()), lr=LR)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=LR)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _policy(self, s, deterministic=False):
        out = self.actor(s)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        if deterministic:
            a = torch.tanh(mean) * self.act_limit
            return a, None
        z = mean + std * torch.randn_like(mean)
        a = torch.tanh(z) * self.act_limit
        log_prob = (torch.distributions.Normal(mean, std).log_prob(z)
                    - torch.log(1 - a.pow(2) / self.act_limit**2 + 1e-6)
                   ).sum(-1, keepdim=True)
        return a, log_prob

    def get_action(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        with torch.no_grad():
            s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            a, _ = self._policy(s, deterministic)
        return a.squeeze(0).cpu().numpy()

    def update(self, buf: ReplayBuffer):
        s, a, r, s2, d = buf.sample(BATCH_SIZE)

        with torch.no_grad():
            a2, lp2 = self._policy(s2)
            sa2 = torch.cat([s2, a2], -1)
            q_next = torch.min(self.q1_tgt(sa2), self.q2_tgt(sa2)) - self.alpha * lp2
            q_target = r + GAMMA * (1 - d) * q_next

        sa = torch.cat([s, a], -1)
        q1_loss = F.mse_loss(self.q1(sa), q_target)
        q2_loss = F.mse_loss(self.q2(sa), q_target)
        self.opt_q.zero_grad(); (q1_loss + q2_loss).backward(); self.opt_q.step()

        a_new, lp = self._policy(s)
        sa_new = torch.cat([s, a_new], -1)
        q_val  = torch.min(self.q1(sa_new), self.q2(sa_new))
        actor_loss = (self.alpha.detach() * lp - q_val).mean()
        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

        for p, pt in zip(list(self.q1.parameters()) + list(self.q2.parameters()),
                         list(self.q1_tgt.parameters()) + list(self.q2_tgt.parameters())):
            pt.data.mul_(1 - TAU); pt.data.add_(TAU * p.data)

        return float(q1_loss), float(actor_loss)


# ─────────────────────────────────────────────
# Phase D+E: 训练 + 评估
# ─────────────────────────────────────────────

def evaluate_on_real(agent: SACAgent, n_eps: int = N_EVAL_EPS) -> float:
    real_env = AeroCheetahEnv(mode="real")
    returns = []
    for ep in range(n_eps):
        obs, _ = real_env.reset(seed=ep + 100)
        total_r = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, _ = real_env.step(a)
            total_r += r
            if d or tr: break
        returns.append(total_r)
    real_env.close()
    return float(np.mean(returns))


def train_sac(train_env_fn, label: str, seed: int = SEED,
              train_steps: int = None, eval_interval: int = None,
              warmup: int = None, n_eval_eps: int = None):
    """训练 SAC，周期性在 real env 上评估，返回 eval 曲线。"""
    _train_steps   = train_steps   or TRAIN_STEPS
    _eval_interval = eval_interval or EVAL_INTERVAL
    _warmup        = warmup        or WARMUP
    _n_eval_eps    = n_eval_eps    or N_EVAL_EPS

    np.random.seed(seed); torch.manual_seed(seed)

    env = train_env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_limit)
    buf   = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)

    obs, _ = env.reset(seed=seed)
    eval_curve = []
    steps_since_eval = 0

    print(f"\n[D] 训练 {label}  ({_train_steps // 1000}k steps, warmup={_warmup})...", flush=True)
    for step in range(1, _train_steps + 1):
        if step < _warmup:
            a = env.action_space.sample()
        else:
            a = agent.get_action(obs)

        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2

        if d or tr:
            obs, _ = env.reset()

        if step >= _warmup and buf.size >= BATCH_SIZE:
            agent.update(buf)

        steps_since_eval += 1
        if steps_since_eval >= _eval_interval:
            ret = evaluate_on_real(agent, n_eps=_n_eval_eps)
            eval_curve.append((step, ret))
            steps_since_eval = 0
            print(f"  step {step:>7d} | real_return = {ret:.1f}", flush=True)

    env.close()
    return eval_curve, agent


# ─────────────────────────────────────────────
# Phase F: 输出
# ─────────────────────────────────────────────

def plot_comparison(curve_raw, curve_corr, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps_r, rets_r = zip(*curve_raw)  if curve_raw  else ([], [])
    steps_c, rets_c = zip(*curve_corr) if curve_corr else ([], [])
    ax.plot(steps_r, rets_r, label="Raw Sim",       color="steelblue",  lw=2)
    ax.plot(steps_c, rets_c, label="Corrected Sim", color="darkorange", lw=2)
    ax.set_xlabel("Training steps"); ax.set_ylabel("Real env return (avg 5 eps)")
    ax.set_title("MVP: AeroCheetah — Corrected vs Raw Sim → Real Transfer")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=100)
    print(f"[F] 对比图已保存 → {out_path}")


def save_sindy_report(corrector, coverage, out_path):
    lines = ["=== SINDy Coverage Report (灰犀牛风险检验) ===\n"]
    lines.append(f"correction_ratio      : {coverage['correction_ratio']:.4f}")
    lines.append(f"  (0=没修正任何东西, 1=修正量与真实残差等量)")
    lines.append(f"RMSE raw sim          : {coverage['rmse_raw_sim']:.5f}")
    lines.append(f"RMSE corrected sim    : {coverage['rmse_corrected_sim']:.5f}")
    lines.append(f"RMSE reduction        : {coverage['rmse_reduction_pct']:.1f}%")
    lines.append(f"\n每个 dim 的 SINDy 拟合 MSE:")
    for i, e in enumerate(corrector.fit_errors):
        lines.append(f"  dim {i:2d}: {e:.5f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(out_path, "w") as f:
        f.write(txt)
    print(f"\n[F] Coverage report → {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MC-WM MVP: Aero-Cheetah State-Only SINDy Correction")
    print("=" * 60)

    # ── A: 收集配对数据
    SA, delta_s = collect_paired_data(N_COLLECT)
    obs_dim = delta_s.shape[1]
    act_dim = SA.shape[1] - obs_dim

    # ── B: SINDy 拟合
    corrector = SINDyStateCorrector(obs_dim)
    corrector.fit(SA, delta_s)

    # 灰犀牛风险检验（用收集数据本身）
    coverage = corrector.correction_coverage(SA, delta_s)

    # ── D+E: 训练对比
    curve_raw, _ = train_sac(
        lambda: AeroCheetahEnv(mode="sim"),
        label="Raw Sim",
        seed=SEED,
    )

    curve_corr, _ = train_sac(
        lambda: CorrectedAeroCheetahEnv(corrector),
        label="Corrected Sim",
        seed=SEED,
    )

    # ── F: 输出
    plot_comparison(
        curve_raw, curve_corr,
        os.path.join(OUT_DIR, "mvp_comparison.png"),
    )
    save_sindy_report(
        corrector, coverage,
        os.path.join(OUT_DIR, "sindy_coverage.txt"),
    )

    # 最终对比总结
    if curve_raw and curve_corr:
        final_raw  = np.mean([r for _, r in curve_raw[-3:]])
        final_corr = np.mean([r for _, r in curve_corr[-3:]])
        print("\n" + "=" * 60)
        print(f"最终 real return (最后3次eval均值):")
        print(f"  Raw sim       : {final_raw:.1f}")
        print(f"  Corrected sim : {final_corr:.1f}")
        gain = final_corr - final_raw
        sign = "✓ 修正有效" if gain > 50 else ("△ 微弱改善" if gain > 0 else "✗ 无改善")
        print(f"  增益          : {gain:+.1f}  → {sign}")
        print("=" * 60)


if __name__ == "__main__":
    main()
