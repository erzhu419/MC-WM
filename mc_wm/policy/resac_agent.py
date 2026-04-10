"""
RE-SAC Agent (PyTorch, gymnasium-compatible)

提取自 LSTM-RL/RE-SAC/sac_ensemble_original_logging.py 的核心思想：
  - Ensemble critic (VectorizedCritic): N 个 Q 函数并行，一次前向
  - LCB policy loss:  L = -(Q_mean + β * Q_std).mean()
    β < 0 → 悲观（避免 Q 过估计）
    β > 0 → 乐观（探索）
  - OOD loss: + β_ood * Q_std（惩罚高方差的 Q 估计）
  - BC loss:  + β_bc * MSE(π(s), a_behavior)（行为克隆正则）

相比原版的简化：
  - 去掉 EmbeddingLayer（bus-specific）
  - 去掉 state_norm / reward_scaling（外部可选）
  - 去掉 argparse 全局变量，改为构造函数参数
  - 适配 gymnasium continuous action spaces

核心公式（来自原版 compute_policy_loss）：
    q_dist = ensemble_Q(s, π(s))           # shape (N_critics, B)
    q_mean = q_dist.mean(0)
    q_std  = q_dist.std(0)
    policy_loss = -(q_mean + β * q_std).mean() + β_bc * MSE(π(s), a_data)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from copy import deepcopy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Vectorized Critic（直接移植自原版）
# ─────────────────────────────────────────────

class VectorizedLinear(nn.Module):
    """N 个线性层并行，权重 shape (N, in, out)。"""
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias   = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        for i in range(ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):               # x: (N, B, in)
        return x @ self.weight + self.bias  # (N, B, out)


class EnsembleCritic(nn.Module):
    """
    Ensemble of N Q-networks, evaluated in one forward pass.
    forward(s, a) → Q values shape (N_critics, B)
    """
    def __init__(self, obs_dim, act_dim, hidden_dim, n_critics):
        super().__init__()
        self.n = n_critics
        self.net = nn.Sequential(
            VectorizedLinear(obs_dim + act_dim, hidden_dim, n_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, n_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, n_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, n_critics),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)                        # (B, obs+act)
        sa = sa.unsqueeze(0).repeat_interleave(self.n, dim=0) # (N, B, obs+act)
        return self.net(sa).squeeze(-1)                        # (N, B)


# ─────────────────────────────────────────────
# Gaussian Policy（标准 SAC actor）
# ─────────────────────────────────────────────

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, act_limit=1.0,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        nn.init.uniform_(self.mean_head.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)

    def forward(self, s):
        h = self.net(s)
        mean    = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, s, eps=1e-6):
        """Reparameterized sample + log_prob (TanhNormal)."""
        mean, log_std = self.forward(s)
        std = log_std.exp()
        z   = Normal(0, 1).sample(mean.shape).to(s.device)
        a0  = torch.tanh(mean + std * z)
        a   = a0 * self.act_limit
        log_prob = (Normal(mean, std).log_prob(mean + std * z)
                    - torch.log(1 - a0.pow(2) + eps)
                    ).sum(dim=-1)
        return a, log_prob, mean

    def get_action(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        with torch.no_grad():
            s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            mean, log_std = self.forward(s)
            if deterministic:
                a = torch.tanh(mean) * self.act_limit
            else:
                std = log_std.exp()
                z = Normal(0, 1).sample(mean.shape).to(s.device)
                a = torch.tanh(mean + std * z) * self.act_limit
        return a.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────
# RE-SAC Agent
# ─────────────────────────────────────────────

class RESACAgent:
    """
    RE-SAC: Ensemble-critic SAC with LCB policy loss.

    Key hyperparameters:
        n_critics  : ensemble size (原版默认 10，MuJoCo 用 5 足够)
        beta       : LCB coefficient (< 0 → pessimistic，原版默认 -2)
        beta_ood   : OOD penalty on Q_std (原版默认 0.01)
        beta_bc    : behavior cloning weight (原版默认 0.001)
        critic_actor_ratio : critic 每 N 步 actor 更新一次 (原版默认 2)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float = 1.0,
        hidden_dim: int = 256,
        n_critics: int = 5,
        beta: float = -2.0,
        beta_ood: float = 0.01,
        beta_bc: float = 0.001,
        critic_actor_ratio: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha_init: float = 0.2,
        max_alpha: float = 0.6,
        device: str = DEVICE,
    ):
        self.gamma = gamma
        self.tau   = tau
        self.beta  = beta
        self.beta_ood = beta_ood
        self.beta_bc  = beta_bc
        self.critic_actor_ratio = critic_actor_ratio
        self.max_alpha = max_alpha
        self.device = device
        self._update_count = 0

        self.actor = GaussianActor(obs_dim, act_dim, hidden_dim, act_limit).to(device)
        self.critic     = EnsembleCritic(obs_dim, act_dim, hidden_dim, n_critics).to(device)
        self.critic_tgt = deepcopy(self.critic)
        for p in self.critic_tgt.parameters():
            p.requires_grad_(False)

        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32, device=device))
        self.target_entropy = -act_dim

        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_alpha  = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return min(self.max_alpha, self.log_alpha.exp().item())

    def update(self, buf):
        self._update_count += 1
        s, a, r, s2, d = buf.sample(256)

        # ── Alpha update
        _, log_prob, _ = self.actor.evaluate(s)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

        # ── Critic update
        with torch.no_grad():
            a2, lp2, _ = self.actor.evaluate(s2)
            q_next = self.critic_tgt(s2, a2)       # (N, B)
            # 用最小 Q（悲观）作为 target（类似 TD3）
            q_next_min = q_next.min(0).values       # (B,)
            q_tgt = r.squeeze(-1) + self.gamma * (1 - d.squeeze(-1)) * (q_next_min - self.alpha * lp2)

        q_pred = self.critic(s, a)                 # (N, B)
        q_tgt_exp = q_tgt.unsqueeze(0).expand_as(q_pred)
        ood_loss = q_pred.std(0).mean()
        critic_loss = F.mse_loss(q_pred, q_tgt_exp.detach()) + self.beta_ood * ood_loss
        self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

        # ── Actor update（每 critic_actor_ratio 步一次）
        if self._update_count % self.critic_actor_ratio == 0:
            a_new, lp_new, _ = self.actor.evaluate(s)
            q_dist = self.critic(s, a_new)          # (N, B)
            q_mean = q_dist.mean(0)
            q_std  = q_dist.std(0)

            # LCB + entropy：
            #   -(Q_mean + β*Q_std) 是 L1-LCB（std 线性进入 loss），
            #   单独用时 policy 可能在低 std 区域无限变尖；
            #   加上 +α*log_π 的 SAC 熵项充当 floor，防止分布坍缩。
            policy_loss = -(q_mean + self.beta * q_std - self.alpha * lp_new).mean()
            # BC 正则（轻微向离线数据靠拢）
            bc_loss = F.mse_loss(a_new, a)
            actor_loss = policy_loss + self.beta_bc * bc_loss

            self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

            # Soft target update
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)

        return float(critic_loss), float(alpha_loss)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.get_action(obs, deterministic)
