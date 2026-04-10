"""
Robust IQL with Confidence-Weighted Critic Loss (dev manual §7.2).

Based on IQL (Implicit Q-Learning, Kostrikov et al. 2021).
Key modification: critic loss is weighted by sample confidence.

High-confidence transitions: standard expectile regression
Low-confidence transitions: push toward pessimism (max_penalty term)

critic_loss = confidence * expectile_loss(Q_target - Q_pred, τ)
            + (1 - confidence) * λ_robust * max_penalty

Source: structure adapted from CORL IQL baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from copy import deepcopy


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """V(s) — state value function."""
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class QNetwork(nn.Module):
    """Twin Q(s, a) — double Q-learning for pessimism."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


class DeterministicActor(nn.Module):
    """Deterministic actor for offline RL (no entropy term needed)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, act_limit: float = 1.0):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden)
        self.act_limit = act_limit

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s)) * self.act_limit


# ---------------------------------------------------------------------------
# IQL loss functions
# ---------------------------------------------------------------------------

def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Asymmetric L2 loss (expectile regression at level τ).
    IQL uses this for the value function update.
    """
    weight = torch.where(diff >= 0, torch.tensor(tau), torch.tensor(1 - tau))
    return (weight * diff ** 2).mean()


def robust_critic_loss(
    q_pred: torch.Tensor,   # (B, 1)
    q_target: torch.Tensor, # (B, 1)
    confidence: torch.Tensor,  # (B, 1)
    tau: float = 0.7,
    lambda_robust: float = 0.5,
    max_penalty: float = 10.0,
) -> torch.Tensor:
    """
    Confidence-weighted critic loss (dev manual §7.2):
        loss = conf * expectile(Q_target - Q_pred, τ)
               + (1 - conf) * λ * max_penalty

    Low confidence → push Q toward pessimism.
    """
    diff = q_target - q_pred
    eloss = torch.where(diff >= 0, tau * diff ** 2, (1 - tau) * diff ** 2)
    standard = confidence * eloss
    pessimism = (1 - confidence) * lambda_robust * max_penalty * torch.ones_like(eloss)
    return (standard + pessimism).mean()


# ---------------------------------------------------------------------------
# RobustIQL
# ---------------------------------------------------------------------------

class RobustIQL:
    """
    IQL with confidence-weighted critic for MC-WM policy learning.

    Args:
        obs_dim: observation dimension
        act_dim: action dimension
        discount: γ
        tau: IQL expectile level (typically 0.7–0.9)
        beta: actor AWR temperature
        lambda_robust: weight on pessimism penalty
        device: "cpu" or "cuda"
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        discount: float = 0.99,
        tau: float = 0.7,
        beta: float = 3.0,
        lambda_robust: float = 0.5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        soft_target_tau: float = 5e-3,
        max_penalty: float = 10.0,
        device: str = "cpu",
    ):
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.lambda_robust = lambda_robust
        self.soft_target_tau = soft_target_tau
        self.max_penalty = max_penalty
        self.device = device

        self.actor = DeterministicActor(obs_dim, act_dim).to(device)
        self.qf    = QNetwork(obs_dim, act_dim).to(device)
        self.qf_target = deepcopy(self.qf).to(device)
        self.vf    = ValueNetwork(obs_dim).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qf_opt    = optim.Adam(self.qf.parameters(), lr=critic_lr)
        self.vf_opt    = optim.Adam(self.vf.parameters(), lr=value_lr)

        # Freeze target
        for p in self.qf_target.parameters():
            p.requires_grad_(False)

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        One gradient step.

        Args:
            batch: from AugmentedBuffer.sample(); must include 'confidence' key.

        Returns:
            dict of scalar losses for logging.
        """
        # Convert to tensors
        s    = torch.tensor(batch["observations"],      dtype=torch.float32, device=self.device)
        a    = torch.tensor(batch["actions"],           dtype=torch.float32, device=self.device)
        r    = torch.tensor(batch["rewards"],           dtype=torch.float32, device=self.device)
        s2   = torch.tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["dones"],             dtype=torch.float32, device=self.device)
        conf = torch.tensor(batch["confidence"],        dtype=torch.float32, device=self.device)

        # ---- Value function update (IQL) ----
        with torch.no_grad():
            q_target_val = self.qf_target.min_q(s, a)
        v_pred = self.vf(s)
        vf_loss = expectile_loss(q_target_val - v_pred, self.tau)
        self.vf_opt.zero_grad()
        vf_loss.backward()
        self.vf_opt.step()

        # ---- Critic update (confidence-weighted) ----
        with torch.no_grad():
            v_next = self.vf(s2)
            q_target = r + self.discount * (1 - done) * v_next
        q1_pred, q2_pred = self.qf(s, a)
        qf_loss = (
            robust_critic_loss(q1_pred, q_target, conf, self.tau, self.lambda_robust, self.max_penalty)
            + robust_critic_loss(q2_pred, q_target, conf, self.tau, self.lambda_robust, self.max_penalty)
        )
        self.qf_opt.zero_grad()
        qf_loss.backward()
        self.qf_opt.step()

        # ---- Actor update (Advantage-Weighted Regression) ----
        with torch.no_grad():
            adv = self.qf_target.min_q(s, a) - self.vf(s)
            exp_adv = torch.exp(self.beta * adv).clamp(max=100.0)
        a_pred = self.actor(s)
        actor_loss = -(exp_adv * F.mse_loss(a_pred, a, reduction="none").mean(-1, keepdim=True)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Soft target update ----
        for p, p_tgt in zip(self.qf.parameters(), self.qf_target.parameters()):
            p_tgt.data.mul_(1 - self.soft_target_tau)
            p_tgt.data.add_(self.soft_target_tau * p.data)

        return {
            "vf_loss": float(vf_loss),
            "qf_loss": float(qf_loss),
            "actor_loss": float(actor_loss),
        }

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s)
        return a.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "qf":    self.qf.state_dict(),
            "vf":    self.vf.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.qf.load_state_dict(ckpt["qf"])
        self.vf.load_state_dict(ckpt["vf"])
        self.qf_target = deepcopy(self.qf)
