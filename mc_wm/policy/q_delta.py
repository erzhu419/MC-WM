"""
QΔ (Residual Bellman Critic): predicts cumulative dynamics gap cost.

QΔ(s,a) = r_Δ + γ E[QΔ(s',a')]

where r_Δ = ||Δ̂(s,a)||² is the SINDy-predicted dynamics gap magnitude.

High QΔ → policy is in a region where sim dynamics diverge from real
         → Q-target should be penalized (Cal-QL style)
Low  QΔ → sim ≈ real → Q-target is trustworthy

This is the "forward problem" use of SINDy:
  NOT correcting sim dynamics, but DETECTING where they're unreliable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


class QDeltaCritic(nn.Module):
    """Small network predicting cumulative dynamics gap cost."""

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, a):
        """Returns QΔ(s,a) shape (B, 1). Always non-negative (gap cost)."""
        sa = torch.cat([s, a], dim=-1)
        return F.softplus(self.net(sa))  # softplus ensures non-negative


class QDeltaModule:
    """
    Manages QΔ training and gap-weighted Q-target computation.

    Usage in training loop:
        1. gap_signal = q_delta_module.compute_gap(s, a)  # from SINDy
        2. q_delta_module.update(s, a, s2, a2, d, gap_signal)  # train QΔ
        3. weights = q_delta_module.get_weights(s, a)  # for Q-target weighting
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128, lr=3e-4,
                 gamma=0.99, tau=5e-3, penalty_scale=0.1, device="cpu"):
        self.gamma = gamma
        self.tau_target = tau
        self.penalty_scale = penalty_scale
        self.device = device

        self.q_delta = QDeltaCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.q_delta_tgt = deepcopy(self.q_delta)
        for p in self.q_delta_tgt.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.q_delta.parameters(), lr=lr)

    def update(self, s, a, s2, a2, d, gap_reward):
        """
        Train QΔ with Bellman backup.

        Args:
            s, a: current state-action (B, dim)
            s2, a2: next state-action (B, dim)
            d: done flags (B, 1)
            gap_reward: r_Δ = ||Δ̂(s,a)||² per transition (B, 1)
        """
        with torch.no_grad():
            q_next = self.q_delta_tgt(s2, a2)
            q_target = gap_reward + self.gamma * (1 - d) * q_next

        q_pred = self.q_delta(s, a)
        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft target update
        for p, pt in zip(self.q_delta.parameters(), self.q_delta_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target)
            pt.data.add_(self.tau_target * p.data)

        return float(loss)

    def get_penalty(self, s, a):
        """
        Returns penalty to subtract from Q-target.
        Higher QΔ → larger penalty → more conservative Q estimate.

        Returns shape (B,) or (B, 1).
        """
        with torch.no_grad():
            q_delta = self.q_delta(s, a)  # (B, 1), non-negative
            return self.penalty_scale * q_delta

    def get_weights(self, s, a, tau_weight=1.0):
        """
        Returns per-transition weights for Q-loss (alternative to penalty).
        w = 1 / (1 + QΔ / τ), smooth sigmoid in [0, 1].

        High QΔ → w ≈ 0 (ignore this transition)
        Low  QΔ → w ≈ 1 (trust this transition)
        """
        with torch.no_grad():
            q_delta = self.q_delta(s, a).squeeze(-1)  # (B,)
            return 1.0 / (1.0 + q_delta / tau_weight)
