"""
QΔ (Residual Bellman Critic) v2: Ensemble + pre-trained on true gap.

Key changes from v1:
1. Ensemble of K QΔ networks (RE-SAC style) → stable, no single-network blowup
2. Pre-trained on TRUE gap from paired data → learns real dynamics distance
3. Penalty clamped → prevents Q-target collapse
4. Frozen after pre-training → no online drift from noisy SINDy proxy

Architecture:
  QΔ_k(s,a) = r_Δ + γ E[QΔ_k(s',a')]  for k=1..K
  penalty = penalty_scale * mean_k(QΔ_k)
  penalty = clamp(penalty, 0, max_penalty)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from copy import deepcopy
import math


class VectorizedLinearSmall(nn.Module):
    """K parallel linear layers."""
    def __init__(self, in_f, out_f, K):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(K, in_f, out_f))
        self.bias = nn.Parameter(torch.empty(K, 1, out_f))
        for i in range(K):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in = in_f
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):  # x: (K, B, in)
        return x @ self.weight + self.bias


class EnsembleQDelta(nn.Module):
    """Ensemble of K QΔ critics, evaluated in one forward pass."""

    def __init__(self, obs_dim, act_dim, hidden_dim=128, K=3):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            VectorizedLinearSmall(obs_dim + act_dim, hidden_dim, K),
            nn.ReLU(),
            VectorizedLinearSmall(hidden_dim, hidden_dim, K),
            nn.ReLU(),
            VectorizedLinearSmall(hidden_dim, 1, K),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)                         # (B, dim)
        sa = sa.unsqueeze(0).repeat_interleave(self.K, dim=0)  # (K, B, dim)
        return F.softplus(self.net(sa).squeeze(-1))             # (K, B) non-negative


class QDeltaModule:
    """
    Ensemble QΔ with pre-training on true gap and clamped penalty.

    Two-phase usage:
    Phase 1 (pre-training): pretrain(paired_trajectories) — uses TRUE Δ
    Phase 2 (policy training): get_penalty(s, a) — frozen, just inference
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128, K=3, lr=3e-4,
                 gamma=0.99, tau=5e-3, penalty_scale=0.1, max_penalty=5.0,
                 device="cpu"):
        self.gamma = gamma
        self.tau_target = tau
        self.penalty_scale = penalty_scale
        self.max_penalty = max_penalty
        self.device = device
        self.K = K
        self._frozen = False

        self.q_delta = EnsembleQDelta(obs_dim, act_dim, hidden_dim, K).to(device)
        self.q_delta_tgt = deepcopy(self.q_delta)
        for p in self.q_delta_tgt.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.q_delta.parameters(), lr=lr)

        # Diagnostics (bounded to prevent unbounded memory growth)
        self._loss_history = deque(maxlen=1000)
        self._penalty_history = deque(maxlen=1000)

    def pretrain(self, trajectories, n_epochs=50, batch_size=256):
        """
        Pre-train QΔ on paired data where TRUE gap is known.

        Args:
            trajectories: list of dicts, each with keys:
                's': (T, obs_dim), 'a': (T, act_dim),
                'gap_reward': (T,) — ||s_real_next - s_sim_next||² per step
                'done': (T,) — episode boundaries
        """
        # Flatten all trajectories into transitions using numpy concatenate
        # (avoids building large Python lists before np.array conversion)
        segs_s, segs_a, segs_s2, segs_a2, segs_r, segs_d = [], [], [], [], [], []
        for traj in trajectories:
            s, a, gr, done = (np.asarray(traj['s']), np.asarray(traj['a']),
                              np.asarray(traj['gap_reward']), np.asarray(traj['done']))
            segs_s.append(s[:-1]); segs_a.append(a[:-1])
            segs_s2.append(s[1:]); segs_a2.append(a[1:])
            segs_r.append(gr[:-1]); segs_d.append(done[:-1])

        s_np   = np.concatenate(segs_s,  axis=0)
        a_np   = np.concatenate(segs_a,  axis=0)
        s2_np  = np.concatenate(segs_s2, axis=0)
        a2_np  = np.concatenate(segs_a2, axis=0)
        r_np   = np.concatenate(segs_r,  axis=0)
        d_np   = np.concatenate(segs_d,  axis=0)
        N = len(s_np)

        s_arr  = torch.FloatTensor(s_np).to(self.device)
        a_arr  = torch.FloatTensor(a_np).to(self.device)
        s2_arr = torch.FloatTensor(s2_np).to(self.device)
        a2_arr = torch.FloatTensor(a2_np).to(self.device)
        r_arr  = torch.FloatTensor(r_np).unsqueeze(-1).to(self.device)
        d_arr  = torch.FloatTensor(d_np).unsqueeze(-1).to(self.device)

        print(f"  QΔ pretrain: {N} transitions, {n_epochs} epochs, K={self.K}")

        for epoch in range(n_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                s, a = s_arr[idx], a_arr[idx]
                s2, a2 = s2_arr[idx], a2_arr[idx]
                r, d = r_arr[idx], d_arr[idx]

                with torch.no_grad():
                    q_next = self.q_delta_tgt(s2, a2)  # (K, B)
                    # Use mean across ensemble for target (stable)
                    q_next_mean = q_next.mean(0)  # (B,)
                    q_target = r.squeeze(-1) + self.gamma * (1 - d.squeeze(-1)) * q_next_mean

                q_pred = self.q_delta(s, a)  # (K, B)
                # Each ensemble member targets the same value
                q_target_exp = q_target.unsqueeze(0).expand_as(q_pred)

                # MSE + OOD penalty (disagreement regularization)
                mse_loss = F.mse_loss(q_pred, q_target_exp.detach())
                ood_loss = q_pred.std(0).mean()
                loss = mse_loss + 0.01 * ood_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Soft target update
                for p, pt in zip(self.q_delta.parameters(), self.q_delta_tgt.parameters()):
                    pt.data.mul_(1 - self.tau_target)
                    pt.data.add_(self.tau_target * p.data)

                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                # Sample penalty stats
                with torch.no_grad():
                    sample_qd = self.q_delta(s_arr[:200], a_arr[:200])  # (K, 200)
                    qd_mean = sample_qd.mean(0)
                    qd_std = sample_qd.std(0)
                print(f"    epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                      f"QΔ mean={qd_mean.mean():.3f} std={qd_std.mean():.3f} "
                      f"max={qd_mean.max():.3f}")

        print(f"  QΔ pretrain done. Final loss: {self._loss_history[-1]:.4f}")

    def freeze(self):
        """Freeze QΔ — no more updates during policy training."""
        self._frozen = True
        for p in self.q_delta.parameters():
            p.requires_grad_(False)

    def update(self, s, a, s2, a2, d, gap_reward):
        """Online update (only if not frozen). Returns loss."""
        if self._frozen:
            return 0.0

        with torch.no_grad():
            q_next = self.q_delta_tgt(s2, a2)  # (K, B)
            q_next_mean = q_next.mean(0)
            q_target = gap_reward.squeeze(-1) + self.gamma * (1 - d.squeeze(-1)) * q_next_mean

        q_pred = self.q_delta(s, a)  # (K, B)
        q_target_exp = q_target.unsqueeze(0).expand_as(q_pred)
        mse_loss = F.mse_loss(q_pred, q_target_exp.detach())
        ood_loss = q_pred.std(0).mean()
        loss = mse_loss + 0.01 * ood_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for p, pt in zip(self.q_delta.parameters(), self.q_delta_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target)
            pt.data.add_(self.tau_target * p.data)

        self._loss_history.append(float(loss))
        return float(loss)

    def get_penalty(self, s, a):
        """
        Clamped penalty from ensemble mean.
        Returns shape (B, 1).
        """
        with torch.no_grad():
            q_all = self.q_delta(s, a)  # (K, B)
            q_mean = q_all.mean(0)       # (B,)
            penalty = self.penalty_scale * q_mean
            penalty = torch.clamp(penalty, 0.0, self.max_penalty)
            self._penalty_history.append(float(penalty.mean()))
            return penalty.unsqueeze(-1)

    def get_diagnostics(self):
        """Return training diagnostics."""
        return {
            "loss_history": list(self._loss_history),
            "penalty_history": list(self._penalty_history),
            "frozen": self._frozen,
        }
