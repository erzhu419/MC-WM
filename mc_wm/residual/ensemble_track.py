"""
Track B: Ensemble NN residual model (flexible, interpolation only).

Adapted from CS-BAPR/sindy-rl/sindy_rl/dynamics.py (EnsembleNetDynamicsModel).
Key changes:
  - Operates on residuals Δ(s,a), not full states
  - Full-tuple outputs: Δs, Δr, Δd together
  - Tracks ensemble disagreement for gate (dev manual §6.2)
  - Aggressively gated: τ_B < τ_A because no OOD guarantee

Disagreement metric: std across ensemble members.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple


class EnsembleMember(nn.Module):
    """Single feedforward net predicting (Δs, Δr, Δd) from (s, a)."""

    def __init__(self, input_dim: int, obs_dim: int, hidden: int = 256):
        super().__init__()
        output_dim = obs_dim + 1 + 1  # Δs + Δr + Δd
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden, output_dim),
        )
        self._obs_dim = obs_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        delta_s = out[:, :self._obs_dim]
        delta_r = out[:, self._obs_dim:self._obs_dim + 1]
        delta_d = out[:, self._obs_dim + 1:]
        return delta_s, delta_r, delta_d


class EnsembleTrack:
    """
    Track B: ensemble of 5 neural nets for residual modeling.
    No OOD guarantee — gated more aggressively than Track A.

    Copied structure from EnsembleNetDynamicsModel (CS-BAPR/sindy-rl),
    adapted to full-tuple residual outputs.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_models: int = 5,
        hidden: int = 256,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.n_models = n_models
        self.device = device

        self.members: List[EnsembleMember] = [
            EnsembleMember(self.input_dim, obs_dim, hidden).to(device)
            for _ in range(n_models)
        ]

    # ------------------------------------------------------------------
    def fit(
        self,
        SA: np.ndarray,
        delta_s: np.ndarray,
        delta_r: np.ndarray,
        delta_d: np.ndarray,
        n_epochs: int = 100,
        lr: float = 3e-4,
        frac_subset: float = 0.8,
    ):
        """Train each ensemble member on a random 80% subset (bootstrap)."""
        N = len(SA)
        SA_t = torch.tensor(SA, dtype=torch.float32).to(self.device)
        delta = np.concatenate([delta_s, delta_r, delta_d], axis=-1)
        delta_t = torch.tensor(delta, dtype=torch.float32).to(self.device)

        for m in self.members:
            idx = np.random.choice(N, size=int(N * frac_subset), replace=False)
            opt = optim.Adam(m.parameters(), lr=lr)
            for _ in range(n_epochs):
                opt.zero_grad()
                ds, dr, dd = m(SA_t[idx])
                pred = torch.cat([ds, dr, dd], dim=-1)
                loss = nn.MSELoss()(pred, delta_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

    # ------------------------------------------------------------------
    def predict(self, s: np.ndarray, a: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns ensemble mean prediction and per-element disagreement.

        Returns dict with keys:
          delta_s (N, obs_dim), delta_r (N, 1), delta_d (N, 1)
          disagreement_s (N,), disagreement_r (N,), disagreement_d (N,)
        """
        SA = np.concatenate([s, a], axis=-1)
        SA_t = torch.tensor(SA, dtype=torch.float32).to(self.device)

        preds_s, preds_r, preds_d = [], [], []
        with torch.no_grad():
            for m in self.members:
                ds, dr, dd = m(SA_t)
                preds_s.append(ds.cpu().numpy())
                preds_r.append(dr.cpu().numpy())
                preds_d.append(dd.cpu().numpy())

        preds_s = np.stack(preds_s)   # (n_models, N, obs_dim)
        preds_r = np.stack(preds_r)   # (n_models, N, 1)
        preds_d = np.stack(preds_d)   # (n_models, N, 1)

        return {
            "delta_s": preds_s.mean(0),
            "delta_r": preds_r.mean(0),
            "delta_d": preds_d.mean(0),
            # Disagreement = mean std across ensemble
            "disagreement_s": preds_s.std(0).mean(-1),  # (N,)
            "disagreement_r": preds_r.std(0).squeeze(-1),
            "disagreement_d": preds_d.std(0).squeeze(-1),
        }
