"""
QDelta-Bellman: proper γ>0 Bellman residual for per-(s,a) model trust.

Motivation
----------
The previous implementation used `QΔ = 1/(1 + MSE(s,a)/τ)` as a per-step
weight — mathematically equivalent to `γ=0` (no Bellman propagation).  While
this worked empirically for rollout weighting, it is NOT a fixed point of any
Bellman operator — merely a normalization of instantaneous model error.

Theoretical alignment (ResidualMDP.lean `bellman_contraction`)
-------------------------------------------------------------
Define the log-confidence Bellman operator:
    QΔ(s,a)  =  log c_step(s,a)  +  γ · E_{s',a'~π} QΔ(s',a')
where c_step(s,a) = 1/(1 + MSE(M_real-pred(s,a), s_real)/τ) ∈ (0, 1].

This is precisely the Bellman operator from the Lean formalization with:
  • reward replaced by log c_step (non-positive, bounded)
  • same policy π, same γ (here set to 0.9 by default)

By `bellman_contraction` the operator is a γ-contraction on (S → ℝ, L∞),
so Banach gives a unique fixed point QΔ* with |QΔ*(s,a)| ≤ -log(min c)/(1-γ).

Why log-transform avoids the γ→1 collapse
-----------------------------------------
With raw confidence `c(s,a) ∈ (0, 1]`, Bellman averaging pushes all c-values
toward their mean (positive support + convex combination → collapse).
With log c ∈ (-∞, 0], averaging log-values is NOT equivalent to averaging
values — relative spatial differences are preserved.  So we get proper
Bellman structure without losing the signal QΔ is meant to carry.

Weight used in MBPO: `w = exp(QΔ(s,a)) ∈ (exp(-‖QΔ‖∞), 1]`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QDeltaNet(nn.Module):
    """
    Log-confidence critic: (s, a) → QΔ(s, a) ≤ 0.

    Uses a soft-clamped output (−softplus) to guarantee QΔ ≤ 0 without
    hard truncation (which breaks gradients near the boundary).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Returns QΔ(s, a) ≤ 0 with shape (B, 1)."""
        x = torch.cat([obs, act], dim=-1)
        raw = self.net(x)
        return -F.softplus(raw)  # ∈ (-∞, 0]


class QDeltaBellman:
    """
    Proper γ > 0 Bellman critic for model confidence.

    Training target:
        QΔ_target(s, a) = log c_step(s, a) + γ · (1 - done) · QΔ(s', a')
    minimized by TD regression on env_buf transitions.

    Usage:
      qd = QDeltaBellman(obs_dim, act_dim, gamma=0.9)
      # Periodically:
      qd.update(env_buf, policy, corrected_model, n_iters=20)
      # At rollout weighting time:
      w = qd.weight(start_states, actions)    # numpy (N,) ∈ (0, 1]
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 128, gamma: float = 0.9,
                 lr: float = 3e-4, tau_half_life: float = 0.0,
                 device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.device = device

        self.net = QDeltaNet(obs_dim, act_dim, hidden).to(device)
        # Target network (Polyak-averaged) to stabilise TD bootstrap.
        self.target = QDeltaNet(obs_dim, act_dim, hidden).to(device)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
        self.tau_polyak = 0.005

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # Running τ (median-based) for c_step normalisation.
        self._tau = 1e-3
        self._tau_half_life = tau_half_life

    def _c_step(self, mse_per_sample: torch.Tensor) -> torch.Tensor:
        """c_step = 1 / (1 + MSE / τ) ∈ (0, 1]."""
        return 1.0 / (1.0 + mse_per_sample / self._tau)

    @torch.no_grad()
    def _update_tau(self, mse_per_sample: torch.Tensor) -> None:
        """EMA on median MSE to normalise c_step adaptively."""
        med = float(mse_per_sample.median().clamp(min=1e-8))
        alpha = 0.1 if self._tau_half_life <= 0.0 else self._tau_half_life
        self._tau = (1 - alpha) * self._tau + alpha * med

    def update(
        self,
        s: torch.Tensor, a: torch.Tensor,
        s2: torch.Tensor, done: torch.Tensor,
        mse_per_sample: torch.Tensor,
        policy_get_next_action,
        n_iters: int = 20,
    ) -> dict:
        """
        TD update on a batch of real transitions (s, a, s', done) with
        pre-computed per-sample MSE of the corrected model on (s, a).

        `policy_get_next_action(s2_tensor) -> Tensor`: deterministic next
        action from current policy (used to bootstrap QΔ at s').
        """
        self._update_tau(mse_per_sample)
        log_c_step = torch.log(self._c_step(mse_per_sample).clamp(min=1e-8)).unsqueeze(-1)
        # Stop grad on targets.
        with torch.no_grad():
            a2 = policy_get_next_action(s2)
            qd_next = self.target(s2, a2)  # (B, 1) ≤ 0
            target = log_c_step + self.gamma * (1 - done).unsqueeze(-1) * qd_next
        last_loss = 0.0
        for _ in range(n_iters):
            pred = self.net(s, a)
            loss = F.mse_loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
        # Polyak update target.
        with torch.no_grad():
            for p, p_tgt in zip(self.net.parameters(), self.target.parameters()):
                p_tgt.data.mul_(1 - self.tau_polyak).add_(p.data, alpha=self.tau_polyak)
        return {"td_loss": last_loss, "tau": self._tau,
                "qd_mean": float(self.net(s, a).mean().detach().item())}

    @torch.no_grad()
    def weight(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Per-(s,a) MBPO rollout weight = exp(QΔ(s,a)) ∈ (0, 1]."""
        return self.net(s, a).exp().squeeze(-1)

    def weight_np(self, s_np, a_np):
        """Numpy interface."""
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a_np, dtype=torch.float32, device=self.device)
        return self.weight(s_t, a_t).cpu().numpy()
