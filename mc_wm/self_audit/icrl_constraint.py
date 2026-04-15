"""
Residual-Aware ICRL: Learn feasibility constraints from real data.

Combines three ICRL insights:
1. Malik 2021: Importance sampling distinguishes "constrained" from "low reward"
2. Hugessen 2024: L1/L2 regularization prevents constraint collapse
3. Critical ICRL 2025: Constraints transfer across environments better than rewards

MC-WM v2 (dynamics-gap optimized):
  In environments WITHOUT spatial constraints (e.g., GravityCheetah),
  standard ICRL learns a signal redundant with QΔ.

  Fix: φ(s, a, Δs) discriminates DYNAMICS, not locations.
  - Expert Δs comes from real env (gravity=1x)
  - Nominal Δs comes from raw M_sim (gravity=2x, no δ correction)
  - φ learns: "does this transition look like real dynamics?"

  This is COMPLEMENTARY to QΔ:
  - QΔ: "is the corrected model accurate here?" (per-step prediction error)
  - φ:  "is this state-action-transition in the real dynamics manifold?"

  Combination: w = QΔ × (0.5 + 0.5×φ)  (soft modulation, not multiplicative kill)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FeasibilityNet(nn.Module):
    """
    φ(s, a, Δs) ∈ [0,1]: probability that transition (s,a)→s' is from real dynamics.

    Two modes:
      - use_transition=True:  input = [s, a, Δs]  (dynamics discriminator)
      - use_transition=False: input = [s, a, conf]  (legacy, model confidence)
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128),
                 use_transition=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_transition = use_transition

        if use_transition:
            input_dim = obs_dim + act_dim + obs_dim  # [s, a, Δs]
        else:
            input_dim = obs_dim + act_dim + 1  # [s, a, conf]

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, obs, acs, extra):
        """
        obs:   (B, obs_dim)
        acs:   (B, act_dim)
        extra: (B, obs_dim) if use_transition else (B, 1) model confidence
        Returns: φ shape (B, 1)
        """
        x = torch.cat([obs, acs, extra], dim=-1)
        return self.net(x)


class ResidualAwareICRL:
    """
    Learn dynamics discriminator φ(s, a, Δs) from:
    - Expert data: real env transitions (s, a, s'_real)
    - Nominal data: raw sim predictions  (s, a, s'_sim) — NO δ correction

    Key difference from v1:
    - Negatives come from raw M_sim, not corrected model buffer
    - φ discriminates dynamics (transition shape), not spatial location
    - This gives a signal COMPLEMENTARY to QΔ in dynamics-gap environments
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128),
                 lr=3e-4, reg_coeff=0.05, use_transition=True,
                 target_kl=10.0, device="cpu", log_fn=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self.reg_coeff = reg_coeff
        self.target_kl = target_kl
        self.use_transition = use_transition

        self.phi_net = FeasibilityNet(
            obs_dim, act_dim, hidden_sizes,
            use_transition=use_transition,
        ).to(device)
        self.optimizer = optim.Adam(self.phi_net.parameters(), lr=lr, weight_decay=1e-5)

        # Expert data (real transitions)
        self._expert_obs = None
        self._expert_acs = None
        self._expert_extra = None  # Δs for transition mode, conf for legacy

        # Normalization stats
        self._obs_mean = None
        self._obs_std = None
        self._delta_mean = None
        self._delta_std = None

        # Diagnostics
        self._train_history = []

    def set_expert_data(self, obs, acs, next_obs=None, model_confidence=None):
        """
        Load expert (real env) data. Called once before training.

        For transition mode: pass next_obs (s'_real).
        For legacy mode: pass model_confidence.
        """
        self._expert_obs = torch.FloatTensor(obs).to(self.device)
        self._expert_acs = torch.FloatTensor(acs).to(self.device)

        if self.use_transition and next_obs is not None:
            delta = next_obs - obs  # Δs = s' - s
            self._expert_extra = torch.FloatTensor(delta).to(self.device)
            # Normalize delta separately (dynamics have different scale than states)
            self._delta_mean = self._expert_extra.mean(0)
            self._delta_std = self._expert_extra.std(0).clamp(min=1e-6)
        elif model_confidence is not None:
            self._expert_extra = torch.FloatTensor(
                model_confidence.reshape(-1, 1)).to(self.device)
        else:
            self._expert_extra = torch.ones(len(obs), 1).to(self.device)

        # Compute observation normalization
        self._obs_mean = self._expert_obs.mean(0)
        self._obs_std = self._expert_obs.std(0).clamp(min=1e-6)

        self._log(f"  ICRL: {len(obs)} expert transitions loaded "
                  f"(mode={'transition' if self.use_transition else 'confidence'})")

    def _normalize_obs(self, obs):
        if self._obs_mean is not None:
            return (obs - self._obs_mean) / self._obs_std
        return obs

    def _normalize_delta(self, delta):
        if self._delta_mean is not None:
            return (delta - self._delta_mean) / self._delta_std
        return delta

    def _prepare_extra(self, extra_np):
        """Convert numpy extra input to normalized tensor.

        - transition mode: (N, obs_dim) → normalized
        - confidence mode: (N,) or (N, 1) → reshaped to (N, 1)
        """
        t = torch.FloatTensor(extra_np).to(self.device)
        if self.use_transition:
            if t.dim() == 2 and t.shape[1] == self.obs_dim:
                return self._normalize_delta(t)
            return t
        # confidence mode: ensure (N, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return t

    def train_constraint(self, nominal_obs, nominal_acs, nominal_extra,
                          n_iters=3, batch_size=512):
        """
        Update φ via plain BCE: label 1 for expert, 0 for nominal.

        Simpler and more stable than IS-weighted log loss:
        - BCE directly pulls φ_expert→1, φ_nominal→0
        - No drift/collapse (seen in IS version where φ_expert decayed to 0.19)
        - Fewer iters (3 vs 10) to prevent overfitting to batch
        """
        n_expert = len(self._expert_obs)
        n_nominal = len(nominal_obs)

        nom_obs_t = torch.FloatTensor(nominal_obs).to(self.device)
        nom_acs_t = torch.FloatTensor(nominal_acs).to(self.device)
        nom_extra_t = self._prepare_extra(nominal_extra)

        exp_extra_norm = (self._normalize_delta(self._expert_extra)
                          if self.use_transition else self._expert_extra)
        nom_extra_norm = (self._normalize_delta(nom_extra_t)
                          if self.use_transition else nom_extra_t)

        eps = 1e-6
        for it in range(n_iters):
            exp_idx = np.random.choice(n_expert, min(batch_size, n_expert), replace=False)
            nom_idx = np.random.choice(n_nominal, min(batch_size, n_nominal), replace=False)

            exp_obs = self._normalize_obs(self._expert_obs[exp_idx])
            exp_acs = self._expert_acs[exp_idx]
            exp_ext = exp_extra_norm[exp_idx]

            nom_obs = self._normalize_obs(nom_obs_t[nom_idx])
            nom_acs = nom_acs_t[nom_idx]
            nom_ext = nom_extra_norm[nom_idx]

            phi_expert = self.phi_net(exp_obs, exp_acs, exp_ext).clamp(eps, 1 - eps)
            phi_nominal = self.phi_net(nom_obs, nom_acs, nom_ext).clamp(eps, 1 - eps)

            # BCE: expert→1, nominal→0
            loss = -(torch.log(phi_expert).mean() + torch.log(1 - phi_nominal).mean())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Diagnostics
        with torch.no_grad():
            n_diag = min(500, n_expert, n_nominal)
            phi_exp_mean = self.phi_net(
                self._normalize_obs(self._expert_obs[:n_diag]),
                self._expert_acs[:n_diag],
                exp_extra_norm[:n_diag]).mean()
            phi_nom_mean = self.phi_net(
                self._normalize_obs(nom_obs_t[:n_diag]),
                nom_acs_t[:n_diag],
                nom_extra_norm[:n_diag]).mean()

        metrics = {
            "phi_expert": float(phi_exp_mean),
            "phi_nominal": float(phi_nom_mean),
            "separation": float(phi_exp_mean - phi_nom_mean),
            "kl": 0.0,  # not tracked in BCE version
            "iters": it + 1,
        }
        self._train_history.append(metrics)
        return metrics

    def get_feasibility(self, obs, acs, delta_s=None, model_confidence=None):
        """
        Get φ(s,a,Δs) for a batch of transitions.
        Returns numpy array (N,) in [0, 1].
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        acs_t = torch.FloatTensor(acs).to(self.device)

        if self.use_transition and delta_s is not None:
            extra_t = self._normalize_delta(
                torch.FloatTensor(delta_s).to(self.device))
        elif model_confidence is not None:
            extra_t = torch.FloatTensor(
                model_confidence.reshape(-1, 1)).to(self.device)
        else:
            extra_t = torch.zeros(len(obs), self.obs_dim if self.use_transition else 1).to(self.device)

        with torch.no_grad():
            obs_norm = self._normalize_obs(obs_t)
            phi = self.phi_net(obs_norm, acs_t, extra_t).squeeze()
        return phi.cpu().numpy()

    def get_soft_weight(self, obs, acs, delta_s=None, model_confidence=None):
        """
        Soft modulation weight: 0.5 + 0.5 × φ  ∈ [0.5, 1.0].

        Designed for multiplicative use with QΔ:
          w = QΔ × (0.5 + 0.5×φ)

        φ≈1 (real-like transition) → weight ≈ 1.0 (trust QΔ fully)
        φ≈0 (sim-like transition)  → weight ≈ 0.5 (halve QΔ weight)

        This prevents φ from killing transitions (min weight = 0.5×QΔ),
        while still giving a useful signal in dynamics-gap environments.
        """
        phi = self.get_feasibility(obs, acs, delta_s, model_confidence)
        return 0.5 + 0.5 * phi

    def get_stats(self):
        if not self._train_history:
            return {}
        latest = self._train_history[-1]
        return {
            "phi_expert": latest["phi_expert"],
            "phi_nominal": latest["phi_nominal"],
            "separation": latest["separation"],
            "n_updates": len(self._train_history),
        }

    def save(self, path):
        """Save φ network + normalization stats to disk for cross-env transfer."""
        torch.save({
            "state_dict": self.phi_net.state_dict(),
            "use_transition": self.use_transition,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "obs_mean": self._obs_mean.cpu() if self._obs_mean is not None else None,
            "obs_std": self._obs_std.cpu() if self._obs_std is not None else None,
            "delta_mean": self._delta_mean.cpu() if self._delta_mean is not None else None,
            "delta_std": self._delta_std.cpu() if self._delta_std is not None else None,
            "train_history": self._train_history,
        }, path)

    def load(self, path, freeze=True):
        """Load φ from disk. If freeze=True, disable further training."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        assert ckpt["use_transition"] == self.use_transition, \
            f"mode mismatch: saved={ckpt['use_transition']} self={self.use_transition}"
        assert ckpt["obs_dim"] == self.obs_dim and ckpt["act_dim"] == self.act_dim, \
            f"dim mismatch: saved=({ckpt['obs_dim']},{ckpt['act_dim']}) self=({self.obs_dim},{self.act_dim})"
        self.phi_net.load_state_dict(ckpt["state_dict"])
        if ckpt["obs_mean"] is not None:
            self._obs_mean = ckpt["obs_mean"].to(self.device)
            self._obs_std = ckpt["obs_std"].to(self.device)
        if ckpt["delta_mean"] is not None:
            self._delta_mean = ckpt["delta_mean"].to(self.device)
            self._delta_std = ckpt["delta_std"].to(self.device)
        self._train_history = list(ckpt.get("train_history", []))
        self._frozen = freeze
        if freeze:
            for p in self.phi_net.parameters():
                p.requires_grad = False
        self._log(f"  ICRL loaded φ from {path} (frozen={freeze}, "
                  f"last sep={self._train_history[-1]['separation']:.3f})" if self._train_history
                  else f"  ICRL loaded φ from {path} (frozen={freeze})")

    @property
    def is_frozen(self):
        return getattr(self, "_frozen", False)
