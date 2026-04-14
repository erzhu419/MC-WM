"""
Residual-Aware ICRL: Learn feasibility constraints from real data.

Combines three ICRL insights:
1. Malik 2021: Importance sampling distinguishes "constrained" from "low reward"
2. Hugessen 2024: L1/L2 regularization prevents constraint collapse
3. Critical ICRL 2025: Constraints transfer across environments better than rewards

MC-WM extension (novel):
  φ(s,a) also conditions on MODEL CONFIDENCE from QΔ/SINDy:
  - Expert went there + model accurate → definitely feasible (φ≈1)
  - Expert avoided it + model accurate + high reward → constrained (φ≈0)
  - Expert avoided it + model inaccurate → uncertain (φ≈0.5)
  - Expert avoided it + low reward → just not worth it (φ≈0.7)

This uses the residual world model as an additional signal that
standard ICRL methods don't have.

Integration: final_weight = QΔ_weight × φ(s,a)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FeasibilityNet(nn.Module):
    """
    φ(s,a) ∈ [0,1]: probability that (s,a) is feasible.

    Architecture: MLP with sigmoid output.
    Input: [s, a] concatenated (optionally + model_confidence).
    Output: scalar φ ∈ [0,1].
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128),
                 use_model_confidence=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_model_confidence = use_model_confidence

        input_dim = obs_dim + act_dim
        if use_model_confidence:
            input_dim += 1  # extra dim for model confidence score

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, obs, acs, model_conf=None):
        """
        obs: (B, obs_dim)
        acs: (B, act_dim)
        model_conf: (B, 1) optional — QΔ/model prediction confidence
        Returns: φ(s,a) shape (B, 1)
        """
        if self.use_model_confidence and model_conf is not None:
            x = torch.cat([obs, acs, model_conf], dim=-1)
        else:
            x = torch.cat([obs, acs], dim=-1)
        return self.net(x)

    def cost(self, obs, acs, model_conf=None):
        """Cost = 1 - φ. High cost = likely constrained."""
        return 1.0 - self.forward(obs, acs, model_conf)


class ResidualAwareICRL:
    """
    Learn feasibility function φ(s,a) from:
    - Expert data (real env transitions) — assumed feasible
    - Nominal data (model rollouts / sim) — may violate constraints
    - Model confidence (QΔ) — additional signal unique to MC-WM

    Training: alternating forward (policy) and backward (constraint) steps.

    Malik-style importance sampling: reweight nominal data by trajectory-level
    likelihood ratio to distinguish "constrained away" from "just low reward".
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128),
                 lr=3e-4, reg_coeff=0.1, use_model_confidence=True,
                 target_kl=10.0, device="cpu", log_fn=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self.reg_coeff = reg_coeff
        self.target_kl = target_kl

        self.phi_net = FeasibilityNet(
            obs_dim, act_dim, hidden_sizes,
            use_model_confidence=use_model_confidence,
        ).to(device)
        self.optimizer = optim.Adam(self.phi_net.parameters(), lr=lr, weight_decay=1e-5)

        # Expert data (real transitions, loaded once)
        self._expert_obs = None
        self._expert_acs = None
        self._expert_conf = None  # model confidence on expert data

        # Normalization stats
        self._obs_mean = None
        self._obs_std = None

        # Diagnostics
        self._train_history = []

    def set_expert_data(self, obs, acs, model_confidence=None):
        """Load expert (real env) data. Called once before training."""
        self._expert_obs = torch.FloatTensor(obs).to(self.device)
        self._expert_acs = torch.FloatTensor(acs).to(self.device)
        if model_confidence is not None:
            self._expert_conf = torch.FloatTensor(
                model_confidence.reshape(-1, 1)).to(self.device)
        else:
            self._expert_conf = torch.ones(len(obs), 1).to(self.device)

        # Compute normalization
        self._obs_mean = self._expert_obs.mean(0)
        self._obs_std = self._expert_obs.std(0).clamp(min=1e-6)

        self._log(f"  ICRL: {len(obs)} expert transitions loaded")

    def _normalize_obs(self, obs):
        if self._obs_mean is not None:
            return (obs - self._obs_mean) / self._obs_std
        return obs

    def train_constraint(self, nominal_obs, nominal_acs, nominal_conf=None,
                          n_iters=10, batch_size=512):
        """
        Backward step: update φ(s,a) to distinguish expert from nominal.

        Uses likelihood-based loss (Malik 2021):
          L = -E_expert[log φ] + E_nominal[w · log φ] + reg
        where w = importance weights (trajectory-level reweighting).
        """
        n_expert = len(self._expert_obs)
        n_nominal = len(nominal_obs)

        nom_obs_t = torch.FloatTensor(nominal_obs).to(self.device)
        nom_acs_t = torch.FloatTensor(nominal_acs).to(self.device)
        if nominal_conf is not None:
            nom_conf_t = torch.FloatTensor(nominal_conf.reshape(-1, 1)).to(self.device)
        else:
            nom_conf_t = torch.ones(n_nominal, 1).to(self.device)

        # Store old predictions for importance sampling
        with torch.no_grad():
            nom_obs_norm = self._normalize_obs(nom_obs_t)
            preds_old = self.phi_net(nom_obs_norm, nom_acs_t, nom_conf_t).squeeze()

        metrics = {}
        for it in range(n_iters):
            # Sample batches
            exp_idx = np.random.choice(n_expert, min(batch_size, n_expert), replace=False)
            nom_idx = np.random.choice(n_nominal, min(batch_size, n_nominal), replace=False)

            exp_obs = self._normalize_obs(self._expert_obs[exp_idx])
            exp_acs = self._expert_acs[exp_idx]
            exp_conf = self._expert_conf[exp_idx]

            nom_obs = self._normalize_obs(nom_obs_t[nom_idx])
            nom_acs = nom_acs_t[nom_idx]
            nom_conf = nom_conf_t[nom_idx]

            # Forward
            phi_expert = self.phi_net(exp_obs, exp_acs, exp_conf)
            phi_nominal = self.phi_net(nom_obs, nom_acs, nom_conf)

            eps = 1e-8
            # Expert loss: maximize log φ(expert) — expert should be feasible
            expert_loss = -torch.log(phi_expert + eps).mean()

            # Nominal loss: importance-weighted
            # Compute IS weights from old vs new predictions
            preds_new_batch = phi_nominal.squeeze()
            preds_old_batch = preds_old[nom_idx]
            is_ratio = (preds_new_batch + eps) / (preds_old_batch + eps)
            is_weights = is_ratio / (is_ratio.sum() + eps) * len(is_ratio)

            # Nominal should have LOW φ (constrained regions)
            nominal_loss = (is_weights.detach() * torch.log(phi_nominal.squeeze() + eps)).mean()

            # Regularization: push expert toward 1.0, nominal toward 0.0
            # (not 0.5 — that was making both sides collapse to ~0.4)
            reg_loss = self.reg_coeff * (
                (phi_expert - 1.0).pow(2).mean() +  # expert should be feasible
                phi_nominal.pow(2).mean()             # nominal should be constrained
            )

            loss = expert_loss + nominal_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # KL divergence check (early stopping à la Malik)
            with torch.no_grad():
                nom_obs_all = self._normalize_obs(nom_obs_t)
                preds_new_all = self.phi_net(nom_obs_all, nom_acs_t, nom_conf_t).squeeze()
                ratio = (preds_new_all + eps) / (preds_old + eps)
                kl = -torch.log(ratio + eps).mean()
                if kl > self.target_kl:
                    break

        # Update old predictions
        with torch.no_grad():
            preds_old = preds_new_all

        # Diagnostics
        with torch.no_grad():
            phi_exp_mean = self.phi_net(
                self._normalize_obs(self._expert_obs[:500]),
                self._expert_acs[:500],
                self._expert_conf[:500]).mean()
            phi_nom_mean = self.phi_net(
                self._normalize_obs(nom_obs_t[:500]),
                nom_acs_t[:500],
                nom_conf_t[:500]).mean()

        metrics = {
            "phi_expert": float(phi_exp_mean),
            "phi_nominal": float(phi_nom_mean),
            "separation": float(phi_exp_mean - phi_nom_mean),
            "kl": float(kl),
            "iters": it + 1,
        }
        self._train_history.append(metrics)
        return metrics

    def get_feasibility(self, obs, acs, model_confidence=None):
        """
        Get φ(s,a) for a batch of transitions.
        Returns numpy array (N,) in [0, 1].
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        acs_t = torch.FloatTensor(acs).to(self.device)
        if model_confidence is not None:
            conf_t = torch.FloatTensor(model_confidence.reshape(-1, 1)).to(self.device)
        else:
            conf_t = torch.ones(len(obs), 1).to(self.device)

        with torch.no_grad():
            obs_norm = self._normalize_obs(obs_t)
            phi = self.phi_net(obs_norm, acs_t, conf_t).squeeze()
        return phi.cpu().numpy()

    def get_constraint_weight(self, obs, acs, model_confidence=None):
        """
        Temperature-calibrated weight in [0, 1].

        Raw φ may not span [0,1] well (e.g., expert φ=0.4, nominal φ=0.2).
        Calibrate: center on expert mean, scale by expert std, sigmoid.
        Result: expert ≈ 0.8, nominal ≈ 0.3.
        """
        phi = self.get_feasibility(obs, acs, model_confidence)

        # Calibrate using expert statistics
        if self._expert_obs is not None and len(self._train_history) > 0:
            phi_expert_mean = self._train_history[-1].get("phi_expert", 0.5)
            # Temperature: spread the distribution around expert mean
            # z = (φ - threshold) / τ, where threshold = midpoint between expert and nominal
            phi_nominal_mean = self._train_history[-1].get("phi_nominal", 0.3)
            threshold = (phi_expert_mean + phi_nominal_mean) / 2
            tau = max(phi_expert_mean - phi_nominal_mean, 0.05)  # separation as scale
            z = (phi - threshold) / tau
            # Sigmoid maps to [0,1]: expert side → ~0.7-0.9, nominal side → ~0.1-0.3
            calibrated = 1.0 / (1.0 + np.exp(-z))
            return calibrated

        return phi

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
