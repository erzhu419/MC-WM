"""
KANResidualAdapter: KAN-based δ model replacing SINDy+NAU/NMU.

Motivation (md/Assumption_upgrade.md):
- SINDy requires predefined basis library (poly2 can't find sin(3θ))
- NAU/NMU needed as separate extrapolation head
- KAN combines discovery + smooth extrapolation in one architecture
  via learnable spline activations on every edge

Interface matches SINDyNAUAdapter so it's drop-in for step2_mbrl_residual.py:
- fit(s, a, ns_sim, r_sim, ns_real, r_real, n_epochs, patience)
- predict_delta(s, a, ns_sim, r_sim) → (Δs, Δr)
- get_ood_bound(s, a) → per-sample bound (approximated via spline smoothness)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from efficient_kan import KAN


class KANResidualAdapter:
    """
    Residual adapter using KAN for state correction + MLP for reward correction.

    Architecture:
      Δs = KAN_state([s, a])     # state correction via spline edges
      Δr = MLP_reward([s, a])    # reward head (simple, avoids KAN overkill on 1D)

    KAN automatically discovers structure in Δs via spline-shaped activations.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(32,),
                 grid_size=8, spline_order=3,
                 device="cpu", log_fn=None, **kwargs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self._log = log_fn or (lambda msg: print(msg, flush=True))

        # KAN for state residual: [s, a] → Δs
        input_dim = obs_dim + act_dim
        layers = [input_dim] + list(hidden_sizes) + [obs_dim]
        self._kan_state = KAN(
            layers_hidden=layers,
            grid_size=grid_size,
            spline_order=spline_order,
        ).to(device)

        # MLP for reward residual
        self._mlp_reward = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        self._opt_state = optim.Adam(self._kan_state.parameters(), lr=1e-3, weight_decay=1e-5)
        self._opt_reward = optim.Adam(self._mlp_reward.parameters(), lr=1e-3, weight_decay=1e-5)

        # Normalization stats (computed on first fit)
        self._s_mean = None; self._s_std = None
        self._L_eff = 1.0  # approximate Lipschitz bound for rollout filtering

        # Compat stub for existing code paths that read _nau_head.L_eff
        outer = self
        class _DummyHead:
            @property
            def L_eff(self_inner): return outer._L_eff
        self._nau_head = _DummyHead()
        self._trained = False
        self._done_trained = False
        # Termination head: P(done | s, a, s') via small MLP + BCE.
        self._done_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        self._done_opt = optim.Adam(self._done_net.parameters(), lr=1e-3)

    def _normalize_s(self, s):
        if self._s_mean is None: return s
        return (s - self._s_mean) / (self._s_std + 1e-6)

    def _fit_normalization(self, s):
        self._s_mean = torch.FloatTensor(s.mean(0)).to(self.device)
        self._s_std = torch.FloatTensor(s.std(0) + 1e-6).to(self.device)

    def fit(self, s, a, ns_sim, r_sim, ns_real, r_real,
            n_epochs=50, patience=15, batch_size=512, real_dones=None, **kwargs):
        """
        Fit KAN state + MLP reward residuals.
          Δs_target = ns_real - ns_sim
          Δr_target = r_real  - r_sim
        """
        s_t = torch.FloatTensor(s).to(self.device)
        a_t = torch.FloatTensor(a).to(self.device)
        ds_target = torch.FloatTensor(ns_real - ns_sim).to(self.device)
        dr_target = torch.FloatTensor((r_real - r_sim).reshape(-1, 1)).to(self.device)

        self._fit_normalization(s)
        s_norm = self._normalize_s(s_t)
        inp = torch.cat([s_norm, a_t], dim=-1)

        n = len(s); idx = np.arange(n)
        n_val = max(1, n // 10); n_tr = n - n_val
        np.random.shuffle(idx)
        idx_tr, idx_val = idx[:n_tr], idx[n_tr:]

        best_val = float("inf"); best_epoch = 0
        for ep in range(1, n_epochs + 1):
            # Train
            self._kan_state.train(); self._mlp_reward.train()
            np.random.shuffle(idx_tr)
            tr_loss_s = tr_loss_r = 0.0; n_batches = 0
            for b in range(0, len(idx_tr), batch_size):
                bi = idx_tr[b:b+batch_size]
                x = inp[bi]
                pred_ds = self._kan_state(x)
                pred_dr = self._mlp_reward(x)
                loss_s = ((pred_ds - ds_target[bi]) ** 2).mean()
                loss_r = ((pred_dr - dr_target[bi]) ** 2).mean()

                self._opt_state.zero_grad(); loss_s.backward(); self._opt_state.step()
                self._opt_reward.zero_grad(); loss_r.backward(); self._opt_reward.step()
                tr_loss_s += loss_s.item(); tr_loss_r += loss_r.item(); n_batches += 1

            # Validate state
            self._kan_state.eval()
            with torch.no_grad():
                val_pred = self._kan_state(inp[idx_val])
                val_loss = ((val_pred - ds_target[idx_val]) ** 2).mean().item()

            if val_loss < best_val - 1e-4:
                best_val = val_loss; best_epoch = ep
            elif ep - best_epoch >= patience:
                break

        # Estimate L_eff from KAN weights (rough proxy for rollout filtering)
        with torch.no_grad():
            total_norm = sum(p.norm().item() for p in self._kan_state.parameters() if p.dim() >= 2)
            self._L_eff = float(np.clip(total_norm * 0.1, 1.0, 500.0))

        self._log(f"  KAN fit: val_MSE={best_val:.5f} @ep{best_epoch}, L_eff={self._L_eff:.2f}")
        self._trained = True

        # Train termination head if real_dones provided.
        if real_dones is not None:
            SAS_t = torch.FloatTensor(
                np.concatenate([s, a, ns_real], axis=-1).astype(np.float32)
            ).to(self.device)
            d_t = torch.FloatTensor(
                np.asarray(real_dones, dtype=np.float32).reshape(-1, 1)
            ).to(self.device)
            for ep in range(50):
                logits = self._done_net(SAS_t)
                loss_d = nn.functional.binary_cross_entropy_with_logits(logits, d_t)
                self._done_opt.zero_grad(); loss_d.backward(); self._done_opt.step()
            n_pos = int(d_t.sum().item())
            self._log(f"  Done MLP: BCE={float(loss_d):.5f} "
                      f"(pos/total={n_pos}/{len(s)}, {100*n_pos/len(s):.1f}%)")
            self._done_trained = True

    def predict_correction(self, states, actions):
        """SINDy-compatible interface: returns (Δs, Δr) as numpy arrays."""
        self._kan_state.eval(); self._mlp_reward.eval()
        s_t = torch.FloatTensor(states).to(self.device)
        a_t = torch.FloatTensor(actions).to(self.device)
        with torch.no_grad():
            inp = torch.cat([self._normalize_s(s_t), a_t], dim=-1)
            ds = self._kan_state(inp).cpu().numpy()
            dr = self._mlp_reward(inp).cpu().numpy().squeeze(-1)
        return ds, dr

    def predict_done(self, states, actions, next_states):
        """P(done | s, a, s') ∈ [0, 1]."""
        if not self._done_trained:
            return np.zeros(len(states))
        SAS = np.concatenate([states, actions, next_states],
                             axis=-1).astype(np.float32)
        with torch.no_grad():
            logits = self._done_net(
                torch.FloatTensor(SAS).to(self.device)).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    def get_active_terms(self):
        """KAN doesn't have discrete terms; return dict-compatible summary."""
        n_params = sum(p.numel() for p in self._kan_state.parameters())
        return {f"KAN_state_dim{i}": [("spline_params", n_params // max(1, self.obs_dim))]
                for i in range(self.obs_dim)}

    def get_ood_bound(self, s, a):
        """Approximate OOD bound per-sample via distance from training mean."""
        s_t = torch.FloatTensor(s).to(self.device)
        if self._s_mean is None:
            return np.ones(len(s)) * self._L_eff
        with torch.no_grad():
            dist = ((s_t - self._s_mean) / (self._s_std + 1e-6)).abs().mean(dim=1)
            bound = (1.0 + dist).cpu().numpy() * self._L_eff
        return bound
