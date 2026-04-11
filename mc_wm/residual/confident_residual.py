"""
Confident Residual Model: MLP ensemble with explicit confidence output.

Core idea: residual model outputs (prediction, confidence).
- Confidence = 1 / (1 + ensemble_disagreement / τ)
- High confidence → trust this prediction → normal Q-learning weight
- Low confidence → don't trust → constrain policy + actively collect data here

Three roles of confidence:
1. Q-loss importance weight: w = confidence
2. Actor constraint: penalty for visiting low-confidence regions
3. Active data collection trigger: refit on low-confidence states
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ResidualPredictor(nn.Module):
    """Single MLP: (s, a) → Δs."""
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, sa):
        return self.net(sa)


class ConfidentResidualModel:
    """
    MLP ensemble residual model with confidence-based gating.

    Outputs:
        prediction: ensemble mean of Δs
        confidence: 1 / (1 + disagreement / τ), in [0, 1]
        disagreement: ensemble std (per-dim and aggregated)

    Supports online refit: add new paired data, re-train.
    """

    def __init__(self, obs_dim, act_dim, K=5, hidden=128,
                 confidence_tau=0.1, device="cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.K = K
        self.confidence_tau = confidence_tau
        self.device = device

        self.models = [ResidualPredictor(self.input_dim, obs_dim, hidden).to(device)
                       for _ in range(K)]
        self.optimizers = [optim.Adam(m.parameters(), lr=1e-3) for m in self.models]

        # Paired data buffer (grows with online collection)
        self._SA_buffer = None
        self._delta_buffer = None
        self._fit_count = 0

        # Calibration: confidence stats on training data
        self._conf_mean = 0.5
        self._conf_std = 0.2

    def fit(self, SA, delta_s, n_epochs=100, batch_size=256, subsample_ratio=0.8):
        """Fit on paired data. Can be called multiple times with growing data."""
        N = len(SA)
        self._SA_buffer = SA.copy()
        self._delta_buffer = delta_s.copy()

        SA_t = torch.FloatTensor(SA).to(self.device)
        ds_t = torch.FloatTensor(delta_s).to(self.device)

        for k, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            n_sub = int(N * subsample_ratio)
            idx = np.random.choice(N, n_sub, replace=True)
            model.train()
            for epoch in range(n_epochs):
                perm = np.random.permutation(n_sub)
                for i in range(0, n_sub, batch_size):
                    bi = idx[perm[i:i+batch_size]]
                    pred = model(SA_t[bi])
                    loss = nn.MSELoss()(pred, ds_t[bi])
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()

        self._fit_count += 1
        self._calibrate(SA_t)

    def add_paired_data(self, SA_new, delta_new):
        """Add new on-policy paired data to the buffer."""
        if self._SA_buffer is None:
            self._SA_buffer = SA_new
            self._delta_buffer = delta_new
        else:
            self._SA_buffer = np.concatenate([self._SA_buffer, SA_new])
            self._delta_buffer = np.concatenate([self._delta_buffer, delta_new])

    def refit(self, n_epochs=50, max_data=10000):
        """Re-fit on accumulated paired data (initial + online)."""
        SA = self._SA_buffer
        ds = self._delta_buffer
        # Keep most recent data if buffer too large
        if len(SA) > max_data:
            SA = SA[-max_data:]
            ds = ds[-max_data:]
        self.fit(SA, ds, n_epochs=n_epochs)

    def _calibrate(self, SA_t):
        """Compute confidence stats on training data."""
        with torch.no_grad():
            conf = self._compute_confidence_tensor(SA_t)
            self._conf_mean = float(conf.mean())
            self._conf_std = float(conf.std()) + 1e-8
        print(f"  Residual model fit #{self._fit_count}: "
              f"N={len(self._SA_buffer)}, "
              f"conf_mean={self._conf_mean:.3f}±{self._conf_std:.3f}")

    def _compute_confidence_tensor(self, SA_t):
        """Confidence from ensemble disagreement. Returns (B,) tensor."""
        preds = torch.stack([m(SA_t) for m in self.models])  # (K, B, obs_dim)
        disagreement = preds.std(0).mean(dim=1)  # (B,) — avg std across dims
        confidence = 1.0 / (1.0 + disagreement / self.confidence_tau)
        return confidence

    def predict_with_confidence(self, s_batch, a_batch):
        """
        Returns (prediction, confidence, disagreement).

        prediction: (B, obs_dim) — ensemble mean
        confidence: (B,) — in [0, 1]
        disagreement: (B, obs_dim) — per-dim ensemble std
        """
        SA = np.concatenate([s_batch, a_batch], axis=-1).astype(np.float32)
        SA_t = torch.FloatTensor(SA).to(self.device)

        with torch.no_grad():
            preds = torch.stack([m(SA_t) for m in self.models])  # (K, B, obs_dim)
            mean_pred = preds.mean(0).cpu().numpy()
            disagreement = preds.std(0).cpu().numpy()
            confidence = self._compute_confidence_tensor(SA_t).cpu().numpy()

        return mean_pred, confidence, disagreement

    def get_confidence(self, s_batch, a_batch):
        """Just confidence, for fast Q-weight computation."""
        SA = np.concatenate([s_batch, a_batch], axis=-1).astype(np.float32)
        SA_t = torch.FloatTensor(SA).to(self.device)
        with torch.no_grad():
            return self._compute_confidence_tensor(SA_t).cpu().numpy()

    def make_confidence_fn(self):
        """
        Returns gap_fn compatible with RESACAgent.
        gap = 1 - confidence (high gap = low confidence).
        """
        def gap_fn(s_batch, a_batch):
            conf = self.get_confidence(s_batch, a_batch)
            return 1.0 - conf  # gap: 0=confident, 1=uncertain
        return gap_fn

    def get_low_confidence_states(self, replay_buffer, n_candidates=1000, threshold=0.3):
        """
        Find states in replay buffer where confidence is lowest.
        These are candidates for active paired data collection.
        Returns indices into replay buffer.
        """
        n = min(n_candidates, replay_buffer.size)
        idx = np.random.randint(0, replay_buffer.size, n)
        s_batch = replay_buffer.s[idx]
        a_batch = replay_buffer.a[idx]
        conf = self.get_confidence(s_batch, a_batch)
        low_mask = conf < threshold
        return idx[low_mask], conf[low_mask]
