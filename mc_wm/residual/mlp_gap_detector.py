"""
MLP Ensemble Gap Detector — replaces SINDy for dynamics gap detection.

SINDy problem: poly2 features extrapolate quadratically → OOD explosion.
MLP with tanh: extrapolates to constant → bounded predictions everywhere.
Ensemble disagreement: natural, bounded gap signal in [0, ∞).

Two gap signals:
  1. prediction magnitude: ||Δ̂(s,a)||² — how large is the predicted gap
  2. ensemble disagreement: std across K models — how uncertain is the prediction

Combined: gap = α * norm(pred_magnitude) + (1-α) * norm(disagreement)

Both are bounded (tanh output), both preserve rank order.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeltaPredictor(nn.Module):
    """Single MLP: (s, a) → Δs prediction."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, sa):
        return self.net(sa)


class MLPGapDetector:
    """
    Ensemble of K MLPs trained on paired (sim, real) data.
    Provides bounded gap signal for importance weighting.

    Usage:
        detector = MLPGapDetector(obs_dim, act_dim, K=5)
        detector.fit(SA, delta_s, n_epochs=100)
        gap_fn = detector.make_gap_fn()
        # gap_fn(s_batch, a_batch) → gap signal in [0, 1]
    """

    def __init__(self, obs_dim, act_dim, K=5, hidden=128, device="cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K = K
        self.device = device
        self.models = [DeltaPredictor(obs_dim, act_dim, hidden).to(device) for _ in range(K)]
        self.optimizers = [optim.Adam(m.parameters(), lr=1e-3) for m in self.models]

        # Calibration stats (set after fit)
        self._gap_mean = 0.0
        self._gap_std = 1.0
        self._dis_mean = 0.0
        self._dis_std = 1.0

    def fit(self, SA, delta_s, n_epochs=100, batch_size=256, subsample_ratio=0.8):
        """
        Train K models on bootstrap subsamples of paired data.

        Args:
            SA: (N, obs_dim + act_dim) — concatenated state-action
            delta_s: (N, obs_dim) — true dynamics gap (s_real_next - s_sim_next)
        """
        N = len(SA)
        SA_t = torch.FloatTensor(SA).to(self.device)
        ds_t = torch.FloatTensor(delta_s).to(self.device)

        for k, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            # Bootstrap subsample
            n_sub = int(N * subsample_ratio)
            idx = np.random.choice(N, n_sub, replace=True)

            model.train()
            for epoch in range(n_epochs):
                perm = np.random.permutation(n_sub)
                epoch_loss = 0.0
                n_batches = 0
                for i in range(0, n_sub, batch_size):
                    batch_idx = idx[perm[i:i+batch_size]]
                    sa = SA_t[batch_idx]
                    ds = ds_t[batch_idx]

                    pred = model(sa)
                    loss = nn.MSELoss()(pred, ds)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss)
                    n_batches += 1

            model.eval()
            final_loss = epoch_loss / max(n_batches, 1)
            if (k + 1) == self.K:
                print(f"  MLP ensemble: K={self.K}, final loss (last model)={final_loss:.5f}")

        # Calibrate gap signal on training data
        self._calibrate(SA_t)

    def _calibrate(self, SA_t):
        """Compute gap signal stats on training data for normalization."""
        with torch.no_grad():
            preds = torch.stack([m(SA_t) for m in self.models])  # (K, N, obs_dim)
            pred_mean = preds.mean(0)  # (N, obs_dim)
            pred_std = preds.std(0)    # (N, obs_dim)

            # Gap signal 1: prediction magnitude
            gap_mag = (pred_mean ** 2).mean(dim=1).cpu().numpy()  # (N,)
            # Gap signal 2: disagreement
            gap_dis = pred_std.mean(dim=1).cpu().numpy()  # (N,)

        self._gap_mean = float(gap_mag.mean())
        self._gap_std = float(gap_mag.std()) + 1e-8
        self._dis_mean = float(gap_dis.mean())
        self._dis_std = float(gap_dis.std()) + 1e-8

        print(f"  Gap calibration: mag_mean={self._gap_mean:.4f}±{self._gap_std:.4f} "
              f"dis_mean={self._dis_mean:.4f}±{self._dis_std:.4f}")

    def predict_gap(self, s_batch, a_batch, alpha=0.5):
        """
        Compute normalized gap signal in [0, 1].

        alpha: weight between magnitude (0) and disagreement (1).
        """
        SA = np.concatenate([s_batch, a_batch], axis=-1).astype(np.float32)
        SA_t = torch.FloatTensor(SA).to(self.device)

        with torch.no_grad():
            preds = torch.stack([m(SA_t) for m in self.models])  # (K, B, obs_dim)
            pred_mean = preds.mean(0)  # (B, obs_dim)
            pred_std = preds.std(0)    # (B, obs_dim)

            # Gap 1: prediction magnitude (z-scored, clipped to [0, 1])
            gap_mag = (pred_mean ** 2).mean(dim=1).cpu().numpy()
            gap_mag_z = (gap_mag - self._gap_mean) / self._gap_std
            gap_mag_norm = np.clip((gap_mag_z + 2) / 5, 0, 1)  # z=-2→0, z=3→1

            # Gap 2: disagreement (z-scored, clipped to [0, 1])
            gap_dis = pred_std.mean(dim=1).cpu().numpy()
            gap_dis_z = (gap_dis - self._dis_mean) / self._dis_std
            gap_dis_norm = np.clip((gap_dis_z + 2) / 5, 0, 1)

            # Combined
            gap = alpha * gap_mag_norm + (1 - alpha) * gap_dis_norm

        return gap

    def make_gap_fn(self, alpha=0.5):
        """Return a gap_fn compatible with RESACAgent."""
        def gap_fn(s_batch, a_batch):
            return self.predict_gap(s_batch, a_batch, alpha=alpha)
        return gap_fn

    def correction_coverage(self, SA, delta_s):
        """Compatibility: compute RMSE reduction on training data."""
        SA_t = torch.FloatTensor(SA).to(self.device)
        with torch.no_grad():
            preds = torch.stack([m(SA_t) for m in self.models])
            pred_mean = preds.mean(0).cpu().numpy()
        raw_rmse = np.sqrt(np.mean(delta_s ** 2))
        corr_rmse = np.sqrt(np.mean((delta_s - pred_mean) ** 2))
        return {
            "rmse_reduction_pct": (1 - corr_rmse / max(raw_rmse, 1e-8)) * 100,
            "raw_rmse": raw_rmse,
            "corr_rmse": corr_rmse,
        }
