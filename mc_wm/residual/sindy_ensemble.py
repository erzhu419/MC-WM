"""
SINDy Ensemble: K bootstrap SINDy correctors with disagreement-based gating.

The single-SINDy corrector has +36% RMSE reduction on training distribution
but -837% on trained policy distribution — it doesn't know it's OOD.

Fix: K models on bootstrap subsamples. When they disagree → OOD → gate closes.
When they agree → in-distribution → gate opens.

This is RE-SAC's dual regularization principle applied to the residual model:
  - Ensemble mean  → correction prediction (like Q-mean)
  - Ensemble std   → correction uncertainty (like Q-std)
  - Gate = f(std)  → only correct when confident (like LCB pessimism)
"""

import numpy as np
import pysindy as ps
from sklearn.linear_model import Ridge
from typing import Dict, Optional


class SINDyEnsembleCorrectorDim:
    """Single-dimension SINDy corrector with K bootstrap models."""

    def __init__(self, K: int = 5, degree: int = 2, threshold: float = 0.05,
                 alpha: float = 0.05, subsample_ratio: float = 0.8):
        self.K = K
        self.degree = degree
        self.threshold = threshold
        self.alpha = alpha
        self.subsample_ratio = subsample_ratio
        self._coefs = []       # K coefficient vectors
        self._library = None   # shared (stateless after transform)
        self._fitted = False

    def fit(self, Theta: np.ndarray, y: np.ndarray):
        """Fit K bootstrap models on pre-computed feature matrix Theta."""
        N = len(Theta)
        n_sub = int(N * self.subsample_ratio)
        self._coefs = []

        for k in range(self.K):
            idx = np.random.choice(N, n_sub, replace=True)
            Theta_k = Theta[idx]
            y_k = y[idx]

            # Ridge + STLSQ
            reg = Ridge(alpha=self.alpha, fit_intercept=False)
            reg.fit(Theta_k, y_k)
            coef = reg.coef_.copy()

            mask = np.abs(coef) > self.threshold
            if np.any(mask):
                reg2 = Ridge(alpha=self.alpha, fit_intercept=False)
                reg2.fit(Theta_k[:, mask], y_k)
                coef_new = np.zeros_like(coef)
                coef_new[mask] = reg2.coef_
                coef = coef_new

            self._coefs.append(coef)

        self._fitted = True

    def predict_ensemble(self, Theta: np.ndarray):
        """Return (K, N) predictions from all ensemble members."""
        return np.stack([Theta @ c for c in self._coefs], axis=0)  # (K, N)

    def predict_mean_std(self, Theta: np.ndarray):
        """Return mean and std across ensemble. Both shape (N,)."""
        preds = self.predict_ensemble(Theta)  # (K, N)
        return preds.mean(axis=0), preds.std(axis=0)


class SINDyEnsembleCorrector:
    """
    Full obs-dim ensemble corrector with disagreement-based gating.

    Replaces SINDyStateCorrector with:
    - K bootstrap SINDy models per dimension
    - predict_with_gate(s, a) → correction + gate value
    - Gate = 1 / (1 + disagreement / tau)  (smooth sigmoid)
    """

    def __init__(self, obs_dim: int, K: int = 5, degree: int = 2,
                 threshold: float = 0.05, alpha: float = 0.05,
                 subsample_ratio: float = 0.8, gate_tau: float = 0.1):
        self.obs_dim = obs_dim
        self.K = K
        self.gate_tau = gate_tau
        self.models = [
            SINDyEnsembleCorrectorDim(K=K, degree=degree, threshold=threshold,
                                       alpha=alpha, subsample_ratio=subsample_ratio)
            for _ in range(obs_dim)
        ]
        self._library = ps.PolynomialLibrary(degree=degree, include_bias=True)
        self._fitted = False
        self.fit_errors = np.zeros(obs_dim)  # compatibility with old API

    def fit(self, SA: np.ndarray, delta_s: np.ndarray):
        """Fit ensemble on paired data. SA: (N, obs+act), delta_s: (N, obs_dim)."""
        self._library.fit(SA)
        Theta = np.asarray(self._library.transform(SA))

        for i, model in enumerate(self.models):
            model.fit(Theta, delta_s[:, i])
            # Compute fit error (full data, mean prediction)
            mean_pred, _ = model.predict_mean_std(Theta)
            self.fit_errors[i] = float(np.mean((delta_s[:, i] - mean_pred) ** 2))

        self._fitted = True

        # Print summary
        n_active = sum(1 for i in range(self.obs_dim) if self.fit_errors[i] < 0.01)
        print(f"  SINDy Ensemble: K={self.K}, {self.obs_dim} dims, "
              f"{n_active} dims with MSE<0.01")
        print(f"  Mean MSE: {self.fit_errors.mean():.5f}, "
              f"Max MSE: {self.fit_errors.max():.5f} (dim {self.fit_errors.argmax()})")

    def _get_theta(self, SA: np.ndarray) -> np.ndarray:
        return np.asarray(self._library.transform(SA))

    def predict(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Mean correction only (compatibility). Returns (obs_dim,)."""
        SA = np.concatenate([s, a])[None]
        Theta = self._get_theta(SA)
        delta = np.array([m.predict_mean_std(Theta)[0][0] for m in self.models])
        return delta

    def predict_batch(self, SA: np.ndarray) -> np.ndarray:
        """Batch mean correction. Returns (N, obs_dim)."""
        Theta = self._get_theta(SA)
        means = np.stack([m.predict_mean_std(Theta)[0] for m in self.models], axis=-1)
        return means  # (N, obs_dim)

    def predict_with_uncertainty(self, s: np.ndarray, a: np.ndarray):
        """
        Returns (correction, per_dim_std, gate_value).

        correction: (obs_dim,) — ensemble mean prediction
        per_dim_std: (obs_dim,) — ensemble disagreement per dim
        gate_value: float in [0, 1] — overall confidence
        """
        SA = np.concatenate([s, a])[None]
        Theta = self._get_theta(SA)

        means = np.zeros(self.obs_dim)
        stds = np.zeros(self.obs_dim)
        for i, model in enumerate(self.models):
            m, s = model.predict_mean_std(Theta)
            means[i] = m[0]
            stds[i] = s[0]

        # Gate: smooth sigmoid based on mean disagreement
        # High disagreement → low gate, low disagreement → high gate
        avg_std = float(stds.mean())
        gate = 1.0 / (1.0 + avg_std / self.gate_tau)

        return means, stds, gate

    def predict_batch_with_uncertainty(self, SA: np.ndarray):
        """
        Batch version. Returns (corrections, per_dim_stds, gate_values).

        corrections: (N, obs_dim)
        per_dim_stds: (N, obs_dim)
        gate_values: (N,)
        """
        Theta = self._get_theta(SA)
        N = len(SA)

        all_means = np.zeros((N, self.obs_dim))
        all_stds = np.zeros((N, self.obs_dim))

        for i, model in enumerate(self.models):
            m, s = model.predict_mean_std(Theta)
            all_means[:, i] = m
            all_stds[:, i] = s

        avg_stds = all_stds.mean(axis=1)  # (N,)
        gates = 1.0 / (1.0 + avg_stds / self.gate_tau)  # (N,)

        return all_means, all_stds, gates

    def correction_coverage(self, SA: np.ndarray, delta_s: np.ndarray) -> dict:
        """Compatibility with SINDyStateCorrector API."""
        pred = self.predict_batch(SA)
        raw_rmse = np.sqrt(np.mean(delta_s ** 2))
        corr_rmse = np.sqrt(np.mean((delta_s - pred) ** 2))
        return {
            "rmse_reduction_pct": (1 - corr_rmse / max(raw_rmse, 1e-8)) * 100,
            "raw_rmse": raw_rmse,
            "corr_rmse": corr_rmse,
        }
