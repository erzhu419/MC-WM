"""
Track A: SINDy + NAU/NMU symbolic residual model.

Adapted from CS-BAPR/sindy-rl/sindy_rl/dynamics.py (EnsembleSINDyDynamicsModel).
Key changes:
  - Operates on residuals Δ(s,a), not full state transitions
  - Each output dimension (Δs_i, Δr, Δd) gets its own SINDy model
  - NAU/NMU head (SymbolicResidualHead) wraps the discovered symbolic features
  - Per-dim fit error ε tracked for gate (dev manual §6.2)
  - Basis library can be extended externally (self-hypothesizing loop hook)
  - SINDy uses STLSQ (sparse threshold least squares) — same as CS-BAPR default
"""

import numpy as np
import pysindy as ps
import warnings
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from mc_wm.networks.nau_nmu import SymbolicResidualHead


# ---------------------------------------------------------------------------
# Default basis libraries
# ---------------------------------------------------------------------------

def make_poly2_library(include_bias: bool = True) -> ps.PolynomialLibrary:
    """Degree-2 polynomial library — the starting point (round 1)."""
    return ps.PolynomialLibrary(degree=2, include_bias=include_bias)


def make_custom_library(feature_fns: list, feature_names: list) -> ps.CustomLibrary:
    """Build a custom library from a list of lambda functions + names."""
    return ps.CustomLibrary(
        library_functions=feature_fns,
        function_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Per-dimension SINDy model
# ---------------------------------------------------------------------------

class SINDyDimModel:
    """
    SINDy model for a single scalar output dimension.

    Wraps pysindy.SINDy to model Δ_i(s, a) where Δ_i is one element
    of the full-tuple residual (one state dim, or Δr, or Δd).
    """

    def __init__(
        self,
        library: ps.feature_library.base.BaseFeatureLibrary = None,
        threshold: float = 0.05,
        alpha: float = 0.05,
    ):
        self.library = library or make_poly2_library()
        self.threshold = threshold
        self.alpha = alpha
        self._fitted: bool = False   # was: self._model (never set after fit — bug fixed)
        self.fit_error: float = float("inf")
        self.feature_names_: List[str] = []
        self._coef: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit SINDy on inputs X (N, feature_dim) and scalar targets y (N,).
        X should be (s, a) concatenation.  y is one column of delta.
        """
        from sklearn.linear_model import Ridge

        # np.asarray() needed: pysindy ≥2.0 returns AxesArray from fit_transform/transform,
        # which breaks matrix multiplication without explicit conversion.
        Theta = np.asarray(self.library.fit_transform(X))

        reg = Ridge(alpha=self.alpha, fit_intercept=False)
        reg.fit(Theta, y)
        coef = reg.coef_.reshape(1, -1)

        # Iterative thresholding (STLSQ)
        mask = np.abs(coef) > self.threshold
        for _ in range(10):
            if not np.any(mask):
                break
            reg2 = Ridge(alpha=self.alpha, fit_intercept=False)
            reg2.fit(Theta[:, mask[0]], y)
            coef_new = np.zeros_like(coef)
            coef_new[0, mask[0]] = reg2.coef_
            coef = coef_new
            mask = np.abs(coef) > self.threshold

        self._coef = coef
        self._mask = mask[0] if np.any(mask) else np.zeros(coef.shape[1], dtype=bool)
        self.feature_names_ = self.library.get_feature_names()
        self._fitted = True

        # Holdout error (10% split)
        N = len(X)
        idx = np.random.permutation(N)
        hold = idx[:max(1, N // 10)]
        y_pred = Theta[hold] @ coef[0]
        self.fit_error = float(np.mean((y[hold] - y_pred) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict Δ_i for a batch of (s, a) inputs. Returns (N,) array."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        Theta = np.asarray(self.library.transform(X))  # np.asarray: AxesArray fix
        return Theta @ self._coef[0]

    def get_active_features(self) -> List[str]:
        """Return names of non-zero SINDy terms."""
        if not self.feature_names_:
            return []
        return [self.feature_names_[i] for i in range(len(self.feature_names_)) if self._mask[i]]

    def set_library(self, library):
        """Replace the library (used by self-hypothesizing loop expansion)."""
        self.library = library


# ---------------------------------------------------------------------------
# Full-tuple SINDy Track
# ---------------------------------------------------------------------------

class SINDyTrack:
    """
    Track A of the dual-track residual model (dev manual §5.1).

    Maintains one SINDyDimModel per output dimension:
      - obs_dim models for Δs
      - 1 model for Δr
      - 1 model for Δd

    After fitting, wraps the SINDy features in a SymbolicResidualHead
    (NAU/NMU) for OOD-bounded prediction and gradient-based fine-tuning.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        library: ps.feature_library.base.BaseFeatureLibrary = None,
        sindy_threshold: float = 0.05,
        sindy_alpha: float = 0.05,
        nau_nmu_hidden: int = 32,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.device = device

        # One SINDy model per output dim — each gets its own library copy
        # (pysindy library objects are stateful after fit_transform;
        #  sharing a single object across models causes shape mismatch errors)
        from copy import deepcopy
        default_lib = library or make_poly2_library()
        self.sindy_s = [
            SINDyDimModel(library=deepcopy(default_lib), threshold=sindy_threshold, alpha=sindy_alpha)
            for _ in range(obs_dim)
        ]
        self.sindy_r = SINDyDimModel(library=deepcopy(default_lib), threshold=sindy_threshold, alpha=sindy_alpha)
        self.sindy_d = SINDyDimModel(library=deepcopy(default_lib), threshold=sindy_threshold, alpha=sindy_alpha)

        # NAU/NMU head (fine-tuning wrapper, initialized lazily after fit)
        self._head_s: Optional[SymbolicResidualHead] = None
        self._head_r: Optional[SymbolicResidualHead] = None
        self._head_d: Optional[SymbolicResidualHead] = None

        # Fit errors per element (used by gate)
        self.eps_s = np.full(obs_dim, float("inf"))
        self.eps_r = float("inf")
        self.eps_d = float("inf")

        # Training center for OOD distance computation
        self._train_center: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, SA: np.ndarray, delta_s: np.ndarray, delta_r: np.ndarray, delta_d: np.ndarray):
        """
        Fit all SINDy models.

        Args:
            SA       shape (N, obs_dim + act_dim)
            delta_s  shape (N, obs_dim)
            delta_r  shape (N, 1)
            delta_d  shape (N, 1)
        """
        self._train_center = SA.mean(axis=0)

        for i, model in enumerate(self.sindy_s):
            model.fit(SA, delta_s[:, i])
            self.eps_s[i] = model.fit_error

        self.sindy_r.fit(SA, delta_r[:, 0])
        self.eps_r = self.sindy_r.fit_error

        self.sindy_d.fit(SA, delta_d[:, 0])
        self.eps_d = self.sindy_d.fit_error

        # Build NAU/NMU heads on top of the number of SINDy features
        n_feat = len(self.sindy_s[0].feature_names_) if self.sindy_s[0].feature_names_ else self.input_dim
        self._head_s = SymbolicResidualHead(n_feat, self.obs_dim).to(self.device)
        self._head_r = SymbolicResidualHead(n_feat, 1).to(self.device)
        self._head_d = SymbolicResidualHead(n_feat, 1).to(self.device)

    def update_library(self, library, element: str = "all"):
        """
        Replace basis library for re-fitting (called by auto_expand).
        Each model gets its own copy (pysindy libraries are stateful).
        element: "all" | "s" | "r" | "d"
        """
        from copy import deepcopy
        if element in ("all", "s"):
            for m in self.sindy_s:
                m.set_library(deepcopy(library))
        if element in ("all", "r"):
            self.sindy_r.set_library(deepcopy(library))
        if element in ("all", "d"):
            self.sindy_d.set_library(deepcopy(library))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, s: np.ndarray, a: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict full-tuple residual for a batch.

        Args:
            s  shape (N, obs_dim)
            a  shape (N, act_dim)

        Returns:
            dict with keys: delta_s (N, obs_dim), delta_r (N, 1), delta_d (N, 1)
        """
        SA = np.concatenate([s, a], axis=-1)
        return self.predict_raw(SA)

    def predict_raw(self, SA: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict from pre-concatenated feature matrix.
        Used when extra columns (from auto-expand) are appended to SA.
        """
        delta_s = np.stack([m.predict(SA) for m in self.sindy_s], axis=-1)
        delta_r = self.sindy_r.predict(SA).reshape(-1, 1)
        delta_d = self.sindy_d.predict(SA).reshape(-1, 1)

        return {"delta_s": delta_s, "delta_r": delta_r, "delta_d": delta_d}

    # ------------------------------------------------------------------
    # OOD bounds
    # ------------------------------------------------------------------

    def ood_distance(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        L2 distance from query point to training distribution center.
        Used by the gate to compute bound = ε + ε_J*d + (L/2)*d².
        Returns shape (N,).
        """
        SA = np.concatenate([s, a], axis=-1)
        if self._train_center is None:
            return np.zeros(len(SA))
        return np.linalg.norm(SA - self._train_center[None, :], axis=-1)

    def get_fit_errors(self) -> Dict[str, np.ndarray]:
        return {
            "eps_s": self.eps_s,
            "eps_r": np.array([self.eps_r]),
            "eps_d": np.array([self.eps_d]),
        }

    def get_active_features(self) -> Dict[str, List[str]]:
        """Summary of non-zero SINDy terms per element."""
        return {
            "delta_s": [m.get_active_features() for m in self.sindy_s],
            "delta_r": self.sindy_r.get_active_features(),
            "delta_d": self.sindy_d.get_active_features(),
        }

    # ------------------------------------------------------------------
    # Fine-tuning via NAU/NMU heads
    # ------------------------------------------------------------------

    def finetune_nau_nmu(
        self,
        SA: np.ndarray,
        delta_s: np.ndarray,
        delta_r: np.ndarray,
        delta_d: np.ndarray,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ):
        """
        After SINDy fitting, fine-tune the NAU/NMU heads on the discovered features.
        The SINDy library serves as a fixed feature extractor; NAU/NMU learns
        the output coefficients with formal L=0/L=2|c| guarantees.
        """
        if self._head_s is None:
            raise RuntimeError("Call fit() before finetune_nau_nmu().")

        # Build SINDy feature matrix
        Theta = self.sindy_s[0].library.transform(SA)
        Theta_t = torch.tensor(Theta, dtype=torch.float32).to(self.device)
        ds_t = torch.tensor(delta_s, dtype=torch.float32).to(self.device)
        dr_t = torch.tensor(delta_r, dtype=torch.float32).to(self.device)
        dd_t = torch.tensor(delta_d, dtype=torch.float32).to(self.device)

        params = (
            list(self._head_s.parameters())
            + list(self._head_r.parameters())
            + list(self._head_d.parameters())
        )
        optimizer = optim.Adam(params, lr=lr)

        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = (
                nn.MSELoss()(self._head_s(Theta_t), ds_t)
                + nn.MSELoss()(self._head_r(Theta_t), dr_t)
                + nn.MSELoss()(self._head_d(Theta_t), dd_t)
                + 0.01 * (
                    self._head_s.regularization_loss()
                    + self._head_r.regularization_loss()
                    + self._head_d.regularization_loss()
                )
            )
            loss.backward()
            optimizer.step()
