"""
NAU/NMU layers adapted for MC-WM residual heads.

Source: CS-BAPR/csbapr/networks/nau_nmu.py
Modifications:
  - Removed SAC actor; added SymbolicResidualHead for residual modeling
  - SymbolicResidualHead takes (s, a) feature vector and predicts scalar residual
  - Exposes compute_ood_bound() to drive the uncertainty gate (Section 6.2 of dev manual)

OOD bound (CS-BAPR Theorem 4.35):
    |Δ̂(s_ood, a) - Δ_true(s_ood, a)| ≤ ε + ε_J*‖d‖ + (L/2)*‖d‖²
where d = s_ood - s_train_center, ε = fit error, ε_J = Jacobian mismatch, L = NMU Lipschitz.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAULayer(nn.Module):
    """
    Neural Arithmetic Unit (Addition/Subtraction).
    Weights constrained toward {-1, 0, 1} via clamp + regularization.
    Derivative Lipschitz constant L = 0 (linear → constant Jacobian).

    Copied verbatim from CS-BAPR/csbapr/networks/nau_nmu.py.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self._reset_parameters()

    def _reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        nn.init.uniform_(self.W, -r, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_clamped = torch.clamp(self.W, -1.0, 1.0)
        return F.linear(x, W_clamped)

    def regularization_loss(self) -> torch.Tensor:
        W_abs = self.W.abs()
        return torch.min(W_abs, (1 - W_abs).abs()).mean()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class NMULayer(nn.Module):
    """
    Neural Multiplication Unit (simplified quadratic).
    f(x) = c·x²,  derivative Lipschitz constant L = 2·max(|c|).

    Copied verbatim from CS-BAPR/csbapr/networks/nau_nmu.py.
    """
    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.coeff = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff * x ** 2

    @property
    def lipschitz_constant(self) -> float:
        return 2 * self.coeff.abs().max().item()

    def extra_repr(self) -> str:
        return f'features={self.features}, L=2|c|'


class SymbolicResidualHead(nn.Module):
    """
    NAU/NMU output head for one scalar dimension of the residual.

    Used by SINDy track (sindy_track.py) to wrap each SINDy feature
    in a differentiable form with formal OOD bounds.

    Architecture (per dev manual §5.2):
        feature_net: [feature_dim → 64 → 32] with LeakyReLU
        NAU head (L=0):  32 → output_dim
        NMU head (L=2|c|): 32 → output_dim
        mixed output = α·NAU + (1-α)·NMU

    Formal OOD bound:
        ‖Δ̂(s_ood) - Δ_true(s_ood)‖ ≤ ε + ε_J·‖d‖ + (L_eff/2)·‖d‖²
    where L_eff = (1-α)·2·max(|c|)·K_g·B_g  (composed Lipschitz).
    """
    def __init__(self, feature_dim: int, output_dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
        )
        self.nau = NAULayer(32, output_dim)
        # NMU operates per-element; project 32 → output_dim first
        self.nmu_proj = nn.Linear(32, output_dim, bias=False)
        self.nmu = NMULayer(output_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_net(x)
        nau_out = self.nau(h)
        nmu_out = self.nmu(self.nmu_proj(h))
        alpha = torch.sigmoid(self.alpha)
        return alpha * nau_out + (1 - alpha) * nmu_out

    def regularization_loss(self) -> torch.Tensor:
        return self.nau.regularization_loss()

    @property
    def L_eff(self) -> float:
        """
        Effective derivative-Lipschitz constant of the composed architecture.
        L_eff = (1-α)·L_nmu·K_g·B_g
        Used by the uncertainty gate to set τ (dev manual §6.2).
        """
        with torch.no_grad():
            alpha = torch.sigmoid(self.alpha).item()
            L_nmu = self.nmu.lipschitz_constant
            K_g = 1.0
            for m in self.feature_net:
                if isinstance(m, nn.Linear):
                    K_g *= torch.linalg.svdvals(m.weight)[0].item()
                elif isinstance(m, nn.LeakyReLU):
                    K_g *= 1.0
        return (1 - alpha) * L_nmu * K_g * K_g  # B_g ≈ K_g

    def compute_ood_bound(self, eps_fit: float, eps_jac: float, dist: float) -> float:
        """
        Returns the OOD error bound at distance `dist` from training support.
        Matches dev manual Eq. (5.2):
            bound = ε_fit + ε_jac·‖d‖ + (L_eff/2)·‖d‖²
        """
        return eps_fit + eps_jac * dist + (self.L_eff / 2) * dist ** 2
