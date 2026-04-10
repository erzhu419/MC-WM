"""
Uncertainty-Gated Correction (dev manual §6).

Two gate types:
  GateA: symbolic (Track A) — uses OOD bound with formal guarantee
  GateB: ensemble (Track B) — uses ensemble disagreement, more aggressive

Combined corrector:
    s'_corrected = s'_sim + g_A * Δ̂_A(s,a) + g_B * Δ̂_B(s,a)

Safety guarantee (Prop §6.3):
    g → 0 as uncertainty grows → graceful fallback to raw sim.
"""

import numpy as np
from typing import Dict, Tuple


class GateA:
    """
    Track A gate: driven by OOD polynomial bound.

    g_A(s, a) = max(0,  1 - bound(s,a) / τ_A)

    where bound = ε_fit + ε_J * ‖d‖ + (L/2) * ‖d‖²

    This is a continuous gate that gracefully decays to 0 as distance grows.
    """

    def __init__(
        self,
        tau_A: float = 1.0,       # threshold τ_A; tune per env
        eps_jac: float = 0.01,    # Jacobian consistency ε_J (default small)
    ):
        self.tau_A = tau_A
        self.eps_jac = eps_jac

    def gate(
        self,
        eps_fit: np.ndarray,    # per-element fit error (scalar or array)
        L_eff: float,           # effective derivative-Lipschitz of NAU/NMU head
        dist: np.ndarray,       # OOD distance (N,)
    ) -> np.ndarray:
        """
        Returns gate values in [0, 1] for a batch.

        Args:
            eps_fit: fit error for this element (scalar)
            L_eff: L_eff from SymbolicResidualHead.L_eff
            dist: OOD distances (N,)

        Returns:
            gate shape (N,) ∈ [0, 1]
        """
        bound = eps_fit + self.eps_jac * dist + (L_eff / 2) * dist ** 2
        g = np.maximum(0.0, 1.0 - bound / self.tau_A)
        return g


class GateB:
    """
    Track B gate: driven by ensemble disagreement.

    g_B(s, a) = σ((τ_B - disagreement) / temperature)

    Lower τ_B than GateA because ensemble has no OOD guarantee.
    """

    def __init__(self, tau_B: float = 0.3, temperature: float = 0.1):
        self.tau_B = tau_B
        self.temperature = temperature

    def gate(self, disagreement: np.ndarray) -> np.ndarray:
        """
        Returns gate values in (0, 1) for a batch.

        Args:
            disagreement: ensemble std (N,)

        Returns:
            gate shape (N,) ∈ (0, 1)
        """
        logit = (self.tau_B - disagreement) / self.temperature
        return 1.0 / (1.0 + np.exp(-logit))   # sigmoid


class UncertaintyGate:
    """
    Combines GateA (symbolic) and GateB (ensemble).
    Used by GatedCorrector.
    """

    def __init__(self, tau_A: float = 1.0, tau_B: float = 0.3,
                 eps_jac: float = 0.01, temperature: float = 0.1):
        self.gate_A = GateA(tau_A=tau_A, eps_jac=eps_jac)
        self.gate_B = GateB(tau_B=tau_B, temperature=temperature)


class GatedCorrector:
    """
    Applies gated correction to sim transitions using Track A + Track B predictions.

    Correction rule (dev manual §6.1):
        s'_c = s'_sim + g_A * Δ̂_A + g_B * Δ̂_B
        r_c  = r_sim  + g_A_r * Δ̂r_A + g_B_r * Δ̂r_B
        d_c  = clip(d_sim + g_A_d * Δ̂d_A + g_B_d * Δ̂d_B, 0, 1)

    confidence = min(g_A, g_B) per sample — used by augmented buffer.
    """

    def __init__(self, gate: UncertaintyGate, sindy_track, ensemble_track):
        self.gate = gate
        self.sindy_track = sindy_track
        self.ensemble_track = ensemble_track

    def correct(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_next_sim: np.ndarray,
        r_sim: np.ndarray,
        d_sim: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute corrected (s', r, d) for a batch of sim transitions.

        Returns dict with keys:
          s_next_corrected  (N, obs_dim)
          r_corrected       (N, 1)
          d_corrected       (N, 1)  clipped to [0, 1]
          confidence        (N,)    min gate value per sample
          gate_A            (N,)    Track A gate
          gate_B            (N,)    Track B gate
        """
        # Track A predictions
        pred_A = self.sindy_track.predict(s, a)
        dist = self.sindy_track.ood_distance(s, a)
        eps = self.sindy_track.get_fit_errors()

        # Track B predictions (includes disagreement)
        pred_B = self.ensemble_track.predict(s, a)

        # Compute gates per element
        # For s: use mean fit error across dims as representative ε
        eps_s_mean = float(eps["eps_s"].mean())
        L_s = 0.5  # default; updated after NAU/NMU fine-tuning
        g_A = self.gate.gate_A.gate(eps_s_mean, L_s, dist)    # (N,)
        g_B = self.gate.gate_B.gate(pred_B["disagreement_s"]) # (N,)

        eps_r = float(eps["eps_r"])
        g_A_r = self.gate.gate_A.gate(eps_r, L_s, dist)
        g_B_r = self.gate.gate_B.gate(pred_B["disagreement_r"])

        eps_d = float(eps["eps_d"])
        g_A_d = self.gate.gate_A.gate(eps_d, L_s, dist)
        g_B_d = self.gate.gate_B.gate(pred_B["disagreement_d"])

        # Apply corrections
        g_A_col = g_A[:, None]
        g_B_col = g_B[:, None]

        s_corr = s_next_sim + g_A_col * pred_A["delta_s"] + g_B_col * pred_B["delta_s"]
        r_corr = r_sim + (g_A_r * g_A_r)[:, None] * pred_A["delta_r"] + (g_B_r * g_B_r)[:, None] * pred_B["delta_r"]
        d_corr = d_sim + g_A_d[:, None] * pred_A["delta_d"] + g_B_d[:, None] * pred_B["delta_d"]
        d_corr = np.clip(d_corr, 0.0, 1.0)

        confidence = np.minimum(g_A, g_B)

        return {
            "s_next_corrected": s_corr,
            "r_corrected": r_corr,
            "d_corrected": d_corr,
            "confidence": confidence,
            "gate_A": g_A,
            "gate_B": g_B,
        }
