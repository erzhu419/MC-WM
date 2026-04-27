"""
Reward-aware feature acceptance for self-hypothesis discovery.

Standard self-hypothesis discovery accepts a feature based on a
correlation threshold with the residual.  This is a *proxy* for what we
care about: does the feature improve downstream RL reward?  RAHD Stage D
replaces the proxy with a direct downstream check.

For each candidate feature ``c`` that already passed orthogonality and
correlation filters:

    1. Build two corrected world models —
         M_with    = M_sim + δ_with_c     (δ refit including c)
         M_without = M_sim + δ_without_c  (δ refit excluding c)
    2. Run K short rollouts of the *current* policy on each model;
       average reward over the K rollouts.
    3. Accept c iff (R_with - R_without) ≥ δ_min, with a t-test gate to
       reject features whose apparent improvement is within noise.

This module supplies the *evaluation harness* (rollouts, statistics);
the actual fit-with/fit-without comparison is left to the caller, since
fitting δ is adapter-specific.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import Callable


class RewardValidator:
    """
    Mini-rollout reward comparator for candidate-feature acceptance.

    Args:
        model_step:  callable(state_np, action_np, model_id) → (next_state_np,
                     reward_np, done_np). Should be deterministic given the
                     same seed.  ``model_id`` is a string slot the validator
                     uses to swap between "with" and "without" models.
        policy_act:  callable(state_np, deterministic=True) → action_np.
        rollout_len: per-rollout horizon (steps).
        n_rollouts:  number of rollouts per model variant.
        seed:        base RNG seed; each rollout uses seed + index so the
                     trajectories are reproducible.
        warmup_states: array of (M, obs_dim) initial states.  Each rollout
                     samples one uniformly without replacement.  Typically
                     the most-recent N real-env start states.
    """

    def __init__(self,
                 model_step: Callable,
                 policy_act: Callable,
                 rollout_len: int = 100,
                 n_rollouts: int = 5,
                 seed: int = 0,
                 warmup_states: np.ndarray | None = None):
        self.model_step = model_step
        self.policy_act = policy_act
        self.rollout_len = rollout_len
        self.n_rollouts = n_rollouts
        self.seed = seed
        self.warmup_states = warmup_states

    def _sample_init_states(self, n: int) -> np.ndarray:
        if self.warmup_states is None or len(self.warmup_states) == 0:
            raise RuntimeError("warmup_states is required for reward validation")
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(self.warmup_states), size=n, replace=False
                         if n <= len(self.warmup_states) else True)
        return self.warmup_states[idx].copy()

    def _rollout(self, init_state: np.ndarray, model_id: str,
                 rng_seed: int) -> float:
        """Single rollout, return cumulative reward."""
        rng = np.random.default_rng(rng_seed)
        s = init_state.copy()
        total = 0.0
        for _ in range(self.rollout_len):
            a = self.policy_act(s[None, :], deterministic=True)[0]
            # Tiny exploration jitter for reproducibility-of-noise but not
            # heavy stochasticity (we want deterministic comparison).
            a = np.clip(a + 1e-3 * rng.standard_normal(a.shape), -1.0, 1.0)
            s_next, r, done = self.model_step(s, a, model_id)
            total += float(r)
            if done:
                break
            s = s_next
        return total

    def compare(self, model_id_with: str = "with",
                model_id_without: str = "without") -> dict:
        """
        Run ``n_rollouts`` rollouts on each model and report the difference.

        Returns:
            dict with mean rewards, std, delta, and a Welch's t-statistic
            so the caller can apply a significance gate.
        """
        starts = self._sample_init_states(self.n_rollouts)
        rewards_with: list[float] = []
        rewards_without: list[float] = []
        for i, init in enumerate(starts):
            rs_with = self._rollout(init, model_id_with, self.seed + i)
            rs_without = self._rollout(init, model_id_without, self.seed + i)
            rewards_with.append(rs_with)
            rewards_without.append(rs_without)

        mu_w = float(np.mean(rewards_with))
        mu_o = float(np.mean(rewards_without))
        s_w = float(np.std(rewards_with, ddof=1)) if len(rewards_with) > 1 else 0.0
        s_o = float(np.std(rewards_without, ddof=1)) if len(rewards_without) > 1 else 0.0
        n = max(1, len(rewards_with))
        # Welch's t: assumes unequal variance; safer with small n_rollouts.
        denom = math.sqrt(s_w ** 2 / n + s_o ** 2 / n) + 1e-9
        t_stat = (mu_w - mu_o) / denom

        return {
            "mu_with": mu_w,
            "mu_without": mu_o,
            "delta": mu_w - mu_o,
            "std_with": s_w,
            "std_without": s_o,
            "t_stat": t_stat,
            "n_rollouts": n,
            "rewards_with": rewards_with,
            "rewards_without": rewards_without,
        }


def accept_feature(stats: dict, delta_min: float = 0.0,
                   t_threshold: float = 1.0) -> bool:
    """
    Default acceptance gate for a candidate feature given compare() stats.

    Accepts iff:
      (1) ``delta = mu_with - mu_without`` ≥ ``delta_min``
      (2) Welch t-statistic ≥ ``t_threshold`` (very lenient by default
          because n_rollouts is small; tune up for tighter selection)

    The choice of (delta_min, t_threshold) matters more for tight basins:
      - delta_min = 0       → accept any positive trend
      - delta_min = 0.5*σ_o → require improvement bigger than baseline noise
      - t_threshold = 1.0   → ~70% one-sided confidence for n=5
      - t_threshold = 1.65  → ~95% one-sided confidence for n=5
    """
    return stats["delta"] >= delta_min and stats["t_stat"] >= t_threshold


def estimate_delta_l_eff(theta_old: np.ndarray, y_old: np.ndarray,
                         theta_with_c: np.ndarray, y: np.ndarray,
                         beta_old: np.ndarray, beta_new: np.ndarray) -> float:
    """
    Helper for RAHD Stage B: estimate ΔL_eff caused by adding a feature.

    Uses the Lipschitz constant of the linear coefficient layer as a proxy:

        L_eff ≈ ‖β‖_∞  +  spectral norm of post-NAU residual map

    Without the NAU head we approximate via the change in coefficient
    magnitudes pre- and post-addition.  Larger jump = more destabilizing.

    This is an approximation: a tighter version would re-fit a small NAU
    pass and directly read its L_eff.  The proxy here is fast and good
    enough for the gate (which only needs a relative comparison across
    candidates within one refit).

    Returns:
        Estimated ΔL_eff ≥ 0.  Larger means the new feature destabilizes
        the coefficient layer more.
    """
    if beta_old.shape[1] != beta_new.shape[1]:
        # Different output-dim ⇒ unable to compare; declare large jump.
        return float("inf")
    # Broadcast old β to the larger feature space (zero-pad missing rows).
    f_new = beta_new.shape[0]
    f_old = beta_old.shape[0]
    if f_old > f_new:
        return float("inf")
    pad = np.zeros((f_new - f_old, beta_old.shape[1]))
    beta_old_pad = np.vstack([beta_old, pad])
    delta = beta_new - beta_old_pad
    # ‖Δβ‖_∞ across all coefficients
    return float(np.max(np.abs(delta)))
