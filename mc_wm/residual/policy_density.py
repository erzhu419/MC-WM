"""
Policy-distribution density estimator for policy-aware residual fitting.

The Residual Simulation Lemma bounds the on-policy value gap by
TV_{d^π}(M_real, M_sim+δ): the divergence is *policy-distribution-weighted*,
not uniform.  Therefore the residual fit should also be d^π-weighted —
giving more weight to (s, a) the current policy actually visits, less
to states it rarely sees.

This module exposes a ``PolicyDensity`` estimator over (s, a) pairs that
returns per-sample weights ``w_i ∈ [w_min, 1]`` for use in weighted least
squares (SINDy) and weighted SGD (NAU/NMU coefficient refit).

Strategy choices, in order of computational cost:

    "recency"   → pure index-based recency weighting (no model);
                  the most recent N transitions get weight 1, older ones
                  decay geometrically.  Cheapest, weakest.

    "knn"       → k-nearest-neighbour density on the most recent rollout
                  buffer in (s, a) space.  Weights ∝ local sample density
                  in the recent π's footprint.  Reasonable middle ground.

    "buffer"    → assume the entire recent-rollout buffer IS a sample
                  from d^π; weight a fit-batch entry by its kNN distance
                  to the rollout buffer.  Closest-to-recent-rollouts gets
                  highest weight.  Best fit-time correspondence to the
                  formal lemma without training a density model.

All three return weights normalized to ``[w_min, 1]``.  ``w_min > 0``
preserves a baseline contribution from off-policy data so the fit doesn't
overfit to the policy's current narrow visitation.
"""

from __future__ import annotations

import numpy as np
from typing import Literal


class PolicyDensity:
    """
    Estimate per-sample weights for residual fit, biased toward the
    current policy's (s, a) visitation.

    Args:
        strategy: which estimator to use; see module docstring.
        w_min:    minimum weight floor in [0, 1].  Default 0.1 → off-policy
                  samples still contribute 10% as much as on-policy.
        recency_horizon: only used by "recency"; how many recent transitions
                  count as "on-policy".  Tail gets exp-decay.
        knn_k:    only used by "knn"/"buffer"; nearest-neighbour count.
    """

    def __init__(self,
                 strategy: Literal["recency", "knn", "buffer"] = "buffer",
                 w_min: float = 0.1,
                 recency_horizon: int = 5000,
                 knn_k: int = 8):
        if not 0.0 <= w_min <= 1.0:
            raise ValueError(f"w_min must be in [0, 1], got {w_min}")
        if strategy not in ("recency", "knn", "buffer"):
            raise ValueError(f"unknown strategy: {strategy}")
        self.strategy = strategy
        self.w_min = w_min
        self.recency_horizon = recency_horizon
        self.knn_k = knn_k
        # Set by ``fit_reference``; the policy-distribution sample bank.
        self._ref_sa: np.ndarray | None = None

    # ─── Reference-distribution setup ────────────────────────────────
    def fit_reference(self, sa_buffer: np.ndarray) -> None:
        """
        Provide the (s, a) sample bank that defines d^π.

        Typically the most-recent-N real transitions from the env buffer.
        For ``strategy="recency"`` this is unused (only fit_batch order
        matters); for "knn"/"buffer" this is the neighbour search target.
        """
        if sa_buffer.ndim != 2:
            raise ValueError(f"sa_buffer must be (N, dim), got {sa_buffer.shape}")
        # Normalize per-dim so distance is scale-invariant.
        # Save mean/std so weights() can re-apply.
        self._ref_mean = sa_buffer.mean(axis=0, keepdims=True)
        self._ref_std = sa_buffer.std(axis=0, keepdims=True) + 1e-6
        self._ref_sa = (sa_buffer - self._ref_mean) / self._ref_std

    # ─── Weight computation ──────────────────────────────────────────
    def weights(self, sa_query: np.ndarray) -> np.ndarray:
        """
        Compute (N,) per-sample weights for the fit batch ``sa_query``.

        Output is normalized to [w_min, 1] (max → 1, min → w_min).
        """
        if sa_query.ndim != 2:
            raise ValueError(f"sa_query must be (N, dim), got {sa_query.shape}")
        n = sa_query.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        if self.strategy == "recency":
            # Geometric decay over fit batch order.  Most recent samples
            # (highest index) get weight 1; oldest get exp(-1) ≈ 0.37.
            half_life = max(1.0, self.recency_horizon / 2.0)
            ages = np.arange(n)[::-1].astype(np.float64)  # 0 = most recent
            raw = np.exp(-ages / half_life)
        else:
            if self._ref_sa is None or self._ref_sa.size == 0:
                # No reference yet (e.g. very early training): degrade to
                # uniform weights.
                return np.ones(n, dtype=np.float64)
            sa_norm = (sa_query - self._ref_mean) / self._ref_std
            # Pairwise squared distances; vectorised for medium batch sizes.
            # For very large refs (>50k) consider sklearn BallTree instead.
            if self._ref_sa.shape[0] > 8000:
                # Subsample reference to keep memory manageable.
                idx = np.random.choice(self._ref_sa.shape[0], 8000, replace=False)
                ref = self._ref_sa[idx]
            else:
                ref = self._ref_sa
            # (n, n_ref) distance matrix in chunks of 1024 to bound memory.
            dists = np.empty((n, ref.shape[0]), dtype=np.float64)
            chunk = 1024
            for i in range(0, n, chunk):
                d = ((sa_norm[i:i + chunk, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
                dists[i:i + chunk] = d
            # k-th nearest distance per query.
            k = max(1, min(self.knn_k, ref.shape[0]))
            kth = np.partition(dists, k - 1, axis=1)[:, :k]
            mean_kth = np.sqrt(kth.mean(axis=1) + 1e-8)
            # Closer to ref → smaller mean_kth → higher weight.  Use a
            # bandwidth based on the global median to keep weights well-scaled.
            h = float(np.median(mean_kth)) + 1e-6
            raw = np.exp(-mean_kth / h)

        # Normalise to [w_min, 1].
        if raw.max() <= 1e-12:
            return np.ones(n, dtype=np.float64)
        normed = raw / raw.max()
        return self.w_min + (1.0 - self.w_min) * normed


def weighted_least_squares(theta: np.ndarray, y: np.ndarray,
                           w: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """
    Weighted ridge regression: minimize Σ w_i ‖y_i - Θ_i β‖² + ridge ‖β‖².

    Used by the SINDy refit when policy-aware weights are provided.

    Args:
        theta: (N, F) feature matrix
        y:     (N, D) targets (state residual per dim)
        w:     (N,) sample weights, expected in (0, 1]
        ridge: L2 regularization strength on β

    Returns:
        beta: (F, D) coefficient matrix
    """
    if theta.ndim != 2 or y.ndim != 2:
        raise ValueError("theta must be (N, F) and y must be (N, D)")
    n, f = theta.shape
    if w.shape != (n,):
        raise ValueError(f"w must be (N,) matching theta; got {w.shape}")
    sw = np.sqrt(np.clip(w, 1e-8, None))
    th_w = theta * sw[:, None]
    y_w = y * sw[:, None]
    # Closed form: β = (Θᵀ W Θ + ridge I)⁻¹ Θᵀ W y
    A = th_w.T @ th_w
    if ridge > 0:
        A = A + ridge * np.eye(f)
    B = th_w.T @ y_w
    # Solve via lstsq for numerical stability when A is near-singular.
    beta, *_ = np.linalg.lstsq(A, B, rcond=None)
    return beta
