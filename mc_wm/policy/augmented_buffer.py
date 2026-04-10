"""
Confidence-Weighted Augmented Buffer (dev manual §7.1).

Combines:
  - Real offline data (confidence = 1.0, full trust)
  - Corrected sim data (confidence = min(gate_A, gate_B), gated trust)

Only sim transitions with confidence > min_threshold are stored.
Used by RobustIQL for training with confidence-weighted losses.

Source: adapted from H2O MixedReplayBuffer (sumo-rl/H2O/SimpleSAC/mixed_replay_buffer.py)
Changes:
  - Removed IS-reweighting discriminator (replaced by gate-based confidence)
  - Added confidence field per transition
  - Added separate partitions for real vs sim (like BusMixedReplayBuffer)
"""

import numpy as np
from typing import Dict, Optional


class AugmentedBuffer:
    """
    Ring buffer with per-transition confidence scores.

    Layout:
        [0, real_size)         — fixed offline real data (confidence=1.0)
        [real_size, capacity)  — growing online corrected-sim data

    Both partitions support sampling; the critic loss weights by confidence.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 500_000,
        min_threshold: float = 0.1,   # drop sim data with confidence < this
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.min_threshold = min_threshold
        self.device = device

        # Pre-allocate numpy arrays (avoids repeated concat)
        self.observations     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations= np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions          = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards          = np.zeros((capacity, 1), dtype=np.float32)
        self.dones            = np.zeros((capacity, 1), dtype=np.float32)
        self.confidence       = np.zeros((capacity, 1), dtype=np.float32)

        self._real_size  = 0   # fixed after loading
        self._total_size = 0   # includes sim
        self._sim_ptr    = 0   # circular pointer for sim partition

    # ------------------------------------------------------------------
    # Loading real offline data
    # ------------------------------------------------------------------

    def load_real(self, observations, actions, next_observations, rewards, dones):
        """
        Load offline real dataset (confidence = 1.0).

        Must be called before any append_sim.
        """
        N = len(observations)
        assert N <= self.capacity, f"Real dataset ({N}) exceeds buffer capacity ({self.capacity})."
        self.observations[:N]      = observations
        self.next_observations[:N] = next_observations
        self.actions[:N]           = actions
        self.rewards[:N]           = rewards.reshape(-1, 1)
        self.dones[:N]             = dones.reshape(-1, 1)
        self.confidence[:N]        = 1.0

        self._real_size = N
        self._total_size = N
        self._sim_ptr = 0

    # ------------------------------------------------------------------
    # Adding corrected sim data
    # ------------------------------------------------------------------

    def append_sim(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: float,
        s_next: np.ndarray,
        d: float,
        confidence: float,
    ):
        """
        Add one corrected sim transition if confidence > min_threshold.
        """
        if confidence < self.min_threshold:
            return

        sim_capacity = self.capacity - self._real_size
        if sim_capacity <= 0:
            return

        idx = self._real_size + (self._sim_ptr % sim_capacity)
        self.observations[idx]      = s
        self.next_observations[idx] = s_next
        self.actions[idx]           = a
        self.rewards[idx]           = r
        self.dones[idx]             = d
        self.confidence[idx]        = confidence

        self._sim_ptr += 1
        self._total_size = min(self.capacity, self._real_size + self._sim_ptr)

    def append_sim_batch(self, corrector_output: Dict, s: np.ndarray, a: np.ndarray,
                         s_next_sim: np.ndarray, r_sim: np.ndarray, d_sim: np.ndarray):
        """
        Batch version: add corrected sim transitions from GatedCorrector output.

        Args:
            corrector_output: output of GatedCorrector.correct()
            s, a: original sim states and actions
            s_next_sim, r_sim, d_sim: raw sim transitions (pre-correction)
        """
        s_corr = corrector_output["s_next_corrected"]
        r_corr = corrector_output["r_corrected"].squeeze(-1)
        d_corr = corrector_output["d_corrected"].squeeze(-1)
        conf   = corrector_output["confidence"]

        for i in range(len(s)):
            self.append_sim(
                s=s[i], a=a[i],
                r=float(r_corr[i]),
                s_next=s_corr[i],
                d=float(d_corr[i]),
                confidence=float(conf[i]),
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, scope: str = "all") -> Dict[str, np.ndarray]:
        """
        Sample a batch.

        Args:
            scope: "all" | "real" | "sim"

        Returns dict with: observations, actions, next_observations, rewards, dones, confidence
        """
        if scope == "real":
            indices = np.random.randint(0, self._real_size, size=batch_size)
        elif scope == "sim":
            sim_size = max(1, self._total_size - self._real_size)
            offsets = np.random.randint(0, sim_size, size=batch_size)
            indices = self._real_size + offsets
        else:
            indices = np.random.randint(0, self._total_size, size=batch_size)

        return {
            "observations":      self.observations[indices],
            "actions":           self.actions[indices],
            "next_observations": self.next_observations[indices],
            "rewards":           self.rewards[indices],
            "dones":             self.dones[indices],
            "confidence":        self.confidence[indices],
        }

    @property
    def real_size(self) -> int:
        return self._real_size

    @property
    def sim_size(self) -> int:
        return max(0, self._total_size - self._real_size)

    def __len__(self):
        return self._total_size
