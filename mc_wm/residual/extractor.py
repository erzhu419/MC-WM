"""
Full-Tuple Residual Extractor.

Collects paired (sim, real) transitions and computes:
    Δ(s, a) = (Δs, Δr, Δd)

where:
    Δs = s'_real - s'_sim    (dynamics gap, shape obs_dim)
    Δr = r_real  - r_sim     (reward gap, scalar)
    Δd = d_real  - d_sim     (termination gap, scalar ∈ {-1, 0, 1})

Source: original (no prior CS-BAPR equivalent — this is MC-WM's novel contribution).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResidualSample:
    """One paired transition with full-tuple residuals."""
    s:        np.ndarray   # state (obs_dim,)
    a:        np.ndarray   # action (act_dim,)
    # Sim side
    s_next_sim: np.ndarray
    r_sim:      float
    d_sim:      float
    # Real side
    s_next_real: np.ndarray
    r_real:       float
    d_real:       float
    # Residuals
    delta_s: np.ndarray   # s'_real - s'_sim
    delta_r: float        # r_real  - r_sim
    delta_d: float        # d_real  - d_sim
    # Optional: time-delay context (populated by extractor if keep_history=True)
    s_prev:  Optional[np.ndarray] = None
    step:    int = 0


class ResidualBuffer:
    """
    Ring buffer of ResidualSamples.

    Stores the last `capacity` samples with optional 2-step history for
    time-delay embedding (Mechanism 1, dev manual §4.3).

    Indexing:
        buf[i] → ResidualSample
        buf.to_arrays() → (SA, delta_s, delta_r, delta_d) numpy arrays
    """

    def __init__(self, capacity: int = 100_000, keep_history: bool = True):
        self.capacity = capacity
        self.keep_history = keep_history
        self._data: List[ResidualSample] = []
        self._ptr = 0

    def __len__(self):
        return len(self._data)

    def append(self, sample: ResidualSample):
        if len(self._data) < self.capacity:
            self._data.append(sample)
        else:
            self._data[self._ptr] = sample
        self._ptr = (self._ptr + 1) % self.capacity

    def to_arrays(self, element: str = "s"):
        """
        Flatten buffer into numpy arrays for SINDy fitting.

        Args:
            element: "s" | "r" | "d" — which tuple element to return as target.

        Returns:
            SA      shape (N, obs_dim + act_dim)  input features
            delta   shape (N, out_dim)            targets
        """
        if len(self._data) == 0:
            raise ValueError("Buffer is empty.")

        SA = np.stack([np.concatenate([s.s, s.a]) for s in self._data])

        if element == "s":
            delta = np.stack([s.delta_s for s in self._data])
        elif element == "r":
            delta = np.array([[s.delta_r] for s in self._data])
        elif element == "d":
            delta = np.array([[s.delta_d] for s in self._data])
        else:
            raise ValueError(f"element must be 's', 'r', or 'd', got {element!r}")

        return SA, delta

    def to_arrays_with_history(self, element: str = "s"):
        """
        Same as to_arrays() but appends s(t-1) to the feature vector.
        Only valid if keep_history=True was set and s_prev is populated.
        Samples without history are skipped.
        """
        valid = [s for s in self._data if s.s_prev is not None]
        if not valid:
            return self.to_arrays(element)

        SA_hist = np.stack([
            np.concatenate([s.s, s.a, s.s_prev])
            for s in valid
        ])

        if element == "s":
            delta = np.stack([s.delta_s for s in valid])
        elif element == "r":
            delta = np.array([[s.delta_r] for s in valid])
        elif element == "d":
            delta = np.array([[s.delta_d] for s in valid])
        else:
            raise ValueError(f"Unknown element: {element!r}")

        return SA_hist, delta

    def get_steps(self) -> np.ndarray:
        """Return array of step indices (used for trajectory-position features)."""
        return np.array([s.step for s in self._data])


class ResidualExtractor:
    """
    Drives a (sim_env, real_env) pair to collect ResidualSamples.

    Workflow:
        extractor = ResidualExtractor(env_pair, buffer)
        for (s, a, s_next_real, r_real, d_real) in offline_dataset:
            extractor.extract(s, a, s_next_real, r_real, d_real)
        # buffer is now populated
    """

    def __init__(self, env_pair, buffer: ResidualBuffer):
        """
        Args:
            env_pair: HPMuJoCoEnvPair (or any object with query_residual(s, a))
            buffer: ResidualBuffer to write into
        """
        self.env_pair = env_pair
        self.buffer = buffer
        self._prev_s = None
        self._step = 0

    def extract(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_next_real: np.ndarray,
        r_real: float,
        d_real: float,
    ):
        """
        Extract one residual sample from the offline real transition (s, a, s', r, d).

        The sim next-state is obtained by querying the sim env from state s.
        The residual is then: Δ = (s_next_real, r_real, d_real) - (s_next_sim, r_sim, d_sim).
        """
        result = self.env_pair.query_residual(s, a)

        sample = ResidualSample(
            s=s.copy(),
            a=a.copy(),
            s_next_sim=result["s_next_sim"],
            r_sim=result["r_sim"],
            d_sim=float(result["d_sim"]),
            s_next_real=s_next_real.copy(),
            r_real=float(r_real),
            d_real=float(d_real),
            delta_s=s_next_real - result["s_next_sim"],
            delta_r=float(r_real) - float(result["r_sim"]),
            delta_d=float(d_real) - float(result["d_sim"]),
            s_prev=self._prev_s.copy() if self._prev_s is not None else None,
            step=self._step,
        )

        self.buffer.append(sample)
        self._prev_s = s.copy()
        self._step += 1

    def reset_history(self):
        """Call at episode boundaries."""
        self._prev_s = None
        self._step = 0

    def extract_dataset(self, offline_dataset):
        """
        Convenience: iterate over a list/dataset of (s, a, s', r, d) tuples.
        Assumes they are ordered by episode (resets history at d=True boundaries).
        """
        for s, a, s_next, r, d in offline_dataset:
            self.extract(s, a, s_next, r, d)
            if d:
                self.reset_history()
