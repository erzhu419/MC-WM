"""
Four Automated Basis Expansion Mechanisms (dev manual §4.3).

Each mechanism is triggered by a specific diagnosis test firing.

Instead of building complex pysindy CustomLibrary objects (which have
compatibility issues with AxesArray/1D inputs), we pre-compute extra feature
columns as numpy arrays and append them to SA. The SINDy library stays as
a simple PolynomialLibrary on the expanded feature space.

Mechanism 1: Time-Delay Embedding    → trigger: autocorrelation
Mechanism 2: Algebraic Feature Cross → trigger: heteroscedasticity
Mechanism 3: Piecewise Logical Mask  → trigger: non-normality (heavy tails)
Mechanism 4: Trajectory Position     → trigger: non-stationarity
"""

import numpy as np
import pysindy as ps
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans

from mc_wm.self_audit.diagnosis import DiagnosisResult


def make_expanded_library(degree: int = 1, include_bias: bool = True):
    """
    After expansion, use degree=1 on (original + extra features).
    The nonlinear terms (x², x³, x|x|, masks) are already in the columns.
    degree=1 avoids combinatorial explosion of poly2 on 30+ features.
    """
    return ps.PolynomialLibrary(degree=degree, include_bias=include_bias)


class AutoExpander:
    """
    Given a list of DiagnosisResults, computes extra feature columns
    to append to SA and returns a new library for the expanded space.

    Usage:
        expander = AutoExpander(obs_dim=17, act_dim=6)
        new_lib, metadata = expander.expand(
            results=diagnosis_results,
            current_library=base_library,
            SA=SA_array,
            remainder=remainder_array,
            steps=step_array,
        )
        # metadata["extra_columns"] = np.ndarray (N, n_new_feats)
        # metadata["extra_names"]   = list of str
        # Caller: SA_expanded = np.hstack([SA, metadata["extra_columns"]])
        #         then fit SINDy with new_lib on SA_expanded
    """

    def __init__(self, obs_dim: int, act_dim: int, T_max: int = 1000):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.T_max = T_max

    def expand(
        self,
        results: List[DiagnosisResult],
        current_library,
        SA: np.ndarray,
        remainder: np.ndarray,
        steps: Optional[np.ndarray] = None,
        SA_hist: Optional[np.ndarray] = None,
    ) -> Tuple[object, dict]:
        """
        Apply expansion mechanisms based on which diagnoses fired.

        Returns:
            new_library: pysindy library for expanded feature space
            metadata: dict with:
                mechanisms_fired: list of str
                extra_columns: np.ndarray (N, n_new) — columns to append to SA
                extra_names: list of str — names for extra columns
                use_history: bool — whether to use SA_hist for time-delay
        """
        metadata = {
            "mechanisms_fired": [],
            "extra_columns": np.zeros((len(SA), 0)),
            "extra_names": [],
        }

        extra_cols = []
        extra_names = []

        # Aggregate: which tests fired across any dimension?
        any_autocorr    = any(r.autocorrelation for r in results)
        any_hetero      = any(r.heteroscedastic for r in results)
        culprits_hetero = [r.heteroscedastic_culprit for r in results
                          if r.heteroscedastic and r.heteroscedastic_culprit is not None]
        any_nonnormal   = any(r.non_normal for r in results)
        culprits_normal = [r.dim for r in results if r.non_normal]
        any_nonstat     = any(r.non_stationary for r in results)

        # ── Mechanism 1: Time-delay embedding
        if any_autocorr:
            metadata["mechanisms_fired"].append("time_delay")
            metadata["use_history"] = True
            # Delay features are data-level: caller appends s_prev columns

        # ── Mechanism 2: Algebraic crossings for hetero culprits
        if any_hetero and culprits_hetero:
            for j in sorted(set(culprits_hetero)):
                col = SA[:, j]
                extra_cols.append(col ** 2)
                extra_names.append(f"x{j}_sq")
                extra_cols.append(col ** 3)
                extra_names.append(f"x{j}_cube")
                extra_cols.append(col * np.abs(col))
                extra_names.append(f"x{j}_signmag")
                # Cross with each action dim
                for k in range(self.act_dim):
                    ak = self.obs_dim + k
                    extra_cols.append(col * SA[:, ak])
                    extra_names.append(f"x{j}_x_a{k}")
            metadata["mechanisms_fired"].append("algebraic_cross")
            metadata["hetero_culprits"] = list(set(culprits_hetero))

        # ── Mechanism 3: Piecewise masks
        if any_nonnormal and culprits_normal:
            for dim in culprits_normal[:3]:   # limit to 3 most offending dims
                feat_idx = min(dim, SA.shape[1] - 1)
                # K-means threshold on |remainder|
                r_col = np.abs(remainder[:, dim]) if dim < remainder.shape[1] else np.abs(remainder[:, 0])
                r_col = r_col.reshape(-1, 1)
                if len(r_col) >= 10:
                    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(r_col)
                    threshold = float(km.cluster_centers_.mean())
                else:
                    threshold = float(np.median(r_col))
                indicator = (SA[:, feat_idx] < threshold).astype(np.float64)
                # Only add if non-degenerate (between 5% and 95% are 1)
                if 0.05 < indicator.mean() < 0.95:
                    extra_cols.append(indicator)
                    extra_names.append(f"mask_x{feat_idx}_lt{threshold:.2f}")
            if any(n.startswith("mask_") for n in extra_names):
                metadata["mechanisms_fired"].append("piecewise_mask")

        # ── Mechanism 4: Trajectory position
        if any_nonstat and steps is not None:
            t_norm = steps.astype(np.float64) / max(self.T_max, 1)
            extra_cols.append(t_norm)
            extra_names.append("t_norm")
            metadata["mechanisms_fired"].append("trajectory_position")

        # ── Assemble extra columns
        if extra_cols:
            metadata["extra_columns"] = np.column_stack(extra_cols)
        else:
            metadata["extra_columns"] = np.zeros((len(SA), 0))
        metadata["extra_names"] = extra_names

        # Library: degree=1 on expanded features (nonlinear terms already pre-computed)
        new_library = make_expanded_library(degree=1)

        return new_library, metadata
