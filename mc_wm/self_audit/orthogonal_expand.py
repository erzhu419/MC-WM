"""
Orthogonal Feature Discovery: find features in the orthogonal complement
of the current SINDy basis that capture remaining structure in the residual.

Key insight: auto-expand adds x³, x·|x| etc., but these are nearly collinear
with existing {1, x, x²} features (correlation > 0.95). They add overfitting
risk without new information.

Correct approach: find the ORTHOGONAL COMPLEMENT of the current feature space,
then discover structure in the residual projected onto that complement.

Method:
1. Compute Θ = current SINDy features (poly2)
2. Compute remainder r = y - Θ·coefs (already orthogonal to Θ by least squares)
3. Generate candidate features from physics/domain knowledge
4. Gram-Schmidt orthogonalize candidates against Θ
5. Keep only candidates with significant correlation to remainder
6. These are the truly new features — orthogonal to existing basis

This replaces blind auto-expand with principled feature selection.
"""

import numpy as np
from typing import List, Tuple, Optional


class OrthogonalExpander:
    """
    Discovers features orthogonal to the current SINDy basis
    that correlate with the residual.
    """

    def __init__(self, obs_dim, act_dim, min_correlation=0.05,
                 max_delta_beta_inf: float | None = None,
                 prev_basis_beta: "np.ndarray | None" = None):
        """
        Args:
            obs_dim/act_dim: env shape
            min_correlation: existing correlation gate.
            max_delta_beta_inf: RAHD Stage B — if set, reject candidates
                whose addition would push the L∞ change in fitted SINDy
                coefficients (compared to the previous round) beyond this
                threshold.  ``None`` disables the gate (legacy behavior).
            prev_basis_beta: (F_prev, D) coefficients from the previous
                fit, required only when ``max_delta_beta_inf`` is set so
                this expander can estimate ΔL_eff.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.min_correlation = min_correlation
        self.max_delta_beta_inf = max_delta_beta_inf
        self.prev_basis_beta = prev_basis_beta

    def expand(self, SA, Theta, remainder, feature_names=None, log_fn=None):
        """
        Find orthogonal features that explain remaining structure.

        Args:
            SA: (N, obs+act) raw state-action data
            Theta: (N, F) current SINDy feature matrix
            remainder: (N, obs_dim) residual after SINDy fit
            feature_names: existing feature names (for logging)

        Returns:
            extra_columns: (N, K) new orthogonal features
            extra_names: list of K names
            diagnostics: dict with analysis details
        """
        _log = log_fn or (lambda msg: print(msg, flush=True))
        N, F = Theta.shape

        # ── Step 1: Generate candidate features
        candidates, cand_names = self._generate_candidates(SA, N)
        _log(f"  OrthExpand: {len(cand_names)} candidates generated")

        # ── Step 2: Orthogonalize each candidate against Θ (Gram-Schmidt)
        # Θ is already column-centered by SINDy. We project each candidate
        # onto the column space of Θ and subtract.
        # Use QR for numerical stability: Θ = Q·R, projection = Q·Q'·c
        Q, R = np.linalg.qr(Theta, mode='reduced')  # Q: (N, F), R: (F, F)

        ortho_candidates = []
        ortho_names = []
        ortho_corrs = []

        for c, name in zip(candidates, cand_names):
            # Project c onto span(Θ) and subtract → get orthogonal component
            proj = Q @ (Q.T @ c)
            c_orth = c - proj

            # Check if candidate has significant orthogonal component
            norm_orth = np.linalg.norm(c_orth)
            norm_orig = np.linalg.norm(c)
            if norm_orig < 1e-8 or norm_orth / norm_orig < 0.1:
                continue  # >90% in span(Θ) → redundant, skip

            c_orth_normed = c_orth / norm_orth

            # Correlation between orthogonal component and remainder
            corrs = np.array([np.abs(np.corrcoef(c_orth_normed, remainder[:, d])[0, 1])
                              for d in range(remainder.shape[1])
                              if not np.isnan(np.corrcoef(c_orth_normed, remainder[:, d])[0, 1])])
            if len(corrs) == 0:
                continue
            max_corr = float(corrs.max())

            if max_corr > self.min_correlation:
                # RAHD Stage B (stability gate): estimate the magnitude of
                # the new SINDy coefficient introduced by accepting this
                # candidate.  Closed-form OLS for the orthogonal component:
                #   β_c ≈ ⟨c_orth, r_d⟩ / ‖c_orth‖²    for each output dim d
                # We use the L∞ over output dims as the ΔL_eff proxy.  A
                # large β_c means the feature is poised to dominate, which
                # historically destabilises NAU (commit 4f53dcd).
                if self.max_delta_beta_inf is not None:
                    norm_sq = float(np.dot(c_orth, c_orth))
                    if norm_sq > 1e-12:
                        beta_c = (c_orth @ remainder) / norm_sq  # (D,)
                        delta_beta_inf = float(np.max(np.abs(beta_c)))
                        if delta_beta_inf > self.max_delta_beta_inf:
                            # Reject — too disruptive to coefficient layer.
                            continue
                # Store RAW candidate (not orthogonalized) for SINDy fitting
                # Orthogonalization was only for selection, not for use
                ortho_candidates.append(c)
                ortho_names.append(name)
                ortho_corrs.append(max_corr)

        # ── Step 3: Rank by correlation, take top K
        if not ortho_candidates:
            _log(f"  OrthExpand: no orthogonal features found above threshold {self.min_correlation}")
            return np.zeros((N, 0)), [], {"n_candidates": len(cand_names), "n_accepted": 0}

        # Sort by correlation
        order = np.argsort(ortho_corrs)[::-1]
        max_features = 10  # cap to prevent feature explosion
        selected = order[:max_features]

        extra_cols = np.column_stack([ortho_candidates[i] for i in selected])
        extra_names = [ortho_names[i] for i in selected]
        extra_corrs = [ortho_corrs[i] for i in selected]

        _log(f"  OrthExpand: {len(extra_names)}/{len(cand_names)} features accepted "
             f"(orthogonal, corr > {self.min_correlation})")
        for name, corr in zip(extra_names[:5], extra_corrs[:5]):
            _log(f"    {name}: max_corr={corr:.4f}")

        diagnostics = {
            "n_candidates": len(cand_names),
            "n_accepted": len(extra_names),
            "correlations": dict(zip(extra_names, extra_corrs)),
        }

        return extra_cols, extra_names, diagnostics

    def _generate_candidates(self, SA, N):
        """
        Generate a diverse set of candidate features.
        These will be orthogonalized against current basis before use.

        Categories:
        1. Time-like features (cumulative, sequential)
        2. Trigonometric (capture periodicity)
        3. Cross-dimensional interactions (beyond poly2)
        4. Threshold/indicator features
        5. Velocity-specific features (physics-informed)
        """
        candidates = []
        names = []
        obs = SA[:, :self.obs_dim]
        act = SA[:, self.obs_dim:]

        # ── Category 1: Time proxies
        # SINDy poly2 has no time awareness. If data is sequential,
        # cumulative features capture time evolution.
        t_norm = np.arange(N, dtype=np.float64) / max(N - 1, 1)
        candidates.append(t_norm); names.append("t_norm")
        candidates.append(t_norm ** 2); names.append("t_sq")
        # t × state interactions: strictly orthogonal to {1, x, x²} if t⊥x
        for i in range(min(self.obs_dim, 6)):  # top 6 dims
            candidates.append(t_norm * obs[:, i]); names.append(f"t_x{i}")

        # ── Category 2: Trigonometric features
        # Low-freq trig ≈ poly, but medium-freq captures periodicity
        for i in range(min(self.obs_dim, 4)):
            for omega in [1.0, 3.0, 5.0]:
                candidates.append(np.sin(omega * obs[:, i])); names.append(f"sin_{omega:.0f}x{i}")
                candidates.append(np.cos(omega * obs[:, i])); names.append(f"cos_{omega:.0f}x{i}")

        # ── Category 3: Higher-order cross terms (beyond poly2)
        # poly2 has x_i·x_j. These add x_i·x_j·x_k (3-way interactions)
        for i in range(min(self.obs_dim, 3)):
            for j in range(i + 1, min(self.obs_dim, 5)):
                for k in range(j + 1, min(self.obs_dim, 7)):
                    candidates.append(obs[:, i] * obs[:, j] * obs[:, k])
                    names.append(f"x{i}_x{j}_x{k}")

        # ── Category 4: State-action higher interactions
        for i in range(min(self.obs_dim, 4)):
            for j in range(min(self.act_dim, 3)):
                candidates.append(obs[:, i] ** 2 * act[:, j])
                names.append(f"x{i}sq_a{j}")

        # ── Category 5: Physics-informed (velocity-specific)
        # Velocity dims for HalfCheetah: obs[8:17]
        vel_start = self.obs_dim // 2
        for i in range(vel_start, min(self.obs_dim, vel_start + 4)):
            v = obs[:, i]
            candidates.append(np.sign(v) * v ** 2); names.append(f"v{i}_signed_sq")
            candidates.append(np.abs(v) ** 0.5 * np.sign(v)); names.append(f"v{i}_sqrt")

        # ── Category 6: Cumulative features
        # Running mean/variance as proxy for trajectory history
        if N > 10:
            for i in range(min(self.obs_dim, 3)):
                cumsum = np.cumsum(obs[:, i]) / (np.arange(N) + 1)
                candidates.append(cumsum); names.append(f"cumavg_x{i}")

        return candidates, names
