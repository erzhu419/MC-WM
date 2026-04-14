"""
Constraint System: monotonically growing set of physical/semantic constraints.

v4 Architecture Component #5:
  C₀ ⊆ C₁ ⊆ C₂ ⊆ ...
  New constraints are added, old ones never removed.

Two sources:
  Role #1 (initial): Physics constraints from env analysis (hardcoded)
  Role #3 (runtime): Audit suspicious corrections, add new rules

Any correction that violates ANY constraint is rejected → fallback to raw sim.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Constraint:
    name: str
    description: str
    check_fn: object  # callable(s, a, s_corrected, r_corrected) → bool (True=violation)
    source: str = "role1"  # "role1" or "role3"
    added_at_step: int = 0


class ConstraintSystem:
    """
    Monotonically growing constraint set.

    Usage:
        cs = ConstraintSystem(env_type="gravity_cheetah")
        # Check if a corrected transition is valid
        ok, violations = cs.check(s, a, s_next_corrected, r_corrected)
        if not ok:
            use raw sim instead

        # Runtime: audit suspicious corrections and add new constraints
        cs.audit_and_expand(suspicious_transitions, step)
    """

    def __init__(self, env_type="gravity_cheetah", log_fn=None):
        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self.constraints: List[Constraint] = []
        self._n_checked = 0
        self._n_rejected = 0

        # Role #1: Generate initial constraints
        if env_type == "gravity_cheetah":
            self._init_gravity_cheetah()
        self._log(f"  ConstraintSystem: {len(self.constraints)} initial constraints (Role #1)")

    def _init_gravity_cheetah(self):
        """
        Role #1: Physics constraints for HalfCheetah.

        HalfCheetah obs (17-dim):
          [0]=rootz (height), [1]=rooty (body angle)
          [2:8]=6 joint angles, [8]=vx, [9]=vz, [10]=va
          [11:17]=6 joint angular velocities

        Gravity is 1x in real (9.81), 2x in sim (19.62).
        Correction should make sim behave like 1x gravity.
        """
        # ── Physical possibility constraints
        self.constraints.append(Constraint(
            name="height_bounds",
            description="Cheetah height must be in [-1, 2] (can't go underground or fly)",
            check_fn=lambda s, a, sc, r: sc[0] < -1.0 or sc[0] > 2.0,
        ))
        self.constraints.append(Constraint(
            name="body_angle",
            description="Body angle |rooty| < 3.0 rad (can't do >1 full rotation per step)",
            check_fn=lambda s, a, sc, r: abs(sc[1]) > 3.0,
        ))
        self.constraints.append(Constraint(
            name="forward_velocity",
            description="|vx| < 15 m/s (physical speed limit for this robot)",
            check_fn=lambda s, a, sc, r: abs(sc[8]) > 15.0,
        ))
        self.constraints.append(Constraint(
            name="vertical_velocity",
            description="|vz| < 15 m/s (can't launch vertically)",
            check_fn=lambda s, a, sc, r: abs(sc[9]) > 15.0,
        ))
        self.constraints.append(Constraint(
            name="angular_velocity",
            description="|angular_vel| < 30 rad/s (physical rotation limit)",
            check_fn=lambda s, a, sc, r: abs(sc[10]) > 30.0,
        ))

        # ── Joint constraints
        for j in range(6):
            dim = 2 + j
            vel_dim = 11 + j
            self.constraints.append(Constraint(
                name=f"joint{j}_angle",
                description=f"Joint {j} angle in [-1.5, 1.5] rad",
                check_fn=lambda s, a, sc, r, d=dim: abs(sc[d]) > 1.5,
            ))
            self.constraints.append(Constraint(
                name=f"joint{j}_velocity",
                description=f"Joint {j} angular velocity |ω| < 40 rad/s",
                check_fn=lambda s, a, sc, r, d=vel_dim: abs(sc[d]) > 40.0,
            ))

        # ── State transition magnitude (not correction — sc is next_state, not corrected_current)
        # Real env stats: joint vels can be ±25, so per-step change can be large
        # Only flag truly extreme transitions
        self.constraints.append(Constraint(
            name="extreme_transition",
            description="Per-dim state change |s'−s| < 30 (filters catastrophic predictions)",
            check_fn=lambda s, a, sc, r: np.max(np.abs(sc - s)) > 30.0
                if len(s) == len(sc) else False,
        ))

        # ── Reward constraints
        self.constraints.append(Constraint(
            name="reward_bounds",
            description="Reward in [-10, 10] (physical reward range)",
            check_fn=lambda s, a, sc, r: r < -10.0 or r > 10.0,
        ))

        # ── Semantic: gravity correction direction
        self.constraints.append(Constraint(
            name="gravity_direction",
            description="Gravity correction should generally reduce acceleration "
                        "(real=1x < sim=2x), so corrected vz should be less negative",
            check_fn=lambda s, a, sc, r: False,  # soft constraint, logged but not rejected
        ))

    def check(self, s, a, s_corrected, r_corrected):
        """
        Check if a corrected transition violates any constraint.

        Returns: (ok: bool, violations: list of constraint names)
        """
        violations = []
        for c in self.constraints:
            try:
                if c.check_fn(s, a, s_corrected, r_corrected):
                    violations.append(c.name)
            except Exception:
                pass  # constraint evaluation error → skip

        self._n_checked += 1
        if violations:
            self._n_rejected += 1

        return len(violations) == 0, violations

    def check_batch(self, states, actions, s_corrected, r_corrected):
        """
        Batch check. Returns (ok_mask, violation_counts).

        ok_mask: (N,) bool — True if transition passes all constraints
        violation_counts: (N,) int — number of violated constraints per transition
        """
        N = len(states)
        ok_mask = np.ones(N, dtype=bool)
        violation_counts = np.zeros(N, dtype=int)

        for i in range(N):
            ok, violations = self.check(
                states[i], actions[i], s_corrected[i], r_corrected[i])
            ok_mask[i] = ok
            violation_counts[i] = len(violations)

        return ok_mask, violation_counts

    def add_constraint(self, name, description, check_fn, step=0):
        """Role #3: Add a new constraint (monotonic growth)."""
        self.constraints.append(Constraint(
            name=name, description=description,
            check_fn=check_fn, source="role3",
            added_at_step=step,
        ))
        self._log(f"  Constraint added (Role #3): {name} — {description}")

    def audit_suspicious(self, states, actions, s_corrected, r_corrected,
                          corrections_magnitude, step=0, threshold_percentile=95):
        """
        Role #3: Find suspicious corrections and potentially add new constraints.

        Suspicious = passes all current constraints BUT correction is unusually large.
        """
        N = len(states)
        ok_mask, _ = self.check_batch(states, actions, s_corrected, r_corrected)

        # Among passing transitions, find those with large corrections
        mag = corrections_magnitude  # (N,) — per-transition correction magnitude
        threshold = np.percentile(mag[ok_mask], threshold_percentile)
        suspicious = ok_mask & (mag > threshold)
        n_suspicious = int(suspicious.sum())

        if n_suspicious == 0:
            return

        self._log(f"  Constraint audit: {n_suspicious} suspicious corrections "
                  f"(pass constraints but |Δ| > p{threshold_percentile}={threshold:.3f})")

        # Analyze suspicious corrections for patterns
        sus_corr = s_corrected[suspicious] - states[suspicious]
        sus_states = states[suspicious]

        # Auto-detect: any dimension consistently at extreme values?
        for dim in range(min(sus_states.shape[1], 17)):
            vals = sus_states[:, dim]
            if np.std(vals) < 0.01:  # all suspicious states cluster in one value
                continue
            # Check if corrections are unreasonably large for this dim
            corr_dim = np.abs(sus_corr[:, dim])
            if corr_dim.mean() > 3.0:  # mean correction > 3 in this dim
                max_val = float(np.percentile(np.abs(sus_states[:, dim]), 99))
                self.add_constraint(
                    name=f"auto_dim{dim}_extreme_corr",
                    description=f"Dim {dim}: correction > 3.0 when |state| > {max_val:.1f}",
                    check_fn=lambda s, a, sc, r, d=dim, mv=max_val:
                        abs(s[d]) > mv and abs(sc[d] - s[d]) > 3.0,
                    step=step,
                )

    def get_stats(self):
        return {
            "n_constraints": len(self.constraints),
            "n_checked": self._n_checked,
            "n_rejected": self._n_rejected,
            "reject_rate": self._n_rejected / max(self._n_checked, 1),
            "role1": sum(1 for c in self.constraints if c.source == "role1"),
            "role3": sum(1 for c in self.constraints if c.source == "role3"),
        }
