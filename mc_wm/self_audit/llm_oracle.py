"""
LLM Oracle: physics-informed feature suggestions for residual model.

Instead of blind polynomial expansion (x², x³, x·|x|), the LLM oracle
suggests features based on physical understanding of the sim-real gap.

Current implementation: hardcoded physics knowledge per environment.
Future: actual LLM API call with environment description + SINDy terms + diagnosis.
"""

import numpy as np
from typing import List, Optional


class LLMOracle:
    """
    Suggests physics-informed features based on environment knowledge.

    Replaces blind auto-expand with targeted features from physical reasoning.
    """

    def __init__(self, env_type="gravity_cheetah", log_fn=None):
        self.env_type = env_type
        self._log = log_fn or (lambda msg: print(msg, flush=True))

    def suggest_features(self, SA, obs_dim=17, act_dim=6):
        """Returns (extra_columns, extra_names, reasoning)."""
        if self.env_type == "gravity_cheetah":
            return self._gravity_cheetah_features(SA, obs_dim, act_dim)
        return np.zeros((len(SA), 0)), [], "No env-specific features"

    def _gravity_cheetah_features(self, SA, obs_dim, act_dim):
        """
        HalfCheetah obs (17-dim, rootx excluded):
          [0]=rootz, [1]=rooty, [2:8]=6 joint angles
          [8]=vx, [9]=vz, [10]=va, [11:17]=6 joint angular vels

        Gravity Δg causes:
          Δv_z = Δg·dt per step (CONSTANT)
          Δz = v_z·dt + ½Δg·dt²
          Torque = m·Δg·L·sin(θ)

        NOT needed: x³ (curve fitting, not physics)
        """
        N = len(SA)
        z = SA[:, 0]       # height
        angle = SA[:, 1]   # body angle
        vx = SA[:, 8]      # forward vel
        vz = SA[:, 9]      # vertical vel (primary gravity dim)
        va = SA[:, 10]     # angular vel

        features = [
            (np.ones(N),        "grav_bias",        "Δv_z = Δg·dt is constant per step"),
            (vz,                "vz",               "vertical velocity: primary gravity variable"),
            (z * vz,            "z_vz",             "height × v_z: kinematic coupling"),
            (np.cos(angle),     "cos_theta",        "gravity torque ∝ cos(θ)"),
            (np.sin(angle),     "sin_theta",        "lateral gravity ∝ sin(θ)"),
            (vz ** 2,           "vz_sq",            "vertical kinetic energy ½mv_z²"),
            (angle * vz,        "theta_vz",         "angle-velocity coupling"),
            (z,                 "z",                "gravitational PE ∝ z"),
            (vx * np.sin(angle),"vx_sin_theta",     "forward vel projected by gravity angle"),
            (va * z,            "va_z",             "angular vel × height: torque coupling"),
        ]

        cols = [f[0] for f in features]
        names = [f[1] for f in features]
        reasons = [f[2] for f in features]

        self._log(f"  LLM Oracle: {len(names)} physics-informed features for {self.env_type}")
        for n, r in zip(names, reasons):
            self._log(f"    {n}: {r}")

        return np.column_stack(cols), names, reasons
