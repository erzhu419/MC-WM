"""
GravityCheetah: HalfCheetah with different gravity between sim and real.

H2O benchmark: sim uses 2x gravity, real uses 1x (default).
This creates a LARGE dynamics gap:
  - 2x gravity → agent falls faster, needs stronger leg forces
  - Policy learned in 2x gravity is overly aggressive in 1x
  - Reward directly affected (velocity-based reward changes)

Unlike CarpetAnt's post-hoc obs modification, this changes the actual
MuJoCo physics engine parameters → reward gap is automatic and large.
"""

import numpy as np
import gymnasium as gym


class GravityCheetahEnv(gym.Env):
    """
    HalfCheetah with configurable gravity.

    mode="sim": gravity_factor applied (default 2.0 = double gravity)
    mode="real": standard gravity (1x)

    This is the H2O benchmark configuration that creates ~30-50% reward gap.
    """

    OBS_DIM = 17
    ACT_DIM = 6
    GRAVITY_FACTOR = 2.0  # sim gravity multiplier (H2O default)

    def __init__(self, mode: str = "sim", gravity_factor: float = None, seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._gf = gravity_factor or self.GRAVITY_FACTOR

        self._env = gym.make("HalfCheetah-v4")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._modified = False

    def _modify_gravity(self):
        """Modify MuJoCo model gravity in-place."""
        if self.mode == "sim" and not self._modified:
            model = self._env.unwrapped.model
            # Default gravity is [0, 0, -9.81]
            model.opt.gravity[2] = -9.81 * self._gf
            self._modified = True

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._modify_gravity()
        return obs, info

    def step(self, action):
        return self._env.step(action)


class GravityCheetahCeilingEnv(GravityCheetahEnv):
    """
    GravityCheetah + physical ceiling constraint z > Z_MAX → termination.

    This adds a SPATIAL constraint orthogonal to the dynamics gap:
      - Dynamics shift: gravity 2x (sim) vs 1x (real) — unchanged
      - Spatial constraint: z > 0.5 terminates episode with zero reward
        (same threshold in both sim and real — a physical barrier)

    Why this exposes ICRL value:
      - In sim (2x gravity), cheetah rarely jumps high → constraint rarely binds
      - In real (1x gravity), cheetah jumps higher → constraint binds
      - Expert (real) data teaches ICRL "don't let z exceed 0.5"
      - This signal is ORTHOGONAL to QΔ (which learns dynamics accuracy)

    Obs layout: HalfCheetah-v4 with exclude_current_positions=True
      obs[0] = z_position (root height) — used for ceiling check
    """

    Z_CEILING = 0.2  # height above which the episode terminates (tight ceiling)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Ceiling constraint: if root z exceeds threshold, terminate with zero reward
        if obs[0] > self.Z_CEILING:
            terminated = True
            reward = 0.0
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info


class GravityCheetahSoftCeilingEnv(GravityCheetahEnv):
    """
    GravityCheetah + SOFT ceiling: continuous penalty instead of terminal.

    Violation of z > 0.2 is penalized as r -= PENALTY_SCALE * (z - 0.2)²
    but episode CONTINUES. This is "Soft Type 2" — the Bellman signal is
    weaker than terminal-r=0, so implicit constraint learning should
    struggle and explicit ICRL should shine.

    Differences from GravityCheetahCeilingEnv:
      - No early termination on z > 0.2
      - Continuous quadratic penalty (strong but not absolute)
      - info['ceiling_hit'] set whenever z > 0.2 (for violation tracking)
    """

    Z_CEILING = 0.2
    PENALTY_SCALE = 10.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        z = float(obs[0])
        excess = max(0.0, z - self.Z_CEILING)
        if excess > 0:
            reward -= self.PENALTY_SCALE * excess * excess
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()
