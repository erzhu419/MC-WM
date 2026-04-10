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

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()
