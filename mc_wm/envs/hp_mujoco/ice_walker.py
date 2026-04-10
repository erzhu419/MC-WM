"""
IceWalker: Walker2d with friction drop at x>5 and softer termination in real mode.

Sim-real gap:
  State gap       → friction scaling 0.8x when x > 5 (lateral position)
                    affects velocity dims [8-17] with piecewise structure
  Reward gap      → none (velocity reward scaling ×0.8 in real when friction drops)
  Termination gap → real termination angle is 60° vs sim's 45° (softer fall threshold)

Expected SINDy discovery:
  Dim vx should show piecewise residual correlated with x position.
  Round 1 (poly2): captures mean drift but misses piecewise structure.
  Round 3 (piecewise mask via normality trigger): 1(x > 5) feature fixes it.
"""

import numpy as np
import gymnasium as gym


class IceWalkerEnv(gym.Env):
    """
    Walker2d wrapper.

    Walker2d obs layout (17-dim):
      [0]    z (height)
      [1]    θ (body angle)
      [2-8]  joint angles (6 joints)
      [9]    vx
      [10]   vz
      [11-17] joint angular velocities

    Friction drop affects velocity dims when lateral x-position > threshold.
    x-position is NOT in obs; we track it via vx integration.
    """

    VEL_DIMS = list(range(9, 17))
    OBS_DIM = 17
    ACT_DIM = 6

    FRICTION_DROP = 0.8          # multiplier on velocity update when on ice
    ICE_THRESHOLD = 5.0          # lateral position beyond which friction drops
    SIM_TERM_ANGLE = 1.0472      # ~60° in radians (Walker2d default: 1.0)
    REAL_TERM_ANGLE = 1.3        # ~75°, softer fall threshold in real

    def __init__(self, mode: str = "sim", seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._env = gym.make("Walker2d-v4")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._x_pos = 0.0        # tracked x position for piecewise trigger
        self._step = 0

    def reset(self, **kwargs):
        self._x_pos = 0.0
        self._step = 0
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1

        if self.mode == "real":
            obs, reward, terminated = self._apply_ice(obs, reward, terminated)
        else:
            # track x for reference
            self._x_pos += obs[9] * self._env.unwrapped.dt

        return obs, reward, terminated, truncated, info

    def _apply_ice(self, obs, reward, terminated):
        obs = obs.copy()
        dt = self._env.unwrapped.dt
        self._x_pos += obs[9] * dt

        on_ice = self._x_pos > self.ICE_THRESHOLD

        if on_ice:
            for i in self.VEL_DIMS:
                obs[i] *= self.FRICTION_DROP
            reward *= self.FRICTION_DROP

        # Softer termination: real falls at larger body angle
        body_angle = abs(obs[1])
        if body_angle > self.REAL_TERM_ANGLE:
            terminated = True

        return obs, reward, terminated

    def get_x_pos(self) -> float:
        return self._x_pos

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()
