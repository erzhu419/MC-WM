"""
AeroCheetah: HalfCheetah with quadratic aerodynamic drag in real mode.

Sim-real gap:
  State gap  → velocity dims get extra deceleration: F_drag = -k_drag * v * |v|
  Reward gap → none (reward is position-based, not velocity-penalized in sim)
  Term. gap  → none

Expected SINDy discovery (Exp 2 in dev manual §9.3):
  SINDy should recover Δvx ≈ -k * vx * |vx|  within 2 rounds
  basis: poly2 finds vx² but not sign — round 2 adds vx*|vx| via heteroscedasticity trigger
"""

import numpy as np
import gymnasium as gym


class AeroCheetahEnv(gym.Env):
    """
    HalfCheetah wrapper.

    HalfCheetah obs layout (17-dim):
      [0]      z (height)
      [1]      θ (body pitch)
      [2-8]    7 joint angles
      [9]      vx (forward velocity)
      [10]     vz (vertical velocity)
      [11-17]  7 joint angular velocities

    Drag acts on velocity dims [9-17] (8 velocity dimensions).
    """

    VEL_DIMS = list(range(9, 17))     # velocity state dims affected by drag
    OBS_DIM = 17
    ACT_DIM = 6

    def __init__(self, mode: str = "sim", k_drag: float = 0.05, seed: int = 0):
        """
        Args:
            mode: "sim" (standard) or "real" (with quadratic drag)
            k_drag: drag coefficient F = -k * v * |v|
        """
        assert mode in ("sim", "real")
        self.mode = mode
        self.k_drag = k_drag
        self._env = gym.make("HalfCheetah-v4")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._step = 0

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        self._step = 0
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1

        if self.mode == "real":
            obs, reward = self._apply_drag(obs, reward)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _apply_drag(self, obs, reward):
        obs = obs.copy()
        dt = self._env.unwrapped.dt  # integration step (0.05 s for HalfCheetah)
        for i in self.VEL_DIMS:
            v = obs[i]
            dv = -self.k_drag * v * abs(v) * dt   # Euler drag correction
            obs[i] = v + dv
        # reward is unaffected in this env
        return obs, reward

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()
