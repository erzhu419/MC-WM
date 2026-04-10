"""
WindHopper: Hopper with sinusoidal side wind in real mode.

Sim-real gap:
  State gap       → sinusoidal lateral force: F_wind = A*sin(2π*t/T) + noise
                    applied to torso vx dim, causing oscillating velocity drift
  Reward gap      → none (forward reward unaffected by sidewind in this simplification)
  Termination gap → wind-induced falls: extra angle check from lateral sway

Expected SINDy discovery:
  Autocorrelation test fires (temporal periodicity in Δvx).
  Round 2: time-delay embedding t/(T) added to basis.
  SINDy finds: Δvx ≈ A*sin(2π*step/T_steps)  (via Fourier basis in expanded library)
"""

import numpy as np
import gymnasium as gym


class WindHopperEnv(gym.Env):
    """
    Hopper wrapper.

    Hopper obs layout (11-dim):
      [0]    z
      [1]    θ (torso angle)
      [2]    θ_thigh
      [3]    θ_leg
      [4]    θ_foot
      [5]    vx
      [6]    vz
      [7]    dθ/dt (torso angular vel)
      [8]    dθ_thigh/dt
      [9]    dθ_leg/dt
      [10]   dθ_foot/dt

    Wind acts on vx [5] and angle [1].
    """

    OBS_DIM = 11
    ACT_DIM = 3

    def __init__(
        self,
        mode: str = "sim",
        wind_amplitude: float = 0.5,   # m/s peak
        wind_period: float = 2.0,       # seconds (period of wind cycle)
        wind_noise_std: float = 0.05,
        seed: int = 0,
    ):
        assert mode in ("sim", "real")
        self.mode = mode
        self.A = wind_amplitude
        self.T = wind_period
        self.noise_std = wind_noise_std
        self._env = gym.make("Hopper-v4")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._t = 0.0
        self._step = 0
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self._t = 0.0
        self._step = 0
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        dt = self._env.unwrapped.dt
        self._t += dt
        self._step += 1

        if self.mode == "real":
            obs, terminated = self._apply_wind(obs, terminated)

        return obs, reward, terminated, truncated, info

    def _apply_wind(self, obs, terminated):
        obs = obs.copy()
        dt = self._env.unwrapped.dt

        # Sinusoidal wind force on vx
        f_wind = self.A * np.sin(2 * np.pi * self._t / self.T)
        f_wind += self._rng.normal(0, self.noise_std)

        obs[5] += f_wind * dt            # vx perturbed
        obs[1] += f_wind * dt * 0.1      # slight torso angle perturbation

        # Wind-induced termination: if angle too large (absolute)
        if abs(obs[1]) > 0.35:           # stricter than sim default (0.2 for Hopper)
            terminated = True

        return obs, terminated

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()
