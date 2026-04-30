"""
Single-factor sweep envs for the gap-taxonomy study (E5).

All envs share the same HalfCheetah-v4 morphology so the only thing
that varies is one of:
    * gravity factor (physics-dynamics gap, on the works-side)
    * observation noise (observation gap, on the fails-side per Theorem 2)
    * actuator scaling (actuator gap, on the fails-side)

Each env exposes the same interface as GravityCheetahSoftCeilingEnv so
step2_mbrl_residual.py can drop them in via ``--env <name>``.
"""

import numpy as np
import gymnasium as gym

from .gravity_cheetah import GravityCheetahEnv, GravityCheetahSoftCeilingEnv


# ─────────────────────────────────────────────────────────────────────────
# Sweep #1 — physics-dynamics gap: gravity factor
# ─────────────────────────────────────────────────────────────────────────
class GravitySweepEnv(GravityCheetahSoftCeilingEnv):
    """Soft-ceiling HalfCheetah with configurable sim gravity factor.

    Subclasses set ``GRAVITY_FACTOR`` to one of {1.5, 2.0, 3.0, 4.0}.
    """
    GRAVITY_FACTOR = 2.0  # default; subclasses override


class GravitySweep15Env(GravitySweepEnv):
    GRAVITY_FACTOR = 1.5


class GravitySweep20Env(GravitySweepEnv):
    GRAVITY_FACTOR = 2.0


class GravitySweep30Env(GravitySweepEnv):
    GRAVITY_FACTOR = 3.0


class GravitySweep40Env(GravitySweepEnv):
    GRAVITY_FACTOR = 4.0


# ─────────────────────────────────────────────────────────────────────────
# Sweep #2 — observation gap: Gaussian noise added to obs[8:17] (velocities)
# ─────────────────────────────────────────────────────────────────────────
class ObsNoiseCheetahEnv(GravityCheetahSoftCeilingEnv):
    """HalfCheetah with i.i.d.\\ Gaussian observation noise on the velocity
    dimensions.  Sim adds noise; real does not.  This is the ``observation
    gap'' case that should fail according to Theorem~\\ref{thm:universal-false}
    because the residual model has no access to the noise generator.
    """

    NOISE_STD = 0.5  # default; subclasses override

    def __init__(self, mode="sim", **kw):
        super().__init__(mode=mode, **kw)
        # Same morphology and gravity as the base soft-ceiling env so the
        # only sim-vs-real difference is observation noise on the sim side.

    def _add_noise(self, obs):
        if self.mode == "sim":
            obs = obs.copy()
            obs[8:17] = obs[8:17] + self.NOISE_STD * np.random.randn(9)
        return obs

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        return self._add_noise(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._add_noise(obs), r, term, trunc, info


class ObsNoise05Env(ObsNoiseCheetahEnv):
    NOISE_STD = 0.5


class ObsNoise10Env(ObsNoiseCheetahEnv):
    NOISE_STD = 1.0


class ObsNoise20Env(ObsNoiseCheetahEnv):
    NOISE_STD = 2.0


class ObsNoise50Env(ObsNoiseCheetahEnv):
    NOISE_STD = 5.0


# ─────────────────────────────────────────────────────────────────────────
# Sweep #3 — actuator gap: action multiplied before being applied
# ─────────────────────────────────────────────────────────────────────────
class ActuatorScalingCheetahEnv(GravityCheetahSoftCeilingEnv):
    """HalfCheetah with sim-side actuator scaling.  Sim multiplies each
    incoming action by ``ACT_SCALE`` before applying; real uses the
    actions as-is.  This is the ``actuator gap'' case where the policy's
    notion of (a → s') is systematically off, but the underlying physics
    is unchanged.
    """

    ACT_SCALE = 1.0

    def __init__(self, mode="sim", **kw):
        super().__init__(mode=mode, **kw)

    def step(self, action):
        if self.mode == "sim":
            action = np.asarray(action, dtype=np.float32) * self.ACT_SCALE
            action = np.clip(action, -1.0, 1.0)
        return super().step(action)


class ActuatorScale05Env(ActuatorScalingCheetahEnv):
    ACT_SCALE = 0.5


class ActuatorScale10Env(ActuatorScalingCheetahEnv):
    ACT_SCALE = 1.0


class ActuatorScale20Env(ActuatorScalingCheetahEnv):
    ACT_SCALE = 2.0
