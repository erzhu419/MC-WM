"""
CarpetAnt: Ant with damped contacts, motor current penalty, and soft termination.

Sim-real gap:
  State gap       → contact forces damped by k_contact=0.5; velocity dims lower
  Reward gap      → motor current penalty: r_real -= λ * sum(action²)
  Termination gap → soft falls don't terminate; real threshold z > 0.15 vs sim's z > 0.2

Expected SINDy discovery:
  Δr shows clear quadratic dependence on action — SINDy poly2 finds this in round 1.
  Δd shows piecewise structure around z=0.17 — round 2 piecewise mask trigger.
  Δs (velocity dims) shows homogeneous damping — round 1 linear SINDy finds multiplier.
"""

import numpy as np
import gymnasium as gym


class CarpetAntEnv(gym.Env):
    """
    Ant wrapper.

    Ant obs layout (111-dim by default, we use the 27-dim external force version):
      gymnasium Ant-v4 with use_contact_forces=False → 27 obs dims
      [0]    z (height)
      [1-4]  quaternion orientation
      [5-12] 8 joint angles
      [13]   vx
      [14]   vy
      [15]   vz
      [16-18] angular velocities
      [19-26] 8 joint angular velocities

    We use Ant-v4 with default 27-dim obs.
    Contact damping approximated as velocity multiplier on all vel dims.
    """

    VEL_DIMS = list(range(13, 27))    # velocity state dims
    OBS_DIM = 27
    ACT_DIM = 8

    CONTACT_DAMP = 0.7        # velocity multiplier due to carpet friction
    MOTOR_PENALTY = 0.01      # λ in r -= λ*‖a‖²
    REAL_Z_THRESH = 0.15      # soft fall threshold (sim uses 0.2)

    def __init__(self, mode: str = "sim", seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._env = gym.make("Ant-v4")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step += 1

        if self.mode == "real":
            obs, reward, terminated = self._apply_carpet(obs, reward, terminated, action)

        return obs, reward, terminated, truncated, info

    def _apply_carpet(self, obs, reward, terminated, action):
        obs = obs.copy()

        # Contact damping on velocity dims
        for i in self.VEL_DIMS:
            obs[i] *= self.CONTACT_DAMP

        # Motor current penalty (reward gap)
        motor_penalty = self.MOTOR_PENALTY * float(np.sum(np.array(action) ** 2))
        reward -= motor_penalty

        # Soft falls: only terminate if z is genuinely very low
        # In real, soft contacts → agent can recover from "falls"
        z = obs[0]
        if z > self.REAL_Z_THRESH and terminated:
            # Sim would have terminated; real would not (soft carpet)
            terminated = False

        return obs, reward, terminated

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()


class CarpetAntSoftCeilingEnv(CarpetAntEnv):
    """
    CarpetAnt + soft velocity cap |vx| > V_CAP (penalizes running too fast).

    Natural safety semantics on carpet: high forward velocity is unsafe on
    soft/slippery ground, so the policy should stay under V_CAP. This is a
    constraint trained policies actually touch (unlike a z-ceiling, which
    quadrupeds never naturally exceed): Ant-v4 SAC reaches vx ≈ 3–5 m/s, so a
    cap at 3.0 binds during normal forward locomotion.

    Interaction with dynamics gap:
      - Sim (no contact damping) → raw vx measured, cap binds often
      - Real (velocities scaled ×0.7 by parent) → damped vx, cap binds less
      → sim policy is punished more for speed; aligns with "soft terrain
        dynamics" story and lets the Pareto gate mediate the trade-off.

    obs[13] = vx (forward velocity in root frame) per Ant-v4 layout.
    """

    V_CAP = 1.5
    PENALTY_SCALE = 1.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        vx = float(obs[13])
        excess = max(0.0, abs(vx) - self.V_CAP)
        if excess > 0:
            reward -= self.PENALTY_SCALE * excess * excess
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info
