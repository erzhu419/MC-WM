"""
AntWallBroken: MC-WM port of ICRL paper's constraint transfer benchmark
(Malik et al. 2021, Section 4.2).

Setup:
  sim:  AntWall  — full Ant (8 working legs) + spatial wall x > -3
  real: AntWallBroken — back legs disabled (action[4:]=0) + same wall

Why this is a good ICRL test:
  - Dynamics gap  : 4 of 8 actuators dead in real → LARGE action→dynamics shift
  - Spatial       : wall at x=-3 exists in BOTH sim and real → constraint TRANSFERS
  - ICRL learns the wall from real expert data, transfers to policy
  - QΔ learns dynamics difference (broken legs → different next-state distribution)
  - Together they cover orthogonal shift axes

Reward (following ICRL paper):
  r = distance_from_origin + healthy_reward - ctrl_cost - contact_cost
  (NOT the standard forward_velocity reward)

Obs: standard Ant-v4 (27-dim, excludes x,y positions from obs).
     x-position tracked via unwrapped.data.qpos[0] for wall checking.
"""

import numpy as np
import gymnasium as gym


class AntWallBrokenEnv(gym.Env):
    """
    Ant with wall termination + (in real) back legs disabled.

    mode="sim":  full actuators, wall at x=-3 terminates
    mode="real": action[4:] = 0, wall at x=-3 terminates
    """

    OBS_DIM = 27
    ACT_DIM = 8
    WALL_X = -3.0
    HEALTHY_REWARD = 1.0

    def __init__(self, mode: str = "sim", seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._env = gym.make("Ant-v4", use_contact_forces=False,
                              terminate_when_unhealthy=True)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def _x_position(self):
        return float(self._env.unwrapped.data.qpos[0])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.mode == "real":
            action[4:] = 0.0  # back legs disabled

        obs, _, terminated, truncated, info = self._env.step(action)

        # ICRL paper reward shaping: distance_from_origin instead of forward_velocity
        x = self._x_position()
        y = float(self._env.unwrapped.data.qpos[1])
        dist = float(np.sqrt(x * x + y * y))
        ctrl_cost = 0.5 * float(np.sum(action ** 2))
        reward = dist + self.HEALTHY_REWARD - ctrl_cost

        # Wall constraint: x < -3 → terminate with zero reward
        if x < self.WALL_X:
            terminated = True
            reward = 0.0
            info["wall_hit"] = True

        info["x_position"] = x
        info["distance_from_origin"] = dist
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()


class AntWallBrokenSoftCeilingEnv(AntWallBrokenEnv):
    """
    AntWallBroken + SOFT z-ceiling (penalizes hopping).

    Keeps the original hard wall termination at x=-3 AND the back-legs dynamics
    gap; adds a continuous quadratic penalty when the ant's root z exceeds
    Z_CEILING. Ant standing z ≈ 0.75, normal locomotion ≤ 0.95; hopping > 1.0
    would be the cheating strategy the constraint suppresses — a behavior
    whose payoff differs between full (sim) and broken (real) actuators.

    info['ceiling_hit'] is flagged on any step with z > Z_CEILING so the
    standard viol/ep metric counts these as safety violations.
    """

    Z_CEILING = 1.0
    PENALTY_SCALE = 10.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        z = float(obs[0])
        excess = max(0.0, z - self.Z_CEILING)
        if excess > 0:
            reward -= self.PENALTY_SCALE * excess * excess
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info
