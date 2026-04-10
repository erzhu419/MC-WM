"""
HPMuJoCoEnvPair: runs sim and real side-by-side for paired residual extraction.

Usage:
    pair = HPMuJoCoEnvPair("aero_cheetah")
    obs_sim, obs_real = pair.reset()
    for s, a in offline_data:
        delta_s, delta_r, delta_d = pair.query_residual(s, a)
"""

import numpy as np
from typing import Tuple, Dict
from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.envs.hp_mujoco.ice_walker import IceWalkerEnv
from mc_wm.envs.hp_mujoco.wind_hopper import WindHopperEnv
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv

ENV_CLS = {
    "aero_cheetah": (AeroCheetahEnv, AeroCheetahEnv),
    "ice_walker":   (IceWalkerEnv,   IceWalkerEnv),
    "wind_hopper":  (WindHopperEnv,  WindHopperEnv),
    "carpet_ant":   (CarpetAntEnv,   CarpetAntEnv),
}


class HPMuJoCoEnvPair:
    """
    Wraps (sim_env, real_env) for paired transition queries.

    The pair is deterministically reset to the same state before each query,
    so Δ(s,a) = T_real(s,a) - T_sim(s,a) is well-defined.
    """

    def __init__(self, env_name: str, seed: int = 42):
        assert env_name in ENV_CLS, f"Unknown env: {env_name}. Choose from {list(ENV_CLS)}"
        sim_cls, real_cls = ENV_CLS[env_name]
        self.env_name = env_name
        self.sim_env = sim_cls(mode="sim", seed=seed)
        self.real_env = real_cls(mode="real", seed=seed)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        kw = {"seed": seed} if seed is not None else {}
        obs_sim, _ = self.sim_env.reset(**kw)
        obs_real, _ = self.real_env.reset(**kw)
        return obs_sim, obs_real

    def query_residual(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Given (s, a) from offline real data, compute the full-tuple residual
        by stepping both envs from the same state.

        Returns dict with keys: delta_s, delta_r, delta_d
          delta_s  shape (obs_dim,)
          delta_r  scalar float
          delta_d  scalar float  (1.0 if termination disagrees, 0.0 otherwise)
        """
        # Step sim
        self._set_env_state(self.sim_env, state)
        s_next_sim, r_sim, d_sim, _, _ = self.sim_env.step(action)

        # Step real
        self._set_env_state(self.real_env, state)
        s_next_real, r_real, d_real, _, _ = self.real_env.step(action)

        delta_s = s_next_real - s_next_sim
        delta_r = np.array([r_real - r_sim])
        delta_d = np.array([float(d_real) - float(d_sim)])

        return {
            "delta_s": delta_s,
            "delta_r": delta_r,
            "delta_d": delta_d,
            "s_next_sim": s_next_sim,
            "s_next_real": s_next_real,
            "r_sim": r_sim,
            "r_real": r_real,
            "d_sim": d_sim,
            "d_real": d_real,
        }

    def _set_env_state(self, env, state: np.ndarray):
        """
        Set the MuJoCo env to a specific state for deterministic paired queries.

        Gymnasium MuJoCo v4/v5 obs layout (standard):
          HalfCheetah : obs = qpos[1:]   (8) + qvel (9)   → nq=9,  nv=9
          Walker2d    : obs = qpos[1:]   (8) + qvel (9)   → nq=9,  nv=9
          Hopper      : obs = qpos[1:]   (5) + qvel (6)   → nq=6,  nv=6
          Ant         : obs = qpos[2:]   (13)+ qvel (14)  → nq=15, nv=14

        Root x-position (qpos[0]) is excluded from obs; we keep it at its
        current sim value (it affects nothing in AeroCheetah/WindHopper/etc.
        since the reward/gap doesn't depend on absolute x).
        """
        unwrapped = env._env.unwrapped
        model = unwrapped.model
        nq = model.nq
        nv = model.nv

        # Determine how many qpos dims appear in obs.
        # Standard gymnasium convention: obs = qpos[skip:] + qvel
        # where skip=1 for most envs, skip=2 for Ant.
        obs_dim = len(state)
        n_qpos_in_obs = obs_dim - nv   # e.g. 17-9=8 for HalfCheetah
        skip = nq - n_qpos_in_obs      # qpos dims not in obs (root positions)

        try:
            data = unwrapped.data
            qpos = data.qpos.copy()   # keep root position as-is
            qvel = data.qvel.copy()

            # Overwrite observable qpos dims
            qpos[skip:] = state[:n_qpos_in_obs]
            # Overwrite all qvel dims
            qvel[:nv] = state[n_qpos_in_obs:n_qpos_in_obs + nv]

            unwrapped.set_state(qpos, qvel)
        except Exception:
            env.reset()

    def close(self):
        self.sim_env.close()
        self.real_env.close()
