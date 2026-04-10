from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.envs.hp_mujoco.ice_walker import IceWalkerEnv
from mc_wm.envs.hp_mujoco.wind_hopper import WindHopperEnv
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.envs.hp_mujoco.env_pair import HPMuJoCoEnvPair

REGISTRY = {
    "aero_cheetah": AeroCheetahEnv,
    "ice_walker": IceWalkerEnv,
    "wind_hopper": WindHopperEnv,
    "carpet_ant": CarpetAntEnv,
}
