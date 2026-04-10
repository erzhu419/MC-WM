"""
Measure sim-to-real gap for ALL HP-MuJoCo envs.
Both state gap AND reward gap per step.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.envs.hp_mujoco.wind_hopper import WindHopperEnv
from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.envs.hp_mujoco.ice_walker import IceWalkerEnv

def measure_gap(env_cls, name, n_steps=2000, seed=42):
    sim = env_cls(mode="sim"); real = env_cls(mode="real")
    os_s, _ = sim.reset(seed=seed); os_r, _ = real.reset(seed=seed)
    ep = 0
    state_gaps, reward_gaps, done_mismatches = [], [], 0

    for _ in range(n_steps):
        a = sim.action_space.sample()
        ns_s, rs, ds, ts, _ = sim.step(a)
        ns_r, rr, dr, tr, _ = real.step(a)

        state_gaps.append(np.abs(ns_r - ns_s).mean())
        reward_gaps.append(abs(rr - rs))
        if (ds or ts) != (dr or tr): done_mismatches += 1

        os_s, os_r = ns_s, ns_r
        if ds or ts or dr or tr:
            ep += 1; os_s, _ = sim.reset(seed=ep+seed); os_r, _ = real.reset(seed=ep+seed)
    sim.close(); real.close()

    sg = np.array(state_gaps); rg = np.array(reward_gaps)
    print(f"\n{name} (obs={sim.observation_space.shape[0]}, act={sim.action_space.shape[0]}):")
    print(f"  State gap:  mean={sg.mean():.4f}  std={sg.std():.4f}  max={sg.max():.4f}")
    print(f"  Reward gap: mean={rg.mean():.4f}  std={rg.std():.4f}  max={rg.max():.4f}")
    print(f"  Reward gap as % of |reward|: {rg.mean()/max(np.abs(np.array([rs])).mean(), 0.01)*100:.1f}%")
    print(f"  Done mismatches: {done_mismatches}/{n_steps} ({done_mismatches/n_steps*100:.1f}%)")
    print(f"  Episodes: {ep}")

for cls, name in [(CarpetAntEnv, "CarpetAnt"),
                   (WindHopperEnv, "WindHopper"),
                   (AeroCheetahEnv, "AeroCheetah"),
                   (IceWalkerEnv, "IceWalker")]:
    try:
        measure_gap(cls, name)
    except Exception as e:
        print(f"\n{name}: ERROR — {e}")
