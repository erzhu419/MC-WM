"""
Week 1 milestone: debug_residuals.py

目标：从4个 HP-MuJoCo 环境收集配对残差，验证结构符合预期，输出 debug_residuals.png。

预期结构：
  AeroCheetah → Δvx 对 vx 呈二次型（异方差性）
  IceWalker   → Δvx 在 x=5 附近有跳变（非正态）
  WindHopper  → Δvx 呈正弦自相关（自相关）
  CarpetAnt   → Δr 对 action 呈二次型；Δvx 均匀阻尼（线性）

运行：conda run -n MC-WM python3 experiments/debug_residuals.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.envs.hp_mujoco.ice_walker   import IceWalkerEnv
from mc_wm.envs.hp_mujoco.wind_hopper  import WindHopperEnv
from mc_wm.envs.hp_mujoco.carpet_ant   import CarpetAntEnv
from mc_wm.self_audit.diagnosis import DiagnosisBattery

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EPISODES = 5
MAX_STEPS  = 200
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "debug_residuals.png")

ENV_PAIRS = {
    "aero_cheetah": (lambda: AeroCheetahEnv(mode="sim"),
                     lambda: AeroCheetahEnv(mode="real")),
    "ice_walker":   (lambda: IceWalkerEnv(mode="sim"),
                     lambda: IceWalkerEnv(mode="real")),
    "wind_hopper":  (lambda: WindHopperEnv(mode="sim"),
                     lambda: WindHopperEnv(mode="real")),
    "carpet_ant":   (lambda: CarpetAntEnv(mode="sim"),
                     lambda: CarpetAntEnv(mode="real")),
}

# ---------------------------------------------------------------------------
# 数据收集：sim / real 同步 step
# ---------------------------------------------------------------------------

def _sample_action(env, obs, env_name: str, rng):
    """
    IceWalker 用偏向前进的混合策略（80% 正向 action、20% 随机），
    确保 agent 能走到 x>5 触发冰面跳变；其他 env 纯随机。
    """
    if env_name == "ice_walker":
        # Walker2d action: [thigh_right, leg_right, foot_right,
        #                   thigh_left,  leg_left,  foot_left]
        # 正向行走：右腿向前（正），左腿向后（负）交替；简化为固定正向偏置
        a = env.action_space.sample()
        if rng.random() < 0.8:
            a = np.clip(a + 0.5, env.action_space.low, env.action_space.high)
        return a
    return env.action_space.sample()


def collect_residuals(env_name):
    sim_fn, real_fn = ENV_PAIRS[env_name]
    sim_env  = sim_fn()
    real_env = real_fn()
    rng = np.random.default_rng(42)

    states, actions = [], []
    delta_s_list, delta_r_list, delta_d_list = [], [], []

    for ep in range(N_EPISODES):
        obs_s, _ = sim_env.reset(seed=ep)
        obs_r, _ = real_env.reset(seed=ep)

        for _ in range(MAX_STEPS):
            a = _sample_action(sim_env, obs_s, env_name, rng)

            obs_s2, r_s, d_s, tr_s, _ = sim_env.step(a)
            obs_r2, r_r, d_r, tr_r, _ = real_env.step(a)

            states.append(np.concatenate([obs_s, a]))
            actions.append(a)
            delta_s_list.append(obs_r2 - obs_s2)
            delta_r_list.append(r_r - r_s)
            delta_d_list.append(float(d_r) - float(d_s))

            obs_s, obs_r = obs_s2, obs_r2
            if d_r or tr_r or d_s or tr_s:
                break

    sim_env.close()
    real_env.close()

    SA      = np.array(states,      dtype=np.float32)
    delta_s = np.array(delta_s_list, dtype=np.float32)
    delta_r = np.array(delta_r_list, dtype=np.float32).reshape(-1, 1)
    delta_d = np.array(delta_d_list, dtype=np.float32).reshape(-1, 1)
    return SA, delta_s, delta_r, delta_d

# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------

def run_diagnosis(env_name, SA, delta_s):
    battery = DiagnosisBattery(alpha=0.05)
    results = battery.run(delta_s, SA)
    print(f"\n{'='*50}\nEnv: {env_name}  (N={len(SA)} samples)")
    print(battery.summarize(results))
    return results

# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def make_debug_plot(all_data):
    n = len(all_data)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle("MC-WM Week 1 Milestone — Full-Tuple Residual Structure", fontsize=13)

    for row, (env_name, (SA, delta_s, delta_r, delta_d)) in enumerate(all_data.items()):
        obs_dim = delta_s.shape[1]
        # 找方差最大的 vel dim（后半段 obs）
        half = obs_dim // 2
        vel_idx = half + int(np.argmax(delta_s[:, half:].var(0)))
        vel_idx = min(vel_idx, obs_dim - 1)

        # Panel 1: Δs[vel] histogram
        ax = axes[row, 0]
        ax.hist(delta_s[:, vel_idx], bins=40, color="steelblue", alpha=0.7)
        ax.set_title(f"{env_name}: Δs[{vel_idx}] hist")
        ax.set_xlabel("Residual"); ax.set_ylabel("Count")

        # Panel 2: Δs[vel] vs s[vel]
        ax = axes[row, 1]
        ax.scatter(SA[:, vel_idx], delta_s[:, vel_idx],
                   alpha=0.25, s=5, color="darkorange")
        ax.set_title(f"{env_name}: Δs[{vel_idx}] vs s[{vel_idx}]")
        ax.set_xlabel(f"s[{vel_idx}]"); ax.set_ylabel("Δs")

        # Panel 3: Δr histogram
        ax = axes[row, 2]
        ax.hist(delta_r[:, 0], bins=40, color="seagreen", alpha=0.7)
        ax.set_title(f"{env_name}: Δr hist")
        ax.set_xlabel("Reward residual"); ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=100)
    print(f"\nSaved → {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_data = {}
    for env_name in ENV_PAIRS:
        print(f"\nCollecting: {env_name} ...", flush=True)
        try:
            SA, delta_s, delta_r, delta_d = collect_residuals(env_name)
            run_diagnosis(env_name, SA, delta_s)
            all_data[env_name] = (SA, delta_s, delta_r, delta_d)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {env_name}: {e}")
            traceback.print_exc()

    if all_data:
        make_debug_plot(all_data)
        print("\n=== Week 1 milestone: debug_residuals.png OK ===")
    else:
        print("所有环境均失败，检查 MuJoCo 安装。")

if __name__ == "__main__":
    main()
