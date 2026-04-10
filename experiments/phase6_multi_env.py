"""
Phase 6: Full MC-WM Pipeline — Multi-Env Experiment

4 个 HP-MuJoCo 环境 × 3 个条件：
  A. Raw Sim (RE-SAC)           — baseline
  B. Gated Corrected Sim (RE-SAC) — SINDy + Gate
  C. Augmented Buffer (RE-SAC)    — Gated Corrected Sim + real offline data 混入 buffer

每个环境：
  1. 收集配对数据 → SINDy 拟合 (state + reward)
  2. 收集 offline real transitions → 作为 augmented buffer 的种子
  3. 训练 RE-SAC (150k steps) × 3 条件
  4. 每 10k 步在 real env 上 eval
  5. 输出对比图

运行：conda run -n MC-WM python3 -u experiments/phase6_multi_env.py --env aero_cheetah
       conda run -n MC-WM python3 -u experiments/phase6_multi_env.py --env all
"""

import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.envs.hp_mujoco.ice_walker import IceWalkerEnv
from mc_wm.envs.hp_mujoco.wind_hopper import WindHopperEnv
from mc_wm.envs.hp_mujoco.carpet_ant import CarpetAntEnv
from mc_wm.policy.resac_agent import RESACAgent
from experiments.mvp_aero_cheetah import SINDyStateCorrector
from experiments.mvp_td3_gated import ReplayBuffer, GatedCorrectedEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_PAIRED       = 5_000     # 配对数据量（SINDy 拟合用）
N_OFFLINE_REAL = 2_000     # offline real data（augmented buffer 种子）
TRAIN_STEPS    = 150_000
EVAL_INTERVAL  = 10_000
N_EVAL_EPS     = 5
WARMUP         = 5_000
BATCH_SIZE     = 256
REPLAY_SIZE    = 300_000

# RE-SAC 超参
N_CRITICS      = 5
BETA_LCB       = -2.0
BETA_OOD       = 0.01
BETA_BC        = 0.001
CRITIC_RATIO   = 2
LR             = 3e-4
HIDDEN         = 256

# Gate 参数
GATE_TAU       = 0.5
GATE_EPS_JAC   = 0.01

OUT_DIR = os.path.dirname(__file__)

# ─────────────────────────────────────────────
# 环境注册
# ─────────────────────────────────────────────
ENV_REGISTRY = {
    "aero_cheetah": AeroCheetahEnv,
    "ice_walker":   IceWalkerEnv,
    "wind_hopper":  WindHopperEnv,
    "carpet_ant":   CarpetAntEnv,
}


# ─────────────────────────────────────────────
# 数据收集
# ─────────────────────────────────────────────

def collect_paired_data(env_cls, n_steps, seed=SEED):
    """并行 sim+real 收集 SINDy 拟合用配对数据。"""
    sim_env  = env_cls(mode="sim")
    real_env = env_cls(mode="real")
    SA_list, delta_s_list, delta_r_list = [], [], []

    obs_s, _ = sim_env.reset(seed=seed)
    obs_r, _ = real_env.reset(seed=seed)
    ep = 0

    for _ in range(n_steps):
        a = sim_env.action_space.sample()
        ns, rs, ds, ts, _ = sim_env.step(a)
        nr, rr, dr, tr, _ = real_env.step(a)
        SA_list.append(np.concatenate([obs_s, a]))
        delta_s_list.append(nr - ns)
        delta_r_list.append(rr - rs)
        obs_s, obs_r = ns, nr
        if ds or ts or dr or tr:
            ep += 1
            obs_s, _ = sim_env.reset(seed=ep + seed)
            obs_r, _ = real_env.reset(seed=ep + seed)

    sim_env.close(); real_env.close()
    SA = np.array(SA_list, dtype=np.float32)
    delta_s = np.array(delta_s_list, dtype=np.float32)
    delta_r = np.array(delta_r_list, dtype=np.float32)
    print(f"  Collected {len(SA)} paired steps ({ep} episodes)", flush=True)
    return SA, delta_s, delta_r


def collect_offline_real(env_cls, n_steps, seed=SEED):
    """在 real env 上用随机策略收集 offline transitions。"""
    env = env_cls(mode="real")
    transitions = []
    obs, _ = env.reset(seed=seed)
    ep = 0

    for _ in range(n_steps):
        a = env.action_space.sample()
        obs2, r, d, tr, _ = env.step(a)
        transitions.append((obs.copy(), a.copy(), r, obs2.copy(), float(d and not tr)))
        obs = obs2
        if d or tr:
            ep += 1
            obs, _ = env.reset(seed=ep + seed + 1000)

    env.close()
    print(f"  Collected {len(transitions)} offline real steps ({ep} episodes)", flush=True)
    return transitions


def seed_buffer_with_real(buf: ReplayBuffer, transitions):
    """将 offline real transitions 注入 replay buffer。"""
    for s, a, r, s2, d in transitions:
        buf.add(s, a, r, s2, d)
    print(f"  Seeded buffer with {len(transitions)} real transitions", flush=True)


# ─────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────

def evaluate_on_real(agent, env_cls, n_eps=N_EVAL_EPS):
    env = env_cls(mode="real")
    returns = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep + 200)
        total_r = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, _ = env.step(a)
            total_r += r
            if d or tr:
                break
        returns.append(total_r)
    env.close()
    return float(np.mean(returns))


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train_resac(env_fn, env_cls, label, seed=SEED,
                offline_real=None):
    """
    Train RE-SAC in a given environment.
    If offline_real is provided, seed the buffer with real transitions.
    """
    np.random.seed(seed); torch.manual_seed(seed)
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = RESACAgent(
        obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit,
        hidden_dim=HIDDEN, n_critics=N_CRITICS, beta=BETA_LCB,
        beta_ood=BETA_OOD, beta_bc=BETA_BC, critic_actor_ratio=CRITIC_RATIO,
        lr=LR, device=DEVICE,
    )
    buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)

    # Augmented buffer: seed with offline real data
    if offline_real is not None:
        seed_buffer_with_real(buf, offline_real)

    obs, _ = env.reset(seed=seed)
    eval_curve = []
    steps_since_eval = 0

    print(f"\n  [{label}] {TRAIN_STEPS//1000}k steps, "
          f"n_critics={N_CRITICS}, β={BETA_LCB}", flush=True)

    for step in range(1, TRAIN_STEPS + 1):
        if step < WARMUP:
            a = env.action_space.sample()
        else:
            a = agent.get_action(obs, deterministic=False)

        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr:
            obs, _ = env.reset()

        if step >= WARMUP and buf.size >= BATCH_SIZE:
            agent.update(buf)

        steps_since_eval += 1
        if steps_since_eval >= EVAL_INTERVAL:
            ret = evaluate_on_real(agent, env_cls)
            eval_curve.append((step, ret))
            steps_since_eval = 0
            gate_info = ""
            if hasattr(env, "mean_gate"):
                gate_info = f"  gate={env.mean_gate():.3f}"
            print(f"    step {step:>7d} | real_return = {ret:7.1f}"
                  f"  α={agent.alpha:.4f}{gate_info}", flush=True)

    env.close()
    return eval_curve


# ─────────────────────────────────────────────
# Run one env
# ─────────────────────────────────────────────

def run_env(env_name: str):
    env_cls = ENV_REGISTRY[env_name]
    print(f"\n{'='*60}", flush=True)
    print(f"  ENV: {env_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── A: 配对数据收集 + SINDy
    print("\n[Phase A] Collecting paired data...", flush=True)
    SA, delta_s, delta_r = collect_paired_data(env_cls, N_PAIRED)
    obs_dim = delta_s.shape[1]

    print("[Phase B] Fitting SINDy corrector...", flush=True)
    corrector = SINDyStateCorrector(obs_dim)
    corrector.fit(SA, delta_s)
    cov = corrector.correction_coverage(SA, delta_s)
    print(f"  RMSE reduction: {cov['rmse_reduction_pct']:.1f}%", flush=True)

    # ── B: Offline real data
    print("\n[Phase B] Collecting offline real data...", flush=True)
    offline_real = collect_offline_real(env_cls, N_OFFLINE_REAL)

    # ── C: Training 3 conditions
    curves = {}

    # C1: Raw Sim
    print("\n[Condition A] Raw Sim (RE-SAC)", flush=True)
    curves["Raw Sim"] = train_resac(
        lambda: env_cls(mode="sim"), env_cls, "Raw Sim",
    )

    # C2: Gated Corrected Sim
    def make_gated():
        sim = env_cls(mode="sim")
        env = GatedCorrectedEnv(corrector, tau=GATE_TAU, eps_jac=GATE_EPS_JAC,
                                sim_env=sim)
        env.set_train_center(SA)
        return env

    print("\n[Condition B] Gated Corrected Sim (RE-SAC)", flush=True)
    curves["Gated Corrected"] = train_resac(
        make_gated, env_cls, "Gated Corrected",
    )

    # C3: Augmented Buffer (Gated + real offline seed)
    print("\n[Condition C] Augmented Buffer (RE-SAC)", flush=True)
    curves["Augmented"] = train_resac(
        make_gated, env_cls, "Augmented Buffer",
        offline_real=offline_real,
    )

    # ── D: 对比图
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Raw Sim": "steelblue", "Gated Corrected": "darkorange", "Augmented": "forestgreen"}
    for label, curve in curves.items():
        if curve:
            steps, rets = zip(*curve)
            ax.plot(steps, rets, label=label, color=colors.get(label, "gray"), lw=2)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(f"Real env return (avg {N_EVAL_EPS} eps)")
    ax.set_title(f"Phase 6: {env_name} — RE-SAC + Gate + Augmented Buffer")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, f"phase6_{env_name}.png")
    plt.savefig(out_fig, dpi=100)
    plt.close()
    print(f"\n  图已保存 → {out_fig}", flush=True)

    # ── E: 总结
    print(f"\n  {'='*50}", flush=True)
    for label, curve in curves.items():
        if curve and len(curve) >= 3:
            avg = np.mean([r for _, r in curve[-3:]])
            print(f"  {label:25s} (最后3次均值): {avg:.1f}", flush=True)
    print(f"  {'='*50}", flush=True)

    return curves


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="aero_cheetah",
                        choices=list(ENV_REGISTRY.keys()) + ["all"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}", flush=True)

    if args.env == "all":
        for name in ENV_REGISTRY:
            try:
                run_env(name)
            except Exception as e:
                print(f"\n  {name} FAILED: {e}", flush=True)
    else:
        run_env(args.env)


if __name__ == "__main__":
    main()
