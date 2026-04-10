"""
MVP v3: RE-SAC + Gated Corrected Sim

改进点 vs mvp_td3_gated.py（TD3 + Gate）：
  1. TD3 → RE-SAC（Ensemble critic + LCB + SAC 熵正则）
     - LCB = -(Q_mean + β*Q_std)，β=-2 悲观
     - +α*log_π 熵项防止 policy 在低 std 区域坍缩（L1-LCB 稳定性修复）
     - 这与用户指出的"只加 LCB 也许不稳定"完全对应
  2. 保留 Gate（OOD 区域修正量自动退为 0）

对比组：
  A. Raw Sim RE-SAC（baseline）
  B. Gated Corrected Sim RE-SAC（我们的方法）

运行：conda run -n MC-WM python3 -u experiments/mvp_resac_gated.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.policy.resac_agent import RESACAgent
from experiments.mvp_aero_cheetah import (
    collect_paired_data, SINDyStateCorrector, DEVICE, SEED
)
from experiments.mvp_td3_gated import (
    ReplayBuffer, GatedCorrectedEnv,
    GATE_TAU_A, GATE_EPS_JAC, OUT_DIR,
    N_COLLECT, EVAL_INTERVAL, N_EVAL_EPS,
)

# ─────────────────────────────────────────────
# Config（可覆盖 td3_gated 的）
# ─────────────────────────────────────────────
TRAIN_STEPS  = 150_000
WARMUP       = 5_000
BATCH_SIZE   = 256
REPLAY_SIZE  = 300_000

# RE-SAC 超参
N_CRITICS    = 5       # ensemble size
BETA_LCB     = -2.0    # β<0 → 悲观 LCB
BETA_OOD     = 0.01
BETA_BC      = 0.001
CRITIC_RATIO = 2       # actor 每 2 步更新一次（和 TD3 POLICY_DELAY 对应）
LR           = 3e-4
HIDDEN       = 256


# ─────────────────────────────────────────────
# Eval on real env
# ─────────────────────────────────────────────

def evaluate_on_real(agent: RESACAgent, n_eps: int = N_EVAL_EPS) -> float:
    real_env = AeroCheetahEnv(mode="real")
    returns = []
    for ep in range(n_eps):
        obs, _ = real_env.reset(seed=ep + 200)
        total_r = 0.0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, _ = real_env.step(a)
            total_r += r
            if d or tr:
                break
        returns.append(total_r)
    real_env.close()
    return float(np.mean(returns))


# ─────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────

def train_resac(env_fn, label: str, seed: int = SEED) -> tuple:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = RESACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_dim=HIDDEN,
        n_critics=N_CRITICS,
        beta=BETA_LCB,
        beta_ood=BETA_OOD,
        beta_bc=BETA_BC,
        critic_actor_ratio=CRITIC_RATIO,
        lr=LR,
        device=DEVICE,
    )
    buf = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE)

    obs, _ = env.reset(seed=seed)
    eval_curve = []
    steps_since_eval = 0

    print(f"\n[Train] {label}  ({TRAIN_STEPS // 1000}k steps, "
          f"n_critics={N_CRITICS}, β={BETA_LCB}, GPU={DEVICE})", flush=True)

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
            ret = evaluate_on_real(agent)
            eval_curve.append((step, ret))
            steps_since_eval = 0
            gate_info = ""
            if hasattr(env, "mean_gate"):
                gate_info = f"  gate={env.mean_gate():.3f}"
            print(f"  step {step:>7d} | real_return = {ret:7.1f}"
                  f"  α={agent.alpha:.4f}{gate_info}", flush=True)

    env.close()
    return eval_curve, agent


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}", flush=True)

    # ── A: 数据收集
    SA, delta_s = collect_paired_data(N_COLLECT)

    # ── B: SINDy
    corrector = SINDyStateCorrector(delta_s.shape[1])
    corrector.fit(SA, delta_s)
    cov = corrector.correction_coverage(SA, delta_s)
    print(f"RMSE reduction: {cov['rmse_reduction_pct']:.1f}%\n", flush=True)

    # ── Raw Sim RE-SAC
    curve_raw, _ = train_resac(
        lambda: AeroCheetahEnv(mode="sim"),
        "Raw Sim (RE-SAC)",
    )

    # ── Gated Corrected Sim RE-SAC
    def make_gated_env():
        env = GatedCorrectedEnv(corrector, tau=GATE_TAU_A, eps_jac=GATE_EPS_JAC)
        env.set_train_center(SA)
        return env

    curve_corr, _ = train_resac(make_gated_env, "Gated Corrected Sim (RE-SAC)")

    # ── 对比图
    fig, ax = plt.subplots(figsize=(10, 5))
    if curve_raw:
        steps, rets = zip(*curve_raw)
        ax.plot(steps, rets, label="Raw Sim (RE-SAC)", color="steelblue", lw=2)
    if curve_corr:
        steps, rets = zip(*curve_corr)
        ax.plot(steps, rets, label="Gated Corrected Sim (RE-SAC)", color="darkorange", lw=2)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Real env return (avg 5 eps)")
    ax.set_title("MVP v3: RE-SAC + Gate — AeroCheetah Sim-to-Real")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "mvp_resac_gated.png")
    plt.savefig(out_fig, dpi=100)
    print(f"\n图已保存 → {out_fig}", flush=True)

    # ── 最终结论
    if curve_raw and curve_corr:
        fr = np.mean([r for _, r in curve_raw[-3:]])
        fc = np.mean([r for _, r in curve_corr[-3:]])
        gain = fc - fr
        if gain > 200:   verdict = "✓✓ 显著改善"
        elif gain > 50:  verdict = "✓  有改善"
        elif gain > 0:   verdict = "△  微弱改善"
        else:            verdict = "✗  无改善，检查 β 和 gate 参数"

        print("\n" + "=" * 50)
        print(f"Raw Sim RE-SAC      (最后3次均值): {fr:.1f}")
        print(f"Gated Corrected     (最后3次均值): {fc:.1f}")
        print(f"增益: {gain:+.1f}   {verdict}")
        print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
