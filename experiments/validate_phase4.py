"""
Phase 4 Milestone 验证：自假设循环端到端测试（无 LLM）。

测试流程：
  1. 收集全元组配对数据（delta_s, delta_r, delta_d）
  2. 填充 ResidualBuffer
  3. 创建 SINDyTrack（poly2 库）+ HypothesisLoop（无 LLM）
  4. 运行循环，打印每轮日志
  5. 验证 Milestone 4

预期结果：
  Round 1: degree=2 库找到 s8² 项 → quality_gate PASS
  或:      Round 1 FAIL → hetero(culprit=8) → 自动扩展 → Round 2 PASS
  全程：   NO LLM 调用
  输出：   accepted_round in [1, 2, 3], reason in {quality_gate, white_noise}

运行：conda run -n MC-WM python3 -u experiments/validate_phase4.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np

from mc_wm.envs.hp_mujoco.aero_cheetah import AeroCheetahEnv
from mc_wm.residual.extractor import ResidualBuffer, ResidualSample
from mc_wm.residual.sindy_track import SINDyTrack
from mc_wm.self_audit.hypothesis_loop import HypothesisLoop

SEED      = 42
N_COLLECT = 5_000

# eps_threshold: quality gate（MSE 单位，不是 RMSE）
# AeroCheetah 速度维 MSE ≈ 0.005–0.02；0.05 是合理的 pass 线
EPS_THRESHOLD = 0.05
MAX_ROUNDS    = 3


# ─────────────────────────────────────────────
# 数据收集：全元组 (delta_s, delta_r, delta_d)
# ─────────────────────────────────────────────

def collect_full_tuple(n_steps: int = N_COLLECT) -> ResidualBuffer:
    """
    并行运行 sim + real 两个 AeroCheetah 环境，收集全元组残差。

    输入特征用 sim 状态（s_sim, a）：
      "给定 sim 当前状态，当采取动作 a 时，real 比 sim 多出多少？"
    这与 MVP 中 collect_paired_data 的定义一致。
    """
    sim_env  = AeroCheetahEnv(mode="sim")
    real_env = AeroCheetahEnv(mode="real")

    obs_dim = sim_env.observation_space.shape[0]
    buf = ResidualBuffer(capacity=n_steps + 100, keep_history=True)

    s_sim,  _ = sim_env.reset(seed=SEED)
    s_real, _ = real_env.reset(seed=SEED)
    s_prev    = None
    step_in_ep = 0
    ep = 0

    print(f"  Collecting {n_steps} full-tuple samples (sim+real)...", flush=True)
    for global_step in range(n_steps):
        a = sim_env.action_space.sample()

        ns_sim,  r_sim,  d_sim,  tr_sim,  _ = sim_env.step(a)
        ns_real, r_real, d_real, tr_real, _ = real_env.step(a)

        sample = ResidualSample(
            s           = s_sim.copy(),
            a           = a.copy(),
            s_next_sim  = ns_sim.copy(),
            r_sim       = float(r_sim),
            d_sim       = float(d_sim),
            s_next_real = ns_real.copy(),
            r_real      = float(r_real),
            d_real      = float(d_real),
            delta_s     = ns_real - ns_sim,
            delta_r     = float(r_real) - float(r_sim),
            delta_d     = float(d_real) - float(d_sim),
            s_prev      = s_prev.copy() if s_prev is not None else None,
            step        = step_in_ep,
        )
        buf.append(sample)

        s_prev = s_sim.copy()
        step_in_ep += 1

        if d_sim or tr_sim or d_real or tr_real:
            ep += 1
            s_sim,  _ = sim_env.reset(seed=ep + SEED)
            s_real, _ = real_env.reset(seed=ep + SEED)
            s_prev    = None
            step_in_ep = 0
        else:
            s_sim  = ns_sim
            s_real = ns_real

    sim_env.close()
    real_env.close()
    print(f"  Buffer filled: {len(buf)} samples", flush=True)
    return buf


# ─────────────────────────────────────────────
# 主验证函数
# ─────────────────────────────────────────────

def run_phase4():
    print("=" * 60, flush=True)
    print("Phase 4 Validation: Self-Hypothesizing Loop (no LLM)", flush=True)
    print(f"  eps_threshold={EPS_THRESHOLD}  max_rounds={MAX_ROUNDS}", flush=True)
    print("=" * 60, flush=True)

    # 收集全元组数据
    buf = collect_full_tuple(N_COLLECT)

    # 从 buffer 获取 obs/act 维度
    SA_check, delta_s_check = buf.to_arrays("s")
    obs_dim = delta_s_check.shape[1]
    act_dim = SA_check.shape[1] - obs_dim
    print(f"\n  obs_dim={obs_dim}  act_dim={act_dim}", flush=True)

    # 数据统计（快速 sanity check）
    print(f"\n  delta_s mean |Δ| per dim (top 5):", flush=True)
    abs_mean = np.abs(delta_s_check).mean(0)
    top5 = np.argsort(abs_mean)[-5:][::-1]
    for idx in top5:
        print(f"    dim {idx:>2d}: mean|Δs|={abs_mean[idx]:.5f}", flush=True)

    # 创建 SINDyTrack
    from mc_wm.residual.sindy_track import make_poly2_library
    track = SINDyTrack(
        obs_dim=obs_dim,
        act_dim=act_dim,
        library=make_poly2_library(),
        sindy_threshold=0.01,
        sindy_alpha=0.05,
    )

    # 创建 HypothesisLoop（无 LLM）
    loop = HypothesisLoop(
        sindy_track=track,
        obs_dim=obs_dim,
        act_dim=act_dim,
        eps_threshold=EPS_THRESHOLD,
        max_rounds=MAX_ROUNDS,
        diagnosis_alpha=0.05,
        llm_oracle=None,          # ← NO LLM
    )

    # ─── 运行循环
    print(f"\n[Running HypothesisLoop]", flush=True)
    logs = loop.run(buf)

    # ─── 打印每轮详情
    print("\n" + "─" * 50, flush=True)
    loop.print_summary()

    for log in logs:
        print(f"\n  --- Round {log.round_num} ---", flush=True)
        print(f"  quality_passed: {log.quality_passed}  reason: {log.reason}", flush=True)

        # eps_s 最大值 + 对应维度
        if len(log.fit_errors["eps_s"]) > 0:
            idx_max = int(np.argmax(log.fit_errors["eps_s"]))
            print(f"  eps_s: max={log.fit_errors['eps_s'].max():.6f} (dim {idx_max})"
                  f"  eps_r={log.fit_errors['eps_r'][0]:.6f}"
                  f"  eps_d={log.fit_errors['eps_d'][0]:.6f}", flush=True)

        # 活跃 SINDy 特征（仅对 dim 8）
        if log.accepted:
            feats = track.get_active_features()
            print(f"  Active SINDy features (dim 8): {feats['delta_s'][8]}", flush=True)
            print(f"  Active SINDy features (delta_r): {feats['delta_r']}", flush=True)

        if log.mechanisms_fired:
            print(f"  Mechanisms fired: {log.mechanisms_fired}", flush=True)

        if log.diagnoses:
            fired = [(d.dim, d.summary()) for d in log.diagnoses if d.any_fired()]
            print(f"  Diagnoses fired ({len(fired)} dims): ", flush=True)
            for dim, s in fired[:5]:
                print(f"    {s}", flush=True)

    # ─── Milestone 4 check
    print("\n" + "=" * 60, flush=True)
    print("Milestone 4 Checks:", flush=True)

    final_log = logs[-1]
    accepted_reason = final_log.reason if final_log.accepted else "not_accepted"

    checks = {
        "Loop terminated (accepted_round set)": loop.accepted_round is not None,
        "Reason is quality_gate, white_noise, or max_rounds":
            accepted_reason in ("quality_gate", "white_noise", "max_rounds"),
        "NO LLM oracle used":
            not any("llm" in log.reason for log in logs),
        "Accepted within max_rounds":
            loop.accepted_round is not None and loop.accepted_round <= MAX_ROUNDS,
        "Diagnosis correctly detected hetero on velocity dims":
            any(d.heteroscedastic and d.dim >= 9 for log in logs for d in log.diagnoses),
    }

    all_pass = True
    for label, passed in checks.items():
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {label}", flush=True)
        if not passed:
            all_pass = False

    print(f"\n  Accepted at round: {loop.accepted_round}  reason: {accepted_reason}",
          flush=True)

    print("=" * 60, flush=True)
    if all_pass:
        print("MILESTONE 4: PASS ✓  (self-hypothesizing loop — no LLM needed)", flush=True)
    else:
        print("MILESTONE 4: PARTIAL — see ✗ items above", flush=True)

    return all_pass


if __name__ == "__main__":
    run_phase4()
