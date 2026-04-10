"""
Phase 3 Milestone 验证：DiagnosisBattery 正确区分有结构 vs 无结构残差维度。

预期（Milestone 3）：
  dim  8（速度维，有二次阻力）: heteroscedastic=True, culprit≈8
  dim  0（无已知 gap 结构）  : CLEAN — 所有测试均为 False

两个阶段的诊断：
  Stage A — 原始残差 (delta_s before SINDy)：
            检验"有结构"能被统计测试检测到
  Stage B — SINDy 余项 (delta_s - SINDy_pred)：
            检验 SINDy 把结构清理掉后余项变 CLEAN

运行：conda run -n MC-WM python3 -u experiments/validate_phase3.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.mvp_aero_cheetah import collect_paired_data, SINDyStateCorrector, SEED
from mc_wm.self_audit.diagnosis import DiagnosisBattery


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_COLLECT = 5_000
DIM_STRUCTURED = 9   # vx (forward velocity) — drag: -k*v*|v|*dt → Δs[9] ∝ v²
DIM_BASELINE   = 0   # z (height) — no known gap structure


def run_phase3():
    print("=" * 60, flush=True)
    print("Phase 3 Validation: DiagnosisBattery", flush=True)
    print("=" * 60, flush=True)

    # ── A: 收集数据
    print(f"\n[A] Collecting {N_COLLECT} paired steps...", flush=True)
    SA, delta_s = collect_paired_data(N_COLLECT)
    obs_dim = delta_s.shape[1]
    act_dim = SA.shape[1] - obs_dim
    print(f"    SA: {SA.shape}  delta_s: {delta_s.shape}", flush=True)

    # ── B: 拟合 SINDy（用于计算余项）
    print(f"\n[B] Fitting SINDy corrector (degree=2)...", flush=True)
    corrector = SINDyStateCorrector(obs_dim)
    corrector.fit(SA, delta_s)
    cov = corrector.correction_coverage(SA, delta_s)
    print(f"    RMSE reduction: {cov['rmse_reduction_pct']:.1f}%", flush=True)

    # 计算余项 = 真实残差 − SINDy 预测（用 batch 版本）
    pred = corrector.predict_batch(SA)   # (N, obs_dim)
    remainder = delta_s - pred

    # ── C: 初始化 DiagnosisBattery
    battery = DiagnosisBattery(alpha=0.05)

    # ─────────────────────────────────────────────
    # Stage A: 原始残差诊断
    # ─────────────────────────────────────────────
    print("\n" + "─" * 50, flush=True)
    print("Stage A: Diagnosis on RAW delta_s (before SINDy)", flush=True)
    print("─" * 50, flush=True)

    results_raw = battery.run(delta_s, SA)

    # 打印全部维度摘要
    for r in results_raw:
        print(f"  {r.summary()}", flush=True)

    r8_raw = results_raw[DIM_STRUCTURED]
    r0_raw = results_raw[DIM_BASELINE]

    print(f"\n  Focus dim {DIM_STRUCTURED} (structured): {r8_raw.summary()}", flush=True)
    print(f"  Focus dim {DIM_BASELINE}  (baseline) : {r0_raw.summary()}", flush=True)

    # ─────────────────────────────────────────────
    # Stage B: SINDy 余项诊断
    # ─────────────────────────────────────────────
    print("\n" + "─" * 50, flush=True)
    print("Stage B: Diagnosis on REMAINDER (delta_s - SINDy pred)", flush=True)
    print("─" * 50, flush=True)

    results_rem = battery.run(remainder, SA)

    for r in results_rem:
        print(f"  {r.summary()}", flush=True)

    r8_rem = results_rem[DIM_STRUCTURED]
    r0_rem = results_rem[DIM_BASELINE]

    print(f"\n  Focus dim {DIM_STRUCTURED} (structured, after SINDy): {r8_rem.summary()}", flush=True)
    print(f"  Focus dim {DIM_BASELINE}  (baseline,   after SINDy): {r0_rem.summary()}", flush=True)

    # ─────────────────────────────────────────────
    # Milestone 3 check
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Milestone 3 Checks:", flush=True)

    checks = {
        f"[A] dim {DIM_STRUCTURED} raw: any structure detected": r8_raw.any_fired(),
        f"[A] dim {DIM_STRUCTURED} raw: heteroscedastic": r8_raw.heteroscedastic,
        f"[A] dim {DIM_BASELINE}  raw: CLEAN (no structure)": not r0_raw.any_fired(),
    }
    # Stage B 是信息性的：余项仍有结构 = Phase 4 循环有必要
    info_checks = {
        f"[B] dim {DIM_STRUCTURED} rem: structure remaining → Phase 4 needed": r8_rem.any_fired(),
    }

    all_pass = True
    for label, passed in checks.items():
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {label}", flush=True)
        if not passed:
            all_pass = False

    for label, val in info_checks.items():
        mark = "△" if val else "○"
        print(f"  {mark}  {label}", flush=True)

    # 额外打印 culprit 信息
    if r8_raw.heteroscedastic and r8_raw.heteroscedastic_culprit is not None:
        culprit = r8_raw.heteroscedastic_culprit
        culprit_type = "state" if culprit < obs_dim else f"action[{culprit - obs_dim}]"
        print(f"\n  Heteroscedasticity culprit: feature {culprit} ({culprit_type})", flush=True)
        print(f"  Expected: feature 8 (velocity dim, state[8])", flush=True)
        if culprit == DIM_STRUCTURED:
            print(f"  ✓ Culprit matches expected velocity dimension", flush=True)
        else:
            print(f"  △ Culprit differs from expected — check correlation structure", flush=True)

    print("=" * 60, flush=True)
    if all_pass:
        print("MILESTONE 3: PASS ✓", flush=True)
    else:
        print("MILESTONE 3: PARTIAL — see ✗ items above", flush=True)

    return all_pass


if __name__ == "__main__":
    run_phase3()
