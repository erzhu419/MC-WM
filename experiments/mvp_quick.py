"""快速测试脚本：10k steps，eval 每 2k，warmup 1k"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import experiments.mvp_aero_cheetah as mvp_mod

mvp_mod.N_COLLECT     = 2_000
mvp_mod.TRAIN_STEPS   = 10_000
mvp_mod.EVAL_INTERVAL = 2_000
mvp_mod.WARMUP        = 1_000
mvp_mod.N_EVAL_EPS    = 3

# 重新导入使模块级常量生效
import importlib
importlib.reload(mvp_mod)

from experiments.mvp_aero_cheetah import (
    collect_paired_data, SINDyStateCorrector, CorrectedAeroCheetahEnv,
    AeroCheetahEnv, train_sac, plot_comparison, SEED
)
import numpy as np

print("=== Phase A: 收集数据 ===", flush=True)
SA, delta_s = collect_paired_data(2_000)

print("=== Phase B: SINDy 拟合 ===", flush=True)
corrector = SINDyStateCorrector(delta_s.shape[1])
corrector.fit(SA, delta_s)
cov = corrector.correction_coverage(SA, delta_s)
print(f"RMSE reduction: {cov['rmse_reduction_pct']:.1f}%", flush=True)

print("=== Phase D: 训练 Raw Sim ===", flush=True)
curve_raw, _ = train_sac(
    lambda: AeroCheetahEnv(mode='sim'),
    'Raw Sim', SEED,
    train_steps=10_000, eval_interval=2_000, warmup=1_000, n_eval_eps=3,
)

print("=== Phase D: 训练 Corrected Sim ===", flush=True)
curve_corr, _ = train_sac(
    lambda: CorrectedAeroCheetahEnv(corrector),
    'Corrected Sim', SEED,
    train_steps=10_000, eval_interval=2_000, warmup=1_000, n_eval_eps=3,
)

if curve_raw and curve_corr:
    fr = np.mean([r for _, r in curve_raw[-2:]])
    fc = np.mean([r for _, r in curve_corr[-2:]])
    gain = fc - fr
    verdict = "✓ 有效" if gain > 30 else ("△ 微弱" if gain > 0 else "✗ 无改善")
    print(f"\n=== 结果 ===", flush=True)
    print(f"Raw sim       : {fr:.1f}", flush=True)
    print(f"Corrected sim : {fc:.1f}", flush=True)
    print(f"增益           : {gain:+.1f}  {verdict}", flush=True)
