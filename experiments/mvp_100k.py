"""正式 MVP：100k steps，GPU，AeroCheetah"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from experiments.mvp_aero_cheetah import (
    collect_paired_data, SINDyStateCorrector, CorrectedAeroCheetahEnv,
    AeroCheetahEnv, train_sac, plot_comparison, save_sindy_report, SEED, DEVICE
)

print(f"Device: {DEVICE}", flush=True)

# ── A: 数据收集（5k 步，足够 SINDy 建模）
SA, delta_s = collect_paired_data(5_000)

# ── B: SINDy
corrector = SINDyStateCorrector(delta_s.shape[1])
corrector.fit(SA, delta_s)
cov = corrector.correction_coverage(SA, delta_s)
print(f"RMSE reduction: {cov['rmse_reduction_pct']:.1f}%\n", flush=True)

# ── D: 训练，100k steps，eval 每 10k
curve_raw, agent_raw = train_sac(
    lambda: AeroCheetahEnv(mode='sim'),
    'Raw Sim', SEED,
    train_steps=100_000, eval_interval=10_000, warmup=5_000, n_eval_eps=5,
)

curve_corr, agent_corr = train_sac(
    lambda: CorrectedAeroCheetahEnv(corrector),
    'Corrected Sim', SEED,
    train_steps=100_000, eval_interval=10_000, warmup=5_000, n_eval_eps=5,
)

# ── F: 输出
plot_comparison(
    curve_raw, curve_corr,
    os.path.join(os.path.dirname(__file__), 'mvp_comparison_100k.png'),
)
save_sindy_report(
    corrector, cov,
    os.path.join(os.path.dirname(__file__), 'sindy_coverage_100k.txt'),
)

# ── 最终结论
if curve_raw and curve_corr:
    fr = np.mean([r for _, r in curve_raw[-3:]])
    fc = np.mean([r for _, r in curve_corr[-3:]])
    gain = fc - fr
    if gain > 100:
        verdict = "✓✓ 显著改善，这条路值得走"
    elif gain > 30:
        verdict = "✓  有改善，继续推进"
    elif gain > 0:
        verdict = "△  微弱改善，谨慎推进"
    else:
        verdict = "✗  无改善，重新评估方案"

    print("\n" + "="*50)
    print(f"Raw sim       (最后3次均值): {fr:.1f}")
    print(f"Corrected sim (最后3次均值): {fc:.1f}")
    print(f"增益: {gain:+.1f}")
    print(f"结论: {verdict}")
    print("="*50, flush=True)
