"""
Multi-seed post-fix ablation chart for §6.5 Ablation Study.
Pairs with tab:ablation-multi.  Uses 3-seed mean ± std.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# (config_label, reward_mean, reward_std, viol_mean, color) on gravity_soft_ceiling
# 3 seeds {42, 123, 456}, post-bug-fix pipeline.
data = [
    ("Full c9 RAHD",              5394, 478,  4.68,  "#264653"),
    ("no Pareto gate",            5299, 677,  1.03,  "#8DB580"),
    ("no Bellman Q$^{\\Delta}$",  5410, 113,  3.09,  "#E76F51"),
    ("no Role \\#5 (HP)",         5351, 223, 13.52,  "#F4A261"),
    ("no LLM (Roles 1--3)",       5274, 311,  1.65,  "#E9C46A"),
]

labels   = [d[0] for d in data]
rewards  = [d[1] for d in data]
rstds    = [d[2] for d in data]
viols    = [d[3] for d in data]
colors   = [d[4] for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

# ── Reward bars (with std error bars) ───────────────────────────────────
x = np.arange(len(labels))
ax1.bar(x, rewards, yerr=rstds, color=colors, edgecolor="black",
        linewidth=0.8, capsize=4)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
ax1.set_ylabel("Return (3-seed mean $\\pm$ std)", fontsize=11)
ax1.axhline(rewards[0], color="black", linewidth=0.7, linestyle=":", alpha=0.6)
ax1.set_ylim(0, max(rewards) * 1.15)
ax1.set_title("Reward (post-fix multi-seed)", fontsize=11)
ax1.grid(axis="y", alpha=0.3)

# Annotate Δ% vs full
for i in range(1, len(rewards)):
    delta_pct = (rewards[i] - rewards[0]) / rewards[0] * 100
    ax1.annotate(f"{delta_pct:+.1f}\\%", (x[i], rewards[i] + rstds[i]),
                 ha="center", va="bottom", fontsize=8.5)

# ── Violation bars ───────────────────────────────────────────────────────
ax2.bar(x, viols, color=colors, edgecolor="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
ax2.set_ylabel("Violations / episode (3-seed mean)", fontsize=11)
ax2.axhline(viols[0], color="black", linewidth=0.7, linestyle=":", alpha=0.6)
ax2.set_ylim(0, max(viols) * 1.15)
ax2.set_title("Violations (post-fix multi-seed)", fontsize=11)
ax2.grid(axis="y", alpha=0.3)

for i in range(1, len(viols)):
    fold = viols[i] / max(viols[0], 1e-3)
    ax2.annotate(f"{fold:.1f}$\\times$", (x[i], viols[i]),
                 ha="center", va="bottom", fontsize=9, color="black")

plt.tight_layout()
plt.savefig(HERE / "ablation_chart_multi.pdf", bbox_inches="tight")
plt.savefig(HERE / "ablation_chart_multi.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'ablation_chart_multi.pdf'}")
