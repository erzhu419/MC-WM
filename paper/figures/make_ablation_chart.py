"""
Figure: ablation study as a bar chart, showing reward drop vs violation
increase for each removed component.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# (config_label, reward, violation) on gravity_soft_ceiling s42.
# All values from actual experiment logs (qdg50 runs + ablation_c2_baseline_as_c9_s42).
# Full c9 (qdg50, s42): 6611 reward, 15.3 viol (qdelta_gamma=0.50)
# no Bellman QΔ  (qdelta_gamma=0, s42): 4700 reward, 16.0 viol  ← new real run
data = [
    ("Full c9",                   6611, 15.3,  "#264653"),
    ("no Pareto gate",            5930,  3.4,  "#8DB580"),
    ("no Bellman Q$^{\\Delta}$",  4700, 16.0,  "#E76F51"),
    ("no Role \\#5 (HP)",         5580,  3.5,  "#F4A261"),
    ("no LLM (Roles 1--3)",       5417,  2.3,  "#E9C46A"),
]

labels  = [d[0] for d in data]
rewards = [d[1] for d in data]
viols   = [d[2] for d in data]
colors  = [d[3] for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

# ── Reward bars ──────────────────────────────────────────────────────────
x = np.arange(len(labels))
bars = ax1.bar(x, rewards, color=colors, edgecolor="black", linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
ax1.set_ylabel("Return (last-3 avg)", fontsize=11)
ax1.axhline(rewards[0], color="black", linewidth=0.7, linestyle=":", alpha=0.6)
ax1.set_ylim(0, 7000)
ax1.set_title("Reward", fontsize=11)
ax1.grid(axis="y", alpha=0.3)

# Annotate deltas
for i in range(1, len(rewards)):
    delta_pct = (rewards[i] - rewards[0]) / rewards[0] * 100
    ax1.annotate(f"{delta_pct:+.0f}\\%", (x[i], rewards[i]),
                 ha="center", va="bottom", fontsize=9, color="black")

# ── Violation bars ───────────────────────────────────────────────────────
bars2 = ax2.bar(x, viols, color=colors, edgecolor="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
ax2.set_ylabel("Violations / episode", fontsize=11)
ax2.axhline(viols[0], color="black", linewidth=0.7, linestyle=":", alpha=0.6)
ax2.set_ylim(0, 20)
ax2.set_title("Violations", fontsize=11)
ax2.grid(axis="y", alpha=0.3)

for i in range(1, len(viols)):
    fold = viols[i] / viols[0]
    ax2.annotate(f"{fold:.1f}$\\times$", (x[i], viols[i]),
                 ha="center", va="bottom", fontsize=9, color="black")

plt.tight_layout()
plt.savefig(HERE / "ablation_chart.pdf", bbox_inches="tight")
plt.savefig(HERE / "ablation_chart.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'ablation_chart.pdf'}")
