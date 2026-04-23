"""
Generate Figure 3: Gap Taxonomy.

Plots c9/c1 return ratio against residual improvement ratio.
Shows the ~35% threshold below which residual correction is not useful.

Data points (s42, final last-3 avg return; residual improvement from Phase 1b
logs; one representative value per env):

| env                           | c1   | c9   | c9/c1 | residual_imp |
| ------------------------------|------|------|-------|--------------|
| carpet_ant_soft_ceiling       | 884  | 782  |  0.88 | 0.55         |
| ant_wall_broken_soft_ceiling  | 1841 | 1728 |  0.94 | 0.08         |
| friction_walker_soft_ceiling  | 262  | 357  |  1.36 | 0.34         |
| gravity_ceiling               | 543  | 5196 |  9.57 | 0.72         |
| gravity_soft_ceiling          | 494  | 6288 | 12.73 | 0.57         |
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

data = [
    # (env_short, c1, c9, residual_imp, color, marker)
    ("carpet",    884,  782,  0.55, "#E63946", "x"),
    ("ant_wall",  1841, 1728, 0.08, "#E63946", "x"),
    ("friction",  262,  357,  0.34, "#F4A261", "s"),
    ("g_ceiling", 543,  5196, 0.72, "#2A9D8F", "o"),
    ("g_soft",    494,  6288, 0.57, "#2A9D8F", "o"),
]

fig, ax = plt.subplots(figsize=(6.5, 4))
for name, c1, c9, imp, color, mk in data:
    ratio = c9 / c1
    ax.scatter(imp, ratio, s=180, color=color, marker=mk, edgecolor="black",
               linewidth=1.0, zorder=3)
    # Offset labels to avoid overlap
    dx, dy = 0.01, 0.3
    if name == "g_ceiling":
        dy = 0.8
    if name == "g_soft":
        dy = -0.9
    if name == "friction":
        dy = 0.4
    if name == "ant_wall":
        dy = -0.25
    if name == "carpet":
        dy = -0.25
    ax.annotate(name, (imp + dx, ratio + dy), fontsize=9, ha="left")

# Horizontal line at ratio = 1.0 (break-even)
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
ax.text(0.95, 1.05, "break-even", fontsize=8, color="gray", ha="right")

# Vertical threshold at residual_imp = 0.35
ax.axvline(0.35, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(0.36, 11.5, "threshold ≈ 0.35", fontsize=9, color="black",
        style="italic", ha="left")

# Shaded regions
ax.axvspan(0.0, 0.35, alpha=0.08, color="red")
ax.axvspan(0.35, 1.0, alpha=0.08, color="green")

# Labels
ax.set_xlabel("Residual improvement on real data\n(RMSE$_{\\mathrm{sim}} - $RMSE$_{\\mathrm{real}}$) / RMSE$_{\\mathrm{sim}}$",
              fontsize=11)
ax.set_ylabel("c9 / c1 return ratio", fontsize=11)
ax.set_title("Gap Taxonomy: residual correction helps iff gap is large",
             fontsize=11)
ax.set_xlim(-0.05, 0.85)
ax.set_ylim(0, 14)
ax.grid(alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2A9D8F",
           markersize=10, label="physics-gap (ours wins)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#F4A261",
           markersize=10, label="moderate gap"),
    Line2D([0], [0], marker="x", color="#E63946", markersize=10,
           markeredgewidth=2, label="obs / actuator gap (ours hurts)"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(HERE / "gap_taxonomy.pdf", bbox_inches="tight")
plt.savefig(HERE / "gap_taxonomy.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'gap_taxonomy.pdf'} and .png")
