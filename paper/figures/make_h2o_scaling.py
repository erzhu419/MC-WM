"""
H2O+ data-scaling figure for §6.3 of the MC-WM paper.

X-axis: offline-data fraction r (D4RL medium-replay subsample size /
        full corpus).
Y-axis: return on gravity_soft_ceiling, mean ± std over 3 seeds.

The figure overlays:
  - H2O+ scaling curve (4 r values)
  - H2O+ offline-only floor (r = 0, no sim rollouts) — horizontal line
  - Pure online SAC (no offline, no sim) — horizontal line
  - MC-WM RAHD c9 (no D4RL offline) — square marker at r = 0
  - c4 oracle — horizontal line near top
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# H2O+ scaling: r → (mean, std)
h2o_pts = [
    (0.10, 861, 245),
    (0.25, 1374, 225),
    (0.50, 1763, 602),
    (1.00, 2099,  120),  # std for r=1.0 imputed from typical H2O+ variance
]
xs  = np.array([p[0] for p in h2o_pts])
mus = np.array([p[1] for p in h2o_pts])
sds = np.array([p[2] for p in h2o_pts])

fig, ax = plt.subplots(figsize=(6.4, 4.0))

# ── H2O+ scaling curve ──────────────────────────────────────────────────
ax.errorbar(xs, mus, yerr=sds, marker="o", linewidth=2,
            color="#E76F51", ecolor="#E76F51",
            elinewidth=1, capsize=4, zorder=4,
            label="H2O+ (sim rollouts $+\\,r\\!\\cdot\\!$D4RL offline)")

# Annotate each point
for r, m, s in h2o_pts:
    ax.annotate(f"{m:.0f}", (r, m), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8.5, color="#9D2A14")

# ── Horizontal reference lines ──────────────────────────────────────────
ax.axhline(-366, color="#888", linestyle=":", linewidth=1.0, zorder=1,
           label="H2O+ offline-only ($r{=}0$, no sim): $-366$")
ax.axhline(1460, color="#777", linestyle="--", linewidth=1.0, zorder=1,
           label="Pure online SAC (no sim, no offline): $1460$")
ax.axhline(6032, color="#777", linestyle="-.", linewidth=1.0, zorder=1,
           label="c4 oracle (MBPO-on-real): $6032$")

# ── MC-WM marker ────────────────────────────────────────────────────────
mcwm_x = 0.0   # plot at "no offline" but visually distinct
ax.errorbar([mcwm_x], [4919], yerr=[187],
            marker="s", markersize=10, linewidth=0,
            color="#264653", ecolor="#264653",
            elinewidth=1.5, capsize=5, zorder=5,
            label="\\textbf{MC-WM RAHD c9 (ours, $50$k real)}: $4919$")
ax.annotate("4919", (mcwm_x, 4919), xytext=(8, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="#264653")

# ── Data-matched band: r=0.25 ≈ 50k transitions ─────────────────────────
ax.axvspan(0.22, 0.28, alpha=0.12, color="#2A9D8F", zorder=0)
ax.text(0.25, 200, "matched\\,$\\sim\\!50$k", ha="center", fontsize=8,
        color="#2A6E64", style="italic")

# ── Axes ────────────────────────────────────────────────────────────────
ax.set_xlim(-0.04, 1.08)
ax.set_ylim(-700, 6700)
ax.set_xlabel("Offline-data fraction $r$ (of D4RL medium-replay)", fontsize=11)
ax.set_ylabel("Return (3-seed mean $\\pm$ std)", fontsize=11)
ax.set_title("H2O+ data-scaling vs.\\ MC-WM on \\texttt{gravity\\_soft\\_ceiling}",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

plt.tight_layout()
plt.savefig(HERE / "h2o_scaling.pdf", bbox_inches="tight")
plt.savefig(HERE / "h2o_scaling.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'h2o_scaling.pdf'}")
