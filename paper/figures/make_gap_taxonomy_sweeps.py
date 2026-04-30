"""
E5 single-factor gap-taxonomy sweeps (HalfCheetah morphology only).

Three single-axis sweeps share the same morphology, soft-ceiling, and
RAHD c9 pipeline; only one factor varies per panel.  Panels:
    (a) physics gap  — gravity factor ∈ {1.5, 2.0, 3.0, 4.0}
    (b) observation  — obs-noise std ∈ {0.5, 1.0, 2.0, 5.0}
    (c) actuator     — actuator scale ∈ {0.5, 1.0, 2.0}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

DATA = {
    "gravity": [
        (1.5, 5327, 402, 20.79),
        (2.0, 5594, 299,  1.76),
        (3.0, 5068, 230,  6.74),
        (4.0, 4936, 300,  0.51),
    ],
    "obs_noise": [
        (0.5, 3358, 235, 14.38),
        (1.0,   94, 114,  0.06),
        (2.0,  -28,  81,  0.99),
        (5.0, -277,  70,  2.49),
    ],
    "actuator": [
        (0.5, 5006, 342,  5.89),
        (1.0, 5251, 265,  1.53),
        (2.0, 5001, 477,  1.83),
    ],
}

REF_FULL_RAHD = 5394   # full RAHD c9 on gravity_soft_ceiling, post-fix
REF_RAW_SIM   = 494    # c1 on gravity_soft_ceiling

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)

PALETTE = {"gravity": "#264653", "obs_noise": "#E76F51", "actuator": "#8DB580"}
LABELS = {
    "gravity":  ("Gravity factor (sim/real)",  "(a) Physics gap — works"),
    "obs_noise": ("Obs noise std on velocities", "(b) Observation gap — fails"),
    "actuator": ("Actuator scale (sim)",         "(c) Actuator gap — robust"),
}

for ax, (key, pts) in zip(axes, DATA.items()):
    xs   = np.array([p[0] for p in pts])
    mus  = np.array([p[1] for p in pts])
    stds = np.array([p[2] for p in pts])
    color = PALETTE[key]

    ax.errorbar(xs, mus, yerr=stds, marker="o", linewidth=2.0,
                color=color, ecolor=color, elinewidth=1.0,
                capsize=4, zorder=4,
                label="MC-WM RAHD c9 (3 seeds)")

    # numeric labels
    for x, m in zip(xs, mus):
        ax.annotate(f"{m:.0f}", (x, m), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=8, color=color)

    ax.axhline(REF_FULL_RAHD, color="#264653", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"full c9 RAHD: {REF_FULL_RAHD}")
    ax.axhline(REF_RAW_SIM, color="#999", linestyle=":", linewidth=0.8,
               alpha=0.7, label=f"raw sim: {REF_RAW_SIM}")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)

    xlabel, title = LABELS[key]
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel("Return (3-seed mean $\\pm$ std)", fontsize=10)
    ax.set_ylim(-700, 6700)
    if key == "gravity":
        ax.set_xticks([1.5, 2.0, 3.0, 4.0])
    elif key == "obs_noise":
        ax.set_xticks([0.5, 1.0, 2.0, 5.0])
    else:
        ax.set_xticks([0.5, 1.0, 2.0])

axes[0].legend(fontsize=8, loc="lower left", framealpha=0.92)

plt.tight_layout()
plt.savefig(HERE / "gap_taxonomy_sweeps.pdf", bbox_inches="tight")
plt.savefig(HERE / "gap_taxonomy_sweeps.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'gap_taxonomy_sweeps.pdf'}")
