"""
Re-fit the gap-taxonomy diagnostic threshold with confidence interval.

Plots: residual-improvement % (gap_state) vs MC-WM RAHD reward, with
points colored by gap type.  Logistic-regression decision boundary
fitted via bootstrap (2000 iterations).

Source data: 11 single-factor sweep envs (E5) + 5 cross-morphology envs
from the original taxonomy table.  Binary label "works" iff post-fix
RAHD reward exceeds 1.5× the raw-sim baseline (c1).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

D = json.loads(open("/tmp/threshold_fit.json").read())
points = D["points"]
a, b, thr = D["fit"]["a"], D["fit"]["b"], D["fit"]["threshold"]
ci = D["bootstrap_ci"]

# Classify each point by gap type for color coding.
def classify(env: str) -> str:
    if env.startswith("gravity") or env in {"gravity_ceiling", "gravity_soft_ceiling"}:
        return "physics"
    if env.startswith("obs_noise"):
        return "obs"
    if env.startswith("actuator"):
        return "actuator"
    if env.startswith("friction"):
        return "physics"
    if env.startswith("carpet") or env.startswith("ant_wall"):
        return "obs_or_actuator"  # original taxonomy negatives
    return "other"

COLORS = {"physics":"#264653", "obs":"#E76F51",
          "actuator":"#8DB580", "obs_or_actuator":"#E9C46A", "other":"#999"}

fig, ax = plt.subplots(figsize=(7.0, 4.4))

# CI band from bootstrap percentiles
ax.axvspan(ci["25"], ci["75"], color="#E9ECEF", alpha=0.6, zorder=0,
           label=f"IQR threshold: [{ci['25']:.2f}, {ci['75']:.2f}]")
ax.axvline(thr, color="#444", linestyle="--", linewidth=1.2, zorder=1,
           label=f"point estimate: {thr:.2f}")
ax.axvline(0.35, color="#999", linestyle=":", linewidth=1.0, zorder=1,
           label="prior heuristic: 0.35")

for p in points:
    cls = classify(p["env"])
    color = COLORS[cls]
    marker = "o" if p["works"] else "X"
    ax.scatter([p["gap"]], [p["reward"]], color=color, marker=marker,
               s=80, edgecolors="black", linewidths=0.8, zorder=4,
               label=None)
    # short label
    short = (p["env"]
             .replace("gravity_sweep_", "g")
             .replace("obs_noise_", "obs")
             .replace("actuator_scale_", "act")
             .replace("_soft_ceiling", "_sc")
             .replace("gravity_ceiling", "g_ceil")
             .replace("gravity_soft_ceiling", "g_soft")
             .replace("friction_walker_sc", "fric_w")
             .replace("carpet_ant_sc", "carpet")
             .replace("ant_wall_broken_sc", "ant_wall"))
    ax.annotate(short, (p["gap"], p["reward"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#444")

# Reward = 0 reference
ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
# Raw-sim band (HalfCheetah)
ax.axhline(494, color="#999", linestyle=":", linewidth=0.6, alpha=0.6)
ax.text(1.0, 540, "c1 raw sim (HalfCheetah)", ha="right", fontsize=7,
        color="#999")

# Legend (one per type, plus thresholds)
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([], [], marker='o', color="w", markerfacecolor="#264653",
                  markersize=8, label="physics gap (works)", markeredgecolor="black"),
    mlines.Line2D([], [], marker='X', color="w", markerfacecolor="#E76F51",
                  markersize=8, label="obs gap (fails ≥σ=1)", markeredgecolor="black"),
    mlines.Line2D([], [], marker='o', color="w", markerfacecolor="#8DB580",
                  markersize=8, label="actuator gap (robust)", markeredgecolor="black"),
    mlines.Line2D([], [], marker='X', color="w", markerfacecolor="#E9C46A",
                  markersize=8, label="cross-morphology negs", markeredgecolor="black"),
    mlines.Line2D([], [], color="#444", linestyle="--",
                  label=f"fitted threshold {thr:.2f}"),
    mlines.Line2D([], [], color="#999", linestyle=":",
                  label="prior heuristic 0.35"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
          framealpha=0.92)

ax.set_xlabel("Residual improvement on real data " r"$(\mathrm{RMSE}_{\mathrm{sim}\to\mathrm{real}}-\mathrm{RMSE}_{\mathrm{real}})\,/\,\mathrm{RMSE}_{\mathrm{sim}\to\mathrm{real}}$",
              fontsize=10)
ax.set_ylabel("Post-fix RAHD c9 return", fontsize=11)
ax.set_title("Re-fit of the gap-taxonomy threshold (16 envs, 3 seeds each)",
             fontsize=11)
ax.set_xlim(-0.05, 1.0)
ax.set_ylim(-700, 6500)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(HERE / "gap_taxonomy_threshold_fit.pdf", bbox_inches="tight")
plt.savefig(HERE / "gap_taxonomy_threshold_fit.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'gap_taxonomy_threshold_fit.pdf'}")
print(f"\nSummary:")
print(f"  Point estimate threshold: {thr:.3f} ({thr*100:.1f}%)")
print(f"  Bootstrap 95% CI: [{ci['5']:.3f}, {ci['95']:.3f}]")
print(f"  Bootstrap IQR:    [{ci['25']:.3f}, {ci['75']:.3f}]")
print(f"  Prior heuristic:  0.35 (within IQR)")
