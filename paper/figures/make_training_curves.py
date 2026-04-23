"""
Figure: training curves for c1 / c9 on the three main envs.

Parses /tmp/step2_{c1,c9}_{env}_s{seed}.log for 'step N | real= ...'
lines and plots mean ± std over seeds.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

STEP_RE = re.compile(r"step\s+(\d+)\s*\|\s*real=\s*([-\d.]+)")

def parse(path):
    """Returns dict {step_int: reward_float} from log file."""
    if not Path(path).exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                s = int(m.group(1)); r = float(m.group(2))
                out[s] = r
    return out


def mean_std(dicts, target_steps):
    """Mean and std across seeds at each target_step; None when missing."""
    means, stds = [], []
    for step in target_steps:
        vals = [d[step] for d in dicts if step in d]
        if len(vals) == 0:
            means.append(np.nan); stds.append(0.0)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    return np.array(means), np.array(stds)


envs = [
    ("gravity_ceiling",              "gravity\\_ceiling"),
    ("gravity_soft_ceiling",         "gravity\\_soft\\_ceiling"),
    ("friction_walker_soft_ceiling", "friction\\_walker\\_soft\\_ceiling"),
]

# ── assemble curves per env ────────────────────────────────────────────────
TARGET_STEPS = list(range(5000, 50001, 5000))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=False)

for ax, (env, env_label) in zip(axes, envs):
    # c1 curves (3 seeds; main log may count as s42)
    c1_logs = []
    for seed in ["", "_s123", "_s456"]:
        p = f"/tmp/step2_c1_{env}{seed}.log"
        d = parse(p)
        if d: c1_logs.append(d)

    # c9 curves
    c9_logs = []
    # Prefer qdg50 variants if they exist, else the base logs
    for seed_suffix in ["_qdg50", "_qdg50_s123", "_qdg50_s456"]:
        p = f"/tmp/step2_c9_{env}{seed_suffix}.log"
        d = parse(p)
        if d: c9_logs.append(d)
    if not c9_logs:
        for seed_suffix in ["", "_s123", "_s456"]:
            p = f"/tmp/step2_c9_{env}{seed_suffix}.log"
            d = parse(p)
            if d: c9_logs.append(d)

    # c4 oracle (only on gravity_soft + friction_walker)
    c4_logs = []
    for seed_suffix in ["", "_s123", "_s456"]:
        p = f"/tmp/step2_c4_{env}{seed_suffix}.log"
        d = parse(p)
        if d: c4_logs.append(d)

    ax.set_title(env_label, fontsize=10)
    ax.set_xlabel("Env step", fontsize=10)
    ax.set_xlim(5000, 50000)
    ax.grid(alpha=0.3)

    def plot_curve(logs, label, color):
        if not logs: return
        m, s = mean_std(logs, TARGET_STEPS)
        valid = ~np.isnan(m)
        x = np.array(TARGET_STEPS)[valid]
        ax.plot(x, m[valid], label=f"{label} (n={len(logs)})",
                color=color, linewidth=2)
        ax.fill_between(x, m[valid]-s[valid], m[valid]+s[valid],
                        color=color, alpha=0.2)

    plot_curve(c1_logs, "c1 (Raw Sim)",  "#E63946")
    plot_curve(c4_logs, "c4 (Oracle)",   "#F4A261")
    plot_curve(c9_logs, "c9 (Ours)",     "#2A9D8F")

    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("Return", fontsize=10)
plt.tight_layout()
plt.savefig(HERE / "training_curves.pdf", bbox_inches="tight")
plt.savefig(HERE / "training_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved {HERE/'training_curves.pdf'}")
