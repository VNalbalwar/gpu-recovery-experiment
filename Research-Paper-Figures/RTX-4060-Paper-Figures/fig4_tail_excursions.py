"""
Figure 4: Extended L2 experiment: tail excursions are trial-dependent

Source: extended_l2_trial_metrics.csv
  gt1_2 = fraction of recovery probes with normalized_latency > 1.2x baseline
  converted to percentage. One value per trial (10 trials x 2 conditions).

Boxplot per condition (Control, L2 interference) with every individual
trial overlaid as a point (n=10 per condition, all shown).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude")
from style import apply_style

apply_style()

df = pd.read_csv("/mnt/user-data/uploads/extended_l2_trial_metrics.csv")
df["gt1_2_pct"] = df["gt1_2"] * 100.0

label_map = {"control": "Control", "interference": "L2 interference"}
df["condition_label"] = df["condition"].map(label_map)

order = ["Control", "L2 interference"]
data = [df.loc[df["condition_label"] == c, "gt1_2_pct"].values for c in order]
colors = {"Control": "#4c72b0", "L2 interference": "#c44e52"}

fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)

bp = ax.boxplot(
    data,
    positions=[1, 2],
    widths=0.42,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.3),
    boxprops=dict(linewidth=0.9),
    whiskerprops=dict(linewidth=0.9),
    capprops=dict(linewidth=0.9),
)
for patch, c in zip(bp["boxes"], order):
    patch.set_facecolor(colors[c])
    patch.set_alpha(0.18)
    patch.set_edgecolor(colors[c])

rng = np.random.default_rng(0)
for i, c in enumerate(order, start=1):
    vals = df.loc[df["condition_label"] == c, "gt1_2_pct"].values
    jitter = rng.uniform(-0.10, 0.10, size=len(vals))
    ax.scatter(
        np.full(len(vals), i) + jitter,
        vals,
        color=colors[c],
        edgecolor="white",
        linewidth=0.4,
        s=28,
        zorder=3,
        label=f"{c} trials (n={len(vals)})",
    )

ax.axhline(0, color="0.6", linewidth=0.7, zorder=0)

ax.set_xticks([1, 2])
ax.set_xticklabels(order)
ax.set_ylabel("Recovery probes > 1.2\u00d7 baseline per trial (%)")
ax.set_title("Extended L2 experiment: tail excursions are trial-dependent", loc="left", fontsize=10.5)
ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"/home/claude/figs/fig4_tail_excursions.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight")

plt.close(fig)
print("Figure 4 done")
