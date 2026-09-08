"""
Figure 5: Extended L2 recovery: tail behavior over the post-interference window

Source: hr_l2_extended.csv, condition == "interference" only (no temporal
control curve, since control trials have no corresponding interference-end
event).

Bins (us): [0,100), [100,250), [250,500), [500,1000), [1000,1500),
           [1500,2000), [2000,3000)
Per bin: median and P95 of normalized_latency across all interference-trial
recovery probes falling in that bin (pooled probe-level statistic, not a
recovery curve).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude")
from style import apply_style, BASELINE_STYLE

apply_style()

df = pd.read_csv("/mnt/user-data/uploads/hr_l2_extended.csv")
df = df.loc[df["condition"] == "interference"].copy()

bin_edges = [0, 100, 250, 500, 1000, 1500, 2000, 3000]
bin_labels = [
    "0\u2013100", "100\u2013250", "250\u2013500", "500\u20131000",
    "1000\u20131500", "1500\u20132000", "2000\u20133000",
]

df["bin"] = pd.cut(
    df["time_since_interference_end_us"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
    include_lowest=True,
)

agg = df.groupby("bin", observed=True)["normalized_latency"].agg(
    median="median",
    p95=lambda s: np.percentile(s, 95),
    n="count",
).reindex(bin_labels)

# Equally-spaced categorical x positions: bin widths are highly uneven
# (100 us to 1000 us), so plotting at true time coordinates crowds the
# narrow early bins. Positions here are ordinal (bin index), not a linear
# time axis -- this is a categorical/binned plot, not a continuous curve.
x_pos = list(range(len(bin_labels)))

fig, ax = plt.subplots(figsize=(6.6, 4.8))

ax.plot(x_pos, agg["median"].values, marker="o", color="#c44e52",
        label="Bin median", linewidth=1.6)
ax.plot(x_pos, agg["p95"].values, marker="s", color="#8172b2",
        linestyle="--", linewidth=1.4, label="Bin P95")

ax.axhline(1.0, **BASELINE_STYLE, label="Baseline (1.0\u00d7)")

ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, rotation=0, fontsize=8.5)
ax.set_xlabel("Time since interference end (\u00b5s) \u2014 bin")
ax.set_ylabel("Normalized victim latency (\u00d7 baseline median)")

fig.suptitle(
    "Extended L2 recovery: tail behavior over the post-interference window",
    x=0.02, y=0.985, ha="left", fontsize=10.3,
)
fig.text(
    0.02, 0.925,
    "Binned post-interference latency statistics; values characterize tail "
    "variability rather than a continuous recovery trajectory.",
    ha="left", va="top", fontsize=7.8, color="0.35", style="italic",
)

import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for x, n in zip(x_pos, agg["n"].values):
    ax.text(x, -0.14, f"n={n}", transform=trans, ha="center", va="top",
            fontsize=7.3, color="0.45")
fig.text(
    0.5, 0.015,
    "n = recovery probes per bin (L2 interference trials only)",
    ha="center", va="bottom", fontsize=7.3, color="0.45", style="italic",
)
ax.set_xlim(-0.5, len(bin_labels) - 0.5)

ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
ax.legend(loc="upper right", fontsize=8.5, ncol=1)
fig.subplots_adjust(top=0.86, bottom=0.18, left=0.13, right=0.97)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"/home/claude/figs/fig5_temporal_tail.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight")

plt.close(fig)
print(agg)
print("Figure 5 done")
