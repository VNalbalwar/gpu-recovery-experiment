"""
Figure 1: Victim slowdown during resource interference

Source: duration_run_metrics.csv, rows where phase == "interference"

x = duration_ms (log scale)
y = median (run-level median latency / baseline median)

Grouped by resource: Memory, L2, Compute.
Each CSV run is one experimental observation -> shown as an individual
open-marker scatter point (small, low alpha). Where multiple runs exist
at a duration, the solid line + filled marker connects the median-of-
run-medians at that duration (a second-order aggregate, not a fitted
trend). Durations with only one run get a single point on the line --
no fabricated error bars.

Presentation-only revision: distinguishes individual-run points (hollow,
small) from the per-duration median line (filled, larger) with an
explicit legend entry, per reviewer request. No values, aggregation, or
statistics changed from the prior version.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
sys.path.insert(0, "/home/claude")
from style import apply_style, RESOURCE_COLORS, BASELINE_STYLE

apply_style()

df = pd.read_csv("/home/claude/duration_run_metrics.csv")
df = df.loc[df["phase"] == "interference"].copy()

resources = ["Memory", "L2", "Compute"]
markers = {"Memory": "o", "L2": "s", "Compute": "^"}

fig, ax = plt.subplots(figsize=(6.6, 4.4))

for res in resources:
    sub = df.loc[df["resource"] == res]
    if sub.empty:
        continue

    # individual run-level observations: hollow, small, semi-transparent
    ax.scatter(
        sub["duration_ms"], sub["median"],
        facecolor="none", edgecolor=RESOURCE_COLORS[res],
        marker=markers[res], s=34, linewidth=1.0, alpha=0.75, zorder=2,
    )

    # central tendency: median across runs at each duration (solid, filled)
    line = sub.groupby("duration_ms")["median"].median().sort_index()
    ax.plot(
        line.index, line.values,
        color=RESOURCE_COLORS[res], marker=markers[res],
        markerfacecolor=RESOURCE_COLORS[res], markeredgecolor="white",
        markeredgewidth=0.6, markersize=6, linewidth=1.8, zorder=3,
        label=f"{res} \u2014 median across runs",
    )

ax.axhline(1.0, **BASELINE_STYLE, label="Baseline (1.0\u00d7)")

ax.set_xscale("log")
ax.set_xticks([5, 10, 25, 50, 100])
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Interference duration (ms)")
ax.set_ylabel("Run median latency / baseline median")
ax.set_title("Victim slowdown during resource interference", loc="left", fontsize=10.8)

# Explicit generic legend entry for the hollow "individual run" marker,
# in addition to the per-resource median-line entries above.
handles, labels = ax.get_legend_handles_labels()
individual_run_handle = mlines.Line2D(
    [], [], marker="o", linestyle="None", markerfacecolor="none",
    markeredgecolor="0.35", markeredgewidth=1.0, markersize=6,
    label="Individual runs (single experimental observation)",
)
handles = [individual_run_handle] + handles
labels = ["Individual runs (single experimental observation)"] + labels
ax.legend(handles, labels, loc="upper left", fontsize=8.2)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"/home/claude/figs/fig1_interference_slowdown.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight")

plt.close(fig)
print("Figure 1 done")
