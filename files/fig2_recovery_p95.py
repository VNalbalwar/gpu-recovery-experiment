"""
Figure 2: Post-interference P95 latency across duration experiments

Source: duration_run_metrics.csv, rows where phase == "recovery"

x = duration_ms (log scale)
y = p95 (run-level P95 latency / baseline median)

Grouped by resource: Memory, L2, Compute.
Each run-level P95 is an individual observation (hollow scatter point).
The line connects the median P95 across runs at each duration -- this is
NOT a confidence interval, and recovery samples across runs are not
pooled into one distribution.

Presentation-only revision: same hollow-marker-vs-solid-line convention
as Figure 1, with explicit "median P95 across runs" wording in the
legend per reviewer request. No values, aggregation, or statistics
changed from the prior version.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
sys.path.insert(0, "/home/claude")
from style import apply_style, RESOURCE_COLORS, BASELINE_STYLE

apply_style()

df = pd.read_csv("/home/claude/duration_run_metrics.csv")
df = df.loc[df["phase"] == "recovery"].copy()

resources = ["Memory", "L2", "Compute"]
markers = {"Memory": "o", "L2": "s", "Compute": "^"}

fig, ax = plt.subplots(figsize=(6.6, 4.4))

for res in resources:
    sub = df.loc[df["resource"] == res]
    if sub.empty:
        continue

    # individual run-level P95 values: hollow, small
    ax.scatter(
        sub["duration_ms"], sub["p95"],
        facecolor="none", edgecolor=RESOURCE_COLORS[res],
        marker=markers[res], s=34, linewidth=1.0, alpha=0.75, zorder=2,
    )

    # median P95 across runs at each duration (solid, filled)
    line = sub.groupby("duration_ms")["p95"].median().sort_index()
    ax.plot(
        line.index, line.values,
        color=RESOURCE_COLORS[res], marker=markers[res],
        markerfacecolor=RESOURCE_COLORS[res], markeredgecolor="white",
        markeredgewidth=0.6, markersize=6, linewidth=1.8, zorder=3,
        label=f"{res} \u2014 median P95 across runs",
    )

ax.axhline(1.0, **BASELINE_STYLE, label="Baseline (1.0\u00d7)")

ax.set_xscale("log")
ax.set_xticks([5, 10, 25, 50, 100])
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Interference duration (ms)")
ax.set_ylabel("Recovery P95 / baseline median")
ax.set_title("Post-interference P95 latency across duration experiments",
             loc="left", fontsize=10.5)

handles, labels = ax.get_legend_handles_labels()
individual_run_handle = mlines.Line2D(
    [], [], marker="o", linestyle="None", markerfacecolor="none",
    markeredgecolor="0.35", markeredgewidth=1.0, markersize=6,
    label="Individual runs (single run-level P95)",
)
handles = [individual_run_handle] + handles
labels = ["Individual runs (single run-level P95)"] + labels
ax.legend(handles, labels, loc="upper left", fontsize=8.2)

ymax = max(df["p95"].max() * 1.08, 1.15)
ax.set_ylim(0.98, ymax)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"/home/claude/figs/fig2_recovery_p95.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight")

plt.close(fig)
print("Figure 2 done")
