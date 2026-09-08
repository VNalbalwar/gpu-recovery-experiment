"""
Figure 3: High-resolution L2 recovery: trial-level behavior

Source: hr_l2_highres_100ms_r1.csv, r2.csv, r3.csv (raw, per-sample)

For each run independently:
  - baseline median computed from phase == "baseline" rows of THAT run
  - recovery samples (phase == "recovery") normalized by that run's own baseline median
  - plotted against time_since_interference_end_us as a distinct trace (no pooling)
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude")
from style import apply_style, BASELINE_STYLE

apply_style()

FILES = {
    "Trial 1": "/mnt/user-data/uploads/hr_l2_highres_100ms_r1.csv",
    "Trial 2": "/mnt/user-data/uploads/hr_l2_highres_100ms_r2.csv",
    "Trial 3": "/mnt/user-data/uploads/hr_l2_highres_100ms_r3.csv",
}

TRIAL_COLORS = {
    "Trial 1": "#1b9e77",
    "Trial 2": "#d95f02",
    "Trial 3": "#7570b3",
}
TRIAL_MARKERS = {"Trial 1": "o", "Trial 2": "s", "Trial 3": "^"}

fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)

for label, path in FILES.items():
    df = pd.read_csv(path)

    baseline = df.loc[df["phase"] == "baseline", "victim_latency_us"]
    baseline_median = baseline.median()

    recov = df.loc[df["phase"] == "recovery"].copy()
    recov["normalized_latency"] = recov["victim_latency_us"] / baseline_median
    recov = recov.sort_values("time_since_interference_end_us")

    ax.plot(
        recov["time_since_interference_end_us"],
        recov["normalized_latency"],
        marker=TRIAL_MARKERS[label],
        markersize=3.0,
        markerfacecolor="none",
        markeredgewidth=0.7,
        linewidth=0.9,
        alpha=0.85,
        color=TRIAL_COLORS[label],
        label=f"{label} (baseline median = {baseline_median:.2f} \u00b5s)",
    )

ax.axhline(1.0, **BASELINE_STYLE, label="Baseline (1.0\u00d7)")

ax.set_xlabel("Time since interference end (\u00b5s)")
ax.set_ylabel("Normalized victim latency (\u00d7 baseline median)")
ax.set_title("High-resolution L2 recovery: trial-level behavior", loc="left")
ax.legend(loc="upper right", fontsize=8)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"/home/claude/figs/fig3_hires_l2_trials.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight")

plt.close(fig)
print("Figure 3 done")
