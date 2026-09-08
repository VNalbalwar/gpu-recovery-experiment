import matplotlib as mpl

def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.grid": True,
        "grid.linestyle": "-",
    })

# Consistent naming/colors across figures
RESOURCE_COLORS = {
    "Memory": "#1f77b4",
    "L2": "#d62728",
    "Compute": "#2ca02c",
}
CONDITION_COLORS = {
    "control": "#4c72b0",
    "interference": "#c44e52",
    "Control": "#4c72b0",
    "L2 interference": "#c44e52",
}
BASELINE_STYLE = dict(color="0.35", linestyle="--", linewidth=1.0, zorder=1)
