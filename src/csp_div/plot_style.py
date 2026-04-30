"""Shared plot styling for CSP trajectory figures.

Conventions used by every trajectory plot in this repo:
  - Trajectories colored by basin: dipper (deep) blue, non-dipper (shallow + mid) red.
  - Open circle (○) at trajectory start, filled dot (●) at end — same color as the line.
  - Log scale on the KL axis, dotted reference line at zero.
  - Two-entry basin legend (with start/end key) instead of per-seed labels.

Visual style mirrors the bar-chart figures in github.com/kmaherx/csp:
  - Libertinus Serif (falls back to system serif if not installed).
  - White background, muted grey edges, only horizontal gridlines, no top/right spines.
  - Bold panel titles, frameless legends.

Importing from one place avoids style drift across the plotting scripts. The
rcParams below take effect on import.
"""
import json
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D


# ── Paper-style rcParams (applied at import time) ──────────────────────

EDGE_COLOR = "#888888"
WHITE = "#FFFFFF"

# Try to register Libertinus Serif from common system locations, then fall back
# to whatever matplotlib's default serif is. The figure is well-styled either
# way; the font is the only difference.
_FONT_NAME = "Libertinus Serif"
_FONT_DIRS = [
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts/truetype/libertinus",
    "/usr/local/share/fonts/libertinus",
    os.path.expanduser("~/.fonts/libertinus"),
]
for _dir in _FONT_DIRS:
    if not os.path.isdir(_dir):
        continue
    for _fn in os.listdir(_dir):
        if _fn.startswith("LibertinusSerif") and _fn.lower().endswith((".otf", ".ttf")):
            try:
                fm.fontManager.addfont(os.path.join(_dir, _fn))
            except Exception:
                pass

_have_libertinus = any(f.name == _FONT_NAME for f in fm.fontManager.ttflist)

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ([_FONT_NAME] if _have_libertinus else []) + [
    "DejaVu Serif", "Times New Roman", "Times", "serif",
]
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["axes.edgecolor"] = EDGE_COLOR
mpl.rcParams["axes.labelcolor"] = "#222222"
mpl.rcParams["xtick.color"] = "#444444"
mpl.rcParams["ytick.color"] = "#444444"
mpl.rcParams["xtick.major.width"] = 0.8
mpl.rcParams["ytick.major.width"] = 0.8
mpl.rcParams["figure.facecolor"] = WHITE
mpl.rcParams["axes.facecolor"] = WHITE
mpl.rcParams["savefig.facecolor"] = WHITE
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.25
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 200


# ── Basin classification ────────────────────────────────────────────────
DEEP_THRESHOLD = -0.5
SHALLOW_THRESHOLD = -0.4

DIPPER_COLOR = "tab:blue"
NONDIPPER_COLOR = "tab:red"
HIGHLIGHT_DEEP = "#c0392b"     # bolder red for "look here" deep example
HIGHLIGHT_SHALLOW = "#1f77b4"  # bolder blue for "look here" shallow example

# Line + endpoint defaults — sized for the compact paper-scale figures
# (~5–6" wide single-panel; ~10" wide 1×2). If you scale the figure up,
# bump these proportionally.
LINE_WIDTH = 1.0
LINE_ALPHA = 0.55
HIGHLIGHT_LW = 2.0
HIGHLIGHT_ALPHA = 0.95

START_SIZE = 18          # open ring
END_SIZE = 30            # filled dot
ENDPOINT_LW = 1.0
EMPHASIS_SIZE = 110      # for "look here" annotated step in figure_basins-style plots


def basin_from_cos(cos: float) -> str:
    """Bucket a cos value into deep / mid / shallow."""
    if cos <= DEEP_THRESHOLD:
        return "deep"
    if cos > SHALLOW_THRESHOLD:
        return "shallow"
    return "mid"


def basin_color(basin: str) -> str:
    """Visual binary: deep = dipper blue; mid + shallow = non-dipper red."""
    return DIPPER_COLOR if basin == "deep" else NONDIPPER_COLOR


def trajectory_basin(traj, min_step: int = 5) -> str:
    """Classify a trajectory by its deepest cos at step >= min_step.

    `traj` is a list of (kl, cos, step) tuples (the format returned by
    load_axis_trajectories). Steps below min_step are ignored so the
    random-init point doesn't artificially classify a trajectory.
    """
    in_train = [t for t in traj if t[2] >= min_step]
    if not in_train:
        return "shallow"
    return basin_from_cos(min(t[1] for t in in_train))


# ── Axis.json loader ────────────────────────────────────────────────────

def load_axis_trajectories(axis_path):
    """Read an axis.json and return ({seed: [(kl, cos, step), ...]}, layer).

    Trajectories are sorted by step. Seed is parsed from the trailing
    'seed_<N>' fragment of the row's group field.
    """
    with open(axis_path) as f:
        d = json.load(f)
    rows = d["rows"]
    layer = d["layer"]
    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].split("_")[-1])
        by_seed.setdefault(seed, []).append((r["kl"], r["proj_cos"], r["step"]))
    for s in by_seed:
        by_seed[s].sort(key=lambda x: x[2])
    return by_seed, layer


def find_point(traj, step):
    """Return (kl, cos) for the trajectory point at the given step."""
    for kl, cos, st in traj:
        if st == step:
            return kl, cos
    raise ValueError(f"step {step} not in trajectory (have {[t[2] for t in traj]})")


# ── Plotting primitives ────────────────────────────────────────────────

def draw_trajectory(ax, xs, ys, color, *, lw=LINE_WIDTH, alpha=LINE_ALPHA,
                    zorder=2, label=None):
    """Draw the line for a single trajectory."""
    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
            zorder=zorder, label=label)


def draw_endpoints(ax, xs, ys, color, *, zorder=5,
                   start_size=START_SIZE, end_size=END_SIZE):
    """Open circle at start, filled dot at end — both in `color`."""
    ax.scatter([xs[0]], [ys[0]], s=start_size,
               facecolor="white", edgecolor=color,
               linewidth=ENDPOINT_LW, zorder=zorder)
    ax.scatter([xs[-1]], [ys[-1]], s=end_size,
               facecolor=color, edgecolor=color,
               linewidth=ENDPOINT_LW, zorder=zorder)


def draw_emphasis(ax, x, y, color, *, zorder=6, size=EMPHASIS_SIZE):
    """A bigger filled marker with a black outline — for highlighting a
    specific step (e.g. the one a behavior-example came from)."""
    ax.scatter([x], [y], s=size, facecolor=color,
               edgecolor="black", linewidth=2.0, zorder=zorder)


# ── Axis cosmetics ─────────────────────────────────────────────────────

def _strip_spines(ax):
    """Remove top + right spines; trim the remaining ones to the data range feel."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)


def style_kl_axis(ax, *, layer=None, ylabel=None, xlabel="KL ↑ (log)"):
    """Standard styling for cos-vs-log-KL trajectory plots."""
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if ylabel is None and layer is not None:
        ylabel = f"cos(L{layer} shift, assistant axis)"
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    # Keep the x-grid on (log-KL is continuous — vertical gridlines help read
    # off coordinates). The bar-chart style this is borrowed from drops it
    # because its x is categorical.
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4, which="both")
    ax.set_axisbelow(True)
    _strip_spines(ax)


def style_pc_axis(ax, *, x_label="PC1", y_label="PC2"):
    """Standard styling for PC×PC scatter plots."""
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.5, linestyle=":", alpha=0.6)
    ax.axvline(0, color=EDGE_COLOR, linewidth=0.5, linestyle=":", alpha=0.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    _strip_spines(ax)


# ── Legend ─────────────────────────────────────────────────────────────

def basin_legend(ax, n_dippers: int, n_nondippers: int, *,
                 loc="best", extra_handles=None):
    """Two-entry population legend with a start/end glyph key.

    Labels are intentionally noncommittal ("Population 1/2") because the
    PCA section of the writeup discovers the two clusters before naming
    them as persona / format basins. Counts are surfaced in the label
    but kept compact.

    Pass extra_handles=[Line2D(...), ...] to append (e.g. for emphasis-
    dot examples in figure_basins / figure_cross_condition).
    """
    handles = [
        Line2D([0], [0], color=DIPPER_COLOR, linewidth=LINE_WIDTH + 0.6,
               label=f"Population 1  (n={n_dippers})"),
        Line2D([0], [0], color=NONDIPPER_COLOR, linewidth=LINE_WIDTH + 0.6,
               label=f"Population 2  (n={n_nondippers})"),
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="white", markeredgecolor=EDGE_COLOR,
               markeredgewidth=ENDPOINT_LW, markersize=5,
               label="○ start    ● end", linestyle=""),
    ]
    if extra_handles:
        handles.extend(extra_handles)
    return ax.legend(handles=handles, loc=loc, fontsize=8, frameon=False)


def panel_title(ax, text):
    """Bold panel title in the paper style."""
    ax.set_title(text, fontsize=12, fontweight="bold", color="#222222")
