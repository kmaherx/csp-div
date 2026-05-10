"""Shared plot styling for CSP figures (both matplotlib and Plotly).

Matplotlib section preserves the conventions from the historical
`plot_style.py` — basin colors, two-population legend, log-KL axis style,
Libertinus Serif font (with fallback). rcParams are applied at module
import so any matplotlib figure produced after `import csp_div.plotting`
inherits the paper style.

Plotly section provides the colormaps + colorscale helpers used by
`5_dashboard.py` so the published dashboard doesn't reach into
matplotlib internals directly.
"""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib import colormaps


# ── Colors and thresholds ───────────────────────────────────────────────

EDGE_COLOR = "#888888"
WHITE = "#FFFFFF"

DEEP_THRESHOLD = -0.5
SHALLOW_THRESHOLD = -0.4

DIPPER_COLOR = "tab:blue"        # deep basin / Population 1
NONDIPPER_COLOR = "tab:red"      # shallow + mid / Population 2

DASHBOARD_LINE_COLOR = "#d4d4d4"  # neutral grey for dashboard polylines


# Line + endpoint sizing
LINE_WIDTH = 1.0
LINE_ALPHA = 0.55
START_SIZE = 18
END_SIZE = 30
ENDPOINT_LW = 1.0
EMPHASIS_SIZE = 110


# ── Font registration (Libertinus Serif if available) ──────────────────

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


# ── Matplotlib rcParams (applied on import) ────────────────────────────

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

def basin_from_cos(cos: float) -> str:
    """Bucket a `proj_cos` into deep / mid / shallow."""
    if cos <= DEEP_THRESHOLD:
        return "deep"
    if cos > SHALLOW_THRESHOLD:
        return "shallow"
    return "mid"


def basin_color(basin: str) -> str:
    """Binary palette: deep = dipper blue, mid + shallow = non-dipper red."""
    return DIPPER_COLOR if basin == "deep" else NONDIPPER_COLOR


def trajectory_basin(
    traj: Sequence[tuple[float, float, int]], min_step: int = 5,
) -> str:
    """Classify a trajectory by its deepest `proj_cos` at step >= min_step.

    `traj` is a list of `(kl, cos, step)` tuples. Pre-min_step steps are
    excluded so the random-init anchor doesn't dominate classification.
    """
    in_train = [t for t in traj if t[2] >= min_step]
    if not in_train:
        return "shallow"
    return basin_from_cos(min(t[1] for t in in_train))


# ── Matplotlib primitives ──────────────────────────────────────────────

def draw_trajectory(
    ax: Axes,
    xs: Sequence[float], ys: Sequence[float],
    color: str,
    *,
    lw: float = LINE_WIDTH, alpha: float = LINE_ALPHA,
    zorder: int = 2, label: str | None = None,
) -> None:
    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
            zorder=zorder, label=label)


def draw_endpoints(
    ax: Axes,
    xs: Sequence[float], ys: Sequence[float],
    color: str,
    *,
    zorder: int = 5,
    start_size: int = START_SIZE, end_size: int = END_SIZE,
) -> None:
    """Open circle at trajectory start, filled dot at end."""
    ax.scatter([xs[0]], [ys[0]], s=start_size,
               facecolor="white", edgecolor=color,
               linewidth=ENDPOINT_LW, zorder=zorder)
    ax.scatter([xs[-1]], [ys[-1]], s=end_size,
               facecolor=color, edgecolor=color,
               linewidth=ENDPOINT_LW, zorder=zorder)


def _strip_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)


def style_kl_axis(
    ax: Axes,
    *,
    layer: int | None = None,
    ylabel: str | None = None,
    xlabel: str = "KL ↑ (log)",
) -> None:
    """Standard styling for cos-vs-log-KL trajectory plots."""
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if ylabel is None and layer is not None:
        ylabel = f"cos(L{layer} shift, assistant axis)"
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4, which="both")
    ax.set_axisbelow(True)
    _strip_spines(ax)


def style_pc_axis(
    ax: Axes,
    *,
    x_label: str = "PC1", y_label: str = "PC2",
) -> None:
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.5, linestyle=":", alpha=0.6)
    ax.axvline(0, color=EDGE_COLOR, linewidth=0.5, linestyle=":", alpha=0.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    _strip_spines(ax)


def basin_legend(
    ax: Axes,
    n_dippers: int,
    n_nondippers: int,
    *,
    loc: str = "best",
    extra_handles: list | None = None,
):
    """Two-entry population legend with start/end glyph key."""
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


def panel_title(ax: Axes, text: str) -> None:
    """Bold panel title in the paper style."""
    ax.set_title(text, fontsize=12, fontweight="bold", color="#222222")


# ── Plotly colorscales (for the interactive dashboard) ─────────────────

_REDS_R = colormaps["Reds_r"]    # t=0 → dark red (persona-aligned)
_RAINBOW = colormaps["rainbow"]  # t=0 → violet (start), t=1 → red (end)


def color_persona(t: float) -> str:
    """Hex color for persona-strength coloring; t in [0, 1]."""
    return to_hex(_REDS_R(max(0.0, min(1.0, t))))


def color_step(t: float) -> str:
    """Hex color for step-progress coloring; t in [0, 1]."""
    return to_hex(_RAINBOW(max(0.0, min(1.0, t))))


def cmap_to_plotly_colorscale(name: str = "rainbow", n: int = 21) -> list[list]:
    """Convert a matplotlib colormap to a Plotly colorscale `[[t, hex], ...]`."""
    cmap = colormaps[name]
    return [[i / (n - 1), to_hex(cmap(i / (n - 1)))] for i in range(n)]


PERSONA_COLORSCALE = cmap_to_plotly_colorscale("Reds_r")
STEP_COLORSCALE = cmap_to_plotly_colorscale("rainbow")
