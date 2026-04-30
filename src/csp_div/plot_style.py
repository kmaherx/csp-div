"""Shared plot styling for CSP trajectory figures.

Conventions used by every trajectory plot in this repo:
  - Trajectories colored by basin: dipper (deep) blue, non-dipper (shallow + mid) red.
  - Open circle (○) at trajectory start, filled dot (●) at end — same color as the line.
  - Log scale on the KL axis, dotted reference line at zero.
  - Two-entry basin legend (with start/end key) instead of per-seed labels.

Importing from one place avoids style drift across the five plotting scripts.
"""
import json

from matplotlib.lines import Line2D


# ── Basin classification ────────────────────────────────────────────────
DEEP_THRESHOLD = -0.5
SHALLOW_THRESHOLD = -0.4

DIPPER_COLOR = "tab:blue"
NONDIPPER_COLOR = "tab:red"
HIGHLIGHT_DEEP = "#c0392b"     # bolder red for "look here" deep example
HIGHLIGHT_SHALLOW = "#1f77b4"  # bolder blue for "look here" shallow example

# Line + endpoint defaults
LINE_WIDTH = 1.4
LINE_ALPHA = 0.55
HIGHLIGHT_LW = 2.6
HIGHLIGHT_ALPHA = 0.95

START_SIZE = 32          # open ring
END_SIZE = 52            # filled dot
ENDPOINT_LW = 1.3
EMPHASIS_SIZE = 220      # for "look here" annotated step in figure_basins-style plots


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

def style_kl_axis(ax, *, layer=None, ylabel=None, xlabel="KL ↑ (log)"):
    """Standard styling for cos-vs-log-KL trajectory plots."""
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel is None and layer is not None:
        ylabel = f"cos(L{layer} shift, assistant axis)"
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.25)


def style_pc_axis(ax, *, x_label="PC1", y_label="PC2"):
    """Standard styling for PC×PC scatter plots."""
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(alpha=0.25)


# ── Legend ─────────────────────────────────────────────────────────────

def basin_legend(ax, n_dippers: int, n_nondippers: int, *,
                 loc="best", extra_handles=None):
    """Two-entry basin legend with a start/end glyph key.

    Pass extra_handles=[Line2D(...), ...] to append (e.g. for emphasis-
    dot examples in figure_basins / figure_cross_condition).
    """
    handles = [
        Line2D([0], [0], color=DIPPER_COLOR, linewidth=LINE_WIDTH + 0.4,
               label=f"dipper (deep)  ·  n={n_dippers}"),
        Line2D([0], [0], color=NONDIPPER_COLOR, linewidth=LINE_WIDTH + 0.4,
               label=f"non-dipper  ·  n={n_nondippers}"),
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=ENDPOINT_LW, markersize=6,
               label="○ start    ● end", linestyle=""),
    ]
    if extra_handles:
        handles.extend(extra_handles)
    return ax.legend(handles=handles, loc=loc, fontsize=9, framealpha=0.95)
