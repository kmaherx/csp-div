"""Re-render axis.png from an existing axis.json.

The full csp_div.analyze_assistant_axis pipeline does a GPU eval pass to
*generate* axis.json. Once that data exists, the figure is just a function
of the JSON — re-rendering with a new style needs no GPU.

X-axis choice:
  --x kl   (default): log KL on x — appropriate for static-teacher runs
                      where KL grows monotonically with training.
  --x step           : training step on x — appropriate for chain-teacher
                      runs where KL is non-monotonic (sawtoothes, then
                      collapses) and step is the only monotonic progression.

Usage:
  python scripts/replot_axis.py results/qwen/axis.json
  python scripts/replot_axis.py --x step results/llama_chain/axis.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp_div.plot_style import (
    EDGE_COLOR, basin_color, basin_legend, draw_endpoints, draw_trajectory,
    draw_step_colored_trajectory, panel_title, style_kl_axis, trajectory_basin,
)


_X_FIELDS = {"kl": "kl", "step": "step", "vanilla_kl": "vanilla_kl"}
_X_LABELS = {
    "kl": "KL ↑ (per-segment, log)",
    "step": "step",
    "vanilla_kl": "KL(student || vanilla)  [log]",
}


def _style_step_axis(ax, ylabel):
    """Linear-step variant of style_kl_axis. Same chrome conventions."""
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)


def replot(axis_json_path, out_path=None, x_kind="kl"):
    if out_path is None:
        out_path = axis_json_path.replace(".json", ".png")

    with open(axis_json_path) as f:
        d = json.load(f)
    rows = d["rows"]
    layer = d["layer"]

    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: r["step"])

    group_basins = {
        g: trajectory_basin([(r["kl"], r["proj_cos"], r["step"]) for r in rs])
        for g, rs in by_group.items()
    }
    n_dippers = sum(1 for b in group_basins.values() if b == "deep")
    n_nondippers = len(group_basins) - n_dippers

    field = _X_FIELDS[x_kind]
    xlabel = _X_LABELS[x_kind]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if x_kind == "step":
        # Grey traces only — step is on x already, no color needed for time.
        for g, rs in sorted(by_group.items()):
            xs_dots = [(r.get(field), r["proj_dot"], r["proj_cos"]) for r in rs]
            xs_dots = [(x, d, c) for (x, d, c) in xs_dots if x is not None]
            if not xs_dots:
                continue
            xs = [v[0] for v in xs_dots]
            dots = [v[1] for v in xs_dots]
            coss = [v[2] for v in xs_dots]
            draw_trajectory(axes[0], xs, dots, "#888888", lw=1.0, alpha=0.45, zorder=2)
            draw_endpoints(axes[0], xs, dots, "#888888", zorder=4)
            draw_trajectory(axes[1], xs, coss, "#888888", lw=1.0, alpha=0.45, zorder=2)
            draw_endpoints(axes[1], xs, coss, "#888888", zorder=4)
        _style_step_axis(axes[0], f"(L{layer} shift) · (assistant axis)")
        _style_step_axis(axes[1], f"cos(L{layer} shift, assistant axis)")
    elif x_kind == "vanilla_kl":
        # Rainbow lines colored by step.
        all_steps = [r["step"] for r in rows]
        cmap = plt.get_cmap("rainbow")
        norm = plt.Normalize(vmin=min(all_steps), vmax=max(all_steps))
        for g, rs in sorted(by_group.items()):
            steps = [r["step"] for r in rs]
            xs_dots = [(r.get(field), r["proj_dot"], r["proj_cos"], s)
                       for r, s in zip(rs, steps)]
            xs_dots = [(x, d, c, s) for (x, d, c, s) in xs_dots if x is not None]
            if len(xs_dots) < 2:
                continue
            xs = [v[0] for v in xs_dots]
            dots = [v[1] for v in xs_dots]
            coss = [v[2] for v in xs_dots]
            ss = [v[3] for v in xs_dots]
            draw_step_colored_trajectory(axes[0], xs, dots, ss, cmap, norm)
            draw_step_colored_trajectory(axes[1], xs, coss, ss, cmap, norm)
        # Need to manually autoscale because LineCollection doesn't trigger it
        for a in axes:
            a.autoscale()
        style_kl_axis(axes[0], xlabel=xlabel,
                      ylabel=f"(L{layer} shift) · (assistant axis)")
        style_kl_axis(axes[1], xlabel=xlabel,
                      ylabel=f"cos(L{layer} shift, assistant axis)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("step", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        # x_kind == "kl" — basin colors (existing default for static-teacher runs)
        for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                     (lambda b: b == "deep", 3)]:
            for g, rs in sorted(by_group.items()):
                if not basin_filter(group_basins[g]):
                    continue
                color = basin_color(group_basins[g])
                xs_dots = [(r.get(field), r["proj_dot"], r["proj_cos"]) for r in rs]
                xs_dots = [(x, d, c) for (x, d, c) in xs_dots if x is not None]
                if not xs_dots:
                    continue
                xs = [v[0] for v in xs_dots]
                dots = [v[1] for v in xs_dots]
                coss = [v[2] for v in xs_dots]
                draw_trajectory(axes[0], xs, dots, color, zorder=zorder)
                draw_endpoints(axes[0], xs, dots, color, zorder=zorder + 2)
                draw_trajectory(axes[1], xs, coss, color, zorder=zorder)
                draw_endpoints(axes[1], xs, coss, color, zorder=zorder + 2)
        style_kl_axis(axes[0], xlabel=xlabel,
                      ylabel=f"(L{layer} shift) · (assistant axis)")
        style_kl_axis(axes[1], xlabel=xlabel,
                      ylabel=f"cos(L{layer} shift, assistant axis)")
        basin_legend(axes[1], n_dippers, n_nondippers, loc="lower right")

    panel_title(axes[0], "Magnitude along assistant axis")
    panel_title(axes[1], "Direction alignment with assistant axis")
    fig.suptitle(
        f"L{layer} shift onto Butanium axis  ·  "
        f"negative = role-play, positive = default-assistant",
        fontsize=10, color="#444444",
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("axis_json", nargs="+",
                        help="One or more axis.json paths. PNG is written next to each.")
    parser.add_argument("--x", choices=["kl", "step", "vanilla_kl"], default="kl",
                        help="What to put on the x-axis: 'kl' (per-segment training KL, "
                             "log — static-teacher default), 'step' (linear, best for "
                             "chain runs since per-segment KL is non-monotonic), or "
                             "'vanilla_kl' (cumulative drift from base, log — cleanest "
                             "cross-experiment readout).")
    args = parser.parse_args()
    for p in args.axis_json:
        replot(p, x_kind=args.x)


if __name__ == "__main__":
    main()
