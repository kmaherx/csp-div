"""Re-render axis.png from an existing axis.json.

The full csp_div.analyze_assistant_axis pipeline does a GPU eval pass to
*generate* axis.json. Once that data exists, the figure is just a function
of the JSON — re-rendering with a new style needs no GPU.

Usage:
  python scripts/replot_axis.py results/qwen/axis.json
  python scripts/replot_axis.py results/qwen/axis.json results/llama/axis.json ...
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp_div.plot_style import (
    basin_color, basin_legend, draw_endpoints, draw_trajectory,
    panel_title, style_kl_axis, trajectory_basin,
)


def replot(axis_json_path, out_path=None):
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for g, rs in sorted(by_group.items()):
            if not basin_filter(group_basins[g]):
                continue
            color = basin_color(group_basins[g])
            kls = [r["kl"] for r in rs]
            dots = [r["proj_dot"] for r in rs]
            coss = [r["proj_cos"] for r in rs]
            draw_trajectory(axes[0], kls, dots, color, zorder=zorder)
            draw_endpoints(axes[0], kls, dots, color, zorder=zorder + 2)
            draw_trajectory(axes[1], kls, coss, color, zorder=zorder)
            draw_endpoints(axes[1], kls, coss, color, zorder=zorder + 2)

    style_kl_axis(axes[0], ylabel=f"(L{layer} shift) · (assistant axis)")
    style_kl_axis(axes[1], ylabel=f"cos(L{layer} shift, assistant axis)")
    panel_title(axes[0], "Magnitude along assistant axis")
    panel_title(axes[1], "Direction alignment with assistant axis")
    basin_legend(axes[1], n_dippers, n_nondippers, loc="lower right")
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
    args = parser.parse_args()
    for p in args.axis_json:
        replot(p)


if __name__ == "__main__":
    main()
