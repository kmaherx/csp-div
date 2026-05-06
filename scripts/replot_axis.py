"""Re-render axis.png from an existing axis.json.

The full csp_div.analyze_assistant_axis pipeline does a GPU eval pass to
*generate* axis.json. Once that data exists, the figure is just a function
of the JSON — re-rendering with a new style needs no GPU.

Usage:
  python scripts/replot_axis.py results/llama/axis.json
  python scripts/replot_axis.py --x step results/llama/axis.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp_div.plot_style import (
    EDGE_COLOR, basin_color, basin_legend, cluster_color, cluster_legend,
    draw_endpoints, draw_trajectory, is_multi_cluster, load_cluster_assignments,
    panel_title, style_kl_axis, trajectory_basin,
)


def _style_step_axis(ax, ylabel):
    """Linear-x styling for step plots (style_kl_axis assumes log)."""
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


def replot(axis_json_path, out_path=None, x_kind="kl", cluster_json=None):
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

    if cluster_json is not None:
        group_basins = load_cluster_assignments(cluster_json)
        # Defensively backfill any missing groups (shouldn't happen but just
        # in case the JSON was generated against a smaller seed set).
        for g in by_group:
            group_basins.setdefault(g, "shallow")
    else:
        group_basins = {
            g: trajectory_basin([(r["kl"], r["proj_cos"], r["step"]) for r in rs])
            for g, rs in by_group.items()
        }

    multi = is_multi_cluster(group_basins)
    color_fn = cluster_color if multi else basin_color
    label_counts = {}
    for v in group_basins.values():
        label_counts[v] = label_counts.get(v, 0) + 1
    n_dippers = label_counts.get("deep", 0)
    n_nondippers = len(group_basins) - n_dippers

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if multi:
        # Draw clusters in reverse rank order so cluster_0 (Population 1) ends on top.
        cluster_order = sorted(
            (k for k in label_counts if k.startswith("cluster_")),
            key=lambda s: -int(s.split("_", 1)[1]),
        )
        for zorder, label in enumerate(cluster_order, start=2):
            color = color_fn(label)
            for g, rs in sorted(by_group.items()):
                if group_basins[g] != label:
                    continue
                xs = [r["step" if x_kind == "step" else "kl"] for r in rs]
                dots = [r["proj_dot"] for r in rs]
                coss = [r["proj_cos"] for r in rs]
                draw_trajectory(axes[0], xs, dots, color, zorder=zorder)
                draw_endpoints(axes[0], xs, dots, color, zorder=zorder + 2)
                draw_trajectory(axes[1], xs, coss, color, zorder=zorder)
                draw_endpoints(axes[1], xs, coss, color, zorder=zorder + 2)
    else:
        for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                     (lambda b: b == "deep", 3)]:
            for g, rs in sorted(by_group.items()):
                if not basin_filter(group_basins[g]):
                    continue
                color = color_fn(group_basins[g])
                xs = [r["step" if x_kind == "step" else "kl"] for r in rs]
                dots = [r["proj_dot"] for r in rs]
                coss = [r["proj_cos"] for r in rs]
                draw_trajectory(axes[0], xs, dots, color, zorder=zorder)
                draw_endpoints(axes[0], xs, dots, color, zorder=zorder + 2)
                draw_trajectory(axes[1], xs, coss, color, zorder=zorder)
                draw_endpoints(axes[1], xs, coss, color, zorder=zorder + 2)

    if x_kind == "step":
        _style_step_axis(axes[0], f"(L{layer} shift) · (assistant axis)")
        _style_step_axis(axes[1], f"cos(L{layer} shift, assistant axis)")
    else:
        style_kl_axis(axes[0], ylabel=f"(L{layer} shift) · (assistant axis)")
        style_kl_axis(axes[1], ylabel=f"cos(L{layer} shift, assistant axis)")
    panel_title(axes[0], "Magnitude along assistant axis")
    panel_title(axes[1], "Direction alignment with assistant axis")
    if multi:
        cluster_legend(axes[1], label_counts, loc="lower right")
    else:
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
    parser.add_argument("--x", choices=["kl", "step"], default="kl",
                        help="X-axis: 'kl' (log per-segment KL — default, original style) "
                             "or 'step' (linear training step).")
    parser.add_argument("--cluster-json", default=None,
                        help="Path to a kmeans_clusters.json (output of "
                             "scripts/compute_kmeans_clusters.py). When given, "
                             "trajectory colors come from this file instead of "
                             "the cosine-threshold trajectory_basin classifier.")
    parser.add_argument("--out-suffix", default="",
                        help="Append to the auto-derived output path before .png "
                             "(e.g. '_kmeans' -> axis_kmeans.png).")
    args = parser.parse_args()
    for p in args.axis_json:
        out = p.replace(".json", f"{args.out_suffix}.png")
        replot(p, out_path=out, x_kind=args.x, cluster_json=args.cluster_json)


if __name__ == "__main__":
    main()
