"""All trajectories colored by basin label — one panel.

Each trajectory is colored by its own deepest-cos basin label (deep → blue,
mid + shallow → red), with an open circle at start and a filled dot at end.

Usage:
  python scripts/figure_basins_by_label.py
  python scripts/figure_basins_by_label.py --axis-paths results/qwen/axis.json
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp_div.plot_style import (
    DIPPER_COLOR, NONDIPPER_COLOR, HIGHLIGHT_ALPHA,
    basin_color, basin_legend, draw_endpoints, draw_trajectory,
    load_axis_trajectories, panel_title, style_kl_axis, trajectory_basin,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-paths", nargs="+",
                        default=["results/qwen_frames/instrumental/axis.json"],
                        help="One or more axis.json files. Each contributes its "
                             "trajectories to the combined plot.")
    parser.add_argument("--out",
                        default="results/qwen_frames/instrumental/figure_basins_by_label.png")
    args = parser.parse_args()

    out_path = os.path.join(ROOT, args.out)

    # Collect (cond_label, seed, traj, basin) for every trajectory
    all_trajs = []
    layer = None
    for ap in args.axis_paths:
        path = os.path.join(ROOT, ap)
        by_seed, this_layer = load_axis_trajectories(path)
        layer = this_layer if layer is None else layer
        cond_label = ap.split("/")[-2]
        for seed, traj in by_seed.items():
            basin = trajectory_basin(traj)
            all_trajs.append((cond_label, seed, traj, basin))

    n_dippers = sum(1 for t in all_trajs if t[3] == "deep")
    n_nondippers = len(all_trajs) - n_dippers

    print(f"Total trajectories: {len(all_trajs)} from {len(args.axis_paths)} condition(s)")
    print(f"  deep    : {n_dippers}")
    print(f"  non-dip : {n_nondippers}")

    fig, ax = plt.subplots(figsize=(6, 4))

    # Non-dippers under, dippers over (so dipper trajectories aren't visually buried)
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for _, _, traj, basin in all_trajs:
            if not basin_filter(basin):
                continue
            kls = [t[0] for t in traj]
            coss = [t[1] for t in traj]
            color = basin_color(basin)
            draw_trajectory(ax, kls, coss, color, zorder=zorder)
            draw_endpoints(ax, kls, coss, color, zorder=zorder + 2)

    style_kl_axis(ax, layer=layer)
    basin_legend(ax, n_dippers, n_nondippers, loc="lower right")

    cond_labels = [ap.split("/")[-2] for ap in args.axis_paths]
    panel_title(ax, " + ".join(cond_labels))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
