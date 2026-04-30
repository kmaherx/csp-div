"""Two-basin trajectory figure with one bolded deep + one bolded shallow example.

All trajectories are colored by basin (deep blue, non-dip red); two seeds are
bolded for emphasis with a large filled dot at the step a behavior example was
taken from.

Defaults pick (both at step 40 for visual symmetry):
  Deep    = seed_9 step 40 (cos −0.677, KL 1.57)
  Shallow = seed_2 step 40 (cos −0.310, KL 1.05)

Usage:
  python scripts/figure_basins.py
  python scripts/figure_basins.py --model llama
  python scripts/figure_basins.py --deep-seed 5 --deep-step 60
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from csp_div.plot_style import (
    HIGHLIGHT_ALPHA, HIGHLIGHT_LW,
    basin_color, basin_legend, draw_emphasis, draw_endpoints,
    draw_trajectory, find_point, load_axis_trajectories, style_kl_axis,
    trajectory_basin,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen",
                        help="Path component under results/ (e.g. 'qwen', 'llama', "
                             "'qwen_frames/instrumental').")
    parser.add_argument("--deep-seed", type=int, default=9)
    parser.add_argument("--deep-step", type=int, default=40)
    parser.add_argument("--shallow-seed", type=int, default=2)
    parser.add_argument("--shallow-step", type=int, default=40)
    parser.add_argument("--out", default=None)
    parser.add_argument("--subsample-every", type=int, default=None,
                        help="Keep only ckpts at multiples of this step.")
    args = parser.parse_args()

    axis_path = os.path.join(ROOT, "results", args.model, "axis.json")
    out_path = args.out or os.path.join(ROOT, "results", args.model, "figure_basins.png")

    by_seed, layer = load_axis_trajectories(axis_path)
    print(f"Loaded {len(by_seed)} trajectories from {axis_path}")

    if args.subsample_every:
        for seed in by_seed:
            by_seed[seed] = [t for t in by_seed[seed] if t[2] % args.subsample_every == 0]
        kept = sum(len(v) for v in by_seed.values())
        print(f"  subsampled to every-{args.subsample_every}: {kept} ckpts total")

    deep_traj = by_seed[args.deep_seed]
    shallow_traj = by_seed[args.shallow_seed]
    deep_pt = find_point(deep_traj, args.deep_step)
    shallow_pt = find_point(shallow_traj, args.shallow_step)
    print(f"  deep    seed_{args.deep_seed}    @ step {args.deep_step}: "
          f"KL={deep_pt[0]:.3f}, cos={deep_pt[1]:+.3f}")
    print(f"  shallow seed_{args.shallow_seed} @ step {args.shallow_step}: "
          f"KL={shallow_pt[0]:.3f}, cos={shallow_pt[1]:+.3f}")

    # Per-trajectory basin labels
    seed_basins = {seed: trajectory_basin(traj) for seed, traj in by_seed.items()}
    n_dippers = sum(1 for b in seed_basins.values() if b == "deep")
    n_nondippers = len(seed_basins) - n_dippers

    fig, ax = plt.subplots(figsize=(9, 6))

    # All trajectories colored by basin (background context).
    # Bolded examples drawn on top with thicker line + emphasis dot.
    bolded = {args.deep_seed, args.shallow_seed}
    for seed, traj in by_seed.items():
        if seed in bolded:
            continue
        kls = [t[0] for t in traj]
        coss = [t[1] for t in traj]
        color = basin_color(seed_basins[seed])
        draw_trajectory(ax, kls, coss, color, zorder=2)
        draw_endpoints(ax, kls, coss, color, zorder=4)

    for seed in (args.deep_seed, args.shallow_seed):
        traj = by_seed[seed]
        kls = [t[0] for t in traj]
        coss = [t[1] for t in traj]
        color = basin_color(seed_basins[seed])
        draw_trajectory(ax, kls, coss, color,
                        lw=HIGHLIGHT_LW, alpha=HIGHLIGHT_ALPHA, zorder=5)
        draw_endpoints(ax, kls, coss, color, zorder=6)

    draw_emphasis(ax, deep_pt[0], deep_pt[1], basin_color(seed_basins[args.deep_seed]))
    draw_emphasis(ax, shallow_pt[0], shallow_pt[1], basin_color(seed_basins[args.shallow_seed]))

    style_kl_axis(ax, layer=layer)
    extra = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=basin_color(seed_basins[args.deep_seed]),
               markeredgecolor="black", markeredgewidth=1.5, markersize=10,
               linestyle="", label=f"deep example: seed_{args.deep_seed} step {args.deep_step}"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=basin_color(seed_basins[args.shallow_seed]),
               markeredgecolor="black", markeredgewidth=1.5, markersize=10,
               linestyle="", label=f"shallow example: seed_{args.shallow_seed} step {args.shallow_step}"),
    ]
    basin_legend(ax, n_dippers, n_nondippers, loc="lower right", extra_handles=extra)

    ax.set_title(args.model, fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
