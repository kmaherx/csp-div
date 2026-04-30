"""Cross-condition basin figure: side-by-side panels with paired same-seed bolding.

All trajectories colored by basin (deep blue, non-dip red); two seeds bolded
in both panels, with an emphasis dot at the chosen step.

Defaults:
  Bolded preserved-deep seed = seed_3 (deep in both panels)
  Bolded "flip" seed = seed_7 (deep under PERSONA, shallow under INSTRUMENTAL)

Usage:
  python scripts/figure_cross_condition.py
  python scripts/figure_cross_condition.py --subsample-every 10
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
    draw_trajectory, find_point, load_axis_trajectories,
    panel_title as _panel_title, style_kl_axis, trajectory_basin,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_panel(ax, by_seed, layer, bolded_a, step_a, bolded_b, step_b, panel_title):
    seed_basins = {seed: trajectory_basin(traj) for seed, traj in by_seed.items()}
    n_dippers = sum(1 for b in seed_basins.values() if b == "deep")
    n_nondippers = len(seed_basins) - n_dippers
    bolded = {bolded_a, bolded_b}

    for seed, traj in by_seed.items():
        if seed in bolded:
            continue
        kls = [t[0] for t in traj]
        coss = [t[1] for t in traj]
        color = basin_color(seed_basins[seed])
        draw_trajectory(ax, kls, coss, color, zorder=2)
        draw_endpoints(ax, kls, coss, color, zorder=4)

    for seed, step in [(bolded_a, step_a), (bolded_b, step_b)]:
        traj = by_seed[seed]
        kls = [t[0] for t in traj]
        coss = [t[1] for t in traj]
        color = basin_color(seed_basins[seed])
        draw_trajectory(ax, kls, coss, color,
                        lw=HIGHLIGHT_LW, alpha=HIGHLIGHT_ALPHA, zorder=5)
        draw_endpoints(ax, kls, coss, color, zorder=6)
        pt = find_point(traj, step)
        draw_emphasis(ax, pt[0], pt[1], color)

    style_kl_axis(ax, layer=layer)
    _panel_title(ax, panel_title)
    extra = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=basin_color(seed_basins[bolded_a]),
               markeredgecolor="black", markeredgewidth=1.4, markersize=9,
               linestyle="", label=f"seed_{bolded_a} @ step {step_a}"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=basin_color(seed_basins[bolded_b]),
               markeredgecolor="black", markeredgewidth=1.4, markersize=9,
               linestyle="", label=f"seed_{bolded_b} @ step {step_b}"),
    ]
    basin_legend(ax, n_dippers, n_nondippers, loc="lower right", extra_handles=extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-axis", default="results/qwen/axis.json")
    parser.add_argument("--instrumental-axis",
                        default="results/qwen_frames/instrumental/axis.json")
    parser.add_argument("--deep-seed", type=int, default=3)
    parser.add_argument("--persona-deep-step", type=int, default=30)
    parser.add_argument("--instrumental-deep-step", type=int, default=40)
    parser.add_argument("--flip-seed", type=int, default=7)
    parser.add_argument("--persona-flip-step", type=int, default=50)
    parser.add_argument("--instrumental-flip-step", type=int, default=40)
    parser.add_argument("--out", default=None)
    parser.add_argument("--subsample-every", type=int, default=None)
    args = parser.parse_args()

    persona_path = os.path.join(ROOT, args.persona_axis)
    instr_path = os.path.join(ROOT, args.instrumental_axis)
    out_path = args.out or os.path.join(
        ROOT, "results", "qwen_frames", "figure_cross_condition.png",
    )

    persona, p_layer = load_axis_trajectories(persona_path)
    instr, i_layer = load_axis_trajectories(instr_path)
    layer = p_layer

    if args.subsample_every:
        for d in (persona, instr):
            for seed in d:
                d[seed] = [t for t in d[seed] if t[2] % args.subsample_every == 0]
        print(f"Subsampled to every-{args.subsample_every}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    plot_panel(axes[0], persona, layer,
               args.deep_seed, args.persona_deep_step,
               args.flip_seed, args.persona_flip_step,
               "PERSONA  (Be / Act / Please / You should)")
    plot_panel(axes[1], instr, layer,
               args.deep_seed, args.instrumental_deep_step,
               args.flip_seed, args.instrumental_flip_step,
               "INSTRUMENTAL  (Use / Apply / Follow / Employ)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
