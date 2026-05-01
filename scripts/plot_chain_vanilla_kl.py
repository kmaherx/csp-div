"""Plot absolute KL(student || vanilla) vs step, one line per seed.

Reads vanilla_kl values from axis.json (populated by analyze_assistant_axis
when run on chain checkpoints). Unlike the per-segment KL trajectory plot
(plot_chain_kl.py), this measures cumulative drift from the base model —
monotonic-ish, less noisy, directly comparable across static and chain runs.

Usage:
  python scripts/plot_chain_vanilla_kl.py
  python scripts/plot_chain_vanilla_kl.py --axis-json results/random_walk/axis.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp_div.plot_style import (
    EDGE_COLOR, basin_color, basin_legend, panel_title, trajectory_basin,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-json",
                        default="results/random_walk/axis.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-y", action="store_true",
                        help="Use log scale on y-axis (vanilla_kl spans many orders).")
    args = parser.parse_args()

    axis_path = args.axis_json if os.path.isabs(args.axis_json) else os.path.join(ROOT, args.axis_json)
    out_path = args.out or axis_path.replace(".json", "_vanilla_kl_vs_step.png").replace("axis_", "")
    if os.path.basename(out_path) == axis_path.replace(".json", "_vanilla_kl_vs_step.png"):
        out_path = os.path.join(os.path.dirname(axis_path), "vanilla_kl_vs_step.png")

    with open(axis_path) as f:
        d = json.load(f)
    rows = d["rows"]

    # Group by seed (parsed from group field), sort by step
    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].replace("seed_", "").split("_")[0])
        by_seed.setdefault(seed, []).append(r)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["step"])

    # Basin per seed (for color)
    seed_basins = {}
    for seed, rs in by_seed.items():
        traj_tuples = [(r["kl"], r["proj_cos"], r["step"]) for r in rs]
        seed_basins[seed] = trajectory_basin(traj_tuples)
    n_dippers = sum(1 for b in seed_basins.values() if b == "deep")
    n_nondippers = len(seed_basins) - n_dippers

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot non-dippers under, dippers over
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for seed, rs in sorted(by_seed.items()):
            if not basin_filter(seed_basins[seed]):
                continue
            steps = [r["step"] for r in rs]
            vkls = [r.get("vanilla_kl") for r in rs]
            if any(v is None for v in vkls):
                continue
            color = basin_color(seed_basins[seed])
            ax.plot(steps, vkls, color=color, linewidth=1.0, alpha=0.55, zorder=zorder)
            # Endpoint markers (start = open, end = filled)
            ax.scatter([steps[0]], [vkls[0]], s=18, facecolor="white",
                       edgecolor=color, linewidth=1.0, zorder=zorder + 2)
            ax.scatter([steps[-1]], [vkls[-1]], s=30, facecolor=color,
                       edgecolor=color, linewidth=1.0, zorder=zorder + 2)

    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("KL(student || vanilla)" + ("  [log]" if args.log_y else ""))
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    panel_title(ax, "vanilla_kl vs step")
    basin_legend(ax, n_dippers, n_nondippers, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
