"""Per-token CSP L2 norm trajectory across training, one line per seed.

Diagnostic plot of CSP per-token L2 norm vs step. Under the default
`randn(L, hidden) * 0.1` init the CSP starts at per-token L2 ≈ 6.4
(~10× the median real-token row norm of ~0.69 on Llama-3.1-8B) — deep OOD
in magnitude. This plot quantifies how much the optimizer drifts the norm
during training; useful for the methods/diagnostics section of the writeup
and for catching pathological blowup if it ever happens.

Reads sp_pos_step*.pt from each seed_<N>/ dir under --csp-dir, computes the
mean per-token L2 norm, plots vs step.

Usage:
  python scripts/plot_csp_norm.py
  python scripts/plot_csp_norm.py --csp-dir results/llama
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from csp_div.plot_style import (
    EDGE_COLOR, basin_color, basin_legend, cluster_color, cluster_legend,
    is_multi_cluster, load_axis_trajectories, load_cluster_assignments,
    panel_title, trajectory_basin,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _step_from_path(p):
    m = re.search(r"sp_pos_step(\d+)\.pt$", p)
    if m:
        return int(m.group(1))
    if p.endswith("sp_pos.pt"):
        return None  # final ckpt — handled separately, step from kl_curve
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csp-dir", default="results/llama",
                        help="Dir containing seed_*/sp_pos*.pt")
    parser.add_argument("--axis-json", default=None,
                        help="Optional axis.json for basin-coloring (deep blue / "
                             "non-dipper red). If unset, all lines grey.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: <csp-dir>/csp_norm_vs_step.png)")
    parser.add_argument("--cluster-json", default=None,
                        help="Path to kmeans_clusters.json. When given, basin "
                             "colors come from this file instead of "
                             "trajectory_basin (axis-json is then optional).")
    parser.add_argument("--out-suffix", default="",
                        help="Append to --out filename before .png (e.g. '_kmeans').")
    args = parser.parse_args()

    csp_dir = args.csp_dir if os.path.isabs(args.csp_dir) else os.path.join(ROOT, args.csp_dir)
    out_path = args.out or os.path.join(csp_dir, "csp_norm_vs_step.png")
    if args.out_suffix:
        out_path = out_path.replace(".png", f"{args.out_suffix}.png")

    # Optional basin coloring — k-means JSON takes precedence over axis-json
    seed_basins = {}
    if args.cluster_json:
        cluster_path = args.cluster_json if os.path.isabs(args.cluster_json) \
            else os.path.join(ROOT, args.cluster_json)
        # JSON is keyed by group string ("llama/seed_<N>"); coerce to seed int
        # to match the rest of this script.
        for g, basin in load_cluster_assignments(cluster_path).items():
            try:
                seed = int(g.split("seed_")[-1].split("_")[0])
                seed_basins[seed] = basin
            except ValueError:
                continue
    elif args.axis_json:
        axis_path = args.axis_json if os.path.isabs(args.axis_json) \
            else os.path.join(ROOT, args.axis_json)
        if os.path.isfile(axis_path):
            by_seed, _ = load_axis_trajectories(axis_path)
            for s, traj in by_seed.items():
                seed_basins[s] = trajectory_basin(traj)

    # Group ckpt paths by seed, compute (step, norm) per ckpt
    seed_dirs = sorted(glob.glob(os.path.join(csp_dir, "seed_*")))
    if not seed_dirs:
        raise SystemExit(f"No seed_*/ under {csp_dir}")

    print(f"Found {len(seed_dirs)} seed dirs under {csp_dir}")
    fig, ax = plt.subplots(figsize=(6, 4))

    multi = is_multi_cluster(seed_basins) if seed_basins else False
    color_fn = cluster_color if multi else basin_color
    label_counts = {}
    n_dippers = 0
    n_nondippers = 0

    for sd in seed_dirs:
        seed = int(os.path.basename(sd).replace("seed_", "").split("_")[0])
        ckpts = sorted(glob.glob(os.path.join(sd, "sp_pos*.pt")))
        steps_norms = []
        for p in ckpts:
            step = _step_from_path(p)
            try:
                c = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                continue
            if step is None:  # sp_pos.pt — derive step from kl_curve length
                step = len(c.get("kl_curve") or []) or 0
            e = c["embedding"].float()
            per_tok_norm = e.norm(dim=-1).mean().item()
            steps_norms.append((step, per_tok_norm))
        steps_norms.sort()
        if len(steps_norms) < 2:
            continue
        steps = [v[0] for v in steps_norms]
        norms = [v[1] for v in steps_norms]
        basin = seed_basins.get(seed, None)
        if multi and basin and basin.startswith("cluster_"):
            color = color_fn(basin)
            label_counts[basin] = label_counts.get(basin, 0) + 1
        elif basin == "deep":
            color = basin_color(basin)
            n_dippers += 1
        elif basin in ("mid", "shallow"):
            color = basin_color(basin)
            n_nondippers += 1
        else:
            color = "#888888"
        ax.plot(steps, norms, color=color, linewidth=1.0, alpha=0.7)
        ax.scatter([steps[0]], [norms[0]], s=18, facecolor="white",
                   edgecolor=color, linewidth=1.0)
        ax.scatter([steps[-1]], [norms[-1]], s=30, facecolor=color,
                   edgecolor=color, linewidth=1.0)

    ax.set_xlabel("step")
    ax.set_ylabel("CSP per-token L2 norm  (mean over L tokens)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    panel_title(ax, "CSP norm vs step")
    if multi and label_counts:
        cluster_legend(ax, label_counts, loc="best")
    elif (args.axis_json or args.cluster_json) and (n_dippers or n_nondippers):
        basin_legend(ax, n_dippers, n_nondippers, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
