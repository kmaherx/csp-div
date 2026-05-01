"""PC1 × PC2 trajectory plot for chain runs, colored + labeled BY SEED.

Companion to analyze_pca_trajectory.py — that one colors by basin
(blue dipper / red non-dipper). This one colors by seed (tab10) and
puts a "seed_N" label at each endpoint, so you can tell which
trajectory is which when reading off behavior or self-verb.

Usage:
  python scripts/figure_chain_pca_by_seed.py
  python scripts/figure_chain_pca_by_seed.py --normalize
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from csp_div.plot_style import (
    EDGE_COLOR, draw_endpoints, draw_trajectory, panel_title, style_pc_axis,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-path",
                        default="results/llama_chain/shifts.pt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize each shift before PCA (direction-only PCs).")
    args = parser.parse_args()

    shifts_path = os.path.join(ROOT, args.shifts_path) if not os.path.isabs(args.shifts_path) else args.shifts_path
    out_path = args.out
    if out_path is None:
        out_dir = os.path.dirname(shifts_path)
        suffix = "_normalized" if args.normalize else ""
        out_path = os.path.join(out_dir, f"figure_pc1_vs_pc2_by_seed{suffix}.png")

    d = torch.load(shifts_path, map_location="cpu", weights_only=True)
    rows = d["rows"]
    print(f"Loaded {len(rows)} rows")

    # Group by (seed, sorted by step)
    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].replace("seed_", "").split("_")[0])
        by_seed.setdefault(seed, []).append(r)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["step"])

    # PCA across all shifts
    X = np.stack([r["shift"].numpy() for r in rows])
    if args.normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_for_pca = X / np.maximum(norms, 1e-8)
    else:
        X_for_pca = X
    pca = PCA(n_components=4)
    Y = pca.fit_transform(X_for_pca)
    print(f"Variance explained: {pca.explained_variance_ratio_}")

    # Map row → PC coords
    pc_for_row = {id(r): Y[i] for i, r in enumerate(rows)}

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")

    for i, seed in enumerate(sorted(by_seed)):
        traj = by_seed[seed]
        pcs1 = [pc_for_row[id(r)][0] for r in traj]
        pcs2 = [pc_for_row[id(r)][1] for r in traj]
        color = cmap(i % 10)
        draw_trajectory(ax, pcs1, pcs2, color, zorder=2)
        draw_endpoints(ax, pcs1, pcs2, color, zorder=4)
        # Label endpoint with seed number
        ax.annotate(f"seed_{seed}",
                    xy=(pcs1[-1], pcs2[-1]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=color,
                    fontweight="bold")

    style_pc_axis(ax, x_label="PC1", y_label="PC2")
    panel_title(ax, "llama_chain — PC1 × PC2 by seed")

    # Compact legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=cmap(i % 10), linewidth=1.4,
               label=f"seed_{seed}")
        for i, seed in enumerate(sorted(by_seed))
    ]
    handles.append(Line2D([0], [0], marker="o", color="white",
                          markerfacecolor="white", markeredgecolor=EDGE_COLOR,
                          markeredgewidth=1.0, markersize=5,
                          label="○ start    ● end", linestyle=""))
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
