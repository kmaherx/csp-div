"""PC trajectory plots with each point colored by step (rainbow / viridis).

Companion to figure_chain_pca_by_seed.py and analyze_pca_trajectory.py.
Those plots show *which* trajectory is which (color = seed or basin).
This one shows *when* in training each point is, so you can see the
temporal sweep of the trajectories — useful when trajectories overlap
in PC space and you can't tell which direction they're going.

Outputs:
  figure_pc1_vs_pc2_rainbow.png       PC1 × PC2 with step coloring
  figure_pc1_vs_vanilla_kl_rainbow.png PC1 × log vanilla_kl with step coloring
  figure_pc2_vs_vanilla_kl_rainbow.png PC2 × log vanilla_kl with step coloring

Usage:
  python scripts/figure_chain_pca_rainbow.py
  python scripts/figure_chain_pca_rainbow.py --shifts-path results/random_walk/shifts.pt
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from csp_div.plot_style import EDGE_COLOR, panel_title, style_pc_axis


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-path",
                        default="results/random_walk/shifts.pt")
    parser.add_argument("--out-prefix", default=None,
                        help="Output prefix (e.g. results/random_walk/figure_chain_rainbow). "
                             "Three files written: _pc1_vs_pc2.png, _pc1_vs_vanilla_kl.png, "
                             "_pc2_vs_vanilla_kl.png.")
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize each shift before PCA (direction-only PCs).")
    parser.add_argument("--cmap", default="viridis",
                        help="Sequential colormap for step coloring. "
                             "Try 'viridis', 'plasma', 'inferno', 'turbo'.")
    args = parser.parse_args()

    shifts_path = args.shifts_path if os.path.isabs(args.shifts_path) \
        else os.path.join(ROOT, args.shifts_path)
    if args.out_prefix is None:
        out_dir = os.path.dirname(shifts_path)
        suffix = "_normalized" if args.normalize else ""
        out_prefix = os.path.join(out_dir, f"figure_chain_rainbow{suffix}")
    else:
        out_prefix = args.out_prefix

    d = torch.load(shifts_path, map_location="cpu", weights_only=True)
    rows = d["rows"]
    print(f"Loaded {len(rows)} rows")

    # Group by seed, sort by step
    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].replace("seed_", "").split("_")[0])
        by_seed.setdefault(seed, []).append(r)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["step"])

    # PCA
    X = np.stack([r["shift"].numpy() for r in rows])
    if args.normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_for_pca = X / np.maximum(norms, 1e-8)
    else:
        X_for_pca = X
    pca = PCA(n_components=4)
    Y = pca.fit_transform(X_for_pca)
    print(f"Variance explained: {pca.explained_variance_ratio_}")

    pc_for_row = {id(r): Y[i] for i, r in enumerate(rows)}

    # Step range for color normalization (shared across all panels)
    all_steps = [r["step"] for r in rows]
    step_min, step_max = min(all_steps), max(all_steps)
    cmap = plt.get_cmap(args.cmap)
    norm = plt.Normalize(vmin=step_min, vmax=step_max)

    def _plot_xy(ax, x_field_fn, y_field_fn):
        # Thin gray connecting lines first (so points sit on top)
        for seed, traj in by_seed.items():
            xs = [x_field_fn(r) for r in traj]
            ys = [y_field_fn(r) for r in traj]
            xs_ys = [(x, y) for (x, y) in zip(xs, ys) if x is not None and y is not None]
            if not xs_ys:
                continue
            xs = [v[0] for v in xs_ys]
            ys = [v[1] for v in xs_ys]
            ax.plot(xs, ys, color="#cccccc", linewidth=0.6, alpha=0.6, zorder=2)
        # Scatter points colored by step
        for r in rows:
            x = x_field_fn(r)
            y = y_field_fn(r)
            if x is None or y is None:
                continue
            ax.scatter([x], [y], s=14, color=cmap(norm(r["step"])),
                       edgecolor="none", alpha=0.85, zorder=3)

    # --- PC1 × PC2 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_xy(ax, lambda r: pc_for_row[id(r)][0], lambda r: pc_for_row[id(r)][1])
    style_pc_axis(ax, x_label="PC1", y_label="PC2")
    panel_title(ax, "PC1 × PC2 (step → color)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("step", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    out = f"{out_prefix}_pc1_vs_pc2.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # --- PC1 × log vanilla_kl ---
    for pc_idx, label in [(0, "PC1"), (1, "PC2")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_xy(ax,
                 lambda r: r.get("vanilla_kl"),
                 lambda r, idx=pc_idx: pc_for_row[id(r)][idx])
        ax.set_xscale("log")
        ax.set_xlabel("KL(student || vanilla)  [log]")
        ax.set_ylabel(label)
        ax.axhline(0, color=EDGE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.grid(True, alpha=0.2, linewidth=0.4, which="both")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(EDGE_COLOR)
        ax.spines["bottom"].set_color(EDGE_COLOR)
        panel_title(ax, f"{label} × vanilla_kl (step → color)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("step", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()
        out = f"{out_prefix}_{label.lower()}_vs_vanilla_kl.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
