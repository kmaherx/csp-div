"""PC trajectory plots with each line segment colored by step (rainbow / viridis).

Companion to figure_chain_pca_by_seed.py and analyze_pca_trajectory.py.
Same start/end glyph convention as those (open circle at start, filled
dot at end), but the connecting line is colored with a sequential
colormap by step — temporal sweep is visible without sacrificing the
trajectory-as-line aesthetic.

Outputs:
  *_pc1_vs_pc2.png             PC1 × PC2 with step-colored lines
  *_pc1_vs_vanilla_kl.png      PC1 × log vanilla_kl with step-colored lines
  *_pc2_vs_vanilla_kl.png      PC2 × log vanilla_kl with step-colored lines

Usage:
  python scripts/figure_chain_pca_rainbow.py
  python scripts/figure_chain_pca_rainbow.py --shifts-path results/random_walk/shifts.pt
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from sklearn.decomposition import PCA

from csp_div.plot_style import EDGE_COLOR, panel_title, style_pc_axis


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _segments_from_xy(xs, ys):
    """Build a (N-1, 2, 2) segment array for LineCollection from xs, ys."""
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def _draw_step_colored_lines(ax, by_seed, x_field_fn, y_field_fn,
                              cmap, norm, lw=1.2, alpha=0.85,
                              start_size=22, end_size=36):
    """For each trajectory, draw a multi-segment line colored by step.
    Add open circle at start, filled dot at end (color = first/last step's cmap)."""
    for seed, traj in by_seed.items():
        steps = [r["step"] for r in traj]
        xs_ys = [(x_field_fn(r), y_field_fn(r), s)
                 for r, s in zip(traj, steps)]
        xs_ys = [(x, y, s) for (x, y, s) in xs_ys if x is not None and y is not None]
        if len(xs_ys) < 2:
            continue
        xs = [v[0] for v in xs_ys]
        ys = [v[1] for v in xs_ys]
        seg_steps = [v[2] for v in xs_ys]

        segments = _segments_from_xy(xs, ys)
        # Use the average of the two endpoints' steps for each segment's color
        seg_colors = [(seg_steps[i] + seg_steps[i + 1]) / 2.0
                      for i in range(len(seg_steps) - 1)]
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            array=np.asarray(seg_colors),
                            linewidth=lw, alpha=alpha, zorder=2)
        ax.add_collection(lc)
        # Endpoints (open at start, filled at end), colored to match the line
        c_start = cmap(norm(seg_steps[0]))
        c_end = cmap(norm(seg_steps[-1]))
        ax.scatter([xs[0]], [ys[0]], s=start_size,
                   facecolor="white", edgecolor=c_start,
                   linewidth=1.2, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=end_size,
                   facecolor=c_end, edgecolor=c_end,
                   linewidth=1.2, zorder=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-path",
                        default="results/random_walk/shifts.pt")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cmap", default="viridis")
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

    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].replace("seed_", "").split("_")[0])
        by_seed.setdefault(seed, []).append(r)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["step"])

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

    all_steps = [r["step"] for r in rows]
    cmap = plt.get_cmap(args.cmap)
    norm = plt.Normalize(vmin=min(all_steps), vmax=max(all_steps))

    # --- PC1 × PC2 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_step_colored_lines(
        ax, by_seed,
        lambda r: pc_for_row[id(r)][0],
        lambda r: pc_for_row[id(r)][1],
        cmap, norm,
    )
    ax.autoscale()
    style_pc_axis(ax, x_label="PC1", y_label="PC2")
    panel_title(ax, "PC1 × PC2 (line color → step)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("step", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    out = f"{out_prefix}_pc1_vs_pc2.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # --- PC{1,2} × log vanilla_kl ---
    for pc_idx, label in [(0, "PC1"), (1, "PC2")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        _draw_step_colored_lines(
            ax, by_seed,
            lambda r: r.get("vanilla_kl"),
            lambda r, idx=pc_idx: pc_for_row[id(r)][idx],
            cmap, norm,
        )
        ax.autoscale()
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
        panel_title(ax, f"{label} × vanilla_kl (line color → step)")
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
