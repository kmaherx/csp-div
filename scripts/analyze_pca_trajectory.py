"""Visualize CSP-shift trajectories in their own data-derived PC space.

Independent of the assistant axis. Loads shift vectors from one or more
shifts.pt files (produced by analyze_assistant_axis.py with its --save-shifts
flag), pools all shifts, fits sklearn PCA, and emits three figures:

  figure_pc1_vs_kl.png   — PC1 on y, log KL on x. One trajectory per (cond, seed).
  figure_pc2_vs_kl.png   — same but PC2 on y.
  figure_pc1_vs_pc2.png  — leading-PC plane scatter with one trajectory per
                           (cond, seed). Open circle at start, filled dot at end.

Trajectories colored by the basin label inherited from the sibling axis.json
(deep ≤ −0.5; shallow > −0.4; mid otherwise; mid is grouped with shallow as
"non-dipper" red).

Usage:
  python scripts/analyze_pca_trajectory.py
  python scripts/analyze_pca_trajectory.py \\
      --shifts-paths results/qwen/shifts.pt \\
                     results/qwen_frames/instrumental/shifts.pt \\
                     results/qwen_frames/prepend/shifts.pt \\
      --out-dir results/qwen_frames/pca
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from csp_div.plot_style import (
    DIPPER_COLOR, NONDIPPER_COLOR,
    basin_color, basin_from_cos, basin_legend, draw_endpoints,
    draw_trajectory, style_kl_axis, style_pc_axis,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_axis_basins(axis_path):
    """Return {group_str: basin_label} from a sibling axis.json.

    Slightly different shape from plot_style.load_axis_trajectories: keyed by
    the full group path (which is what shifts.pt records use), not by seed int.
    """
    with open(axis_path) as f:
        rows = json.load(f)["rows"]
    by_group = {}
    for r in rows:
        if r["step"] < 5:
            continue
        by_group.setdefault(r["group"], []).append(r["proj_cos"])
    return {g: basin_from_cos(min(cs)) for g, cs in by_group.items()}


def load_shifts(shifts_paths):
    """Load shifts.pt files and matching axis.json basin labels.

    Returns (records, basins_per_group) where:
      records: list of dicts {cond, group, seed, ckpt, step, kl, shift (np)}
      basins_per_group: {group_str: basin_label}
    """
    all_records = []
    all_basins = {}
    for path in shifts_paths:
        full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        cond_label = os.path.basename(os.path.dirname(full_path))
        d = torch.load(full_path, map_location="cpu", weights_only=True)
        print(f"Loaded {path}: {len(d['rows'])} rows, layer={d['layer']}, "
              f"hidden_dim={d['mean_vanilla'].shape[0]}")

        axis_json = os.path.join(os.path.dirname(full_path), "axis.json")
        if os.path.isfile(axis_json):
            all_basins.update(load_axis_basins(axis_json))

        for r in d["rows"]:
            seed = int(r["group"].split("/")[-1].replace("seed_", "").split("_")[0])
            all_records.append({
                "cond": cond_label,
                "group": r["group"],
                "seed": seed,
                "ckpt": r["ckpt"],
                "step": r["step"],
                "kl": r["kl"],
                "shift": r["shift"].numpy(),
            })
    return all_records, all_basins


def group_by_trajectory(records):
    """Return {(cond, group): [records, sorted by step]}."""
    by_traj = {}
    for r in records:
        by_traj.setdefault((r["cond"], r["group"]), []).append(r)
    for k in by_traj:
        by_traj[k].sort(key=lambda r: r["step"])
    return by_traj


def basin_counts(by_traj, basins):
    n_dippers = sum(1 for (_, g) in by_traj if basins.get(g, "shallow") == "deep")
    return n_dippers, len(by_traj) - n_dippers


def plot_pc_vs_kl(records, basins, pc_idx, out_path):
    """One panel: PC<pc_idx> on y, log KL on x. One trajectory per (cond, seed)."""
    by_traj = group_by_trajectory(records)
    n_dippers, n_nondippers = basin_counts(by_traj, basins)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Non-dippers under, dippers over
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for (cond, group), traj in by_traj.items():
            basin = basins.get(group, "shallow")
            if not basin_filter(basin):
                continue
            color = basin_color(basin)
            kls = [r["kl"] for r in traj]
            pcs = [r["pc"][pc_idx] for r in traj]
            draw_trajectory(ax, kls, pcs, color, zorder=zorder)
            draw_endpoints(ax, kls, pcs, color, zorder=zorder + 2)

    style_kl_axis(ax, ylabel=f"PC{pc_idx + 1}")
    basin_legend(ax, n_dippers, n_nondippers, loc="best")
    cond_labels = sorted({r["cond"] for r in records})
    ax.set_title(" + ".join(cond_labels), fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pc1_vs_pc2(records, basins, out_path):
    """PC1×PC2 trajectory plot — open circle start, filled dot end."""
    by_traj = group_by_trajectory(records)
    n_dippers, n_nondippers = basin_counts(by_traj, basins)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Non-dippers under, dippers over
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for (cond, group), traj in by_traj.items():
            basin = basins.get(group, "shallow")
            if not basin_filter(basin):
                continue
            color = basin_color(basin)
            pcs1 = [r["pc"][0] for r in traj]
            pcs2 = [r["pc"][1] for r in traj]
            draw_trajectory(ax, pcs1, pcs2, color, zorder=zorder)
            draw_endpoints(ax, pcs1, pcs2, color, zorder=zorder + 2)

    style_pc_axis(ax, x_label="PC1", y_label="PC2")
    basin_legend(ax, n_dippers, n_nondippers, loc="best")
    cond_labels = sorted({r["cond"] for r in records})
    ax.set_title(" + ".join(cond_labels), fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-paths", nargs="+",
                        default=[
                            "results/qwen/shifts.pt",
                            "results/qwen_frames/instrumental/shifts.pt",
                            "results/qwen_frames/prepend/shifts.pt",
                        ])
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--out-dir", default="results/qwen_frames/pca")
    parser.add_argument("--per-condition", action="store_true",
                        help="Also write per-condition figures using the same pooled PC basis.")
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize each shift before PCA (direction-only PCs).")
    args = parser.parse_args()

    available = [p for p in args.shifts_paths
                 if os.path.isfile(p if os.path.isabs(p) else os.path.join(ROOT, p))]
    missing = [p for p in args.shifts_paths if p not in available]
    if missing:
        print(f"  missing (skipping): {missing}")
    if not available:
        raise SystemExit("No shifts.pt files found.")

    records, basins = load_shifts(available)
    print(f"\nTotal: {len(records)} ckpts, "
          f"{len(set((r['cond'], r['group']) for r in records))} trajectories")

    X = np.stack([r["shift"] for r in records])
    print(f"Shift matrix shape: {X.shape}")
    if args.normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_for_pca = X / norms
        print(f"Normalized: shift norm range was [{norms.min():.2f}, {norms.max():.2f}] -> all 1.0")
    else:
        X_for_pca = X
    pca = PCA(n_components=min(args.n_components, X.shape[0], X.shape[1]))
    Y = pca.fit_transform(X_for_pca)
    print(f"Variance explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative:         {np.cumsum(pca.explained_variance_ratio_)}")

    for i, r in enumerate(records):
        r["pc"] = Y[i]

    out_dir = os.path.join(ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    plot_pc_vs_kl(records, basins, 0, os.path.join(out_dir, "figure_pc1_vs_kl.png"))
    plot_pc_vs_kl(records, basins, 1, os.path.join(out_dir, "figure_pc2_vs_kl.png"))
    plot_pc1_vs_pc2(records, basins, os.path.join(out_dir, "figure_pc1_vs_pc2.png"))

    if args.per_condition:
        cond_subdir = "pca_normalized" if args.normalize else "pca"
        for cond in sorted({r["cond"] for r in records}):
            cond_records = [r for r in records if r["cond"] == cond]
            cond_path = next(p for p in available
                             if os.path.basename(os.path.dirname(p if os.path.isabs(p) else os.path.join(ROOT, p))) == cond)
            cond_dir = os.path.dirname(cond_path if os.path.isabs(cond_path) else os.path.join(ROOT, cond_path))
            cond_out = os.path.join(cond_dir, cond_subdir)
            os.makedirs(cond_out, exist_ok=True)
            print(f"\n--- per-condition: {cond} ({len(cond_records)} ckpts) -> {cond_out} ---")
            plot_pc_vs_kl(cond_records, basins, 0, os.path.join(cond_out, "figure_pc1_vs_kl.png"))
            plot_pc_vs_kl(cond_records, basins, 1, os.path.join(cond_out, "figure_pc2_vs_kl.png"))
            plot_pc1_vs_pc2(cond_records, basins, os.path.join(cond_out, "figure_pc1_vs_pc2.png"))

    summary = {
        "n_components": pca.n_components_,
        "n_records": len(records),
        "hidden_dim": int(X.shape[1]),
        "normalized": bool(args.normalize),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_explained": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "shifts_paths": available,
        "n_per_condition": {
            c: sum(1 for r in records if r["cond"] == c)
            for c in sorted({r["cond"] for r in records})
        },
        "n_per_basin": {
            b: sum(1 for r in records if basins.get(r["group"], "shallow") == b)
            for b in ("deep", "mid", "shallow")
        },
    }
    with open(os.path.join(out_dir, "pca_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {os.path.join(out_dir, 'pca_summary.json')}")


if __name__ == "__main__":
    main()
