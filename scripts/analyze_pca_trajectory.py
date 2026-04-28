"""Visualize CSP-shift trajectories in their own data-derived PC space.

Independent of the assistant axis. Loads shift vectors from one or
more shifts.pt files (produced by analyze_assistant_axis.py with
its --save-shifts flag), pools all shifts across all (condition,
seed, ckpt) combinations, fits sklearn PCA, and emits three figures:

  figure_pc1_vs_kl.png   — PC1 on y, log KL on x. One trajectory per
                           (cond, seed). Colored blue (dipper) /
                           red (non-dipper) by basin label inherited
                           from axis.json.
  figure_pc2_vs_kl.png   — same but PC2 on y.
  figure_pc1_vs_pc2.png  — scatter of all (cond, seed, ckpt) points
                           in the leading PC plane. Marker size
                           scales with log(KL) so trajectory direction
                           is visible (small early, big late). Thin
                           gray segments connect consecutive ckpts
                           within each (cond, seed).

Basin classification (deep ≤ −0.5; shallow > −0.4; mid otherwise) is
read from the sibling axis.json next to each shifts.pt file. Mid is
visually merged into "non-dipper" red, matching figure_basins_by_label.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEEP_THRESHOLD = -0.5
SHALLOW_THRESHOLD = -0.4


def basin_from_cos(cos):
    if cos <= DEEP_THRESHOLD:
        return "deep"
    if cos > SHALLOW_THRESHOLD:
        return "shallow"
    return "mid"


def load_axis_basins(axis_path):
    """Return {group_str: basin_label} from a sibling axis.json."""
    with open(axis_path) as f:
        rows = json.load(f)["rows"]
    by_group = {}
    for r in rows:
        if r["step"] < 5:
            continue
        by_group.setdefault(r["group"], []).append(r["proj_cos"])
    return {g: basin_from_cos(min(cs)) for g, cs in by_group.items()}


def load_shifts(shifts_paths):
    """Load shifts.pt files and the matching axis.json basin labels.

    Returns (records, basins_per_group) where:
      records: list of dicts {cond_label, group, seed, ckpt, step, kl, shift (np)}
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

        # Sibling axis.json for basin labels
        axis_json = os.path.join(os.path.dirname(full_path), "axis.json")
        if os.path.isfile(axis_json):
            basins = load_axis_basins(axis_json)
            all_basins.update(basins)

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


def trajectory_color(basin):
    """blue for dippers (deep), red for non-dippers (shallow + mid)."""
    if basin == "deep":
        return "tab:blue"
    return "tab:red"


def group_by_trajectory(records):
    """Return {(cond, group): [records, sorted by step]}."""
    by_traj = {}
    for r in records:
        by_traj.setdefault((r["cond"], r["group"]), []).append(r)
    for k in by_traj:
        by_traj[k].sort(key=lambda r: r["step"])
    return by_traj


def plot_pc_vs_kl(records, basins, pc_idx, out_path, alpha=0.5):
    """One panel: PC<pc_idx> on y, log KL on x. One trajectory per (cond, seed)."""
    by_traj = group_by_trajectory(records)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Bucket trajectories by basin for layered drawing (non-dippers under, dippers over)
    n_dippers = n_non = 0
    for (cond, group), traj in by_traj.items():
        basin = basins.get(group, "shallow")
        color = trajectory_color(basin)
        if basin == "deep":
            n_dippers += 1
        else:
            n_non += 1

    first_red = first_blue = True
    for color_priority, basin_filter in [("red", lambda b: b != "deep"),
                                          ("blue", lambda b: b == "deep")]:
        for (cond, group), traj in by_traj.items():
            basin = basins.get(group, "shallow")
            if not basin_filter(basin):
                continue
            color = trajectory_color(basin)
            kls = [r["kl"] for r in traj]
            pcs = [r["pc"][pc_idx] for r in traj]
            label = None
            if color == "tab:blue" and first_blue:
                label = f"dippers (deep) — {n_dippers} trajectories"
                first_blue = False
            elif color == "tab:red" and first_red:
                label = f"non-dippers (shallow + mid) — {n_non} trajectories"
                first_red = False
            zorder = 3 if color == "tab:blue" else 2
            ax.plot(kls, pcs, color=color, linewidth=1.5, alpha=alpha,
                    label=label, zorder=zorder)

    # Start + end emphasis
    for (cond, group), traj in by_traj.items():
        for r in (traj[0], traj[-1]):
            ax.scatter([r["kl"]], [r["pc"][pc_idx]], s=80,
                       facecolor="white", edgecolor="black",
                       linewidth=1.8, zorder=5)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KL ↑ (log)", fontsize=12)
    ax.set_ylabel(f"PC{pc_idx + 1}", fontsize=12)
    cond_labels = sorted({r["cond"] for r in records})
    ax.set_title(
        f"PC{pc_idx + 1} vs KL — {' + '.join(cond_labels)} pooled\n"
        f"{len(by_traj)} trajectories | {len(records)} ckpts | "
        f"PCA fit on all pooled shifts",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pc1_vs_pc2(records, basins, out_path, alpha=0.5):
    """PC1×PC2 scatter with marker-size encoding KL and per-trajectory connectors."""
    by_traj = group_by_trajectory(records)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Connectors first (drawn under markers)
    for (cond, group), traj in by_traj.items():
        basin = basins.get(group, "shallow")
        color = trajectory_color(basin)
        pcs1 = [r["pc"][0] for r in traj]
        pcs2 = [r["pc"][1] for r in traj]
        ax.plot(pcs1, pcs2, color=color, linewidth=1.0, alpha=alpha * 0.6,
                zorder=2)

    # Marker size scales with log(KL): small early, big late.
    all_kl = np.array([r["kl"] for r in records])
    log_kl = np.log10(np.maximum(all_kl, 0.01))
    log_kl_min, log_kl_max = log_kl.min(), log_kl.max()
    s_min, s_max = 8, 80

    for color_priority, basin_filter in [("red", lambda b: b != "deep"),
                                          ("blue", lambda b: b == "deep")]:
        for (cond, group), traj in by_traj.items():
            basin = basins.get(group, "shallow")
            if not basin_filter(basin):
                continue
            color = trajectory_color(basin)
            for r in traj:
                lk = np.log10(max(r["kl"], 0.01))
                size = s_min + (s_max - s_min) * (lk - log_kl_min) / max(log_kl_max - log_kl_min, 1e-6)
                ax.scatter([r["pc"][0]], [r["pc"][1]],
                           s=size, color=color, alpha=alpha,
                           edgecolor="none", zorder=3 if color == "tab:blue" else 2)

    # Start + end emphasis (white-with-black-outline) — small for start, big for end
    for (cond, group), traj in by_traj.items():
        ax.scatter([traj[0]["pc"][0]], [traj[0]["pc"][1]], s=70,
                   facecolor="white", edgecolor="black", linewidth=1.6, zorder=5)
        ax.scatter([traj[-1]["pc"][0]], [traj[-1]["pc"][1]], s=120,
                   facecolor="white", edgecolor="black", linewidth=2.0, zorder=5)

    ax.axhline(0, color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    cond_labels = sorted({r["cond"] for r in records})
    ax.set_title(
        f"PC1 × PC2 — {' + '.join(cond_labels)} pooled\n"
        f"marker size scales with log(KL); blue = dippers, red = non-dippers; "
        f"large open dots = trajectory endpoints",
        fontsize=11,
    )
    ax.grid(alpha=0.3)

    # Two-tier legend: basin colors + size scale
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
               markersize=8, alpha=alpha, label='dipper (deep)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red',
               markersize=8, alpha=alpha, label='non-dipper (shallow + mid)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=4, label=f'KL ≈ {10**log_kl_min:.2f} (early)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label=f'KL ≈ {10**log_kl_max:.0f} (late)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='trajectory start (small) / end (large)'),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=9, framealpha=0.95)
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
                        ],
                        help="One or more shifts.pt files. PCA is fit on the pooled shifts.")
    parser.add_argument("--n-components", type=int, default=4,
                        help="Number of PCs to keep (PC1 + PC2 always plotted; rest are reported).")
    parser.add_argument("--out-dir", default="results/qwen_frames/pca",
                        help="Where to write the pooled figures.")
    parser.add_argument("--per-condition", action="store_true",
                        help="Also write per-condition figures (using the same pooled PC basis "
                             "so they're directly comparable). Output goes to "
                             "<shifts_path's parent>/pca/figure_*.png for each condition.")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    # Filter to existing paths so partial runs are OK
    available = [p for p in args.shifts_paths
                 if os.path.isfile(p if os.path.isabs(p) else os.path.join(ROOT, p))]
    missing = [p for p in args.shifts_paths if p not in available]
    if missing:
        print(f"  missing (skipping): {missing}")
    if not available:
        raise SystemExit("No shifts.pt files found.")

    records, basins = load_shifts(available)
    print(f"\nTotal: {len(records)} ckpts, {len(set((r['cond'], r['group']) for r in records))} trajectories")

    X = np.stack([r["shift"] for r in records])  # (n_total, hidden_dim)
    print(f"Shift matrix shape: {X.shape}")
    pca = PCA(n_components=min(args.n_components, X.shape[0], X.shape[1]))
    Y = pca.fit_transform(X)
    print(f"Variance explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative:         {np.cumsum(pca.explained_variance_ratio_)}")

    for i, r in enumerate(records):
        r["pc"] = Y[i]

    out_dir = os.path.join(ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    plot_pc_vs_kl(records, basins, 0, os.path.join(out_dir, "figure_pc1_vs_kl.png"), args.alpha)
    plot_pc_vs_kl(records, basins, 1, os.path.join(out_dir, "figure_pc2_vs_kl.png"), args.alpha)
    plot_pc1_vs_pc2(records, basins, os.path.join(out_dir, "figure_pc1_vs_pc2.png"), args.alpha)

    # Per-condition figures (same pooled PC basis, just one condition's trajectories)
    if args.per_condition:
        for cond in sorted({r["cond"] for r in records}):
            cond_records = [r for r in records if r["cond"] == cond]
            # Find the original shifts.pt path for this condition to derive the output dir
            cond_path = next(p for p in available
                             if os.path.basename(os.path.dirname(p if os.path.isabs(p) else os.path.join(ROOT, p))) == cond)
            cond_dir = os.path.dirname(cond_path if os.path.isabs(cond_path) else os.path.join(ROOT, cond_path))
            cond_out = os.path.join(cond_dir, "pca")
            os.makedirs(cond_out, exist_ok=True)
            print(f"\n--- per-condition: {cond} ({len(cond_records)} ckpts) -> {cond_out} ---")
            plot_pc_vs_kl(cond_records, basins, 0, os.path.join(cond_out, "figure_pc1_vs_kl.png"), args.alpha)
            plot_pc_vs_kl(cond_records, basins, 1, os.path.join(cond_out, "figure_pc2_vs_kl.png"), args.alpha)
            plot_pc1_vs_pc2(cond_records, basins, os.path.join(cond_out, "figure_pc1_vs_pc2.png"), args.alpha)

    summary = {
        "n_components": pca.n_components_,
        "n_records": len(records),
        "hidden_dim": int(X.shape[1]),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_explained": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "shifts_paths": available,
        "n_per_condition": {
            c: sum(1 for r in records if r["cond"] == c)
            for c in sorted({r["cond"] for r in records})
        },
        "n_per_basin": {
            b: sum(1 for r in records
                   if basins.get(r["group"], "shallow") == b)
            for b in ("deep", "mid", "shallow")
        },
    }
    with open(os.path.join(out_dir, "pca_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {os.path.join(out_dir, 'pca_summary.json')}")


if __name__ == "__main__":
    main()
