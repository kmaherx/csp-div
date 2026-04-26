"""PCA over all CSP embeddings, plot PC1×PC2 trajectories per seed/run.

Looks at every sp_pos*.pt under results/, fits PCA on the full set, and draws
a line per seed/run connecting that seed's checkpoints in step order, colored
by KL. Tests visually whether the diverse seed trajectories converge in
embedding space.

Usage: python analyze_pca_trajectory.py [--out plot.png] [--results-dir results]
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_step(ckpt_name, kl_curve_len):
    """Parse step number. sp_pos_step100.pt -> 100. sp_pos.pt -> kl_curve_len."""
    m = re.match(r"sp_pos_step(\d+)\.pt$", ckpt_name)
    if m:
        return int(m.group(1))
    return kl_curve_len  # final ckpt


RUN_LABELS = {
    "divergent":      "run1 (s42, 500 steps)",
    "divergent_run2": "run2 (s123, 500 steps)",
    "divergent_run3": "run3 (s7, 100 steps)",
    "divergent_run4": "run4 (s99, 100 steps)",
    "divergent_run5": "run5 (s333, 100 steps)",
}


def label_for(group):
    """Pretty label for a group/dir name."""
    if group in RUN_LABELS:
        return RUN_LABELS[group]
    if group.startswith("early_stop/seed_"):
        return f"seed_{group.split('_')[-1]} (KL≤10)"
    return group


def collect_checkpoints(results_dir):
    """Walk results/early_stop/seed_*/, return list of dicts:
       {group, label, step, kl, embedding (1D np)}.
    Limited to the clean early-stop-KL=10 batch — earlier runs (run1-5 with
    different settings) are excluded."""
    items = []
    pattern = os.path.join(results_dir, "early_stop", "seed_*", "sp_pos*.pt")
    for path in sorted(glob.glob(pattern)):
        rel = os.path.relpath(path, results_dir)
        ckpt_name = os.path.basename(path)
        # Group = parent dir, e.g. "early_stop/seed_3"
        group = os.path.dirname(rel)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  skip {rel}: {e}")
            continue
        emb = ckpt["embedding"].flatten().float().numpy()
        kl = ckpt.get("final_kl") or 0.0
        kl_curve = ckpt.get("kl_curve") or []
        step = parse_step(ckpt_name, len(kl_curve))
        items.append({
            "group": group, "ckpt": ckpt_name, "step": step,
            "kl": float(kl), "emb": emb,
        })
    return items


def draw_panel(ax, Y, items, by_group, pc_x, pc_y, evr, norm, cmap):
    for group, idxs in by_group.items():
        pts = np.array([[Y[i, pc_x], Y[i, pc_y]] for i in idxs])
        kls = np.array([items[i]["kl"] for i in idxs])
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_kls = (kls[:-1] + kls[1:]) / 2
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.6)
            lc.set_array(seg_kls)
            ax.add_collection(lc)
        ax.scatter(pts[:, 0], pts[:, 1], c=kls, cmap=cmap, norm=norm,
                   s=30, edgecolor="black", linewidth=0.4, zorder=3)
        last = pts[-1]
        ax.annotate(label_for(group),
                    xy=last, xytext=(5, 5), textcoords="offset points",
                    fontsize=7, alpha=0.8)
    ax.set_xlabel(f"PC{pc_x+1} ({evr[pc_x]:.1%} var)")
    ax.set_ylabel(f"PC{pc_y+1} ({evr[pc_y]:.1%} var)")
    ax.set_title(f"PC{pc_x+1} × PC{pc_y+1}")
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "results", "pca_trajectory.png"))
    parser.add_argument("--n-components", type=int, default=6)
    args = parser.parse_args()

    print(f"Scanning {args.results_dir}...")
    items = collect_checkpoints(args.results_dir)
    print(f"  found {len(items)} checkpoints across "
          f"{len(set(i['group'] for i in items))} groups")

    X = np.stack([i["emb"] for i in items])
    print(f"  embedding matrix: {X.shape}")
    pca = PCA(n_components=args.n_components)
    Y = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print(f"  explained variance per PC: " +
          ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr)))
    print(f"  cumulative through PC{args.n_components}: {evr.sum():.1%}")

    by_group = {}
    for idx, it in enumerate(items):
        by_group.setdefault(it["group"], []).append(idx)
    for g in by_group:
        by_group[g].sort(key=lambda i: items[i]["step"])

    all_kls = np.array([i["kl"] for i in items])
    norm = matplotlib.colors.LogNorm(
        vmin=max(all_kls.min(), 0.05), vmax=all_kls.max(),
    )
    cmap = plt.get_cmap("viridis")

    # Three panels: PC1×PC2, PC3×PC4, PC5×PC6
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    pairs = [(0, 1), (2, 3), (4, 5)]
    for ax, (px, py) in zip(axes, pairs):
        draw_panel(ax, Y, items, by_group, px, py, evr, norm, cmap)

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("KL ↑ (log scale)")
    fig.suptitle(
        f"CSP optimization trajectories — {len(items)} checkpoints across "
        f"{len(by_group)} runs, connected in step order  "
        f"(cum. var through PC6: {evr.sum():.1%})",
        fontsize=12,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")

    import json
    json_out = args.out.replace(".png", ".json")
    rec = {
        "n_components": args.n_components,
        "explained_variance_ratio": evr.tolist(),
        "items": [
            {"group": it["group"], "ckpt": it["ckpt"], "step": it["step"],
             "kl": it["kl"], "pca": Y[idx].tolist()}
            for idx, it in enumerate(items)
        ],
    }
    with open(json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved: {json_out}")


if __name__ == "__main__":
    main()
