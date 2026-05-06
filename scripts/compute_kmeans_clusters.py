"""Compute k=2 k-means clusters on per-seed trajectories in PC space.

Loads shifts.pt, fits PCA the same way as analyze_pca_trajectory.py (default
--normalize), then for each seed flattens its trajectory across the leading
--n-pcs PCs into a single feature vector and runs k-means with k=2.

The cluster label is aligned with the existing cosine-threshold convention
(plot_style.basin_from_cos): the cluster whose members are predominantly
"deep" under the cosine classifier is relabeled "deep" (blue dipper); the
other is "shallow" (red non-dipper). This keeps the red/blue theme stable
across the cosine-classified and k-means-classified figures.

Output: JSON {group: "deep"|"shallow"} that downstream plot scripts can
consume via their --cluster-json flag.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from csp_div.plot_style import basin_from_cos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_shifts_and_basins(shifts_paths, axis_json_path):
    """Return (records, cosine_basins).

    records: list of {group, step, shift (np)} pooled across all shifts.pt files.
    cosine_basins: {group: "deep"|"mid"|"shallow"} from the sibling axis.json
                   (using the existing trajectory_basin convention).
    """
    records = []
    for path in shifts_paths:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        d = torch.load(full, map_location="cpu", weights_only=True)
        for r in d["rows"]:
            records.append({
                "group": r["group"],
                "step": r["step"],
                "shift": r["shift"].numpy(),
            })

    with open(axis_json_path) as f:
        rows = json.load(f)["rows"]
    by_group = {}
    for r in rows:
        if r["step"] < 5:
            continue
        by_group.setdefault(r["group"], []).append(r["proj_cos"])
    cosine_basins = {g: basin_from_cos(min(cs)) for g, cs in by_group.items()}
    return records, cosine_basins


def trajectory_features(records, n_pcs, normalize, step_range=None):
    """Run PCA on pooled shifts, then return {group: feature_vec, np}.

    step_range = (min, max) inclusive: per-seed features only include steps
    in that range. PCA is always fit on the full record set so the PC basis
    stays comparable across runs.
    """
    X = np.stack([r["shift"] for r in records])
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-8)
    pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]))
    Y = pca.fit_transform(X)  # (n_records, n_pcs)

    # Build per-group trajectory feature: stack PC coords sorted by step,
    # then flatten. All trajectories in this study share the same step grid
    # (21 ckpts) so flattened features are aligned across seeds.
    by_group = {}
    for i, r in enumerate(records):
        if step_range is not None:
            lo, hi = step_range
            if not (lo <= r["step"] <= hi):
                continue
        by_group.setdefault(r["group"], []).append((r["step"], Y[i]))
    features = {}
    for g, items in by_group.items():
        items.sort(key=lambda x: x[0])
        features[g] = np.concatenate([pc for _, pc in items])
    return features, pca.explained_variance_ratio_.tolist()


def _deep_score(cluster_id, kmeans_labels, groups, cosine_basins):
    """Fraction of cosine-deep trajectories in this cluster (higher = deeper)."""
    members = [g for g, cid in zip(groups, kmeans_labels) if cid == cluster_id]
    if not members:
        return -1.0
    return sum(1 for g in members
               if cosine_basins.get(g, "shallow") == "deep") / len(members)


def align_labels_k2(kmeans_labels, groups, cosine_basins):
    """k=2: emit 'deep'/'shallow' aligned to cosine convention."""
    score = {cid: _deep_score(cid, kmeans_labels, groups, cosine_basins)
             for cid in (0, 1)}
    deep_id = 0 if score[0] >= score[1] else 1
    return {g: ("deep" if cid == deep_id else "shallow")
            for g, cid in zip(groups, kmeans_labels)}


def align_labels_kn(kmeans_labels, groups, cosine_basins, k):
    """k>=3: emit 'cluster_0'..'cluster_{k-1}' sorted by deep-fraction
    descending so cluster_0 is the deepest. The plot palette indexes into
    these positions (cluster_0=blue, cluster_1=red, cluster_2=purple, …)."""
    score = {cid: _deep_score(cid, kmeans_labels, groups, cosine_basins)
             for cid in range(k)}
    # Sort cluster IDs by score descending: deepest first
    ranked = sorted(range(k), key=lambda cid: -score[cid])
    rank_of = {cid: i for i, cid in enumerate(ranked)}
    return {g: f"cluster_{rank_of[cid]}"
            for g, cid in zip(groups, kmeans_labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-paths", nargs="+",
                        default=["results/llama/shifts.pt"])
    parser.add_argument("--axis-json", default="results/llama/axis.json",
                        help="Sibling axis.json for cosine-threshold label "
                             "alignment (deep cluster gets blue).")
    parser.add_argument("--n-pcs", type=int, default=2,
                        help="Number of leading PCs to flatten into the per-seed feature.")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize each shift before PCA (matches "
                             "analyze_pca_trajectory --normalize).")
    parser.add_argument("--step-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                        default=None,
                        help="Restrict per-seed features to ckpts with step in "
                             "[MIN, MAX] inclusive. PCA is still fit on all "
                             "records (PC basis unchanged). Use to focus k-means "
                             "on the divergent middle portion of the trajectory.")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of clusters. k=2 emits 'deep'/'shallow' "
                             "labels (aligned with cosine convention); k>=3 "
                             "emits 'cluster_0'..'cluster_{k-1}' sorted by "
                             "deep-fraction descending.")
    parser.add_argument("--seed", type=int, default=0,
                        help="K-means RNG seed (n_init=10 by default).")
    parser.add_argument("--out", default="results/llama/kmeans_clusters.json")
    args = parser.parse_args()

    available = [p for p in args.shifts_paths
                 if os.path.isfile(p if os.path.isabs(p) else os.path.join(ROOT, p))]
    if not available:
        raise SystemExit(f"No shifts.pt files found in: {args.shifts_paths}")

    records, cosine_basins = load_shifts_and_basins(available, args.axis_json)
    print(f"Loaded {len(records)} ckpt rows across {len(available)} shifts.pt file(s)")
    print(f"cosine_basins: {sum(1 for b in cosine_basins.values() if b == 'deep')} deep / "
          f"{sum(1 for b in cosine_basins.values() if b != 'deep')} non-deep "
          f"(of {len(cosine_basins)} groups)")

    features, var_ratio = trajectory_features(
        records, n_pcs=args.n_pcs, normalize=args.normalize,
        step_range=tuple(args.step_range) if args.step_range else None,
    )
    print(f"PCA explained variance: {[round(v, 4) for v in var_ratio]}")
    if args.step_range:
        print(f"Step-range filter: {args.step_range}")

    groups = sorted(features.keys())
    F = np.stack([features[g] for g in groups])  # (n_seeds, n_pcs * n_steps)
    print(f"K-means input: {F.shape}  (n_seeds={F.shape[0]}, "
          f"feature_dim={args.n_pcs} PCs * {F.shape[1] // args.n_pcs} steps)")

    km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed)
    labels = km.fit_predict(F)

    if args.k == 2:
        aligned = align_labels_k2(labels, groups, cosine_basins)
    else:
        aligned = align_labels_kn(labels, groups, cosine_basins, args.k)

    # Cluster sizes + per-cluster cosine-deep fraction (for diagnostics)
    cluster_sizes = {cid: int((labels == cid).sum()) for cid in range(args.k)}
    deep_frac = {cid: _deep_score(cid, labels, groups, cosine_basins)
                 for cid in range(args.k)}
    print(f"\nK-means cluster sizes (raw cluster IDs): {cluster_sizes}")
    print(f"Per-cluster cosine-deep fraction: "
          f"{ {cid: round(f, 3) for cid, f in deep_frac.items()} }")

    label_counts = {}
    for v in aligned.values():
        label_counts[v] = label_counts.get(v, 0) + 1
    print(f"\nFinal labels: {label_counts}")

    # Agreement with cosine-threshold classifier (only meaningful for k=2)
    if args.k == 2:
        agree = sum(1 for g in aligned
                    if (aligned[g] == "deep") == (cosine_basins.get(g, "shallow") == "deep"))
        print(f"Agreement with cosine-threshold classifier: {agree}/{len(aligned)} "
              f"({100*agree/len(aligned):.1f}%)")
    else:
        agree = None

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "method": "kmeans",
        "k": args.k,
        "n_pcs": args.n_pcs,
        "normalize": args.normalize,
        "step_range": list(args.step_range) if args.step_range else None,
        "seed": args.seed,
        "explained_variance_ratio": var_ratio,
        "cluster_sizes_raw": cluster_sizes,
        "cosine_deep_fraction_per_cluster": deep_frac,
        "label_counts": label_counts,
        "assignments": aligned,
    }
    if agree is not None:
        payload["agreement_with_cosine_threshold"] = f"{agree}/{len(aligned)}"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
