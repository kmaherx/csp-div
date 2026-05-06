"""Interactive 3D PCA trajectory plot (PC1/PC2/PC3) with hover labels.

Reads shifts.pt, fits PCA on the pooled (normalized) shifts to define PC
space the same way analyze_pca_trajectory.py does, builds one Plotly line
trajectory per seed, colored by an external cluster JSON. Per-seed hover
shows the most descriptive (highest unique-word-count) behavior and
self-verb responses across all available step files — these are the
"best single" examples for the seed, deliberately taken from mid-training
where the persona is interpretable rather than the collapsed final ckpt.

Output is a standalone HTML file so the user can rotate / pan / zoom on
mobile.

Usage:
  python scripts/plot_pc_3d_interactive.py
  python scripts/plot_pc_3d_interactive.py \
      --shifts-paths results/llama/shifts.pt \
      --cluster-json results/llama/kmeansmid_clusters.json \
      --csp-dir results/llama \
      --out results/llama/pca_normalized/figure_pc3d_kmeansmid.html
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

from csp_div.plot_style import (
    CLUSTER_PALETTE, DIPPER_COLOR, NONDIPPER_COLOR, load_cluster_assignments,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def best_in_file(json_path, response_key, condition_key="divergent-in-pos"):
    """Pick the most illustrative response within a single eval file.

    "Most illustrative" = highest unique-word count among the prompts in
    that one file. Long diverse responses beat short collapsed ones
    ("Be Be Be ..."). Returns {prompt, text, approach} or None.
    """
    if not os.path.isfile(json_path):
        return None
    try:
        data = json.load(open(json_path))
    except Exception:
        return None
    items = data.get(condition_key, [])
    best = None
    best_score = -1
    for it in items:
        text = it.get(response_key, "") or ""
        score = len(set(text.split()))
        if score > best_score:
            best_score = score
            best = {
                "text": text,
                "prompt": it.get("prompt", ""),
                "approach": it.get("approach", ""),
            }
    return best


def per_ckpt_responses(eval_dir, ckpt_steps):
    """Return {step: {behav: ..., sv: ...}} for each ckpt step.

    ckpt_steps is the list of integer steps that exist for the seed (e.g.
    [0, 5, 10, ..., 100]). We map each to behavior_step{N}.json and
    self_verb_step{N}.json — except the final step uses the no-suffix files.
    """
    if not os.path.isdir(eval_dir):
        return {}
    max_step = max(ckpt_steps) if ckpt_steps else 0
    out = {}
    for step in ckpt_steps:
        if step == max_step:
            # Final ckpt → behavior.json / self_verb.json
            b_path = os.path.join(eval_dir, "behavior.json")
            sv_path = os.path.join(eval_dir, "self_verb.json")
        else:
            b_path = os.path.join(eval_dir, f"behavior_step{step}.json")
            sv_path = os.path.join(eval_dir, f"self_verb_step{step}.json")
        out[step] = {
            "behav": best_in_file(b_path, "response_csp"),
            "sv": best_in_file(sv_path, "response"),
        }
    return out


def load_pcs(shifts_path, n_pcs=3, normalize=True):
    """Load shifts.pt and project to top n_pcs (normalized PCA, matching
    analyze_pca_trajectory.py)."""
    d = torch.load(shifts_path, map_location="cpu", weights_only=True)
    rows = d["rows"]
    X = np.stack([r["shift"].numpy() for r in rows])
    if normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    pca = PCA(n_components=n_pcs)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    return rows, Y, var


# matplotlib's "tab:*" names aren't valid in plotly — translate to hex.
_TAB_HEX = {
    "tab:blue":   "#1f77b4",
    "tab:red":    "#d62728",
    "tab:purple": "#9467bd",
    "tab:green":  "#2ca02c",
    "tab:orange": "#ff7f0e",
}


def _to_hex(color):
    return _TAB_HEX.get(color, color)


def cluster_to_color(cluster_label):
    """Map cluster label to a hex string for plotly."""
    if cluster_label == "deep":
        return _to_hex(DIPPER_COLOR)
    if cluster_label == "shallow":
        return _to_hex(NONDIPPER_COLOR)
    if cluster_label.startswith("cluster_"):
        idx = int(cluster_label.split("_", 1)[1])
        return _to_hex(CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)])
    return "#888888"


def cluster_display_name(cluster_label):
    """Human-readable population label for legend entries."""
    if cluster_label == "deep":
        return "Population 1 (deep)"
    if cluster_label == "shallow":
        return "Population 2 (shallow)"
    if cluster_label.startswith("cluster_"):
        idx = int(cluster_label.split("_", 1)[1])
        return f"Population {idx + 1}"
    return cluster_label


def truncate(text, n=240):
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n] + "…"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-paths", nargs="+",
                        default=["results/llama/shifts.pt"])
    parser.add_argument("--cluster-json",
                        default="results/llama/kmeansmid_clusters.json")
    parser.add_argument("--csp-dir", default="results/llama",
                        help="Dir holding seed_*/eval/*.json files.")
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--no-normalize", dest="normalize",
                        action="store_false")
    parser.add_argument("--out",
                        default="results/llama/pca_normalized/figure_pc3d_kmeansmid.html")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    # Plot only supports a single shifts.pt for now (one PCA basis); pool would
    # complicate hover-text mapping.
    if len(args.shifts_paths) != 1:
        raise SystemExit("Pass exactly one --shifts-paths file for the 3D view")
    shifts_path = args.shifts_paths[0]
    if not os.path.isabs(shifts_path):
        shifts_path = os.path.join(ROOT, shifts_path)

    rows, Y, var = load_pcs(shifts_path, n_pcs=args.n_pcs,
                            normalize=args.normalize)
    print(f"Loaded {len(rows)} ckpts, projected to {args.n_pcs} PCs")
    print(f"PC variance explained: {var.tolist()} (cum {np.cumsum(var).tolist()})")

    cluster_path = args.cluster_json if os.path.isabs(args.cluster_json) \
        else os.path.join(ROOT, args.cluster_json)
    assignments = load_cluster_assignments(cluster_path)
    print(f"Loaded cluster assignments for {len(assignments)} groups from {cluster_path}")

    csp_dir = args.csp_dir if os.path.isabs(args.csp_dir) \
        else os.path.join(ROOT, args.csp_dir)

    # Group rows by seed, cache best responses per seed
    by_seed = {}
    for i, r in enumerate(rows):
        seed = int(r["group"].split("seed_")[-1].split("_")[0])
        by_seed.setdefault(seed, {"group": r["group"], "rows": []})
        by_seed[seed]["rows"].append({
            "step": r["step"], "kl": r["kl"],
            "pc": Y[i], "ckpt": r["ckpt"],
        })
    for s in by_seed:
        by_seed[s]["rows"].sort(key=lambda x: x["step"])

    print(f"\nBuilding per-ckpt hover text from seed eval files...")
    for seed, info in sorted(by_seed.items()):
        eval_dir = os.path.join(csp_dir, f"seed_{seed}", "eval")
        ckpt_steps = [r["step"] for r in info["rows"]]
        info["per_ckpt"] = per_ckpt_responses(eval_dir, ckpt_steps)

    # Assemble flat dataframe — per-ckpt hover (each row's behavior/self_verb
    # is the most illustrative response *within that ckpt's file*).
    records = []
    for seed, info in sorted(by_seed.items()):
        cluster = assignments.get(info["group"], "shallow")
        color = cluster_to_color(cluster)
        cluster_name = cluster_display_name(cluster)

        for row in info["rows"]:
            ev = info["per_ckpt"].get(row["step"], {})
            b = ev.get("behav") or {}
            sv = ev.get("sv") or {}
            behav_hover = (
                f"<b>behavior</b> (best of file):<br>"
                f"<i>Q:</i> {truncate(b.get('prompt', ''), 120)}<br>"
                f"<i>A:</i> {truncate(b.get('text', ''), 240)}"
            ) if b else "behavior: (none for this ckpt)"
            sv_hover = (
                f"<b>self-verb</b> (best of file, "
                f"{b.get('approach', '') or sv.get('approach', '')}):<br>"
                f"<i>Q:</i> {truncate(sv.get('prompt', ''), 120)}<br>"
                f"<i>A:</i> {truncate(sv.get('text', ''), 240)}"
            ) if sv else "self-verb: (none for this ckpt)"

            records.append({
                "seed": seed,
                "step": row["step"],
                "kl": row["kl"],
                "PC1": row["pc"][0],
                "PC2": row["pc"][1],
                "PC3": row["pc"][2],
                "cluster": cluster,
                "cluster_name": cluster_name,
                "color": color,
                "behavior": behav_hover,
                "self_verb": sv_hover,
            })

    df = pd.DataFrame(records)
    print(f"\nDataframe: {len(df)} rows, {df['seed'].nunique()} seeds, "
          f"{df['cluster'].nunique()} clusters")

    # Build figure: one line trace per seed (so hover groups by seed naturally,
    # and rotation/zoom keeps the lines crisp). Color comes from the cluster.
    fig = go.Figure()
    cluster_legend_done = set()
    cluster_order = sorted(df["cluster"].unique(),
                           key=lambda c: 999 if c == "shallow" else (
                               -1 if c == "deep" else int(c.split("_", 1)[1])
                           ))
    for cluster in cluster_order:
        seeds_in_cluster = sorted(df[df["cluster"] == cluster]["seed"].unique())
        cluster_name = cluster_display_name(cluster)
        color = cluster_to_color(cluster)
        for seed in seeds_in_cluster:
            sub = df[df["seed"] == seed].sort_values("step")
            show_legend = cluster not in cluster_legend_done
            cluster_legend_done.add(cluster)
            customdata = sub[["seed", "step", "kl", "behavior", "self_verb"]].values
            fig.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode="lines+markers",
                name=cluster_name,
                legendgroup=cluster,
                showlegend=show_legend,
                line=dict(color=color, width=3),
                marker=dict(size=3, color=color, opacity=0.7),
                customdata=customdata,
                hovertemplate=(
                    "<b>seed %{customdata[0]}</b>  ·  step %{customdata[1]}  ·  "
                    "KL=%{customdata[2]:.2f}<br>"
                    "PC1=%{x:.2f}  PC2=%{y:.2f}  PC3=%{z:.2f}<br><br>"
                    "%{customdata[3]}<br><br>"
                    "%{customdata[4]}<extra></extra>"
                ),
            ))
            # Endpoint emphasis: open ring at start, filled larger dot at end
            fig.add_trace(go.Scatter3d(
                x=[sub["PC1"].iloc[0]], y=[sub["PC2"].iloc[0]], z=[sub["PC3"].iloc[0]],
                mode="markers", showlegend=False, legendgroup=cluster,
                marker=dict(size=4, color="white", line=dict(color=color, width=2)),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=[sub["PC1"].iloc[-1]], y=[sub["PC2"].iloc[-1]], z=[sub["PC3"].iloc[-1]],
                mode="markers", showlegend=False, legendgroup=cluster,
                marker=dict(size=6, color=color),
                hoverinfo="skip",
            ))

    var_str = " · ".join(f"PC{i+1} {100*v:.1f}%" for i, v in enumerate(var))
    fig.update_layout(
        title=f"PC1/PC2/PC3 trajectories (n={df['seed'].nunique()} seeds)  ·  {var_str}",
        scene=dict(
            xaxis_title=f"PC1 ({100*var[0]:.1f}%)",
            yaxis_title=f"PC2 ({100*var[1]:.1f}%)",
            zaxis_title=f"PC3 ({100*var[2]:.1f}%)",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=800,
    )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
