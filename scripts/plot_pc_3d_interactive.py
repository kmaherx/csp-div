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


def pick_in_file(json_path, response_key, condition_key="divergent-in-pos",
                 prompt_prefix=None):
    """Return one response from an eval file.

    If prompt_prefix is given, return the (single) item whose prompt starts
    with that prefix — this pins the same prompt across every seed/ckpt for
    apples-to-apples comparison. If no prompt matches or prompt_prefix is
    None, fall back to the highest-unique-word-count item (avoids picking
    a collapsed "Be Be Be ..." reply).
    """
    if not os.path.isfile(json_path):
        return None
    try:
        data = json.load(open(json_path))
    except Exception:
        return None
    items = data.get(condition_key, [])
    if not items:
        return None

    if prompt_prefix is not None:
        for it in items:
            if (it.get("prompt", "") or "").startswith(prompt_prefix):
                return {
                    "text": it.get(response_key, "") or "",
                    "prompt": it.get("prompt", ""),
                    "approach": it.get("approach", ""),
                }
        # fall through to score-based pick if no match

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


# Pinned prompts for consistent hover comparison across seeds/ckpts/frames.
# Behavior: the "didn't satisfy my request" prompt — non-trivial, calls for
# adjusting a previous suggestion, surfaces persona well.
# Self-verb: the first multi_frame prompt — asks for the shared theme across
# all 4 syntactic frames, so its prompt text is identical across frames.
BEHAV_PROMPT_PREFIX = "Your suggestion doesn't account for the challenges"
SV_PROMPT_PREFIX = "Find the theme shared by these instructions"


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
            "behav": pick_in_file(b_path, "response_csp",
                                  prompt_prefix=BEHAV_PROMPT_PREFIX),
            "sv":    pick_in_file(sv_path, "response",
                                  prompt_prefix=SV_PROMPT_PREFIX),
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
    parser.add_argument("--score-json", default=None,
                        help="Path to a per-(group, ckpt) score JSON (e.g. "
                             "compute_sae_recon_error.py output). When given, "
                             "markers are colored by --score-field on a "
                             "continuous colormap; lines stay in cluster color "
                             "for population context.")
    parser.add_argument("--score-field", default="rel_err",
                        help="Field name in each score JSON row to use as the "
                             "continuous color value (default 'rel_err').")
    parser.add_argument("--score-label", default=None,
                        help="Colorbar label. Defaults to '<score_field> "
                             "(<method>)' from the score JSON metadata.")
    parser.add_argument("--score-colorscale", default="Viridis",
                        help="Plotly colorscale name (Viridis, Plasma, Inferno, ...).")
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

    # Optional continuous score for marker coloring (e.g. SAE recon error).
    score_lookup = {}      # {(group, ckpt): float}
    score_method = None
    if args.score_json:
        sp = args.score_json if os.path.isabs(args.score_json) \
            else os.path.join(ROOT, args.score_json)
        with open(sp) as f:
            sd = json.load(f)
        score_method = sd.get("method", "score")
        for r in sd.get("rows", []):
            score_lookup[(r["group"], r["ckpt"])] = float(r[args.score_field])
        if not score_lookup:
            raise SystemExit(f"--score-json {sp} had no rows / missing field "
                             f"{args.score_field!r}")
        print(f"Loaded {len(score_lookup)} scores from {sp} "
              f"(method={score_method}, field={args.score_field})")

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

    # Assemble flat dataframe. Hover-displayed text lives in customdata as
    # plain strings — JS in the HTML wrapper renders them into the sidebar.
    # No truncation here: the sidebar handles overflow with vertical scroll.
    records = []
    for seed, info in sorted(by_seed.items()):
        cluster = assignments.get(info["group"], "shallow")
        color = cluster_to_color(cluster)
        cluster_name = cluster_display_name(cluster)

        for row in info["rows"]:
            ev = info["per_ckpt"].get(row["step"], {})
            b = ev.get("behav") or {}
            sv = ev.get("sv") or {}
            score = score_lookup.get((info["group"], row["ckpt"]))
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
                "score": score,
                "behav_prompt": b.get("prompt", "") if b else "",
                "behav_text": b.get("text", "") if b else "(no behavior file for this ckpt)",
                "sv_prompt": sv.get("prompt", "") if sv else "",
                "sv_text": sv.get("text", "") if sv else "(no self-verb file for this ckpt)",
                "sv_approach": (sv.get("approach", "") if sv else ""),
            })

    df = pd.DataFrame(records)
    print(f"\nDataframe: {len(df)} rows, {df['seed'].nunique()} seeds, "
          f"{df['cluster'].nunique()} clusters")

    # Build figure: one line trace per seed (so hover groups by seed naturally,
    # and rotation/zoom keeps the lines crisp). Color comes from the cluster.
    # The default tooltip is intentionally minimal — the rich behavior /
    # self-verb text goes to the right-side sidebar via JS in the HTML wrapper.
    fig = go.Figure()
    cluster_legend_done = set()
    cluster_order = sorted(df["cluster"].unique(),
                           key=lambda c: 999 if c == "shallow" else (
                               -1 if c == "deep" else int(c.split("_", 1)[1])
                           ))
    # Marker styling: continuous colormap if score given, else cluster color.
    use_score = bool(score_lookup) and df["score"].notna().any()
    score_min = df["score"].min() if use_score else None
    score_max = df["score"].max() if use_score else None
    colorbar_label = args.score_label or (
        f"{args.score_field} ({score_method})" if use_score else None
    )

    if use_score:
        # Pre-compute per-(score) hex colors so line segments and markers
        # share the same colormap eval (plotly's Scatter3d line.color is
        # scalar, so we draw 20 mini-segments per trajectory to fake a
        # gradient line). We DO NOT use cluster colors in this mode.
        from plotly.colors import sample_colorscale

        # Honor matplotlib's "_r" suffix convention to reverse a named
        # colorscale. sample_colorscale doesn't take reversescale=True, so
        # we invert the normalized t coordinate ourselves; for the colorbar
        # trace we pass reversescale=True at the trace level.
        cs_name = args.score_colorscale
        cs_reverse = cs_name.endswith("_r")
        if cs_reverse:
            cs_name = cs_name[:-2]

        def _norm(v):
            if score_max == score_min:
                return 0.5
            return float((v - score_min) / (score_max - score_min))

        def color_for(v):
            t = _norm(v)
            if cs_reverse:
                t = 1.0 - t
            return sample_colorscale(cs_name, t)[0]

        all_seeds = sorted(df["seed"].unique())
        for seed in all_seeds:
            sub = df[df["seed"] == seed].sort_values("step").reset_index(drop=True)
            # Per-trace customdata is just [seed, step, kl] — the heavy text
            # fields live in a global JS lookup table embedded in the HTML
            # wrapper, so we don't duplicate them across ~1000 line segments.
            customdata = sub[["seed", "step", "kl"]].values

            # Gradient-line trick: per-segment 2-point traces would give
            # smooth gradient but at 50 seeds × 20 segments = 1000 traces
            # plotly's overhead makes rendering molasses. Instead, bucket
            # segment colors into N_BUCKETS levels and merge consecutive
            # same-bucket segments into one polyline. Drops trace count ~10×
            # with no perceptible visual difference (each color bucket is
            # ~6% of the colormap range).
            N_BUCKETS = 16
            seg_buckets = [
                int(_norm(0.5 * (sub["score"].iloc[i] + sub["score"].iloc[i + 1]))
                    * (N_BUCKETS - 1) + 0.5)
                for i in range(len(sub) - 1)
            ]
            run_start = 0
            for i in range(1, len(seg_buckets) + 1):
                if i == len(seg_buckets) or seg_buckets[i] != seg_buckets[run_start]:
                    # Polyline covers points [run_start .. i] (i+1 points,
                    # connecting i segments that all share the same bucket).
                    bucket = seg_buckets[run_start]
                    bucket_t = bucket / max(1, N_BUCKETS - 1)
                    if cs_reverse:
                        bucket_t = 1.0 - bucket_t
                    color = sample_colorscale(cs_name, bucket_t)[0]
                    fig.add_trace(go.Scatter3d(
                        x=sub["PC1"].iloc[run_start:i + 1],
                        y=sub["PC2"].iloc[run_start:i + 1],
                        z=sub["PC3"].iloc[run_start:i + 1],
                        mode="lines",
                        line=dict(color=color, width=6),
                        showlegend=False,
                        customdata=customdata[run_start:i + 1],
                        hovertemplate=(
                            "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
                        ),
                    ))
                    run_start = i

            # Marker trace — per-point gradient + customdata for hover.
            fig.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=4,
                    color=sub["score"].tolist(),
                    cmin=score_min, cmax=score_max,
                    colorscale=cs_name,
                    reversescale=cs_reverse,
                    opacity=0.9,
                    showscale=False,
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
                ),
            ))
            # Endpoint emphasis: enlarged open ring at start (white fill,
            # gradient-colored ring), black-filled dot at end. Black end
            # marker is universal across trajectories so the convergence
            # is visually consistent regardless of the per-trajectory color.
            start_color = color_for(sub["score"].iloc[0])
            fig.add_trace(go.Scatter3d(
                x=[sub["PC1"].iloc[0]], y=[sub["PC2"].iloc[0]], z=[sub["PC3"].iloc[0]],
                mode="markers", showlegend=False,
                marker=dict(size=8, color="white",
                            line=dict(color=start_color, width=2)),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=[sub["PC1"].iloc[-1]], y=[sub["PC2"].iloc[-1]], z=[sub["PC3"].iloc[-1]],
                mode="markers", showlegend=False,
                marker=dict(size=6, color="black"),
                hoverinfo="skip",
            ))

        # One shared colorbar driven by an invisible Scatter3d trace.
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=0.001, color=[score_min, score_max],
                cmin=score_min, cmax=score_max,
                colorscale=cs_name,
                reversescale=cs_reverse,
                showscale=True,
                colorbar=dict(
                    title=dict(text=colorbar_label, side="right"),
                    thickness=14, len=0.6, x=1.02,
                ),
                opacity=0,
            ),
            hoverinfo="skip",
        ))
    else:
        # Cluster-discrete coloring (the original kmeansmid/kmeans3 mode).
        for cluster in cluster_order:
            seeds_in_cluster = sorted(df[df["cluster"] == cluster]["seed"].unique())
            cluster_name = cluster_display_name(cluster)
            color = cluster_to_color(cluster)
            for seed in seeds_in_cluster:
                sub = df[df["seed"] == seed].sort_values("step")
                show_legend = cluster not in cluster_legend_done
                cluster_legend_done.add(cluster)
                customdata = sub[["seed", "step", "kl"]].values
                fig.add_trace(go.Scatter3d(
                    x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                    mode="lines+markers",
                    name=cluster_name,
                    legendgroup=cluster,
                    showlegend=show_legend,
                    line=dict(color=color, width=6),
                    marker=dict(size=3, color=color, opacity=0.7),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Scatter3d(
                    x=[sub["PC1"].iloc[0]], y=[sub["PC2"].iloc[0]], z=[sub["PC3"].iloc[0]],
                    mode="markers", showlegend=False, legendgroup=cluster,
                    marker=dict(size=8, color="white",
                                line=dict(color=color, width=2)),
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter3d(
                    x=[sub["PC1"].iloc[-1]], y=[sub["PC2"].iloc[-1]], z=[sub["PC3"].iloc[-1]],
                    mode="markers", showlegend=False, legendgroup=cluster,
                    marker=dict(size=6, color="black"),
                    hoverinfo="skip",
                ))

    var_str = " · ".join(f"PC{i+1} {100*v:.1f}%" for i, v in enumerate(var))
    fig.update_layout(
        title=dict(
            text=(
                f"PC1/PC2/PC3 trajectories  (n={df['seed'].nunique()} seeds)"
                f"  ·  {var_str}"
                "<br><span style='font-size:13px;color:#555'>"
                "○ Starting Points  ·  ● Ending Points"
                "</span>"
            ),
            x=0.02, xanchor="left",
        ),
        scene=dict(
            xaxis_title=f"PC1 ({100*var[0]:.1f}%)",
            yaxis_title=f"PC2 ({100*var[1]:.1f}%)",
            zaxis_title=f"PC3 ({100*var[2]:.1f}%)",
            # Equal physical scale on all 3 axes so rotation doesn't
            # visually distort PC magnitudes.
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
        autosize=True,
    )

    # Build a global lookup table {f"{seed}_{step}": {fields}} so the JS hover
    # handler can fetch the heavy behavior/self-verb text from a single shared
    # store rather than from per-trace customdata. Massive size reduction.
    cell_data = {}
    for r in records:
        cell_data[f"{r['seed']}_{r['step']}"] = {
            "behav_prompt": r["behav_prompt"],
            "behav_text":   r["behav_text"],
            "sv_prompt":    r["sv_prompt"],
            "sv_text":      r["sv_text"],
            "sv_approach":  r["sv_approach"],
            "cluster_name": r["cluster_name"],
        }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_sidebar_html(fig, out_path, cell_data)
    print(f"\nSaved: {out_path}")


# ── Custom HTML with right-side sidebar ─────────────────────────────────

SIDEBAR_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CSP PC1/PC2/PC3 trajectories</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; height: 100%; font-family: 'Libertinus Serif', Georgia, serif; }
  #wrap { display: flex; height: 100vh; }
  #plot { flex: 1; min-width: 0; }
  #sidebar {
    width: 460px; padding: 16px 18px;
    overflow-y: auto; box-sizing: border-box;
    border-left: 1px solid #ddd; background: #fafafa;
  }
  #sidebar h2 {
    margin: 0 0 4px 0; font-size: 16px; color: #222;
  }
  #sidebar .meta { font-size: 12px; color: #666; margin-bottom: 12px; }
  .panel {
    margin-bottom: 12px;
    padding: 10px 12px; background: #fff;
    border: 1px solid #e0e0e0; border-radius: 4px;
  }
  .panel-title {
    font-size: 11px; font-weight: 600;
    color: #555; text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 6px;
  }
  .panel .prompt {
    color: #999; font-size: 11.5px; margin-bottom: 6px;
    font-style: italic;
  }
  .panel .response {
    font-size: 12.5px; line-height: 1.4; color: #222;
    white-space: pre-wrap; word-wrap: break-word;
    max-height: 38vh; overflow-y: auto;
  }
  .placeholder { color: #aaa; font-style: italic; }
</style>
</head>
<body>
<div id="wrap">
  <div id="plot"></div>
  <div id="sidebar">
    <h2 id="title"><span class="placeholder">Hover a point to see outputs</span></h2>
    <div id="meta" class="meta"></div>
    <div class="panel">
      <div class="panel-title">Behavior (best in ckpt)</div>
      <div id="behav-prompt" class="prompt"></div>
      <div id="behav-text" class="response"><span class="placeholder">—</span></div>
    </div>
    <div class="panel">
      <div class="panel-title">Self-verb (best in ckpt)</div>
      <div id="sv-prompt" class="prompt"></div>
      <div id="sv-text" class="response"><span class="placeholder">—</span></div>
    </div>
  </div>
</div>
<script>
  var fig = __FIGURE_JSON__;
  var cellData = __CELL_DATA_JSON__;
  Plotly.newPlot('plot', fig.data, fig.layout, {responsive: true, displaylogo: false});

  var titleEl  = document.getElementById('title');
  var metaEl   = document.getElementById('meta');
  var bpEl     = document.getElementById('behav-prompt');
  var btEl     = document.getElementById('behav-text');
  var spEl     = document.getElementById('sv-prompt');
  var stEl     = document.getElementById('sv-text');

  function setText(el, txt, prefix) {
    if (txt === undefined || txt === null || txt === '') {
      el.innerHTML = '<span class="placeholder">—</span>';
    } else {
      el.textContent = (prefix || '') + txt;
    }
  }

  document.getElementById('plot').on('plotly_hover', function(ev) {
    if (!ev.points || !ev.points.length) return;
    var d = ev.points[0].customdata;
    if (!d) return;
    var seed = d[0], step = d[1], kl = d[2];
    // Optional 4th customdata field is an explicit cellData lookup key.
    // Fallback to seed_step for back-compat with the original plot.
    var lookupKey = d[3] || (seed + '_' + step);
    var info = cellData[lookupKey] || {};
    titleEl.textContent = 'Seed ' + seed;
    var cname = info.cluster_name || '';
    metaEl.textContent = (cname ? cname + '  ·  ' : '')
      + 'step ' + step + '  ·  KL=' + Number(kl).toFixed(2);
    setText(bpEl, info.behav_prompt, 'Q: ');
    setText(btEl, info.behav_text, '');
    var svPrefix = info.sv_approach ? '[' + info.sv_approach + '] ' : '';
    setText(spEl, svPrefix + (info.sv_prompt || ''), 'Q: ');
    setText(stEl, info.sv_text, '');
  });
</script>
</body>
</html>
"""


def write_sidebar_html(fig, out_path, cell_data):
    """Write a custom HTML page with the plotly figure on the left and a
    fixed-position text sidebar on the right driven by plotly_hover events.
    cell_data is a {f'{seed}_{step}': {fields}} lookup so per-trace
    customdata can stay tiny instead of duplicating the heavy text fields
    onto every line segment."""
    fig_json = fig.to_json()  # serializable JSON (lists, not numpy arrays)
    cell_json = json.dumps(cell_data)
    html = (SIDEBAR_HTML_TEMPLATE
            .replace("__FIGURE_JSON__", fig_json)
            .replace("__CELL_DATA_JSON__", cell_json))
    with open(out_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
