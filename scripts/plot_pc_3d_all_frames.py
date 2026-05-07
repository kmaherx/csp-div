"""Combined 3D PC trajectory plot across all 4 syntactic frames (Be, Act,
Please, You should). All trajectories rendered as plain grey lines —
no per-trajectory color, no per-ckpt dots, just the line shape so the
eye reads overall structure across 200 trajectories.

Hover behavior preserved: each line trace carries customdata at every
joint, so hovering anywhere on a trajectory snaps to the nearest ckpt
and updates the right-side sidebar with that (frame, seed, step)'s
behavior + self-verb outputs.

PCA basis: fit ONCE on the pooled shifts (4 × 1050 = 4200 rows), so
PC1/PC2/PC3 are the dominant directions across the full ablation —
not per-frame. This is the right basis for visually comparing whether
frames inhabit the same manifold or pull apart.

Usage:
  python scripts/plot_pc_3d_all_frames.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

# Reuse the helpers + sidebar template from the per-frame script.
from plot_pc_3d_interactive import (  # type: ignore
    BEHAV_PROMPT_PREFIX, SV_PROMPT_PREFIX,
    SIDEBAR_HTML_TEMPLATE, per_ckpt_responses,
    truncate, write_sidebar_html,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Default frames + their result dirs (all relative to project root).
DEFAULT_FRAMES = [
    ("be",        "results/llama"),
    ("act",       "results/llama_act"),
    ("please",    "results/llama_please"),
    ("youshould", "results/llama_youshould"),
]


FRAME_DISPLAY = {
    "be":        "Be {sp}.",
    "act":       "Act {sp}.",
    "please":    "Please {sp}.",
    "youshould": "You should {sp}.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", action="append", nargs=2,
                        metavar=("SLUG", "DIR"), default=None,
                        help="Override the default frame list. Repeat: "
                             "--frame be results/llama --frame act results/llama_act ...")
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--out",
                        default="results/all_frames/figure_pc3d_all_frames.html")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    frames = args.frame or DEFAULT_FRAMES

    # ── 1. Pool shifts across frames; fit one PCA on the union. ────────
    pooled_X = []
    row_index = []  # parallel list of (frame_slug, group, ckpt, step, kl)
    for slug, base in frames:
        shifts_path = base if os.path.isabs(base) else os.path.join(ROOT, base)
        shifts_path = os.path.join(shifts_path, "shifts.pt")
        if not os.path.isfile(shifts_path):
            raise SystemExit(f"missing {shifts_path}")
        d = torch.load(shifts_path, map_location="cpu", weights_only=True)
        for r in d["rows"]:
            pooled_X.append(r["shift"].numpy())
            row_index.append((slug, r["group"], r["ckpt"], int(r["step"]),
                              float(r["kl"])))
        print(f"  loaded {len(d['rows'])} rows from {slug}: {shifts_path}")

    X = np.stack(pooled_X)
    if args.normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-8)

    pca = PCA(n_components=args.n_pcs)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"\nPooled PCA on {X.shape[0]} rows × {X.shape[1]}-d shifts")
    print(f"  variance: {[round(v, 4) for v in var]} (cum {[round(c, 4) for c in np.cumsum(var)]})")

    # ── 2. Group rows into per-(frame, seed) trajectories. ─────────────
    by_traj = {}
    for i, (slug, group, ckpt, step, kl) in enumerate(row_index):
        try:
            seed = int(group.split("seed_")[-1].split("_")[0])
        except ValueError:
            continue
        key = (slug, seed)
        by_traj.setdefault(key, {
            "slug": slug, "seed": seed, "group": group, "rows": [],
        })
        by_traj[key]["rows"].append({
            "step": step, "kl": kl, "ckpt": ckpt, "pc": Y[i],
        })
    for k in by_traj:
        by_traj[k]["rows"].sort(key=lambda r: r["step"])
    print(f"  {len(by_traj)} trajectories total")

    # ── 3. Per-(frame, seed) hover text (best-of-file with the pinned prompts).
    print(f"\nLoading per-ckpt eval files for hover ...")
    cell_data = {}
    for (slug, seed), info in by_traj.items():
        # Match the dir from frames (we may have it relative or absolute)
        base = next((b for s, b in frames if s == slug), None)
        if base is None:
            continue
        if not os.path.isabs(base):
            base = os.path.join(ROOT, base)
        eval_dir = os.path.join(base, f"seed_{seed}", "eval")
        ckpt_steps = [r["step"] for r in info["rows"]]
        per_ckpt = per_ckpt_responses(eval_dir, ckpt_steps)
        for step, ev in per_ckpt.items():
            b = ev.get("behav") or {}
            sv = ev.get("sv") or {}
            cell_data[f"{slug}_{seed}_{step}"] = {
                "behav_prompt": b.get("prompt", "") if b else "",
                "behav_text":   b.get("text", "")
                    if b else "(no behavior file for this ckpt)",
                "sv_prompt":    sv.get("prompt", "") if sv else "",
                "sv_text":      sv.get("text", "")
                    if sv else "(no self-verb file for this ckpt)",
                "sv_approach":  sv.get("approach", "") if sv else "",
                # Reusing the cluster_name slot for the frame label so the
                # sidebar meta line shows e.g. "Frame: Act {sp}.".
                "cluster_name": f"Frame: {FRAME_DISPLAY.get(slug, slug)}",
            }

    # ── 4. Build the figure: 1 grey line + start + end markers per traj. ──
    GREY_LINE = "#888888"
    GREY_RING = "#666666"
    BLACK_END = "#000000"

    fig = go.Figure()
    for (slug, seed), info in sorted(by_traj.items()):
        rows = info["rows"]
        pc1 = [r["pc"][0] for r in rows]
        pc2 = [r["pc"][1] for r in rows]
        pc3 = [r["pc"][2] for r in rows]
        # Per-vertex customdata: [seed, step, kl, lookup_key]. Plotly snaps
        # hover to the nearest vertex on the line, so even though there are
        # no markers drawn at intermediate ckpts, hover still resolves to
        # the closest one and pulls its text from cellData.
        customdata = [
            [seed, r["step"], r["kl"], f"{slug}_{seed}_{r['step']}"]
            for r in rows
        ]
        fig.add_trace(go.Scatter3d(
            x=pc1, y=pc2, z=pc3,
            mode="lines",
            line=dict(color=GREY_LINE, width=2.5),
            opacity=0.7,
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
        # Start marker — open ring, grey
        fig.add_trace(go.Scatter3d(
            x=[pc1[0]], y=[pc2[0]], z=[pc3[0]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color="white",
                        line=dict(color=GREY_RING, width=1.5)),
            customdata=[customdata[0]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
        # End marker — black filled
        fig.add_trace(go.Scatter3d(
            x=[pc1[-1]], y=[pc2[-1]], z=[pc3[-1]],
            mode="markers", showlegend=False,
            marker=dict(size=5, color=BLACK_END),
            customdata=[customdata[-1]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))

    var_str = " · ".join(f"PC{i+1} {100*v:.1f}%" for i, v in enumerate(var))
    n_traj = len(by_traj)
    n_frames = len({s for s, _ in frames})
    fig.update_layout(
        title=dict(
            text=(
                f"PC1/PC2/PC3 trajectories  "
                f"({n_frames} frames × {n_traj // n_frames} seeds = {n_traj} trajectories)"
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
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        autosize=True,
    )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_sidebar_html(fig, out_path, cell_data)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
