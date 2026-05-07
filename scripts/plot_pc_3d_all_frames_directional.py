"""All-frames combined 3D PC plot with direction-of-flow indicated by a
grey-shade gradient along each trajectory: light grey at the start,
near-black at the end. Same "all grey" aesthetic as the plain version —
just multiple shades of grey rather than one — so direction reads at a
glance without arrows or extra glyphs.

Sister script to plot_pc_3d_all_frames.py. Same data layout (4 frames
× 50 seeds = 200 trajectories), same shared PCA basis fit on the
pooled shifts, same hover behavior at every joint.
"""
import argparse
import os

import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib import colormaps
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA

from plot_pc_3d_interactive import (  # type: ignore
    BEHAV_PROMPT_PREFIX, SV_PROMPT_PREFIX,
    per_ckpt_responses, write_sidebar_html,
)
from plot_pc_3d_all_frames import DEFAULT_FRAMES, FRAME_DISPLAY  # type: ignore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# matplotlib's 'coolwarm' runs blue (t=0) → white (t=0.5) → red (t=1).
# t=0 = trajectory start (cool blue), t=1 = trajectory end (warm red).
_COOLWARM = colormaps["coolwarm"]


def progress_color(t):
    return to_hex(_COOLWARM(t))


START_COLOR = progress_color(0.0)  # cool blue
END_COLOR   = progress_color(1.0)  # warm red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", action="append", nargs=2,
                        metavar=("SLUG", "DIR"), default=None)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--n-buckets", type=int, default=8,
                        help="How many discrete grey shades to use along each "
                             "trajectory. Higher = smoother gradient, more traces.")
    parser.add_argument("--out",
                        default="results/all_frames/figure_pc3d_all_frames_directional.html")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    frames = args.frame or DEFAULT_FRAMES

    # ── 1. Pool + PCA (same as the plain all-frames script). ───────────
    pooled_X = []
    row_index = []
    for slug, base in frames:
        shifts_path = os.path.join(
            base if os.path.isabs(base) else os.path.join(ROOT, base),
            "shifts.pt",
        )
        d = torch.load(shifts_path, map_location="cpu", weights_only=True)
        for r in d["rows"]:
            pooled_X.append(r["shift"].numpy())
            row_index.append((slug, r["group"], r["ckpt"], int(r["step"]),
                              float(r["kl"])))
        print(f"  loaded {len(d['rows'])} rows from {slug}")

    X = np.stack(pooled_X)
    if args.normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-8)
    pca = PCA(n_components=args.n_pcs)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"\nPooled PCA: variance {[round(v, 4) for v in var]}  "
          f"(cum {[round(c, 4) for c in np.cumsum(var)]})")

    # ── 2. Group into per-(frame, seed) trajectories. ───────────────────
    by_traj = {}
    for i, (slug, group, ckpt, step, kl) in enumerate(row_index):
        try:
            seed = int(group.split("seed_")[-1].split("_")[0])
        except ValueError:
            continue
        by_traj.setdefault((slug, seed), {
            "slug": slug, "seed": seed, "group": group, "rows": [],
        })["rows"].append({"step": step, "kl": kl, "ckpt": ckpt, "pc": Y[i]})
    for k in by_traj:
        by_traj[k]["rows"].sort(key=lambda r: r["step"])
    print(f"  {len(by_traj)} trajectories")

    # ── 3. Hover text from per-frame eval files. ───────────────────────
    print(f"\nLoading per-ckpt eval files for hover ...")
    cell_data = {}
    for (slug, seed), info in by_traj.items():
        base = next((b for s, b in frames if s == slug), None)
        if base is None:
            continue
        if not os.path.isabs(base):
            base = os.path.join(ROOT, base)
        eval_dir = os.path.join(base, f"seed_{seed}", "eval")
        per_ckpt = per_ckpt_responses(eval_dir, [r["step"] for r in info["rows"]])
        suffix = FRAME_SUFFIX.get(slug, "")
        for step, ev in per_ckpt.items():
            b = ev.get("behav") or {}
            sv = ev.get("sv") or {}
            # Append the per-frame eval suffix to the behavior question so the
            # displayed Q reflects what the model actually saw at eval time
            # (the question + frame template like "Be §.").
            behav_prompt = b.get("prompt", "") if b else ""
            if behav_prompt and suffix:
                behav_prompt = f"{behav_prompt} {suffix}"
            cell_data[f"{slug}_{seed}_{step}"] = {
                "behav_prompt": behav_prompt,
                "behav_text":   b.get("text", "")
                    if b else "(no behavior file for this ckpt)",
                "sv_prompt":    sv.get("prompt", "") if sv else "",
                "sv_text":      sv.get("text", "")
                    if sv else "(no self-verb file for this ckpt)",
                "sv_approach":  sv.get("approach", "") if sv else "",
                "cluster_name": f"Frame: {FRAME_DISPLAY.get(slug, slug)}",
            }

    # ── 4. Render: per-bucket polylines + start/end markers. ───────────
    fig = go.Figure()
    for (slug, seed), info in sorted(by_traj.items()):
        rows = info["rows"]
        n = len(rows)
        pcs = [r["pc"] for r in rows]
        customdata = [
            [seed, r["step"], r["kl"], f"{slug}_{seed}_{r['step']}"]
            for r in rows
        ]

        # Per-segment grey bucket via midpoint progress (i+0.5)/(n-1).
        # Then merge consecutive same-bucket segments into one polyline.
        seg_buckets = [
            int(((i + 0.5) / max(1, n - 1)) * (args.n_buckets - 1) + 0.5)
            for i in range(n - 1)
        ]
        run_start = 0
        for i in range(1, len(seg_buckets) + 1):
            if i == len(seg_buckets) or seg_buckets[i] != seg_buckets[run_start]:
                bucket = seg_buckets[run_start]
                t = bucket / max(1, args.n_buckets - 1)
                color = progress_color(t)
                # Polyline covers points [run_start .. i].
                seg_pcs = pcs[run_start:i + 1]
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in seg_pcs],
                    y=[p[1] for p in seg_pcs],
                    z=[p[2] for p in seg_pcs],
                    mode="lines",
                    line=dict(color=color, width=3),
                    opacity=0.85,
                    showlegend=False,
                    customdata=customdata[run_start:i + 1],
                    hovertemplate=(
                        "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
                    ),
                ))
                run_start = i

        # Start marker — filled dot in the colormap start color (cool blue)
        fig.add_trace(go.Scatter3d(
            x=[pcs[0][0]], y=[pcs[0][1]], z=[pcs[0][2]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color=START_COLOR),
            customdata=[customdata[0]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
        # End marker — filled dot in the colormap end color (warm red)
        fig.add_trace(go.Scatter3d(
            x=[pcs[-1][0]], y=[pcs[-1][1]], z=[pcs[-1][2]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color=END_COLOR),
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
                f"<span style='color:{START_COLOR};font-size:18px;vertical-align:middle'>●</span> Starting Points  ·  "
                f"<span style='color:{END_COLOR};font-size:18px;vertical-align:middle'>●</span> Ending Points  ·  "
                "coolwarm gradient = direction of flow"
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
