"""All-frames combined 3D PC plot with line gradient colored by
cos(shift, assistant axis) — i.e. how persona-aligned each ckpt's
shift is, drawn from the per-frame axis.json files.

Sister script to plot_pc_3d_all_frames_directional.py. Same data
layout (4 frames × 50 seeds = 200 trajectories), same shared PCA basis
fit on the pooled shifts, same hover behavior, same per-frame Q
suffix and curated-pick self-verbs. Differences vs the directional
version:

  * Line color encodes proj_cos (axis.json) rather than step progress.
  * Reds_r colormap: dark red = most-negative cos = most persona-aligned;
    pale = near-zero (no axial movement). Matches the per-frame
    figure_pc3d_kmeansmid_axiscos.html convention.
  * Start markers = open ring (white fill, dark grey border).
  * End markers = filled black.
  * Colorbar on the right labeled "Persona Alignment".
"""
import argparse
import json
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

# § matches the self-verb prompt display convention; appended to the
# behavior question so what's displayed mirrors the eval-time concatenation.
FRAME_SUFFIX = {k: v.format(sp="§") for k, v in FRAME_DISPLAY.items()}

# Reds_r: t=0 → dark red, t=1 → pale (almost white). We map the
# most-negative cos to t=0 so dark red = persona-aligned.
_REDS_R = colormaps["Reds_r"]


def color_for_t(t):
    return to_hex(_REDS_R(max(0.0, min(1.0, t))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", action="append", nargs=2,
                        metavar=("SLUG", "DIR"), default=None)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--n-buckets", type=int, default=16,
                        help="Discrete color buckets along each trajectory; "
                             "consecutive same-bucket segments get merged into "
                             "single polylines for render speed.")
    parser.add_argument("--out",
                        default="results/all_frames/figure_pc3d_all_frames_axiscos.html")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    frames = args.frame or DEFAULT_FRAMES

    # ── 1. Pool + PCA (same as the sister scripts). ────────────────────
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

    # ── 2. Per-frame axis.json: build (slug, group, ckpt) → proj_cos lookup
    proj_cos = {}
    for slug, base in frames:
        axis_path = os.path.join(
            base if os.path.isabs(base) else os.path.join(ROOT, base),
            "axis.json",
        )
        if not os.path.isfile(axis_path):
            print(f"  WARN: no axis.json for {slug}; trajectories in this "
                  f"frame will fall back to t=0.5 grey-mid color")
            continue
        d = json.load(open(axis_path))
        for r in d["rows"]:
            proj_cos[(slug, r["group"], r["ckpt"])] = float(r["proj_cos"])
    score_min = min(proj_cos.values())
    score_max = max(proj_cos.values())
    print(f"  proj_cos range across all frames: [{score_min:.3f}, {score_max:.3f}]")

    def t_for_cos(c):
        if score_max == score_min:
            return 0.5
        return float((c - score_min) / (score_max - score_min))

    # ── 3. Group into per-(frame, seed) trajectories. ───────────────────
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

    # ── 4. Curated agent picks for self-verb hover. ────────────────────
    manual_path = os.path.join(ROOT, "results/all_frames/manual_self_verb.json")
    manual_picks = {}
    if os.path.isfile(manual_path):
        manual_picks = json.load(open(manual_path))
        n_pick = sum(1 for v in manual_picks.values() if not v.get("skipped"))
        n_skip = sum(1 for v in manual_picks.values() if v.get("skipped"))
        print(f"  loaded {len(manual_picks)} curated entries "
              f"({n_pick} picks, {n_skip} skips) from {manual_path}")

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
            behav_prompt = b.get("prompt", "") if b else ""
            if behav_prompt and suffix:
                behav_prompt = f"{behav_prompt} {suffix}"

            key = f"{slug}_{seed}_{step}"
            curated = manual_picks.get(key)
            if curated and not curated.get("skipped"):
                sv_prompt   = curated.get("sv_prompt", "")
                sv_text     = curated.get("sv_text", "")
                sv_approach = curated.get("sv_approach", "")
            else:
                sv_prompt   = sv.get("prompt", "") if sv else ""
                sv_text     = (sv.get("text", "") if sv
                               else "(no self-verb file for this ckpt)")
                sv_approach = sv.get("approach", "") if sv else ""

            cell_data[key] = {
                "behav_prompt": behav_prompt,
                "behav_text":   b.get("text", "")
                    if b else "(no behavior file for this ckpt)",
                "sv_prompt":    sv_prompt,
                "sv_text":      sv_text,
                "sv_approach":  sv_approach,
                "cluster_name": f"Frame: {FRAME_DISPLAY.get(slug, slug)}",
            }

    # ── 5. Render: per-bucket polylines + start (open ring) + end (black).
    fig = go.Figure()
    for (slug, seed), info in sorted(by_traj.items()):
        rows = info["rows"]
        n = len(rows)
        if n < 2:
            continue
        pcs = [r["pc"] for r in rows]
        ckpts = [r["ckpt"] for r in rows]
        scores = [proj_cos.get((slug, info["group"], c), 0.0) for c in ckpts]
        customdata = [
            [seed, r["step"], r["kl"], f"{slug}_{seed}_{r['step']}"]
            for r in rows
        ]

        # Bucket each segment by midpoint cos, then merge consecutive
        # same-bucket segments into one polyline. Same trick as the
        # directional plot — keeps trace count manageable.
        seg_buckets = [
            int(t_for_cos(0.5 * (scores[i] + scores[i + 1]))
                * (args.n_buckets - 1) + 0.5)
            for i in range(n - 1)
        ]
        run_start = 0
        for i in range(1, len(seg_buckets) + 1):
            if i == len(seg_buckets) or seg_buckets[i] != seg_buckets[run_start]:
                bucket = seg_buckets[run_start]
                t = bucket / max(1, args.n_buckets - 1)
                color = color_for_t(t)
                seg_pcs = pcs[run_start:i + 1]
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in seg_pcs],
                    y=[p[1] for p in seg_pcs],
                    z=[p[2] for p in seg_pcs],
                    mode="lines",
                    line=dict(color=color, width=6),
                    opacity=0.85,
                    showlegend=False,
                    customdata=customdata[run_start:i + 1],
                    hovertemplate=(
                        "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
                    ),
                ))
                run_start = i

        # Start marker — open ring (white fill, dark border)
        fig.add_trace(go.Scatter3d(
            x=[pcs[0][0]], y=[pcs[0][1]], z=[pcs[0][2]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color="white",
                        line=dict(color="#333333", width=1.5)),
            customdata=[customdata[0]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
        # End marker — black filled
        fig.add_trace(go.Scatter3d(
            x=[pcs[-1][0]], y=[pcs[-1][1]], z=[pcs[-1][2]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color="black"),
            customdata=[customdata[-1]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))

    # Colorbar (invisible carrier trace; using matplotlib-style "_r" semantics
    # via reversescale on the plotly side so the bar matches the line gradient).
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers", showlegend=False,
        marker=dict(
            size=0.001, color=[score_min, score_max],
            cmin=score_min, cmax=score_max,
            colorscale="Reds",
            reversescale=True,
            showscale=True,
            colorbar=dict(
                title=dict(text="Persona Alignment", side="right"),
                thickness=14, len=0.6, x=1.02,
            ),
            opacity=0,
        ),
        hoverinfo="skip",
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
                "<span style='font-size:18px;vertical-align:middle'>○</span> Starting Points  ·  "
                "<span style='color:#000;font-size:18px;vertical-align:middle'>●</span> Ending Points  ·  "
                "Reds_r gradient = cos(shift, assistant axis)"
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
