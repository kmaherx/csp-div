"""PC plot with two trajectories highlighted, everything else as backdrop.

Same PCA basis as pc_clean.py (L2-normalized shifts across all 4 frames,
capped at step 50). Background trajectories are drawn as edges only,
colored by the trajectory's average assistant-axis projection (Reds_r
cmap) at 0.5 alpha. Two chosen trajectories are drawn on top as a
chain: black connecting edges with black-outlined markers; the marker
fill is the per-step persona color.

Default highlights: (be, 16) and (youshould, 4). These illustrate the
arc-vs-flat trajectory contrast in this space.

Run:
    python figures/pc_highlighted.py
    python figures/pc_highlighted.py --highlight be:16,youshould:4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Module-level rcParams in plotting.py apply on import.
import csp_div.plotting  # noqa: F401, E402


FRAME_SLUGS = ["be", "act", "please", "youshould"]


def parse_highlight(s: str) -> list[tuple[str, int]]:
    """Preserve order so --hl-edge-colors lines up positionally."""
    pairs: list[tuple[str, int]] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        slug, seed = token.split(":")
        pairs.append((slug.strip(), int(seed)))
    return pairs


def parse_colors(s: str) -> list[str]:
    return [c.strip() for c in s.split(",") if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--max-step", type=int, default=50)
    ap.add_argument(
        "--highlight", type=parse_highlight, default="be:12,be:4",
        help="Comma-separated slug:seed pairs to draw on top.",
    )
    ap.add_argument(
        "--cmap", default="Reds_r",
        help="Matplotlib colormap name used to map persona-strength "
             "(normalized cos) to point colors. Default is Reds_r "
             "(dark red = persona, near-white = default). Pass Greys_r "
             "for a monochrome black-to-white version.",
    )
    ap.add_argument(
        "--hl-edge-colors", type=parse_colors, default=[],
        help="Comma-separated list of edge / marker-outline colors for "
             "the highlighted trajectories, in the same order as "
             "--highlight. Defaults to black for all if omitted.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path. If omitted, defaults to "
             "results/llama/all_frames/pc_highlighted.png for Reds_r and "
             "pc_highlighted_<cmap>.png for any other cmap.",
    )
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--bg-alpha", type=float, default=0.3,
        help="Alpha for background trajectory edges.",
    )
    ap.add_argument(
        "--bg-linewidth", type=float, default=2.0,
        help="Linewidth for background trajectory edges.",
    )
    ap.add_argument(
        "--bg-color", default=None,
        help="If set, render every background edge with this single color "
             "(e.g. '#666666') instead of using --cmap per-edge. Useful "
             "for narrative figures where the persona-strength gradient "
             "shouldn't be revealed yet.",
    )
    ap.add_argument(
        "--bg-markers", action="store_true",
        help="Draw a marker at every background checkpoint, colored by "
             "the cmap, no outline. Off by default.",
    )
    ap.add_argument(
        "--bg-marker-size", type=float, default=200,
        help="Marker size for background checkpoints when --bg-markers "
             "is set.",
    )
    ap.add_argument(
        "--bg-color-by", choices=["persona", "step"], default="persona",
        help="What background segments / markers are colored by. "
             "'persona' uses --cmap on the assistant-axis cos at each "
             "checkpoint (the dashboard's default mode). 'step' uses "
             "matplotlib 'rainbow' on the optimization step normalized "
             "to [0, --max-step].",
    )
    ap.add_argument(
        "--hl-linewidth", type=float, default=2.5,
        help="Linewidth for both highlighted edges and marker outlines.",
    )
    ap.add_argument(
        "--hl-marker-size", type=float, default=300,
        help="Marker size for highlighted trajectory points.",
    )
    ap.add_argument(
        "--hl-marker-color", default=None,
        help="If set, fill every highlighted marker with this color "
             "(e.g. 'white') instead of the per-step cmap. Useful for "
             "narrative figures that want flat beads.",
    )
    ap.add_argument(
        "--hl-arrow-scale", type=float, default=15,
        help="mutation_scale for the per-segment arrowheads on the "
             "highlighted trajectories. Bigger = larger arrowheads.",
    )
    ap.add_argument(
        "--hl-arrow-shrink", type=float, default=14,
        help="Shrink the arrow tip by this many points before reaching "
             "the next marker, so the arrowhead sits just outside the "
             "marker circle rather than getting hidden behind it.",
    )
    ap.add_argument(
        "--hl-only-steps", default=None,
        help="If set, restrict the highlight render to checkpoints whose "
             "step is in this comma-separated list (e.g. '0' or '0,50'). "
             "Arrows are suppressed; only the matching markers are drawn. "
             "Useful for single-point assets that pair with rainbow plots.",
    )
    args = ap.parse_args()
    if isinstance(args.highlight, str):
        args.highlight = parse_highlight(args.highlight)
    if isinstance(args.hl_edge_colors, str):
        args.hl_edge_colors = parse_colors(args.hl_edge_colors)
    if args.out is None:
        if args.cmap == "Reds_r":
            args.out = Path("results/llama/all_frames/pc_highlighted.png")
        else:
            stem = f"pc_highlighted_{args.cmap.lower().replace('_', '')}.png"
            args.out = Path("results/llama/all_frames") / stem

    cmap = mpl.colormaps[args.cmap]
    step_cmap = mpl.colormaps["rainbow"]

    def color_for(t: float) -> str:
        return mcolors.to_hex(cmap(max(0.0, min(1.0, t))))

    def bg_color_for(row: dict) -> str:
        if args.bg_color is not None:
            return args.bg_color
        if args.bg_color_by == "step":
            t = row["step"] / max(args.max_step, 1)
            return mcolors.to_hex(step_cmap(max(0.0, min(1.0, t))))
        return color_for(t_cos(row["cos"]))

    hl_color_map: dict[tuple[str, int], str] = {}
    for i, key in enumerate(args.highlight):
        hl_color_map[key] = (
            args.hl_edge_colors[i]
            if i < len(args.hl_edge_colors)
            else "black"
        )

    base_dir = args.results_dir / "llama"

    # ── Pool shifts across all four frames ─────────────────────────────
    pooled_X: list[np.ndarray] = []
    row_index: list[tuple[str, str, str, int]] = []
    for slug in FRAME_SLUGS:
        shifts_path = base_dir / slug / "shifts.pt"
        if not shifts_path.is_file():
            print(f"  WARN: {shifts_path} missing, skipping frame {slug}")
            continue
        d = torch.load(shifts_path, map_location="cpu", weights_only=True)
        for r in d["rows"]:
            step = int(r["step"])
            if step > args.max_step:
                continue
            pooled_X.append(r["shift"].numpy())
            row_index.append((slug, r["group"], r["ckpt"], step))

    if not pooled_X:
        raise SystemExit("No shifts loaded — check --results-dir")

    X = np.stack(pooled_X)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    print(f"Pooled PCA over {len(X)} rows; var ratio "
          f"{[round(v, 3) for v in pca.explained_variance_ratio_]}")

    # ── Cos projections for persona coloring ───────────────────────────
    proj_cos: dict[tuple[str, str, str], float] = {}
    for slug in FRAME_SLUGS:
        axis_path = base_dir / slug / "axis.json"
        if not axis_path.is_file():
            continue
        for r in json.loads(axis_path.read_text())["rows"]:
            proj_cos[(slug, r["group"], r["ckpt"])] = float(r["proj_cos"])
    if not proj_cos:
        raise SystemExit("No axis.json found — run pipeline/4_axis.py first")

    cos_min = min(proj_cos.values())
    cos_max = max(proj_cos.values())

    def t_cos(c: float) -> float:
        return 0.5 if cos_max == cos_min else (c - cos_min) / (cos_max - cos_min)

    # ── Group rows into (slug, seed) trajectories ──────────────────────
    trajs: dict[tuple[str, int], list[dict]] = {}
    for i, (slug, group, ckpt, step) in enumerate(row_index):
        try:
            seed = int(group.split("seed_")[-1].split("_")[0])
        except ValueError:
            continue
        c = proj_cos.get((slug, group, ckpt), 0.0)
        trajs.setdefault((slug, seed), []).append({
            "step": step, "pc": Y[i], "cos": c,
        })
    for k in trajs:
        trajs[k].sort(key=lambda r: r["step"])
    print(f"  {len(trajs)} trajectories ({len(args.highlight)} highlighted)")

    # ── Render ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(args.size, args.size))

    # Background: each individual edge colored by the cos of its preceding
    # checkpoint (so the color reflects persona-strength at the start of
    # each step), unless --bg-color overrides with a single flat color.
    segments: list[list[tuple[float, float]]] = []
    seg_colors: list[str] = []
    bg_pts_x: list[float] = []
    bg_pts_y: list[float] = []
    bg_pts_c: list[str] = []
    # When --hl-only-steps restricts the highlight to a single marker,
    # we want the rest of that seed's checkpoints to remain in the
    # background — otherwise switching which seed is highlighted
    # changes the visible background. For full-chain highlights we
    # still skip the highlighted seed so its chain doesn't render twice.
    bg_skip_keys = set() if args.hl_only_steps is not None else set(args.highlight)

    for key, rows in trajs.items():
        if key in bg_skip_keys:
            continue
        for i in range(len(rows) - 1):
            p1, p2 = rows[i], rows[i + 1]
            segments.append([
                (float(p1["pc"][0]), float(p1["pc"][1])),
                (float(p2["pc"][0]), float(p2["pc"][1])),
            ])
            seg_colors.append(bg_color_for(p1))
        if args.bg_markers:
            for r in rows:
                bg_pts_x.append(float(r["pc"][0]))
                bg_pts_y.append(float(r["pc"][1]))
                bg_pts_c.append(bg_color_for(r))
    if segments:
        lc = LineCollection(
            segments, colors=seg_colors, alpha=args.bg_alpha,
            linewidth=args.bg_linewidth, zorder=1, capstyle="round",
        )
        ax.add_collection(lc)
    if args.bg_markers and bg_pts_x:
        ax.scatter(
            bg_pts_x, bg_pts_y, c=bg_pts_c,
            s=args.bg_marker_size, edgecolors="none", zorder=2,
        )

    # Highlighted: each segment is its own FancyArrowPatch, so we get a
    # black shaft followed by a small filled arrowhead pointing at the
    # next marker. shrinkB pulls the arrowhead back from the marker
    # boundary so the marker doesn't cover the arrow tip. Markers are
    # drawn on top, filling the gap between consecutive segments.
    only_steps: set[int] | None = None
    if args.hl_only_steps is not None:
        only_steps = {int(s.strip()) for s in args.hl_only_steps.split(",") if s.strip()}

    for key in args.highlight:
        if key not in trajs:
            print(f"  WARN: highlight {key} not found in data")
            continue
        edge_color = hl_color_map[key]
        rows = trajs[key]
        if only_steps is None:
            for i in range(len(rows) - 1):
                p1, p2 = rows[i], rows[i + 1]
                arrow = FancyArrowPatch(
                    (float(p1["pc"][0]), float(p1["pc"][1])),
                    (float(p2["pc"][0]), float(p2["pc"][1])),
                    arrowstyle="-|>",
                    mutation_scale=args.hl_arrow_scale,
                    color=edge_color,
                    linewidth=args.hl_linewidth,
                    shrinkA=0,
                    shrinkB=args.hl_arrow_shrink,
                    zorder=10,
                )
                ax.add_patch(arrow)
        for r in rows:
            if only_steps is not None and r["step"] not in only_steps:
                continue
            fill = (
                args.hl_marker_color
                if args.hl_marker_color is not None
                else color_for(t_cos(r["cos"]))
            )
            ax.scatter(
                r["pc"][0], r["pc"][1],
                c=fill,
                s=args.hl_marker_size,
                edgecolors=edge_color,
                linewidths=args.hl_linewidth,
                zorder=11,
            )

    # Force the view to fit the full data extent: when there are zero
    # highlighted trajectories there are no scatter calls, so dataLim
    # isn't otherwise expanded by them. autoscale_view triggers a fit
    # from whatever's already been added (LineCollection + patches).
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                pad_inches=0, transparent=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
