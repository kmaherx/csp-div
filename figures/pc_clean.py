"""Standalone clean version of the dashboard's PC plot.

Renders the same trajectories the dashboard shows (50 seeds × 4 frames,
pooled PCA over residual-stream shifts at L16), colored by persona
strength (the dashboard's default). Strips every bit of chrome — no
axes, no grid, no labels, no legend, no colorbar — so the figure can be
dropped into a writeup as pure data.

Run:
    python figures/pc_clean.py [--max-step 50] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

# Reuse the dashboard's persona-color helper so the colors match
# exactly. Module-level rcParams in plotting.py also apply.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from csp_div.plotting import color_persona  # noqa: E402


FRAME_SLUGS = ["be", "act", "please", "youshould"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--max-step", type=int, default=50,
        help="Filter to checkpoints with step ≤ this value, matching the "
             "dashboard's 0–50 window. Pass a large number to include all.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/llama/all_frames/pc_clean.png"),
    )
    ap.add_argument(
        "--size", type=float, default=10.0,
        help="Figure side in inches (square).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

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
    # L2-normalize each shift before PCA, matching the dashboard default
    # (parser.set_defaults(normalize=True) in pipeline/5_dashboard.py).
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
    print(f"  {len(trajs)} trajectories")

    # ── Render ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(args.size, args.size))
    edge_color = "#d4d4d4"

    for rows in trajs.values():
        xs = [r["pc"][0] for r in rows]
        ys = [r["pc"][1] for r in rows]
        ax.plot(xs, ys, color=edge_color, linewidth=1.2, zorder=1, alpha=1.0)
        for r in rows:
            ax.scatter(
                r["pc"][0], r["pc"][1],
                c=color_persona(t_cos(r["cos"])),
                s=200, edgecolors="none", zorder=2,
            )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                pad_inches=0, transparent=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
