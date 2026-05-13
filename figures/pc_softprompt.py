"""PCA over the actual soft prompts (not their downstream activations).

For each (seed, step) checkpoint, load the soft prompt at
results/llama/be/seed_{N}/sp_pos_step{K}.pt (training is frame-agnostic
and saved under the canonical `be/` slot), flatten its (L, hidden)
parameters to a single vector, stack across all (seed, step) cells,
and fit a 2-component PCA. Plot every checkpoint as a colored dot.

Color comes from the per-step persona-cos averaged across the 4
evaluation frames (since the soft prompt itself doesn't have a frame).

Per-step soft prompts are gitignored (see .gitignore), so this script
needs to run on a machine that has them locally — typically the
training pod, not a fresh clone.

Run:
    python figures/pc_softprompt.py
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import csp_div.plotting  # noqa: F401, E402


FRAME_SLUGS = ["be", "act", "please", "youshould"]


def parse_steps(s: str) -> set[int] | None:
    """None means 'include every step'."""
    if s.strip().lower() == "all":
        return None
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--steps", type=parse_steps, default="all",
        help="Which training steps to include. 'all' uses every saved "
             "checkpoint (up to --max-step); pass a comma-separated list "
             "like '0' or '0,25,50' to restrict to specific steps.",
    )
    ap.add_argument(
        "--max-step", type=int, default=50,
        help="Upper bound on step values when --steps is 'all'.",
    )
    ap.add_argument("--cmap", default="Reds_r")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG. Defaults to pc_softprompt_<tag>.png in "
             "results/llama/all_frames/, where <tag> is 'all' or "
             "'step{N}' / 'steps{N1}_{N2}' based on --steps.",
    )
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--marker-size", type=float, default=250)
    ap.add_argument(
        "--bg-alpha", type=float, default=1.0,
        help="Alpha for the dots.",
    )
    args = ap.parse_args()
    if isinstance(args.steps, str):
        args.steps = parse_steps(args.steps)

    if args.out is None:
        if args.steps is None:
            tag = "all"
        elif len(args.steps) == 1:
            tag = f"step{next(iter(args.steps))}"
        else:
            tag = "steps" + "_".join(str(s) for s in sorted(args.steps))
        args.out = Path("results/llama/all_frames") / f"pc_softprompt_{tag}.png"

    base_dir = args.results_dir / "llama"
    be_dir = base_dir / "be"
    if not be_dir.is_dir():
        raise SystemExit(f"{be_dir} not found")

    # ── Discover (seed, step) pairs with a soft prompt checkpoint ──────
    rows: list[tuple[int, int, np.ndarray]] = []
    seed_dirs = sorted(be_dir.glob("seed_*"))
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("seed_")[-1])
        except ValueError:
            continue
        for sp_path in sorted(seed_dir.glob("sp_pos_step*.pt")):
            stem = sp_path.stem  # e.g. sp_pos_step25
            try:
                step = int(stem.split("step")[-1])
            except ValueError:
                continue
            if args.steps is None:
                if step > args.max_step:
                    continue
            elif step not in args.steps:
                continue
            sp = torch.load(sp_path, map_location="cpu", weights_only=True)
            # Some checkpoints save the bare tensor, others a state-dict.
            if isinstance(sp, dict):
                sp = next(iter(sp.values()))
            rows.append((seed, step, sp.detach().float().flatten().numpy()))

    if not rows:
        raise SystemExit(
            "No sp_pos_step*.pt files found. They're gitignored — run this "
            "on a machine that did the training."
        )

    X = np.stack([r[2] for r in rows])
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    print(f"PCA over {len(X)} soft prompts (dim {X.shape[1]}); var ratio "
          f"{[round(v, 3) for v in pca.explained_variance_ratio_]}")

    # ── Per-(seed, step) cos averaged across frames ────────────────────
    cos_by_cell: dict[tuple[int, int], list[float]] = {}
    for slug in FRAME_SLUGS:
        axis_path = base_dir / slug / "axis.json"
        if not axis_path.is_file():
            continue
        for r in json.loads(axis_path.read_text())["rows"]:
            ckpt = r["ckpt"]
            if "step" not in ckpt:
                continue
            try:
                step = int(ckpt.split("step")[-1].replace(".pt", ""))
            except ValueError:
                continue
            seed = int(r["group"].split("seed_")[-1].split("_")[0])
            cos_by_cell.setdefault((seed, step), []).append(float(r["proj_cos"]))
    if not cos_by_cell:
        raise SystemExit("No axis.json files found — run pipeline/4_axis.py first")
    cos_avg = {k: float(np.mean(v)) for k, v in cos_by_cell.items()}

    cos_vals = [cos_avg.get((s, st), 0.0) for s, st, _ in rows]
    cos_min, cos_max = min(cos_vals), max(cos_vals)

    def t_cos(c: float) -> float:
        return 0.5 if cos_max == cos_min else (c - cos_min) / (cos_max - cos_min)

    cmap = mpl.colormaps[args.cmap]

    def color_for(t: float) -> str:
        return mcolors.to_hex(cmap(max(0.0, min(1.0, t))))

    # ── Render ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(args.size, args.size))
    colors = [color_for(t_cos(c)) for c in cos_vals]
    ax.scatter(
        Y[:, 0], Y[:, 1], c=colors,
        s=args.marker_size, edgecolors="none",
        alpha=args.bg_alpha, zorder=2,
    )

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
