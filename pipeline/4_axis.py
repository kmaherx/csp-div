"""Compute residual-stream shift projections onto the Butanium assistant axis.

Stage B (post-everything) script. No model load — pure aggregation:

  1. For each frame, read either:
     - the consolidated `shifts.pt` produced by an earlier run, OR
     - per-(seed, step) `shift_step{K}.pt` files written by `2_generate.py`
       plus the frame's `vanilla_baseline.pt`. The script pools these
       into the consolidated `shifts.pt` format expected by the dashboard.
  2. Download (once) the Butanium assistant axis vector at L16 and project
     each shift onto it. Per-row scalars (`shift_norm`, `proj_dot`,
     `proj_cos`) go into `axis.json`.
  3. Render a 2-panel matplotlib figure (`axis.png`) of (KL × proj_dot)
     and (KL × proj_cos), colored by deep/shallow basin.

Skip-if-exists: a frame's outputs are skipped iff `axis.json` AND `axis.png`
already exist. Pass `--force` to regenerate.

Usage:
    python pipeline/4_axis.py --frames be,act,please,youshould
    python pipeline/4_axis.py --frames be --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from csp_div.config import AXIS_LAYER, AXIS_REPO, FRAMES_BY_SLUG, frame_results_dir
from csp_div.plotting import (
    basin_color,
    basin_legend,
    draw_endpoints,
    draw_trajectory,
    panel_title,
    style_kl_axis,
    trajectory_basin,
)


def parse_frames(arg: str) -> list[str]:
    slugs = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in slugs if s not in FRAMES_BY_SLUG]
    if unknown:
        raise SystemExit(f"Unknown frame slug(s): {unknown}. "
                         f"Known: {list(FRAMES_BY_SLUG)}")
    return slugs


def load_butanium_axis(layer: int, device: torch.device) -> tuple[torch.Tensor, float]:
    print(f"Loading assistant axis from {AXIS_REPO}...")
    axis_path = hf_hub_download(
        repo_id=AXIS_REPO,
        filename="assistant_axis.pt", repo_type="dataset",
    )
    full_axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    axis = full_axis[layer].float().to(device)
    return axis, axis.norm().item()


def pool_per_cell_shifts(frame_dir: Path, results_dir: Path) -> dict:
    """Walk `seed_*/shift_step*.pt` files under `frame_dir` and rebuild the
    consolidated `shifts.pt` format. Reads the single shared vanilla
    baseline at `<results_dir>/vanilla_baseline.pt` (frame-agnostic: the
    vanilla teacher uses no frame, so its acts are the same regardless
    of which eval frame conditioned the CSP).
    """
    baseline_path = results_dir / "vanilla_baseline.pt"
    if not baseline_path.is_file():
        raise SystemExit(
            f"Missing vanilla baseline at {baseline_path}. Run 2_generate.py "
            f"first to produce per-cell shifts + this baseline."
        )
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)

    rows: list[dict] = []
    cell_files = sorted(frame_dir.glob("seed_*/shift_step*.pt"))
    for cell_path in cell_files:
        seed_name = cell_path.parent.name             # "seed_3"
        step_name = cell_path.stem                    # "shift_step5"
        step = int(step_name.replace("shift_step", ""))
        cell = torch.load(cell_path, map_location="cpu", weights_only=True)
        rows.append({
            "group": f"{frame_dir.name}/{seed_name}",
            "ckpt":  f"sp_pos_step{step}.pt" if step != cell.get("final_step", -1)
                     else "sp_pos.pt",
            "step":  step,
            "kl":    float(cell["kl"]),
            "shift": cell["shift"].cpu(),
            "mean_csp": cell["mean_csp"].cpu(),
        })
    print(f"  {frame_dir.name}: pooled {len(rows)} per-cell shifts")
    return {
        "layer": int(baseline["layer"]),
        "n_eval_prompts": int(baseline["n_eval_prompts"]),
        "max_new_tokens": int(baseline["max_new_tokens"]),
        "mean_vanilla": baseline["mean_vanilla"].cpu(),
        "rows": rows,
    }


def project_onto_axis(
    shifts: dict, axis: torch.Tensor, axis_norm: float,
) -> list[dict]:
    """Produce `axis.json`-style rows from `shifts.pt` content."""
    rows: list[dict] = []
    for r in shifts["rows"]:
        shift = r["shift"].to(axis.device).float()
        shift_norm = float(shift.norm().item())
        proj_dot = float((shift @ axis).item())
        proj_cos = (proj_dot / (shift_norm * axis_norm)) if shift_norm > 0 else 0.0
        rows.append({
            "group": r["group"], "ckpt": r["ckpt"],
            "step":  int(r["step"]), "kl": float(r["kl"]),
            "shift_norm": shift_norm,
            "proj_dot": proj_dot,
            "proj_cos": proj_cos,
        })
    return rows


def render_axis_figure(rows: list[dict], layer: int, out_path: Path) -> None:
    """Two-panel figure: (KL × proj_dot) and (KL × proj_cos), trajectories
    colored by deep/shallow basin assignment."""
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: r["step"])

    group_basins = {
        g: trajectory_basin([(r["kl"], r["proj_cos"], r["step"]) for r in rs])
        for g, rs in by_group.items()
    }
    n_dippers = sum(1 for b in group_basins.values() if b == "deep")
    n_nondippers = len(group_basins) - n_dippers

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for g, rs in sorted(by_group.items()):
            if not basin_filter(group_basins[g]):
                continue
            color = basin_color(group_basins[g])
            kls = [r["kl"] for r in rs]
            dots = [r["proj_dot"] for r in rs]
            coss = [r["proj_cos"] for r in rs]
            draw_trajectory(axes[0], kls, dots, color, zorder=zorder)
            draw_endpoints(axes[0], kls, dots, color, zorder=zorder + 2)
            draw_trajectory(axes[1], kls, coss, color, zorder=zorder)
            draw_endpoints(axes[1], kls, coss, color, zorder=zorder + 2)

    style_kl_axis(axes[0], ylabel=f"(L{layer} shift) · (assistant axis)")
    style_kl_axis(axes[1], ylabel=f"cos(L{layer} shift, assistant axis)")
    panel_title(axes[0], "Magnitude along assistant axis")
    panel_title(axes[1], "Direction alignment with assistant axis")
    basin_legend(axes[1], n_dippers, n_nondippers, loc="lower right")
    fig.suptitle(
        f"L{layer} shift onto Butanium axis  ·  "
        f"negative = role-play, positive = default-assistant",
        fontsize=10, color="#444444",
    )
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def process_frame(
    slug: str, results_dir: Path, layer: int,
    axis: torch.Tensor, axis_norm: float, force: bool,
) -> None:
    frame_dir = frame_results_dir(results_dir, slug)
    if not frame_dir.is_dir():
        print(f"  {slug}: directory {frame_dir} missing, skipping")
        return

    shifts_path = frame_dir / "shifts.pt"
    axis_json = frame_dir / "axis.json"
    axis_png = frame_dir / "axis.png"

    if not force and axis_json.is_file() and axis_png.is_file():
        print(f"  {slug}: outputs already present at {frame_dir}, skipping "
              f"(use --force to regenerate)")
        return

    # Read or rebuild the consolidated shifts.
    if shifts_path.is_file():
        shifts = torch.load(shifts_path, map_location="cpu", weights_only=True)
        print(f"  {slug}: loaded existing shifts.pt ({len(shifts['rows'])} rows)")
    else:
        shifts = pool_per_cell_shifts(frame_dir, results_dir)
        torch.save(shifts, shifts_path)
        print(f"  {slug}: wrote consolidated shifts.pt → {shifts_path}")

    # Project + emit axis.json.
    rows = project_onto_axis(shifts, axis, axis_norm)
    axis_json.write_text(json.dumps({"layer": layer, "rows": rows}, indent=2))
    print(f"  {slug}: wrote {axis_json}")

    # Render the figure.
    render_axis_figure(rows, layer, axis_png)
    print(f"  {slug}: wrote {axis_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames", default="be,act,please,youshould",
        help="Comma-separated frame slugs to process (default: all four).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        help="Root directory containing llama/, llama_act/, ...",
    )
    parser.add_argument(
        "--layer", type=int, default=AXIS_LAYER,
        help="Layer index on the Butanium axis to project onto.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate axis.json + axis.png even if they already exist.",
    )
    args = parser.parse_args()

    slugs = parse_frames(args.frames)
    print(f"Processing frames: {slugs}")
    axis, axis_norm = load_butanium_axis(args.layer, torch.device("cpu"))
    print(f"  axis[{args.layer}]: shape={tuple(axis.shape)}, ‖·‖={axis_norm:.3f}")
    for slug in slugs:
        print(f"\n=== frame={slug} ===")
        process_frame(slug, args.results_dir, args.layer,
                      axis, axis_norm, args.force)


if __name__ == "__main__":
    main()
