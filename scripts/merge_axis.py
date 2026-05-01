"""Merge per-machine axis projection outputs into combined axis.json + shifts.pt.

When the 50-seed random-walk run is split across 5 machines via the
`--seed-range` arg of analyze_assistant_axis, each machine writes its
own axis_<lo>_<hi>.png + axis_<lo>_<hi>.json + shifts_<lo>_<hi>.pt.
This script reads all of those, concatenates the rows, and writes the
canonical axis.json + shifts.pt that the rest of the pipeline expects
(replot_axis.py, analyze_pca_trajectory.py, etc.).

Usage:
  python scripts/merge_axis.py results/random_walk
  python scripts/merge_axis.py --dir results/random_walk

The dir must contain at least one `axis_*.json` file; matching
`shifts_*.pt` files are optional but expected. Mean_vanilla is taken
from the first shifts file (all machines should have computed the same
mean_vanilla on the same eval prompts — a sanity check warns if norms
differ).
"""
import argparse
import glob
import json
import os
import re

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default=None,
                        help="Directory containing axis_*.json + shifts_*.pt")
    parser.add_argument("--dir", dest="dir_flag", default=None,
                        help="Same as positional dir.")
    args = parser.parse_args()
    target = args.dir or args.dir_flag
    if target is None:
        raise SystemExit("Pass the dir containing axis_*.json files")

    json_paths = sorted(glob.glob(os.path.join(target, "axis_*.json")))
    if not json_paths:
        raise SystemExit(f"No axis_*.json files in {target}")
    print(f"Found {len(json_paths)} per-range axis JSONs:")
    for p in json_paths:
        print(f"  {p}")

    all_rows = []
    layer = None
    for jp in json_paths:
        with open(jp) as f:
            d = json.load(f)
        if layer is None:
            layer = d["layer"]
        elif d["layer"] != layer:
            raise SystemExit(f"layer mismatch in {jp}: {d['layer']} != {layer}")
        all_rows.extend(d["rows"])

    # Deduplicate by (group, ckpt) — defensive in case ranges overlap
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["group"], r["ckpt"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    if len(deduped) != len(all_rows):
        print(f"  deduplicated {len(all_rows) - len(deduped)} duplicate rows")

    out_json = os.path.join(target, "axis.json")
    with open(out_json, "w") as f:
        json.dump({"layer": layer, "rows": deduped}, f, indent=2)
    print(f"\nWrote {out_json}  ({len(deduped)} rows)")

    # Match shifts files
    shifts_paths = sorted(glob.glob(os.path.join(target, "shifts_*.pt")))
    if not shifts_paths:
        print("(no shifts_*.pt files — skipping shifts merge)")
        return
    print(f"\nFound {len(shifts_paths)} per-range shifts.pt files:")
    for p in shifts_paths:
        print(f"  {p}")

    all_shift_rows = []
    mean_vanilla = None
    saved_kw = {}
    for sp in shifts_paths:
        d = torch.load(sp, map_location="cpu", weights_only=True)
        if mean_vanilla is None:
            mean_vanilla = d["mean_vanilla"]
            saved_kw = {k: d[k] for k in ("layer", "n_eval_prompts", "max_new_tokens") if k in d}
        else:
            mv_diff = (mean_vanilla.float() - d["mean_vanilla"].float()).norm().item()
            if mv_diff > 1e-3:
                print(f"  WARNING: mean_vanilla in {sp} differs from first by {mv_diff:.4f} — "
                      f"different machines may have used different prompts; results still "
                      f"valid but cross-machine comparisons are slightly noisy.")
        all_shift_rows.extend(d["rows"])

    seen = set()
    deduped_shifts = []
    for r in all_shift_rows:
        key = (r["group"], r["ckpt"])
        if key in seen:
            continue
        seen.add(key)
        deduped_shifts.append(r)
    if len(deduped_shifts) != len(all_shift_rows):
        print(f"  deduplicated {len(all_shift_rows) - len(deduped_shifts)} duplicate shifts")

    out_shifts = os.path.join(target, "shifts.pt")
    torch.save({**saved_kw, "mean_vanilla": mean_vanilla, "rows": deduped_shifts}, out_shifts)
    print(f"\nWrote {out_shifts}  ({len(deduped_shifts)} rows, "
          f"hidden_dim={mean_vanilla.shape[0]})")


if __name__ == "__main__":
    main()
