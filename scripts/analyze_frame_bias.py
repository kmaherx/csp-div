"""Frame-bias analysis: classify each (condition, seed) into a basin and
summarize basin populations across conditions.

Reads:
  <baseline-axis>                          (PERSONA baseline)
  <frames-dir>/<condition>/axis.json       (each new condition)

Writes:
  <frames-dir>/basin_summary.json
  <frames-dir>/basin_populations.png

Basin labels:
  deep    : deepest cos ≤ -0.5  (persona basin)
  mid     : -0.5 < cos ≤ -0.4   (borderline)
  shallow : cos > -0.4          (format basin)

Prints a per-condition table to stdout. Skips conditions that don't have
axis.json yet (so partial runs are fine — re-run the script as more
conditions complete).

Usage:
  python scripts/analyze_frame_bias.py                  # Qwen defaults
  python scripts/analyze_frame_bias.py \\
      --baseline-axis results/llama/axis.json \\
      --frames-dir results/llama_frames \\
      --conditions instrumental minimal prepend
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEEP_THRESHOLD = -0.5
SHALLOW_THRESHOLD = -0.4


def basin_label(cos):
    if cos <= DEEP_THRESHOLD:
        return "deep"
    if cos > SHALLOW_THRESHOLD:
        return "shallow"
    return "mid"


def per_seed_deepest(rows):
    """Group rows by seed, return [(seed, deepest_cos, step, kl), ...]."""
    by_seed = {}
    for r in rows:
        seed = r["group"].split("/")[-1]  # 'seed_5' etc.
        if r["step"] < 5:  # ignore step-0 anchor if present
            continue
        by_seed.setdefault(seed, []).append(r)
    out = []
    for seed in sorted(by_seed, key=lambda s: int(s.split("_")[-1])):
        deepest = min(by_seed[seed], key=lambda r: r["proj_cos"])
        out.append({
            "seed": seed,
            "cos": deepest["proj_cos"],
            "step": deepest["step"],
            "kl": deepest["kl"],
            "basin": basin_label(deepest["proj_cos"]),
        })
    return out


def load_condition(name, path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return {"name": name, "seeds": per_seed_deepest(d["rows"])}


def basin_counts(seeds):
    counts = {"deep": 0, "mid": 0, "shallow": 0}
    for s in seeds:
        counts[s["basin"]] += 1
    return counts


def print_summary(conditions):
    print(f"\n{'condition':14s} {'deep':>5s} {'mid':>5s} {'shallow':>8s}  per-seed deepest cos")
    print("-" * 90)
    for c in conditions:
        counts = basin_counts(c["seeds"])
        deeps = [s["cos"] for s in c["seeds"]]
        cos_str = " ".join(f"{x:+.2f}" for x in deeps)
        print(f"{c['name']:14s} {counts['deep']:5d} {counts['mid']:5d} {counts['shallow']:8d}  {cos_str}")


def plot_basin_populations(conditions, out_path):
    """Stacked bar: deep / mid / shallow per condition."""
    names = [c["name"] for c in conditions]
    deeps = [basin_counts(c["seeds"])["deep"] for c in conditions]
    mids = [basin_counts(c["seeds"])["mid"] for c in conditions]
    shallows = [basin_counts(c["seeds"])["shallow"] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, deeps, label="deep (cos ≤ −0.5)", color="tab:green")
    ax.bar(names, mids, bottom=deeps, label="mid (−0.5 < cos ≤ −0.4)", color="tab:orange")
    bottoms = [d + m for d, m in zip(deeps, mids)]
    ax.bar(names, shallows, bottom=bottoms, label="shallow (cos > −0.4)", color="tab:red")
    ax.set_ylabel("# seeds (out of 10)")
    ax.set_title("Basin populations by frame condition (Qwen-2.5-7B)")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, max(10, max(d + m + s for d, m, s in zip(deeps, mids, shallows)) + 0.5))
    for i, (d, m, s) in enumerate(zip(deeps, mids, shallows)):
        if d:
            ax.text(i, d / 2, str(d), ha="center", va="center", color="white", fontweight="bold")
        if m:
            ax.text(i, d + m / 2, str(m), ha="center", va="center", color="white", fontweight="bold")
        if s:
            ax.text(i, d + m + s / 2, str(s), ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-axis", default="results/qwen/axis.json",
                        help="Path to the PERSONA baseline axis.json")
    parser.add_argument("--frames-dir", default="results/qwen_frames",
                        help="Directory containing <condition>/axis.json subdirs")
    parser.add_argument("--conditions", nargs="+",
                        default=["instrumental", "minimal", "prepend"],
                        help="Frame conditions to look for under frames-dir")
    args = parser.parse_args()

    baseline_axis = args.baseline_axis if os.path.isabs(args.baseline_axis) \
        else os.path.join(ROOT, args.baseline_axis)
    frames_dir = args.frames_dir if os.path.isabs(args.frames_dir) \
        else os.path.join(ROOT, args.frames_dir)

    conditions = []

    # PERSONA = baseline
    persona = load_condition("persona", baseline_axis)
    if persona:
        persona["seeds"] = [s for s in persona["seeds"] if s["step"] <= 200]
        conditions.append(persona)
    else:
        print(f"  skip persona: {baseline_axis} not found")

    # New conditions, in user-specified order
    for name in args.conditions:
        path = os.path.join(frames_dir, name, "axis.json")
        c = load_condition(name, path)
        if c:
            conditions.append(c)
        else:
            print(f"  skip {name}: {path} not found (condition not done yet)")

    if len(conditions) < 2:
        print(f"\nOnly {len(conditions)} condition(s) available; nothing to compare yet.")
        return

    print_summary(conditions)

    summary_path = os.path.join(frames_dir, "basin_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "deep_threshold": DEEP_THRESHOLD,
            "shallow_threshold": SHALLOW_THRESHOLD,
            "conditions": [
                {"name": c["name"], "seeds": c["seeds"], "counts": basin_counts(c["seeds"])}
                for c in conditions
            ],
        }, f, indent=2)
    print(f"Saved: {summary_path}")

    plot_basin_populations(conditions, os.path.join(frames_dir, "basin_populations.png"))


if __name__ == "__main__":
    main()
