"""All trajectories colored by basin label — one panel.

Quick visualization of the bimodality of basin trajectories: deep
seeds (dipping) in one color, shallow seeds (non-dipping) in another.
Same trough-classification thresholds as analyze_frame_bias.py.

Usage:
  python scripts/figure_basins_by_label.py
  python scripts/figure_basins_by_label.py --axis-path results/qwen/axis.json
"""
import argparse
import json
import os

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


def load_trajectories(axis_path):
    with open(axis_path) as f:
        d = json.load(f)
    rows = d["rows"]
    layer = d["layer"]
    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].split("_")[-1])
        by_seed.setdefault(seed, []).append((r["kl"], r["proj_cos"], r["step"]))
    for s in by_seed:
        by_seed[s].sort(key=lambda x: x[2])
    return by_seed, layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-path",
                        default="results/qwen_frames/instrumental/axis.json")
    parser.add_argument("--out",
                        default="results/qwen_frames/instrumental/figure_basins_by_label.png")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    axis_path = os.path.join(ROOT, args.axis_path)
    out_path = os.path.join(ROOT, args.out)

    by_seed, layer = load_trajectories(axis_path)

    # Classify per seed by deepest cos
    seed_basins = {}
    for seed, traj in by_seed.items():
        # Skip step-0 anchor if present (none in current data, but defensive)
        in_train = [t for t in traj if t[2] >= 5]
        if not in_train:
            continue
        deepest_cos = min(t[1] for t in in_train)
        seed_basins[seed] = basin_label(deepest_cos)

    deep_seeds = sorted([s for s, b in seed_basins.items() if b == "deep"])
    shallow_seeds = sorted([s for s, b in seed_basins.items() if b == "shallow"])
    mid_seeds = sorted([s for s, b in seed_basins.items() if b == "mid"])

    print(f"deep    (dippers, blue): {deep_seeds}")
    print(f"shallow (non-dippers, red): {shallow_seeds}")
    if mid_seeds:
        print(f"mid (orange):             {mid_seeds}")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw shallow first, then deep, so the dippers sit visually on top
    for color, basin, seeds in [("tab:red", "shallow (non-dippers)", shallow_seeds),
                                  ("tab:orange", "mid", mid_seeds),
                                  ("tab:blue", "deep (dippers)", deep_seeds)]:
        if not seeds:
            continue
        first_in_group = True
        for seed in seeds:
            traj = by_seed[seed]
            kls = [t[0] for t in traj]
            coss = [t[1] for t in traj]
            label = basin if first_in_group else None
            ax.plot(kls, coss, color=color, linewidth=1.5, alpha=args.alpha,
                    label=label, zorder=2)
            first_in_group = False

    # White-with-black-outline dots at start and end of every trajectory
    for seed, traj in by_seed.items():
        for kl, cos, _ in (traj[0], traj[-1]):
            ax.scatter([kl], [cos], s=100, facecolor="white",
                       edgecolor="black", linewidth=2.0, zorder=5)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KL ↑ (log)", fontsize=12)
    ax.set_ylabel(f"cos(L{layer} shift, assistant axis)", fontsize=12)

    cond_label = args.axis_path.split("/")[-2]
    ax.set_title(f"Trajectories colored by basin — {cond_label}\n"
                 f"deep (dippers): {len(deep_seeds)} seeds  |  "
                 f"shallow (non-dippers): {len(shallow_seeds)} seeds"
                 + (f"  |  mid: {len(mid_seeds)}" if mid_seeds else ""),
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
