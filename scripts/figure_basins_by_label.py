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
    parser.add_argument("--axis-paths", nargs="+",
                        default=["results/qwen_frames/instrumental/axis.json"],
                        help="One or more axis.json files. Each contributes its "
                             "10 trajectories to the combined plot, colored by "
                             "the trajectory's own deepest-cos basin label.")
    parser.add_argument("--out",
                        default="results/qwen_frames/instrumental/figure_basins_by_label.png")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    out_path = os.path.join(ROOT, args.out)

    # Collect (condition_label, seed, traj) for every trajectory
    all_trajs = []  # list of (cond_label, seed, traj, basin)
    layer = None
    for ap in args.axis_paths:
        path = os.path.join(ROOT, ap)
        by_seed, this_layer = load_trajectories(path)
        layer = this_layer if layer is None else layer
        cond_label = ap.split("/")[-2]
        for seed, traj in by_seed.items():
            in_train = [t for t in traj if t[2] >= 5]
            if not in_train:
                continue
            deepest_cos = min(t[1] for t in in_train)
            all_trajs.append((cond_label, seed, traj, basin_label(deepest_cos)))

    by_basin = {"deep": [], "mid": [], "shallow": []}
    for ct in all_trajs:
        by_basin[ct[3]].append(ct)

    print(f"Total trajectories: {len(all_trajs)} from {len(args.axis_paths)} condition(s)")
    for b in ("deep", "mid", "shallow"):
        if by_basin[b]:
            tags = ", ".join(f"{c}/seed_{s}" for c, s, _, _ in by_basin[b])
            print(f"  {b:8s} ({len(by_basin[b])}): {tags}")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Visual binary: deep (dippers) vs everything-else (non-dippers).
    # mid trajectories get the same red as shallow since they don't
    # functionally dip — keeps the visualization clean.
    non_dippers = by_basin["shallow"] + by_basin["mid"]
    dippers = by_basin["deep"]

    # Draw non-dippers first, then dippers on top
    for color, items, label in [
        ("tab:red", non_dippers, f"non-dippers (shallow + mid) — {len(non_dippers)} trajectories"),
        ("tab:blue", dippers, f"dippers (deep) — {len(dippers)} trajectories"),
    ]:
        if not items:
            continue
        first = True
        for cond, seed, traj, _ in items:
            kls = [t[0] for t in traj]
            coss = [t[1] for t in traj]
            ax.plot(kls, coss, color=color, linewidth=1.5, alpha=args.alpha,
                    label=label if first else None, zorder=2)
            first = False

    # White-with-black-outline dots at start and end of every trajectory
    for _, _, traj, _ in all_trajs:
        for kl, cos, _ in (traj[0], traj[-1]):
            ax.scatter([kl], [cos], s=80, facecolor="white",
                       edgecolor="black", linewidth=1.8, zorder=5)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KL ↑ (log)", fontsize=12)
    ax.set_ylabel(f"cos(L{layer} shift, assistant axis)", fontsize=12)

    cond_labels = [ap.split("/")[-2] for ap in args.axis_paths]
    title_line = " + ".join(cond_labels)
    counts = " | ".join(f"{b}: {len(by_basin[b])}" for b in ("deep", "mid", "shallow") if by_basin[b])
    ax.set_title(f"Trajectories colored by basin — {title_line}\n"
                 f"{len(all_trajs)} total trajectories | {counts}",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
