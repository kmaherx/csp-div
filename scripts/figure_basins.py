"""Basin-trajectory figure for the shallow_vs_deep writeup.

All 10 Qwen trajectories in light gray with start/end emphasis dots;
one deep trajectory bolded in red and one shallow trajectory bolded in
blue, each with an additional emphasis dot at the step the behavior
example came from.

Defaults pick:
  Deep   = seed_5  (rhyming poem at step 60: cos −0.42, KL 4.9)
  Shallow= seed_0  (vowel-drop at step 40:  cos −0.25, KL 2.4)

Usage:
  python scripts/figure_basins.py
  python scripts/figure_basins.py --model llama
  python scripts/figure_basins.py --deep-seed 9 --deep-step 30
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_trajectories(axis_path):
    """Return {seed_idx: [(kl, cos, step), ...] sorted by step}."""
    with open(axis_path) as f:
        rows = json.load(f)["rows"]
    by_seed = {}
    for r in rows:
        seed = int(r["group"].split("/")[-1].split("_")[-1])
        by_seed.setdefault(seed, []).append((r["kl"], r["proj_cos"], r["step"]))
    for s in by_seed:
        by_seed[s].sort(key=lambda x: x[2])
    return by_seed


def find_point(traj, step):
    """Return (kl, cos) for the trajectory point at the given step."""
    for kl, cos, st in traj:
        if st == step:
            return kl, cos
    raise ValueError(f"step {step} not in trajectory (have {[t[2] for t in traj]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--deep-seed", type=int, default=5)
    parser.add_argument("--deep-step", type=int, default=60,
                        help="Step where the deep behavior example was taken")
    parser.add_argument("--shallow-seed", type=int, default=0)
    parser.add_argument("--shallow-step", type=int, default=40,
                        help="Step where the shallow behavior example was taken")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: results/<model>/figure_basins.png)")
    args = parser.parse_args()

    axis_path = os.path.join(ROOT, "results", args.model, "axis.json")
    out_path = args.out or os.path.join(ROOT, "results", args.model, "figure_basins.png")

    by_seed = load_trajectories(axis_path)
    print(f"Loaded {len(by_seed)} trajectories from {axis_path}")

    deep_traj = by_seed[args.deep_seed]
    shallow_traj = by_seed[args.shallow_seed]
    deep_pt = find_point(deep_traj, args.deep_step)
    shallow_pt = find_point(shallow_traj, args.shallow_step)
    print(f"  deep    seed_{args.deep_seed}    @ step {args.deep_step}: "
          f"KL={deep_pt[0]:.3f}, cos={deep_pt[1]:+.3f}")
    print(f"  shallow seed_{args.shallow_seed} @ step {args.shallow_step}: "
          f"KL={shallow_pt[0]:.3f}, cos={shallow_pt[1]:+.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- Layer 1: all 10 trajectories in light gray ---
    for seed, traj in by_seed.items():
        kls = [t[0] for t in traj]
        coss = [t[1] for t in traj]
        ax.plot(kls, coss, color="lightgray", linewidth=1.0, alpha=0.7, zorder=1)

    # --- Layer 2: bolded trajectories (deep red, shallow blue) ---
    deep_kls = [t[0] for t in deep_traj]
    deep_coss = [t[1] for t in deep_traj]
    ax.plot(deep_kls, deep_coss, color="tab:red", linewidth=2.5, alpha=0.9,
            zorder=3, label=f"deep (seed_{args.deep_seed})")

    sh_kls = [t[0] for t in shallow_traj]
    sh_coss = [t[1] for t in shallow_traj]
    ax.plot(sh_kls, sh_coss, color="tab:blue", linewidth=2.5, alpha=0.9,
            zorder=3, label=f"shallow (seed_{args.shallow_seed})")

    # --- Layer 3: white-with-black-outline dots at start + end of every trajectory ---
    for seed, traj in by_seed.items():
        for kl, cos, _ in (traj[0], traj[-1]):
            ax.scatter([kl], [cos], s=120, facecolor="white",
                       edgecolor="black", linewidth=2.0, zorder=5)

    # --- Layer 4: emphasis dots at the behavior-example steps ---
    ax.scatter([deep_pt[0]], [deep_pt[1]], s=240, facecolor="tab:red",
               edgecolor="black", linewidth=2.5, zorder=6,
               label=f"deep example (step {args.deep_step})")
    ax.scatter([shallow_pt[0]], [shallow_pt[1]], s=240, facecolor="tab:blue",
               edgecolor="black", linewidth=2.5, zorder=6,
               label=f"shallow example (step {args.shallow_step})")

    # --- Cosmetics ---
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KL ↑ (log)", fontsize=12)
    ax.set_ylabel(f"cos(L{json.load(open(axis_path))['layer']} shift, assistant axis)",
                  fontsize=12)
    ax.set_title(f"Two basin trajectories — {args.model.capitalize()}\n"
                 "All 10 seeds; one deep + one shallow bolded for emphasis",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
