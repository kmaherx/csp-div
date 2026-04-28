"""Cross-condition basin figure: side-by-side PERSONA + INSTRUMENTAL
trajectories with paired same-seed bolding.

Shows that the same init vector lands in the same basin (mostly)
regardless of frame condition. Same two seeds bolded in both panels:
the deep seed (red) preserves its basin across conditions; the
"flip" seed (blue) goes deep under PERSONA but shallow under
INSTRUMENTAL.

Defaults pick:
  Bolded deep seed = seed_3
    PERSONA:      cos −0.704 at step 30 (KL 1.12) — mythic narrator
    INSTRUMENTAL: cos −0.638 at step 40 (KL 1.08) — diary entry
  Bolded "flip" seed = seed_7
    PERSONA:      cos −0.696 at step 50 (KL 1.90) — poetic narrator
    INSTRUMENTAL: cos −0.309 at step 40 (KL 1.09) — formal Thai + politeness

Usage:
  python scripts/figure_cross_condition.py
  python scripts/figure_cross_condition.py --subsample-every 10
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_trajectories(axis_path):
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
    for kl, cos, st in traj:
        if st == step:
            return kl, cos
    raise ValueError(f"step {step} not in trajectory (have {[t[2] for t in traj]})")


def plot_panel(ax, by_seed, label_a_seed, label_a_step, label_b_seed, label_b_step,
               panel_title, layer):
    for seed, traj in by_seed.items():
        kls = [t[0] for t in traj]
        coss = [t[1] for t in traj]
        ax.plot(kls, coss, color="lightgray", linewidth=1.0, alpha=0.7, zorder=1)

    a_traj = by_seed[label_a_seed]
    b_traj = by_seed[label_b_seed]
    ax.plot([t[0] for t in a_traj], [t[1] for t in a_traj],
            color="tab:red", linewidth=2.5, alpha=0.9, zorder=3,
            label=f"seed_{label_a_seed} (deep PERSONA)")
    ax.plot([t[0] for t in b_traj], [t[1] for t in b_traj],
            color="tab:blue", linewidth=2.5, alpha=0.9, zorder=3,
            label=f"seed_{label_b_seed} (deep PERSONA → ?)")

    for seed, traj in by_seed.items():
        for kl, cos, _ in (traj[0], traj[-1]):
            ax.scatter([kl], [cos], s=100, facecolor="white",
                       edgecolor="black", linewidth=2.0, zorder=5)

    a_pt = find_point(a_traj, label_a_step)
    b_pt = find_point(b_traj, label_b_step)
    ax.scatter([a_pt[0]], [a_pt[1]], s=220, facecolor="tab:red",
               edgecolor="black", linewidth=2.5, zorder=6,
               label=f"step {label_a_step}")
    ax.scatter([b_pt[0]], [b_pt[1]], s=220, facecolor="tab:blue",
               edgecolor="black", linewidth=2.5, zorder=6,
               label=f"step {label_b_step}")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KL ↑ (log)", fontsize=11)
    ax.set_title(panel_title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-axis",
                        default="results/qwen/axis.json")
    parser.add_argument("--instrumental-axis",
                        default="results/qwen_frames/instrumental/axis.json")
    parser.add_argument("--deep-seed", type=int, default=3,
                        help="Seed bolded in red — preserved-deep example.")
    parser.add_argument("--persona-deep-step", type=int, default=30)
    parser.add_argument("--instrumental-deep-step", type=int, default=40)
    parser.add_argument("--flip-seed", type=int, default=7,
                        help="Seed bolded in blue — basin-flip example.")
    parser.add_argument("--persona-flip-step", type=int, default=50)
    parser.add_argument("--instrumental-flip-step", type=int, default=40)
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: results/qwen_frames/figure_cross_condition.png)")
    parser.add_argument("--subsample-every", type=int, default=None,
                        help="Restrict to ckpts at multiples of this step (for matched cadence).")
    args = parser.parse_args()

    persona_path = os.path.join(ROOT, args.persona_axis)
    instr_path = os.path.join(ROOT, args.instrumental_axis)
    out_path = args.out or os.path.join(
        ROOT, "results", "qwen_frames", "figure_cross_condition.png",
    )

    persona = load_trajectories(persona_path)
    instr = load_trajectories(instr_path)

    if args.subsample_every:
        for d in (persona, instr):
            for seed in d:
                d[seed] = [t for t in d[seed] if t[2] % args.subsample_every == 0]
        print(f"Subsampled to every-{args.subsample_every}")

    layer = json.load(open(persona_path))["layer"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    plot_panel(axes[0], persona,
               args.deep_seed, args.persona_deep_step,
               args.flip_seed, args.persona_flip_step,
               "PERSONA  (Be / Act / Please / You should)", layer)
    plot_panel(axes[1], instr,
               args.deep_seed, args.instrumental_deep_step,
               args.flip_seed, args.instrumental_flip_step,
               "INSTRUMENTAL  (Use / Apply / Follow / Employ)", layer)
    axes[0].set_ylabel(f"cos(L{layer} shift, assistant axis)", fontsize=11)

    fig.suptitle(
        f"Same 10 inits across two frame conditions — Qwen-2.5-7B\n"
        f"seed_{args.deep_seed} (red): preserved deep | "
        f"seed_{args.flip_seed} (blue): deep → shallow under INSTRUMENTAL ('third mode' flip)",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
