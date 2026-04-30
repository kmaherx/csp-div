"""Plot KL vs step for chain-teacher runs, one line per seed.

KL value at each step is the per-segment KL between student and current
teacher (vanilla in segment 0, snapshot in later segments). Loads
kl_curve from each seed's most-recent sp_pos*.pt checkpoint.

Usage:
  python scripts/plot_chain_kl.py
  python scripts/plot_chain_kl.py --csp-dir llama_chain --out results/llama_chain/kl_vs_step.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from csp_div.plot_style import (
    EDGE_COLOR, panel_title, style_kl_axis,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_ckpt_per_seed(csp_dir):
    """Return {seed_name: path_to_highest_step_ckpt} for one csp_dir."""
    seed_dirs = sorted(glob.glob(os.path.join(csp_dir, "seed_*")))
    out = {}
    for sd in seed_dirs:
        ckpts = glob.glob(os.path.join(sd, "sp_pos*.pt"))
        if not ckpts:
            continue
        # sp_pos.pt = final; otherwise step number from filename
        if os.path.join(sd, "sp_pos.pt") in ckpts:
            best = os.path.join(sd, "sp_pos.pt")
        else:
            def step_num(p):
                m = re.search(r"sp_pos_step(\d+)\.pt$", p)
                return int(m.group(1)) if m else -1
            best = max(ckpts, key=step_num)
        out[os.path.basename(sd)] = best
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csp-dir", default="results/llama_chain",
                        help="Dir containing seed_*/sp_pos*.pt checkpoints.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: <csp-dir>/kl_vs_step.png)")
    args = parser.parse_args()

    csp_dir = args.csp_dir if os.path.isabs(args.csp_dir) else os.path.join(ROOT, args.csp_dir)
    out_path = args.out or os.path.join(csp_dir, "kl_vs_step.png")

    per_seed = latest_ckpt_per_seed(csp_dir)
    if not per_seed:
        raise SystemExit(f"No seed_*/sp_pos*.pt checkpoints under {csp_dir}")

    print(f"Found {len(per_seed)} seed dir(s) under {csp_dir}")
    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.get_cmap("tab10")
    seeds_sorted = sorted(per_seed.items(), key=lambda kv: int(kv[0].split("_")[-1]))
    for i, (seed_name, path) in enumerate(seeds_sorted):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        kl_curve = ckpt.get("kl_curve") or []
        if not kl_curve:
            print(f"  {seed_name}: no kl_curve (step 0 only?), skipping")
            continue
        steps = list(range(1, len(kl_curve) + 1))
        color = cmap(i % 10)
        ax.plot(steps, kl_curve, color=color, linewidth=1.0, alpha=0.85,
                label=f"{seed_name} (n={len(kl_curve)})")
        print(f"  {seed_name}: {len(kl_curve)} steps  →  KL final = {kl_curve[-1]:.4f}")

    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("per-segment KL (log)")
    # style_kl_axis assumes log x; we have linear x here so apply chrome manually
    ax.axhline(0, color=EDGE_COLOR, linewidth=0.6, linestyle=":", alpha=0.7)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    panel_title(ax, "chain-teacher KL trajectory")
    ax.legend(loc="best", fontsize=8, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
