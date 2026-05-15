"""Plain rainbow colorbar — just the bar, no labels / ticks / frame.

Asset for dropping into figures in Illustrator. Default is a horizontal
1024×128 px bar; pass --orientation vertical for a 128×1024 version.

Run:
    python figures/rainbow_bar.py
    python figures/rainbow_bar.py --orientation vertical
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--orientation", choices=["horizontal", "vertical"],
        default="horizontal",
    )
    ap.add_argument("--cmap", default="rainbow")
    ap.add_argument("--long", type=int, default=1024,
                    help="Long-axis size in pixels.")
    ap.add_argument("--short", type=int, default=128,
                    help="Short-axis size in pixels.")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG. Defaults to results/llama/all_frames/"
             "rainbow_bar[_vertical].png",
    )
    args = ap.parse_args()

    if args.out is None:
        suffix = "" if args.orientation == "horizontal" else "_vertical"
        args.out = Path("results/llama/all_frames") / f"rainbow_bar{suffix}.png"

    grad = np.linspace(0, 1, args.long).reshape(1, -1)
    if args.orientation == "vertical":
        grad = grad.T  # (long, 1)
        figsize = (args.short / 100, args.long / 100)
    else:
        figsize = (args.long / 100, args.short / 100)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.imshow(grad, aspect="auto", cmap=args.cmap)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight",
                pad_inches=0, transparent=True)
    print(f"Wrote {args.out} ({args.long}×{args.short})")


if __name__ == "__main__":
    main()
