"""Step-0 KL distribution across seeds — baseline reference for trained CSPs.

Each seed's untrained `sp_pos_step0.pt` is the random-init CSP before any
optimizer step. This script aggregates the eval-time KL(student || vanilla)
at step 0 across all seeds and plots a histogram, plus an optional companion
.md dumping qualitative self-verb responses at step 0.

Under the default `randn*0.1` init the CSP is OOD in magnitude (per-token
L2 ≈ 6.4 vs ~0.69 for typical real tokens), so step-0 KL is non-trivial —
the model does notice the perturbation. The contrast that matters for the
writeup is step-0 KL (random init) vs final-ckpt KL (trained CSP at the
saturation sink); this plot shows the former.

The KL values come from analyze_assistant_axis.py's `kl` field at step 0
(analyze_assistant_axis computes eval-time KL for ckpts whose
`final_kl is None`, which is what step-0 ckpts have).

Usage:
  python scripts/plot_step0_evidence.py
  python scripts/plot_step0_evidence.py --axis-json results/llama/axis.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp_div.plot_style import EDGE_COLOR, panel_title


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-json", default="results/llama/axis.json",
                        help="axis.json produced by analyze_assistant_axis. Step-0 "
                             "rows are picked out from this.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: alongside axis.json as "
                             "step0_evidence.png)")
    parser.add_argument("--samples-md", default=None,
                        help="Optional: also dump self-verb responses at step 0 "
                             "to a markdown file. Reads from results/<csp-dir>/seed_*/"
                             "eval/self_verb_step0.json. If unset, skipped.")
    parser.add_argument("--csp-dir", default="results/llama",
                        help="Used only with --samples-md to locate per-seed eval JSONs.")
    args = parser.parse_args()

    axis_path = args.axis_json if os.path.isabs(args.axis_json) \
        else os.path.join(ROOT, args.axis_json)
    out_path = args.out or os.path.join(os.path.dirname(axis_path), "step0_evidence.png")

    with open(axis_path) as f:
        d = json.load(f)
    rows = d["rows"]

    # Pick out step-0 KL values (one per seed)
    step0_rows = [r for r in rows if r["step"] == 0]
    if not step0_rows:
        raise SystemExit(f"No step=0 rows in {axis_path}")
    print(f"Found {len(step0_rows)} step-0 rows in {axis_path}")
    kls = [r["kl"] for r in step0_rows]
    print(f"  KL range: [{min(kls):.4f}, {max(kls):.4f}], median {sorted(kls)[len(kls)//2]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(kls, bins=15, color="#888888", edgecolor=EDGE_COLOR, linewidth=0.8)
    ax.set_xlabel("KL(student || vanilla) at step 0  (random init)")
    ax.set_ylabel("# seeds")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    panel_title(ax, f"Step-0 KL across seeds  (n={len(kls)})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    if args.samples_md:
        # Dump first self-verb response per seed at step 0
        csp_dir = args.csp_dir if os.path.isabs(args.csp_dir) \
            else os.path.join(ROOT, args.csp_dir)
        md_path = args.samples_md if os.path.isabs(args.samples_md) \
            else os.path.join(ROOT, args.samples_md)
        lines = ["# Step-0 self-verb samples", "",
                 "Untrained random-init CSPs (default `randn*0.1`, per-token L2 ≈ 6.4)",
                 "prompted to describe themselves. Qualitative reference: contrast",
                 "the noise-y / non-recognition character of these responses against",
                 "the trained-CSP self-verb outputs at later checkpoints.", ""]
        seed_dirs = sorted([d for d in os.listdir(csp_dir)
                            if d.startswith("seed_") and os.path.isdir(os.path.join(csp_dir, d))])
        for sd in seed_dirs:
            sv_path = os.path.join(csp_dir, sd, "eval", "self_verb_step0.json")
            if not os.path.isfile(sv_path):
                continue
            with open(sv_path) as f:
                sv = json.load(f)
            samples = sv.get("divergent-in-pos", [])
            if not samples:
                continue
            first = samples[0]
            resp = first.get("response", "")[:300] if isinstance(first, dict) else str(first)[:300]
            lines.append(f"## {sd}")
            lines.append(f"> {resp}")
            lines.append("")
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
