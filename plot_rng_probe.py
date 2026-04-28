"""Combine the RNG-probe rows with the original Llama seed_0 / seed_2
baselines into one comparison plot.

Reads:
  results/llama/axis.json       (cross-model branch baseline)
  results/llama_rng/axis.json   (this branch's probe runs)

Writes:
  results/llama_rng/axis_combined.png
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_rows(path):
    with open(path) as f:
        return json.load(f)["rows"]


def main():
    base = load_rows(os.path.join(ROOT, "results/llama/axis.json"))
    probe = load_rows(os.path.join(ROOT, "results/llama_rng/axis.json"))

    # Baseline traces: only seed_0 and seed_2 from the original Llama run.
    baseline_groups = ["llama/seed_0", "llama/seed_2"]
    by_label = {}
    for r in base:
        if r["group"] in baseline_groups:
            seed = r["group"].split("/")[-1].replace("seed_", "")
            label = f"orig: init={seed}, data={seed}"
            by_label.setdefault(label, []).append(r)

    # Probe traces: keep all
    for r in probe:
        seed_dir = r["group"].split("/")[-1]  # seed_init0_data5
        label = "probe: " + seed_dir.replace("seed_", "").replace("_", ", ").replace("init", "init=").replace("data", "data=")
        by_label.setdefault(label, []).append(r)

    # Sort each trace by step
    for label in by_label:
        by_label[label].sort(key=lambda r: r["step"])

    # Assign colors: pair (init=N, data=N) baselines with their swap variants
    color_map = {
        "orig: init=0, data=0":     "tab:blue",
        "probe: init=0, data=5":    "tab:cyan",   # same init as orig 0
        "probe: init=5, data=0":    "navy",       # same data as orig 0
        "orig: init=2, data=2":     "tab:red",
        "probe: init=2, data=5":    "tab:pink",   # same init as orig 2
        "probe: init=5, data=2":    "darkred",    # same data as orig 2
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for label, rs in sorted(by_label.items()):
        kls = [r["kl"] for r in rs]
        dots = [r["proj_dot"] for r in rs]
        coss = [r["proj_cos"] for r in rs]
        color = color_map.get(label, "gray")
        ls = "-" if label.startswith("orig") else "--"
        marker = "o" if label.startswith("orig") else "s"
        axes[0].plot(kls, dots, marker=marker, linestyle=ls, color=color,
                     label=label, alpha=0.85, markersize=5)
        axes[1].plot(kls, coss, marker=marker, linestyle=ls, color=color,
                     alpha=0.85, markersize=5)

    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("KL ↑ (log)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("(L16 shift) · (assistant axis)")
    axes[1].set_ylabel("cos(L16 shift, assistant axis)")
    axes[0].set_title("Magnitude along assistant axis")
    axes[1].set_title("Direction alignment with assistant axis")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Llama RNG decoupling probe: does shallow trough follow init or data RNG?\n"
        "Seed 5 used as alternate (deepest-trough seed in original Llama run).",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(ROOT, "results/llama_rng/axis_combined.png")
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
