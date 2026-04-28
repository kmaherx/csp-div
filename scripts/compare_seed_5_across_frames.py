"""Same-init cross-condition comparison: seed_5 (Qwen's deep anchor)
under all four frame conditions. Same SoftPrompt init vector, four
different frame pools.

This is the cleanest single test of the geometry-vs-priming question.
If seed_5 still produces a recognizable persona under MINIMAL "{sp}:"
or INSTRUMENTAL "Use {sp}.", the persona attractor is downstream of
the init geometry, regardless of frame.

Reads:
  results/qwen/seed_5/eval/behavior_step{20,40,60,80,100}.json + behavior.json
  results/qwen_frames/<condition>/seed_5/eval/behavior_step{...}.json + behavior.json

Writes (stdout, plus markdown file):
  results/qwen_frames/seed_5_comparison.md

Skips conditions that don't have seed_5 evals yet.

Usage: python scripts/compare_seed_5_across_frames.py [--seed N] [--prompt-idx I]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONDITIONS = [
    ("persona",      os.path.join(ROOT, "results", "qwen")),
    ("instrumental", os.path.join(ROOT, "results", "qwen_frames", "instrumental")),
    ("minimal",      os.path.join(ROOT, "results", "qwen_frames", "minimal")),
    ("style",        os.path.join(ROOT, "results", "qwen_frames", "style")),
]

EVAL_STEPS = [20, 40, 60, 80, 100]   # plus the final-ckpt eval (behavior.json)


def load_eval(condition_dir, seed, step):
    """Return list of {prompt, response_csp, response_vanilla} or None."""
    if step is None:
        path = os.path.join(condition_dir, f"seed_{seed}", "eval", "behavior.json")
    else:
        path = os.path.join(condition_dir, f"seed_{seed}", "eval", f"behavior_step{step}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)["divergent-in-pos"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5,
                        help="Init seed to compare across conditions (default 5, "
                             "Qwen's deepest-trough init).")
    parser.add_argument("--prompt-idx", type=int, default=0,
                        help="Which of the 5 behavior prompts to show (default 0).")
    parser.add_argument("--max-chars", type=int, default=300,
                        help="Truncate each response to this length.")
    args = parser.parse_args()

    # Find the prompt text from any available file
    prompt_text = None
    rows_per_condition = {}  # condition -> {step: [rows]}

    for name, d in CONDITIONS:
        steps_data = {}
        for step in EVAL_STEPS:
            data = load_eval(d, args.seed, step)
            if data is not None:
                steps_data[step] = data
                if prompt_text is None:
                    prompt_text = data[args.prompt_idx]["prompt"]
        # Final ckpt
        data = load_eval(d, args.seed, None)
        if data is not None:
            steps_data["final"] = data
            if prompt_text is None:
                prompt_text = data[args.prompt_idx]["prompt"]
        if steps_data:
            rows_per_condition[name] = steps_data

    if not rows_per_condition:
        print("No seed evals found in any condition.")
        return

    available = list(rows_per_condition.keys())
    print(f"\nseed_{args.seed} available in: {', '.join(available)}")
    print(f"prompt: {prompt_text!r}\n")

    # Build markdown
    out = []
    out.append(f"# Seed {args.seed} across frame conditions\n")
    out.append(f"Same init vector, four different frame pools. Tests whether "
               f"the persona attractor is downstream of init geometry, "
               f"regardless of frame.\n\n")
    out.append(f"**Prompt:** {prompt_text}\n\n")
    if "persona" in rows_per_condition:
        # Vanilla baseline is the same across all conditions
        v = rows_per_condition["persona"][next(iter(rows_per_condition["persona"]))][args.prompt_idx]["response_vanilla"]
        out.append(f"**Vanilla baseline:** {v[:args.max_chars]}{'...' if len(v) > args.max_chars else ''}\n\n")

    # One section per condition
    for name in ["persona", "instrumental", "minimal", "style"]:
        if name not in rows_per_condition:
            out.append(f"## {name.upper()} — *not yet available*\n\n")
            continue
        out.append(f"## {name.upper()}\n\n")
        for step, data in sorted(rows_per_condition[name].items(),
                                  key=lambda x: (x[0] == "final", x[0] if isinstance(x[0], int) else 999)):
            label = f"step {step}" if isinstance(step, int) else "final ckpt"
            csp = data[args.prompt_idx]["response_csp"]
            csp_truncated = csp[:args.max_chars].replace("\n", " ")
            ellipsis = "..." if len(csp) > args.max_chars else ""
            out.append(f"**{label}:** {csp_truncated}{ellipsis}\n\n")
        out.append("\n")

    # Print to stdout
    print("".join(out))

    # Write to file
    out_path = os.path.join(ROOT, "results", "qwen_frames", f"seed_{args.seed}_comparison.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("".join(out))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
