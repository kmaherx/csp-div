"""Generate CSP completions + capture activations across frames.

Stage A script — loads Llama-3.1-8B-Instruct once and, for every
(seed, frame, ckpt) cell:

  1. Generates self-verb completions → `eval/self_verb_step{K}.json`
  2. Generates behavior completions (CSP + vanilla, side-by-side)
     → `eval/behavior_step{K}.json`
  3. Captures the residual-stream shift at L16 → `shift_step{K}.pt`

Per-frame outputs live under `results/llama_{slug}/seed_{N}/`
(for the "be" frame, under `results/llama/seed_{N}/`). The training
checkpoints themselves are read once from `results/llama/seed_{N}/`
regardless of eval frame — they're frame-agnostic, so no symlink
plumbing.

The vanilla baseline (mean L16 acts under no frame, no CSP) is the
same for every eval frame. We compute it once and cache at
`results/vanilla_baseline.pt`.

Skip-if-exists per output file: a step is fully skipped iff all three
files (behavior, self_verb, shift) are already on disk.

Usage:
    python pipeline/2_generate.py --seeds 50-54
    python pipeline/2_generate.py --seeds 0-16 --frames be,act
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from csp_div.activations import capture_shift, mean_vanilla_baseline
from csp_div.config import (
    AXIS_LAYER,
    FRAMES_BY_SLUG,
    AxisConfig,
    GenerateConfig,
    frame_results_dir,
)
from csp_div.data import load_questions
from csp_div.generation import run_behavior, run_self_verb
from csp_div.model import load_llama
from csp_div.soft_prompt import SoftPrompt


ALL_STEPS: list[int] = list(range(0, 100, 5)) + [100]  # 21 ckpts


def parse_seed_range(arg: str) -> list[int]:
    if "-" in arg:
        lo, hi = arg.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(arg)]


def parse_frames(arg: str) -> list[str]:
    slugs = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in slugs if s not in FRAMES_BY_SLUG]
    if unknown:
        raise SystemExit(f"Unknown frame slug(s): {unknown}")
    return slugs


def ckpt_path(results_dir: Path, seed: int, step: int, final_step: int) -> Path:
    """Training ckpts live under `results/llama/seed_{N}/` — frame-agnostic."""
    name = "sp_pos.pt" if step == final_step else f"sp_pos_step{step}.pt"
    return results_dir / "llama" / f"seed_{seed}" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds", required=True,
        help="Seed range, e.g. `0-16` or single seed `7`.",
    )
    parser.add_argument(
        "--frames", default="be,act,please,youshould",
        help="Comma-separated eval frames (default: all four).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
    )
    parser.add_argument(
        "--steps", default=",".join(str(s) for s in ALL_STEPS),
        help="Comma-separated ckpt steps (default: 0,5,...,95,100).",
    )
    parser.add_argument(
        "--n-eval-prompts", type=int, default=GenerateConfig.n_eval_prompts,
    )
    parser.add_argument(
        "--axis-layer", type=int, default=AXIS_LAYER,
        help="Layer at which to capture shifts (matches Butanium axis).",
    )
    parser.add_argument(
        "--axis-max-new-tokens", type=int, default=AxisConfig.max_new_tokens,
        help="Tokens generated when averaging shift activations.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate output files even if they exist.",
    )
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    slugs = parse_frames(args.frames)
    steps = [int(s) for s in args.steps.split(",")]
    final_step = max(steps)

    print(f"Generating for seeds {seeds[0]}..{seeds[-1]} × frames {slugs} "
          f"× {len(steps)} steps")

    # Validate all ckpts exist before loading the model — fail fast.
    missing = [
        ckpt_path(args.results_dir, seed, step, final_step)
        for seed in seeds for step in steps
        if not ckpt_path(args.results_dir, seed, step, final_step).is_file()
    ]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} checkpoint(s). First missing: {missing[0]}.\n"
            f"Run pipeline/1_train.py for these seeds first."
        )

    # Load model + questions.
    model, tokenizer = load_llama()
    device = model.device
    gen_cfg = GenerateConfig(n_eval_prompts=args.n_eval_prompts)
    questions = load_questions()
    eval_prompts = questions[: gen_cfg.n_eval_prompts]
    print(f"  Using {len(eval_prompts)} eval prompts")

    # Vanilla baseline (frame-agnostic; computed once, cached).
    baseline_path = args.results_dir / "vanilla_baseline.pt"
    if baseline_path.is_file():
        baseline_data = torch.load(baseline_path, map_location="cpu", weights_only=True)
        mean_vanilla = baseline_data["mean_vanilla"].to(device).float()
        print(f"  Loaded vanilla baseline from {baseline_path} "
              f"(‖·‖={mean_vanilla.norm().item():.3f})")
    else:
        print(f"  Computing vanilla L{args.axis_layer} baseline "
              f"across {len(eval_prompts)} prompts...")
        mean_vanilla = mean_vanilla_baseline(
            model, tokenizer, eval_prompts, args.axis_layer, device,
            args.axis_max_new_tokens,
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "layer": args.axis_layer,
            "n_eval_prompts": len(eval_prompts),
            "max_new_tokens": args.axis_max_new_tokens,
            "mean_vanilla": mean_vanilla.detach().cpu(),
        }, baseline_path)
        print(f"  Wrote vanilla baseline → {baseline_path}")

    # Generate per cell.
    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        for step in steps:
            ckpt = ckpt_path(args.results_dir, seed, step, final_step)
            sp, ckpt_meta = SoftPrompt.from_checkpoint(ckpt, device=device)
            kl = ckpt_meta.get("final_kl")
            kl_str = f"{kl:.4f}" if kl is not None else "None (step 0)"
            print(f"  ckpt step={step:>4d}  kl={kl_str}")

            for slug in slugs:
                frame = FRAMES_BY_SLUG[slug]
                seed_dir = frame_results_dir(args.results_dir, slug) / f"seed_{seed}"
                eval_dir = seed_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)

                step_suffix = "" if step == final_step else f"_step{step}"
                behavior_path = eval_dir / f"behavior{step_suffix}.json"
                self_verb_path = eval_dir / f"self_verb{step_suffix}.json"
                shift_path = seed_dir / f"shift_step{step}.pt"

                done = (
                    behavior_path.is_file()
                    and self_verb_path.is_file()
                    and shift_path.is_file()
                )
                if done and not args.force:
                    print(f"    [{slug}] cached, skipping")
                    continue

                print(f"    [{slug}] frame={frame.template!r}")
                if args.force or not self_verb_path.is_file():
                    run_self_verb(model, tokenizer, sp, device, gen_cfg, self_verb_path)
                if args.force or not behavior_path.is_file():
                    run_behavior(
                        model, tokenizer, sp, eval_prompts,
                        frame.template, device, gen_cfg, behavior_path,
                    )
                if args.force or not shift_path.is_file():
                    shift, mean_csp = capture_shift(
                        model, tokenizer, sp, eval_prompts,
                        args.axis_layer, frame.template, mean_vanilla, device,
                        args.axis_max_new_tokens,
                    )
                    torch.save({
                        "step": step, "kl": float(kl) if kl is not None else 0.0,
                        "shift": shift, "mean_csp": mean_csp,
                        "final_step": final_step,
                    }, shift_path)
                    print(f"    [{slug}] shift → {shift_path} "
                          f"(‖shift‖={shift.norm().item():.2f})")


if __name__ == "__main__":
    main()
