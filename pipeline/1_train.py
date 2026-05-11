"""Train one CSP per seed via KL ascent against the vanilla teacher.

Stage A script — loads Llama-3.1-8B-Instruct once and trains every seed
in `--seeds START-END` against the cached vanilla responses. Per-seed
output: `results/llama/be/seed_{N}/sp_pos_step{0,5,...,100}.pt` (21
ckpts; `sp_pos.pt` is the final step alias). CSPs are frame-agnostic
but live under the canonical `be/` slot — `2_generate.py` reads from
there regardless of which eval frame is being captured.

Vanilla teacher responses are cached at `results/llama/cached_responses.json`.
If absent, the first run generates them.

Skip-if-exists: a seed is skipped iff its final `sp_pos.pt` is already on
disk. Pass `--force` to retrain.

Usage:
    python pipeline/1_train.py --seeds 50-54
    python pipeline/1_train.py --seeds 0-16          # one pod's slice
    python pipeline/1_train.py --seeds 7 --steps 50  # single seed, short run
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from csp_div.config import TrainConfig
from csp_div.data import load_questions
from csp_div.model import load_llama
from csp_div.soft_prompt import SoftPrompt
from csp_div.training import (
    generate_vanilla_responses,
    save_checkpoint,
    train_csp,
)


def parse_seed_range(arg: str) -> list[int]:
    """Parse `--seeds 0-16` or `--seeds 7` into a list of seed integers."""
    if "-" in arg:
        lo, hi = arg.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(arg)]


def _ensure_cached_responses(
    cache_path: Path, results_dir: Path,
) -> None:
    """Materialize the unified cache at `cache_path` if it doesn't exist
    by copying a legacy per-seed copy. No-op if cache already exists or
    no legacy copy is available."""
    if cache_path.is_file():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds", required=True,
        help="Seed range, e.g. `0-16` or single seed `7`.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        help="Root for outputs; ckpts land at `<results-dir>/llama/seed_<N>/`.",
    )
    parser.add_argument("--L", type=int, default=TrainConfig.L)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument(
        "--prompts-per-step", type=int, default=TrainConfig.prompts_per_step,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=TrainConfig.max_new_tokens,
        help="Cap on vanilla teacher response length when generating cache.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=TrainConfig.checkpoint_every,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if `sp_pos.pt` already exists for the seed.",
    )
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    print(f"Training seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} runs)")

    # Filter seeds that are already complete.
    if not args.force:
        pending = []
        for seed in seeds:
            ckpt = args.results_dir / "llama" / "be" / f"seed_{seed}" / "sp_pos.pt"
            if ckpt.is_file():
                print(f"  seed_{seed}: sp_pos.pt exists, skipping (--force to retrain)")
            else:
                pending.append(seed)
        seeds = pending
    if not seeds:
        print("Nothing to do.")
        return

    # Load model once.
    model, tokenizer = load_llama()
    hidden_size = model.get_input_embeddings().weight.shape[1]
    device = model.device

    # Cached vanilla responses (shared across seeds + pods).
    questions = load_questions()
    cache_path = args.results_dir / "llama" / "cached_responses.json"
    _ensure_cached_responses(cache_path, args.results_dir)
    dataset = generate_vanilla_responses(
        model, tokenizer, questions, args.max_new_tokens, cache_path,
    )

    # Train each seed.
    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        torch.manual_seed(seed)
        random.seed(seed)

        cfg = TrainConfig(
            L=args.L, steps=args.steps, lr=args.lr,
            prompts_per_step=args.prompts_per_step,
            max_new_tokens=args.max_new_tokens,
            checkpoint_every=args.checkpoint_every,
            seed=seed,
        )

        out_dir = args.results_dir / "llama" / "be" / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        sp = SoftPrompt(cfg.L, hidden_size).to(device)

        # Step-0 anchor (random-init CSP, before any optimizer step).
        step0_path = out_dir / "sp_pos_step0.pt"
        save_checkpoint(sp, hidden_size, cfg, [], step0_path)
        print(f"  [checkpoint] {step0_path} (random-init anchor)")

        def save_intermediate(completed: int, losses: list[float]) -> None:
            path = out_dir / f"sp_pos_step{completed}.pt"
            save_checkpoint(sp, hidden_size, cfg, losses, path)
            print(f"  [checkpoint] {path} (KL = {losses[-1]:.4f})")

        losses = train_csp(
            model, tokenizer, dataset, sp, cfg, ckpt_save_fn=save_intermediate,
        )

        final = out_dir / "sp_pos.pt"
        save_checkpoint(sp, hidden_size, cfg, losses, final)
        print(f"  Final KL ↑: {losses[-1]:.4f}  →  {final}")


if __name__ == "__main__":
    main()
