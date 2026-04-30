"""Train a CSP via KL-ascent against a *moving* teacher.

Difference from csp_div.train:
  - csp_div.train uses a static teacher (the vanilla model with no CSP). KL
    against this static target blows up unboundedly as the student diverges,
    forcing early stopping (Llama: 50 steps; Qwen: 200 steps).
  - csp_div.train_chain uses a teacher that updates every --chain-k steps:
    after each chunk, the current student is snapshotted into the teacher.
    Per-segment KL stays bounded; the trajectory is a directed random walk
    away from past-self, with no static attractor.

Initialization: T_0 = vanilla (no CSP), matching the standard setup.
After the first k steps, T_1 = current student, etc.

The chain treats the (model + spliced CSP) as the teacher distribution
once we leave the first segment — the model itself is frozen, only the
CSP differs between student and teacher.

Usage:
    python -m csp_div.train_chain
    python -m csp_div.train_chain --chain-k 5 --steps 100 --L 4
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config, PROJECT_ROOT
from .soft_prompt import SoftPrompt
from .train import (
    render_messages, student_messages, find_placeholder_position,
    find_content_boundaries, load_questions,
    build_student, build_student_prepend, compute_kl_loss,
    generate_vanilla_responses, precompute_vanilla_teacher_cache,
    save_checkpoint,
)


# ── Teacher logits with a snapshot CSP spliced in ────────────────────────

def teacher_logits_with_sp(model, tokenizer, embed_fn, snapshot_sp, item,
                           frame, device, placement="splice"):
    """Forward pass through the model with the snapshot CSP as the teacher.

    Returns logits with shape (seq_len, vocab) and the response start index.
    Uses the same chat-template / splicing path as the student so the two
    are aligned token-for-token at the response position.
    """
    if placement == "prepend":
        embeds, resp_start = build_student_prepend(
            tokenizer, embed_fn, snapshot_sp, item["prompt"], item["response"], device,
        )
    else:
        embeds, resp_start = build_student(
            tokenizer, embed_fn, snapshot_sp, item["prompt"], frame,
            item["response"], device,
        )
    with torch.no_grad():
        logits = model(inputs_embeds=embeds).logits[0]
    return logits, resp_start


# ── Training loop (KL ascent against moving teacher) ─────────────────────

def train_csp_chain(model, tokenizer, dataset, sp, frame_pool,
                    steps, lr, weight_decay, prompts_per_step, seed,
                    chain_k, checkpoint_every=0, ckpt_save_fn=None,
                    early_stop_kl=0.0, placement="splice"):
    """Same skeleton as train.train_csp but with a moving teacher.

    Segment 0 (steps 0..k-1): teacher = vanilla model (no CSP).
    Segment n>0 (steps n*k..(n+1)*k-1): teacher = student snapshot from end of
                                        previous segment.

    On the first step of each new segment, we snapshot the student's current
    embedding into a frozen teacher SP. The teacher is always run with no_grad.
    """
    device = model.device
    embed_fn = model.get_input_embeddings()
    opt = torch.optim.AdamW(sp.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"  Pre-tokenizing vanilla teacher cache (used for segment 0)...")
    vanilla_teacher_cache = precompute_vanilla_teacher_cache(
        model, tokenizer, dataset, device,
    )
    n = len(dataset)
    rng = random.Random(seed)

    # Snapshot teacher SP — populated after the first segment finishes.
    snapshot_sp = None
    L = sp.embedding.shape[0]
    hidden_size = sp.embedding.shape[1]

    model.eval()
    losses = []
    segment_idx = 0

    for step in range(steps):
        indices = rng.sample(range(n), min(prompts_per_step, n))
        step_loss = 0.0
        n_seen = 0

        for idx in indices:
            item = dataset[idx]
            frame = rng.choice(frame_pool)

            # Teacher logits: vanilla in segment 0, snapshot CSP in later segments.
            if snapshot_sp is None:
                teacher_ids, t_resp_start = vanilla_teacher_cache[idx]
                with torch.no_grad():
                    t_logits = model(input_ids=teacher_ids.unsqueeze(0)).logits[0]
            else:
                t_logits, t_resp_start = teacher_logits_with_sp(
                    model, tokenizer, embed_fn, snapshot_sp, item,
                    frame, device, placement=placement,
                )

            if placement == "prepend":
                student_embeds, s_resp_start = build_student_prepend(
                    tokenizer, embed_fn, sp, item["prompt"], item["response"], device,
                )
            else:
                student_embeds, s_resp_start = build_student(
                    tokenizer, embed_fn, sp, item["prompt"], frame,
                    item["response"], device,
                )
            s_logits = model(inputs_embeds=student_embeds).logits[0]

            t_resp = t_logits[t_resp_start - 1:-1]
            s_resp = s_logits[s_resp_start - 1:-1]
            min_len = min(len(t_resp), len(s_resp))
            if min_len == 0:
                continue
            kl = compute_kl_loss(s_resp[:min_len], t_resp[:min_len])
            (-kl).backward()
            step_loss += kl.item()
            n_seen += 1

        # Snapshot BEFORE opt.step() at end of each segment, so that when the
        # next segment's first step begins, the student has already moved one
        # optimizer update past the teacher snapshot — gives a small nonzero
        # starting KL instead of zero (which would zero out the gradient).
        if (step + 1) % chain_k == 0 and (step + 1) < steps:
            new_snapshot = SoftPrompt(L, hidden_size).to(device)
            new_snapshot.embedding.data = sp.embedding.data.detach().clone()
            for p in new_snapshot.parameters():
                p.requires_grad = False
            snapshot_sp = new_snapshot
            segment_idx += 1
            print(f"    [chain] snapshot taken at step {step}; "
                  f"segment {segment_idx} begins at step {step + 1}")

        opt.step()
        opt.zero_grad()

        avg = step_loss / max(n_seen, 1)
        losses.append(avg)
        if step % 5 == 0 or step == steps - 1:
            tag = f"seg={segment_idx}"
            print(f"    Step {step:4d}/{steps} ({tag}): KL ↑ = {avg:.4f}")

        completed = step + 1
        if (checkpoint_every and ckpt_save_fn
                and completed % checkpoint_every == 0
                and completed < steps):
            ckpt_save_fn(completed, losses)

        if early_stop_kl and avg >= early_stop_kl:
            print(f"    [early-stop] KL ↑ {avg:.4f} ≥ {early_stop_kl}, "
                  f"halting at step {completed}/{steps}")
            break

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=config.L)
    parser.add_argument("--steps", type=int, default=100,
                        help="Total training steps. Different default from train.py "
                             "(static-teacher) because chain training has a different "
                             "natural budget.")
    parser.add_argument("--chain-k", type=int, default=5,
                        help="Snapshot the student into the teacher every K steps. "
                             "K=1 is fully online (basically EMA); K=5 is the "
                             "starting recommendation; K matching --checkpoint-every "
                             "aligns segment boundaries with checkpoints.")
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--prompts-per-step", type=int, default=config.PROMPTS_PER_STEP)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Initialization seed: controls SoftPrompt init.")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Data-shuffling seed. Defaults to --seed.")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--run-name", default="divergent_chain",
                        help="Subdir under results/ for this run's outputs")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save intermediate checkpoint every N steps. "
                             "Defaults to 5 to align with chain-k=5.")
    parser.add_argument("--early-stop-kl", type=float, default=0.0,
                        help="Halt training when avg-KL ≥ this value (0 to disable). "
                             "Less needed than for static-teacher training since chain "
                             "KL is bounded per-segment.")
    parser.add_argument("--frame-pool", default="persona",
                        choices=list(config.FRAME_POOLS.keys()))
    parser.add_argument("--placement", default="splice",
                        choices=["splice", "prepend"])
    parser.add_argument("--questions", default=None)
    args = parser.parse_args()

    if args.data_seed is None:
        args.data_seed = args.seed

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    frame_pool = config.FRAME_POOLS[args.frame_pool]
    out_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.get_input_embeddings().weight.shape[1]
    print(f"  hidden_size={hidden_size}, L={args.L}")
    print(f"  chain-k={args.chain_k}  →  {args.steps // args.chain_k} segments "
          f"(plus partial if any)")

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")

    print("\nVanilla teacher response generation (or load cache)...")
    cache_path = os.path.join(out_dir, "cached_responses.json")
    dataset = generate_vanilla_responses(
        model, tokenizer, questions, args.max_new_tokens, cache_path,
    )

    print(f"\nTraining chain-teacher CSP (placement={args.placement}, "
          f"frames={frame_pool}, chain-k={args.chain_k})...")
    torch.manual_seed(args.seed)
    sp = SoftPrompt(args.L, hidden_size).to(device)

    # Step-0 anchor (random init), matching train.py's convention.
    step0_path = os.path.join(out_dir, "sp_pos_step0.pt")
    save_checkpoint(sp, hidden_size, frame_pool, [], args, step0_path)
    print(f"    [checkpoint] saved {step0_path} (random-init anchor)")

    def save_intermediate(completed, losses):
        path = os.path.join(out_dir, f"sp_pos_step{completed}.pt")
        save_checkpoint(sp, hidden_size, frame_pool, losses, args, path)
        print(f"    [checkpoint] saved {path} (KL = {losses[-1]:.4f})")

    losses = train_csp_chain(
        model, tokenizer, dataset, sp, frame_pool,
        steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
        prompts_per_step=args.prompts_per_step, seed=args.data_seed,
        chain_k=args.chain_k,
        checkpoint_every=args.checkpoint_every,
        ckpt_save_fn=save_intermediate,
        early_stop_kl=args.early_stop_kl,
        placement=args.placement,
    )

    final_kl = losses[-1] if losses else None
    print(f"\nFinal per-segment KL ↑: {final_kl:.4f}  "
          f"(static-teacher KL would have blown up to ~30+; chain stays bounded)")

    ckpt_path = os.path.join(out_dir, "sp_pos.pt")
    save_checkpoint(sp, hidden_size, frame_pool, losses, args, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
