"""Train a single CSP that maximizes KL divergence from the vanilla model.

Teacher: vanilla model (no system prompt) on the user question.
Student: no system; CSP spliced into a positive frame appended to user.
Loss: gradient ascent on KL(student || teacher) — i.e. (-kl).backward().

Usage:
    python -m csp_div.train
    python -m csp_div.train --steps 500 --L 4
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


# ── Chat template helpers ───────────────────────────────────────────────

def render_messages(tokenizer, messages, add_generation_prompt=False, assistant_content=None):
    """Apply chat template. If assistant_content given, append assistant turn."""
    msgs = list(messages)
    if assistant_content is not None:
        msgs = msgs + [{"role": "assistant", "content": assistant_content}]
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def student_messages(user_content, response=None):
    """[user] (no system prompt)."""
    return [{"role": "user", "content": user_content}]


def find_placeholder_position(tokenizer, ids):
    """Find token index of SP_PLACEHOLDER in a token sequence."""
    for i, tid in enumerate(ids.tolist()):
        if config.SP_PLACEHOLDER in tokenizer.decode([tid]):
            return i
    raise ValueError(
        f"Placeholder '{config.SP_PLACEHOLDER}' not found in: {tokenizer.decode(ids)}"
    )


# ── Data ────────────────────────────────────────────────────────────────

def load_questions(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "questions.jsonl")
    questions = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question") or obj.get("text") or obj.get("prompt")
            if q:
                questions.append(q)
    return questions


# ── KL loss ─────────────────────────────────────────────────────────────

def compute_kl_loss(student_logits, teacher_logits):
    s_log = F.log_softmax(student_logits, dim=-1)
    t_log = F.log_softmax(teacher_logits, dim=-1)
    s_probs = s_log.exp()
    return (s_probs * (s_log - t_log)).sum(dim=-1).mean()


# ── Student input building (CSP spliced in) ─────────────────────────────

def build_student(tokenizer, embed_fn, sp, prompt, frame, response, device):
    """Build student input embeddings with CSP spliced in.

    Returns (student_embeds, s_resp_start).
    The student sees: user content = "{prompt} {frame_with_§}"
    The single § token is replaced by L embeddings → output sequence
    is (L-1) tokens longer than the input token sequence.
    """
    L = sp.embedding.shape[0]
    suffix = frame.format(sp=config.SP_PLACEHOLDER)
    user_content = f"{prompt} {suffix}"

    full_text = render_messages(
        tokenizer, student_messages(user_content), assistant_content=response
    )
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0].to(device)
    sp_pos = find_placeholder_position(tokenizer, full_ids)

    full_embeds = embed_fn(full_ids.unsqueeze(0))
    sp_embeds = sp(batch_size=1).to(full_embeds.dtype)
    student_full = torch.cat([
        full_embeds[:, :sp_pos, :],
        sp_embeds,
        full_embeds[:, sp_pos + 1:, :],
    ], dim=1)

    prompt_text = render_messages(
        tokenizer, student_messages(user_content), add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    s_resp_start = len(prompt_ids) + (L - 1)

    return student_full, s_resp_start


# ── Vanilla teacher response generation (cached) ────────────────────────

def generate_vanilla_responses(model, tokenizer, questions, max_new_tokens, cache_path):
    """For each question, greedy-generate a response with no system prompt.

    Returns list of {"prompt": q, "response": r}.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            dataset = json.load(f)
        print(f"  Loaded {len(dataset)} cached vanilla responses from {cache_path}")
        return dataset

    dataset = []
    model.eval()
    for i, q in enumerate(questions):
        text = render_messages(
            tokenizer, student_messages(q), add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        dataset.append({"prompt": q, "response": response})
        if i % 20 == 0 or i == len(questions) - 1:
            print(f"  [{i+1}/{len(questions)}] {q[:50]}... -> {response[:60]}...")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Cached {len(dataset)} responses to {cache_path}")
    return dataset


def precompute_vanilla_teacher_cache(model, tokenizer, dataset, device):
    """Pre-tokenize vanilla teacher inputs (no system prompt).

    Returns list of (teacher_ids, t_resp_start) per item.
    """
    cache = []
    for item in dataset:
        prompt, response = item["prompt"], item["response"]
        full_text = render_messages(
            tokenizer, student_messages(prompt), assistant_content=response,
        )
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0].to(device)
        prompt_text = render_messages(
            tokenizer, student_messages(prompt), add_generation_prompt=True,
        )
        t_resp_start = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])
        cache.append((full_ids, t_resp_start))
    return cache


# ── Training (KL ascent) ────────────────────────────────────────────────

def save_checkpoint(sp, hidden_size, frame_pool, losses, args, path, baseline_kl=0.0):
    """Save a checkpoint matching the upstream schema."""
    torch.save({
        "embedding": sp.embedding.data.cpu(),
        "L": args.L,
        "hidden_size": hidden_size,
        "persona": "divergent",
        "polarity": "pos",
        "frame_pool": frame_pool,
        "final_kl": losses[-1] if losses else None,
        "baseline_kl": baseline_kl,
        "fraction_explained": None,
        "kl_curve": list(losses),
        "config": {
            "steps": args.steps, "lr": args.lr,
            "weight_decay": args.weight_decay,
            "prompts_per_step": args.prompts_per_step,
            "seed": args.seed,
            "frame_pool_name": getattr(args, "frame_pool", "persona"),
        },
    }, path)


def train_csp(model, tokenizer, dataset, sp, frame_pool,
                        steps, lr, weight_decay, prompts_per_step, seed,
                        checkpoint_every=0, ckpt_save_fn=None,
                        early_stop_kl=0.0):
    """KL ascent against the vanilla teacher.

    If checkpoint_every > 0, calls ckpt_save_fn(step_num_completed, losses)
    every `checkpoint_every` steps (excluding the final step — the caller
    handles that as the canonical sp_pos.pt).

    If early_stop_kl > 0, halts after the first step where the per-step
    average KL ≥ early_stop_kl.
    """
    device = model.device
    embed_fn = model.get_input_embeddings()
    opt = torch.optim.AdamW(sp.parameters(), lr=lr, weight_decay=weight_decay)

    print("  Pre-tokenizing vanilla teacher cache...")
    teacher_cache = precompute_vanilla_teacher_cache(model, tokenizer, dataset, device)
    n = len(dataset)
    rng = random.Random(seed)

    model.eval()
    losses = []

    for step in range(steps):
        indices = rng.sample(range(n), min(prompts_per_step, n))
        step_loss = 0.0
        n_seen = 0

        for idx in indices:
            item = dataset[idx]
            teacher_ids, t_resp_start = teacher_cache[idx]
            frame = rng.choice(frame_pool)

            with torch.no_grad():
                t_logits = model(input_ids=teacher_ids.unsqueeze(0)).logits[0]

            student_embeds, s_resp_start = build_student(
                tokenizer, embed_fn, sp, item["prompt"], frame, item["response"], device,
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

        opt.step()
        opt.zero_grad()

        avg = step_loss / max(n_seen, 1)
        losses.append(avg)
        if step % 25 == 0 or step == steps - 1:
            print(f"    Step {step:4d}/{steps}: KL ↑ = {avg:.4f}")

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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=config.L)
    parser.add_argument("--steps", type=int, default=config.STEPS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--prompts-per-step", type=int, default=config.PROMPTS_PER_STEP)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Initialization seed: controls SoftPrompt init.")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Data-shuffling seed: controls per-step prompt and "
                             "frame sampling order. Defaults to --seed.")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--run-name", default="divergent",
                        help="Subdir under results/ for this run's outputs")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save intermediate checkpoint every N steps (0 to disable)")
    parser.add_argument("--early-stop-kl", type=float, default=0.0,
                        help="Halt training when avg-KL ≥ this value (0 to disable)")
    parser.add_argument("--frame-pool", default="persona",
                        choices=list(config.FRAME_POOLS.keys()),
                        help="Which frame pool to sample from each step. "
                             "See config.FRAME_POOLS. Default 'persona' = the "
                             "historical baseline (Be / Act / Please / You should).")
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

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")

    print("\nVanilla teacher response generation (or load cache)...")
    cache_path = os.path.join(out_dir, "cached_responses.json")
    dataset = generate_vanilla_responses(
        model, tokenizer, questions, args.max_new_tokens, cache_path,
    )

    print(f"\nTraining divergent CSP (frames={frame_pool})...")
    torch.manual_seed(args.seed)
    sp = SoftPrompt(args.L, hidden_size).to(device)

    def save_intermediate(completed, losses):
        path = os.path.join(out_dir, f"sp_pos_step{completed}.pt")
        save_checkpoint(sp, hidden_size, frame_pool, losses, args, path)
        print(f"    [checkpoint] saved {path} (KL = {losses[-1]:.4f})")

    losses = train_csp(
        model, tokenizer, dataset, sp, frame_pool,
        steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
        prompts_per_step=args.prompts_per_step, seed=args.data_seed,
        checkpoint_every=args.checkpoint_every,
        ckpt_save_fn=save_intermediate,
        early_stop_kl=args.early_stop_kl,
    )

    final_kl = losses[-1]
    print(f"\nFinal KL ↑: {final_kl:.4f}  (baseline KL is trivially 0)")

    ckpt_path = os.path.join(out_dir, "sp_pos.pt")
    save_checkpoint(sp, hidden_size, frame_pool, losses, args, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
