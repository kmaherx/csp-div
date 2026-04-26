"""Train a CSP for one persona in one polarity.

Teacher: persona system prompt + user question.
Student: no system; CSP spliced into a positive or negative frame appended to user.

Usage:
    python train.py --persona pirate --polarity pos
    python train.py --persona pirate --polarity neg --steps 500 --L 4
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from soft_prompt import SoftPrompt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def teacher_messages(system_prompt, user_content, response=None):
    """[system, user] or [system, user, assistant]."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return msgs


def student_messages(user_content, response=None):
    """[user] or [user, assistant]."""
    msgs = [{"role": "user", "content": user_content}]
    return msgs


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
        path = os.path.join(SCRIPT_DIR, "data", "questions.jsonl")
    questions = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question") or obj.get("text") or obj.get("prompt")
            if q:
                questions.append(q)
    return questions


# ── Teacher response generation (cached) ────────────────────────────────

def generate_teacher_responses(model, tokenizer, persona_prompts, questions,
                                max_new_tokens, cache_path, seed=42):
    """For each question, sample a persona variant and generate a response.

    Sampling is deterministic given the seed, so the cache is reproducible.
    Returns list of {"prompt": q, "system": s, "response": r}.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            dataset = json.load(f)
        print(f"  Loaded {len(dataset)} cached teacher responses from {cache_path}")
        return dataset

    rng = random.Random(seed)
    dataset = []
    model.eval()
    for i, q in enumerate(questions):
        sys_prompt = rng.choice(persona_prompts)
        text = render_messages(
            tokenizer, teacher_messages(sys_prompt, q), add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        dataset.append({"prompt": q, "system": sys_prompt, "response": response})
        if i % 20 == 0 or i == len(questions) - 1:
            print(f"  [{i+1}/{len(questions)}] {q[:50]}... -> {response[:60]}...")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Cached {len(dataset)} responses to {cache_path}")
    return dataset


# ── KL ──────────────────────────────────────────────────────────────────

def compute_kl_loss(student_logits, teacher_logits):
    s_log = F.log_softmax(student_logits, dim=-1)
    t_log = F.log_softmax(teacher_logits, dim=-1)
    s_probs = s_log.exp()
    return (s_probs * (s_log - t_log)).sum(dim=-1).mean()


# ── Training ────────────────────────────────────────────────────────────

def precompute_teacher_cache(model, tokenizer, dataset, device):
    """Pre-tokenize teacher inputs and find resp_start.

    Cached: list of (teacher_ids, t_resp_start) per item.
    """
    cache = []
    for item in dataset:
        prompt, system, response = item["prompt"], item["system"], item["response"]
        full_text = render_messages(
            tokenizer, teacher_messages(system, prompt), assistant_content=response
        )
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0].to(device)
        prompt_text = render_messages(
            tokenizer, teacher_messages(system, prompt), add_generation_prompt=True
        )
        t_resp_start = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])
        cache.append((full_ids, t_resp_start))
    return cache


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


def train_csp(model, tokenizer, dataset, sp, frame_pool,
              steps, lr, weight_decay, prompts_per_step, seed):
    """Run KL distillation. Subsample prompts_per_step per step.

    Each prompt gets a random frame from frame_pool per step.
    """
    device = model.device
    embed_fn = model.get_input_embeddings()
    opt = torch.optim.AdamW(sp.parameters(), lr=lr, weight_decay=weight_decay)

    print("  Pre-tokenizing teacher cache...")
    teacher_cache = precompute_teacher_cache(model, tokenizer, dataset, device)
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
            kl.backward()
            step_loss += kl.item()
            n_seen += 1

        opt.step()
        opt.zero_grad()

        avg = step_loss / max(n_seen, 1)
        losses.append(avg)
        if step % 25 == 0 or step == steps - 1:
            print(f"    Step {step:4d}/{steps}: KL = {avg:.4f}")

    return losses


def compute_baseline_kl(model, tokenizer, dataset, max_items=None):
    """KL(no instruction || persona teacher) — the gap CSP must close."""
    device = model.device
    items = dataset if max_items is None else dataset[:max_items]
    total = 0.0
    n = 0
    for item in items:
        prompt, system, response = item["prompt"], item["system"], item["response"]
        teacher_text = render_messages(
            tokenizer, teacher_messages(system, prompt), assistant_content=response
        )
        teacher_ids = tokenizer(teacher_text, return_tensors="pt").input_ids[0].to(device)
        t_prompt_text = render_messages(
            tokenizer, teacher_messages(system, prompt), add_generation_prompt=True
        )
        t_resp_start = len(tokenizer(t_prompt_text, return_tensors="pt").input_ids[0])

        student_text = render_messages(
            tokenizer, student_messages(prompt), assistant_content=response
        )
        student_ids = tokenizer(student_text, return_tensors="pt").input_ids[0].to(device)
        s_prompt_text = render_messages(
            tokenizer, student_messages(prompt), add_generation_prompt=True
        )
        s_resp_start = len(tokenizer(s_prompt_text, return_tensors="pt").input_ids[0])

        with torch.no_grad():
            t_logits = model(input_ids=teacher_ids.unsqueeze(0)).logits[0]
            s_logits = model(input_ids=student_ids.unsqueeze(0)).logits[0]
        t_resp = t_logits[t_resp_start - 1:-1]
        s_resp = s_logits[s_resp_start - 1:-1]
        min_len = min(len(t_resp), len(s_resp))
        if min_len > 0:
            total += compute_kl_loss(s_resp[:min_len], t_resp[:min_len]).item()
            n += 1
    return total / max(n, 1)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, choices=list(config.PERSONAS.keys()))
    parser.add_argument("--polarity", required=True, choices=["pos", "neg"])
    parser.add_argument("--L", type=int, default=config.L)
    parser.add_argument("--steps", type=int, default=config.STEPS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--prompts-per-step", type=int, default=config.PROMPTS_PER_STEP)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--questions", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    persona_prompts = config.PERSONAS[args.persona]
    frame_pool = config.POSITIVE_FRAMES if args.polarity == "pos" else config.NEGATIVE_FRAMES
    persona_dir = os.path.join(args.results_dir, args.persona)
    os.makedirs(persona_dir, exist_ok=True)

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

    print("\nTeacher response generation (or load cache)...")
    cache_path = os.path.join(persona_dir, "cached_responses.json")
    dataset = generate_teacher_responses(
        model, tokenizer, persona_prompts, questions,
        args.max_new_tokens, cache_path, seed=args.seed,
    )

    print("\nBaseline KL (no instruction vs persona teacher)...")
    baseline_kl = compute_baseline_kl(model, tokenizer, dataset, max_items=50)
    print(f"  baseline_kl ≈ {baseline_kl:.4f} (estimated on first 50 items)")

    print(f"\nTraining CSP (persona={args.persona}, polarity={args.polarity}, "
          f"frames={frame_pool})...")
    torch.manual_seed(args.seed)
    sp = SoftPrompt(args.L, hidden_size).to(device)

    losses = train_csp(
        model, tokenizer, dataset, sp, frame_pool,
        steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
        prompts_per_step=args.prompts_per_step, seed=args.seed,
    )

    final_kl = losses[-1]
    fe = 1.0 - final_kl / baseline_kl if baseline_kl > 0 else None
    print(f"\nFinal KL: {final_kl:.4f}, baseline KL: {baseline_kl:.4f}, "
          f"FE: {fe:.1%}" if fe is not None else f"\nFinal KL: {final_kl:.4f}")

    ckpt_path = os.path.join(persona_dir, f"sp_{args.polarity}.pt")
    torch.save({
        "embedding": sp.embedding.data.cpu(),
        "L": args.L,
        "hidden_size": hidden_size,
        "persona": args.persona,
        "polarity": args.polarity,
        "frame_pool": frame_pool,
        "final_kl": final_kl,
        "baseline_kl": baseline_kl,
        "fraction_explained": fe,
        "kl_curve": losses,
        "config": {
            "steps": args.steps, "lr": args.lr,
            "weight_decay": args.weight_decay,
            "prompts_per_step": args.prompts_per_step,
            "seed": args.seed,
        },
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
