"""KL-ascent training loop.

The student model sees a CSP spliced into a sampled frame. The teacher
is the vanilla model (no system prompt) on the same user prompt. We
maximize `KL(student || teacher)` via `(-kl).backward()`.

Teacher responses are greedy-generated once per pod and cached to
`cached_responses.json` so repeat seeds on the same pod reuse them.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import POSITIVE_FRAMES, TrainConfig
from .frames import (
    build_student_for_training,
    render_messages,
    student_messages,
)
from .soft_prompt import SoftPrompt


CkptSaveFn = Callable[[int, list[float]], None]


# ── KL loss ─────────────────────────────────────────────────────────────

def kl_divergence(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Token-wise KL(student || teacher), averaged over the response tokens."""
    s_log = F.log_softmax(student_logits, dim=-1)
    t_log = F.log_softmax(teacher_logits, dim=-1)
    s_probs = s_log.exp()
    return (s_probs * (s_log - t_log)).sum(dim=-1).mean()


# ── Vanilla teacher responses (cached) ──────────────────────────────────

def generate_vanilla_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    max_new_tokens: int,
    cache_path: str | Path,
) -> list[dict]:
    """Greedy-generate one response per question with no system prompt.

    Cached as a list of `{"prompt", "response"}` dicts. Cache is reused
    across all seeds on the same pod (and across pods, since `/workspace`
    is shared).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        dataset = json.loads(cache_path.read_text())
        print(f"  Loaded {len(dataset)} cached vanilla responses from {cache_path}")
        return dataset

    dataset: list[dict] = []
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(dataset, indent=2))
    print(f"  Cached {len(dataset)} responses to {cache_path}")
    return dataset


def _precompute_teacher_cache(
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[dict],
    device: torch.device,
) -> list[tuple[torch.Tensor, int]]:
    """Pre-tokenize teacher inputs (vanilla user → assistant response) and
    note where the response starts. Done once at the top of training so
    every step only does forward passes."""
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


# ── Training loop ───────────────────────────────────────────────────────

def save_checkpoint(
    sp: SoftPrompt,
    hidden_size: int,
    config: TrainConfig,
    losses: list[float],
    path: str | Path,
) -> None:
    """Save a CSP checkpoint with metadata. Schema matches the historical
    one so existing analysis scripts still load these."""
    torch.save({
        "embedding": sp.embedding.data.cpu(),
        "L": config.L,
        "hidden_size": hidden_size,
        "persona": "divergent",
        "polarity": "pos",
        "frame_pool": POSITIVE_FRAMES,
        "final_kl": losses[-1] if losses else None,
        "kl_curve": list(losses),
        "config": {
            "steps": config.steps, "lr": config.lr,
            "weight_decay": config.weight_decay,
            "prompts_per_step": config.prompts_per_step,
            "seed": config.seed,
            "frame_pool_name": "persona",
            "placement": "splice",
        },
    }, path)


def compute_eval_kl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sp: SoftPrompt,
    dataset: list[dict],
    frame: str,
    device: torch.device,
    n_prompts: int = 10,
) -> float:
    """Eval-time KL between CSP-conditioned student and vanilla teacher.

    Used to give step-0 ckpts a KL value (their `final_kl` is None because
    no training step has run). Same formulation as the training loop but
    no backward pass and only the first `n_prompts` cached items.
    """
    if not dataset:
        return 0.0
    embed_fn = model.get_input_embeddings()
    sample = dataset[:n_prompts]
    teacher_cache = _precompute_teacher_cache(tokenizer, sample, device)
    total = 0.0
    n_seen = 0
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(sample):
            teacher_ids, t_resp_start = teacher_cache[i]
            student_embeds, s_resp_start = build_student_for_training(
                tokenizer, embed_fn, sp, item["prompt"], frame,
                item["response"], device,
            )
            t_logits = model(input_ids=teacher_ids.unsqueeze(0)).logits[0]
            s_logits = model(inputs_embeds=student_embeds).logits[0]
            t_resp = t_logits[t_resp_start - 1:-1]
            s_resp = s_logits[s_resp_start - 1:-1]
            min_len = min(len(t_resp), len(s_resp))
            if min_len == 0:
                continue
            kl = kl_divergence(s_resp[:min_len], t_resp[:min_len])
            total += kl.item()
            n_seen += 1
    return total / max(n_seen, 1)


def train_csp(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[dict],
    sp: SoftPrompt,
    config: TrainConfig,
    ckpt_save_fn: CkptSaveFn | None = None,
) -> list[float]:
    """KL-ascent loop. Returns the per-step average KL trace."""
    device = model.device
    embed_fn = model.get_input_embeddings()
    opt = torch.optim.AdamW(
        sp.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    print("  Pre-tokenizing vanilla teacher cache...")
    teacher_cache = _precompute_teacher_cache(tokenizer, dataset, device)
    n = len(dataset)
    rng = random.Random(config.seed)

    model.eval()
    losses: list[float] = []

    for step in range(config.steps):
        indices = rng.sample(range(n), min(config.prompts_per_step, n))
        step_loss = 0.0
        n_seen = 0

        for idx in indices:
            item = dataset[idx]
            teacher_ids, t_resp_start = teacher_cache[idx]
            frame = rng.choice(POSITIVE_FRAMES)

            with torch.no_grad():
                t_logits = model(input_ids=teacher_ids.unsqueeze(0)).logits[0]

            student_embeds, s_resp_start = build_student_for_training(
                tokenizer, embed_fn, sp, item["prompt"], frame, item["response"], device,
            )
            s_logits = model(inputs_embeds=student_embeds).logits[0]

            t_resp = t_logits[t_resp_start - 1:-1]
            s_resp = s_logits[s_resp_start - 1:-1]
            min_len = min(len(t_resp), len(s_resp))
            if min_len == 0:
                continue
            kl = kl_divergence(s_resp[:min_len], t_resp[:min_len])
            (-kl).backward()
            step_loss += kl.item()
            n_seen += 1

        opt.step()
        opt.zero_grad()

        avg = step_loss / max(n_seen, 1)
        losses.append(avg)
        if step % 25 == 0 or step == config.steps - 1:
            print(f"    Step {step:4d}/{config.steps}: KL ↑ = {avg:.4f}")

        completed = step + 1
        if (
            ckpt_save_fn
            and config.checkpoint_every
            and completed % config.checkpoint_every == 0
            and completed < config.steps
        ):
            ckpt_save_fn(completed, losses)

    return losses
