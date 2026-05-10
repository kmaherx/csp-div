"""CSP generation: behavior + self-verb completions for one (CSP, frame).

`2_generate.py` calls these to produce the per-(seed, frame, ckpt) JSON
files consumed by the judge and dashboard.

Output schema preserves the historical `divergent-in-pos` wrapper so
existing checkpoints' eval JSONs stay readable by both old and new code:

    behavior_step{K}.json:
      {"divergent-in-pos": [
        {"prompt": "...", "response_vanilla": "...", "response_csp": "..."},
        ...
      ]}

    self_verb_step{K}.json:
      {"divergent-in-pos": [
        {"approach": "multi_frame"|"single_frame", "prompt": "...", "response": "..."},
        ...
      ]}
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import POSITIVE_FRAMES, SP_PLACEHOLDER, GenerateConfig
from .frames import (
    render_messages,
    splice_csp_into_frame,
    splice_csp_multi,
    student_messages,
)
from .soft_prompt import SoftPrompt


# ── Self-verb prompts ───────────────────────────────────────────────────

MULTI_FRAME_TEMPLATES = [
    "Find the theme shared by these instructions: {frames} The shared theme is:",
    "These instructions all mean the same thing: {frames} In one word or phrase, they are asking me to:",
    "Rewrite these instructions as a single short command: {frames} Command:",
    "These instructions all say the same thing: {frames} Write one clear instruction that captures their meaning:",
    "Summarize these instructions into one directive: {frames} Directive:",
]


def multi_frame_prompts() -> list[str]:
    joined = " ".join(f.format(sp=SP_PLACEHOLDER) for f in POSITIVE_FRAMES)
    return [t.format(frames=joined) for t in MULTI_FRAME_TEMPLATES]


def single_frame_prompts() -> list[str]:
    """One 'In plain English, explain this command' prompt per frame."""
    return [
        f"In plain English, explain this command: {f.format(sp=SP_PLACEHOLDER)}"
        for f in POSITIVE_FRAMES
    ]


def verbalization_prompts() -> list[tuple[str, str]]:
    """All self-verb prompts as `(approach, prompt)` pairs.

    5 multi_frame + |POSITIVE_FRAMES| single_frame = 9 prompts.
    """
    return (
        [("multi_frame", p) for p in multi_frame_prompts()]
        + [("single_frame", p) for p in single_frame_prompts()]
    )


# ── Greedy generation (with kv cache) ───────────────────────────────────

def generate_greedy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    inputs_embeds: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    max_new_tokens: int,
) -> str:
    """Greedy decode up to `max_new_tokens`, return decoded string."""
    out_ids: list[int] = []
    past = None
    next_tok: torch.Tensor | None = None
    for i in range(max_new_tokens):
        if i == 0:
            kw = (
                {"inputs_embeds": inputs_embeds}
                if inputs_embeds is not None
                else {"input_ids": input_ids}
            )
            out = model(**kw, use_cache=True)
        else:
            out = model(
                input_ids=next_tok.unsqueeze(0),
                past_key_values=past, use_cache=True,
            )
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
        out_ids.append(int(next_tok.item()))
        if int(next_tok.item()) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


# ── Run modes (one CSP, one frame) ──────────────────────────────────────

def run_self_verb(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sp: SoftPrompt,
    device: torch.device,
    cfg: GenerateConfig,
    out_path: str | Path,
) -> dict:
    """Self-verb completions across all 9 verbalization prompts."""
    embed_fn = model.get_input_embeddings()
    results: list[dict] = []
    for approach, vp in verbalization_prompts():
        combined = splice_csp_multi(tokenizer, embed_fn, sp, vp, device)
        with torch.no_grad():
            resp = generate_greedy(
                model, tokenizer, inputs_embeds=combined,
                max_new_tokens=cfg.max_new_tokens_verb,
            )
        print(f"    [{approach}] Q: {vp[:80]}")
        print(f"              A: {resp[:120]}")
        results.append({"approach": approach, "prompt": vp, "response": resp})
    out = {"divergent-in-pos": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"  Saved self-verb → {out_path}")
    return out


def run_behavior(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sp: SoftPrompt,
    prompts: list[str],
    eval_frame: str,
    device: torch.device,
    cfg: GenerateConfig,
    out_path: str | Path,
) -> dict:
    """Behavior completions: CSP-conditioned + vanilla, side-by-side on
    the first `cfg.n_behavior_samples` eval prompts."""
    embed_fn = model.get_input_embeddings()
    eval_suffix = eval_frame.format(sp=SP_PLACEHOLDER)
    sample = prompts[: cfg.n_behavior_samples]
    results: list[dict] = []
    for prompt in sample:
        user_csp = f"{prompt} {eval_suffix}"
        combined, _, _ = splice_csp_into_frame(
            tokenizer, embed_fn, sp, user_csp, device,
        )
        with torch.no_grad():
            resp_csp = generate_greedy(
                model, tokenizer, inputs_embeds=combined,
                max_new_tokens=cfg.max_new_tokens_behavior,
            )

        text_vanilla = render_messages(
            tokenizer, student_messages(prompt), add_generation_prompt=True,
        )
        ids_vanilla = tokenizer(text_vanilla, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            resp_vanilla = generate_greedy(
                model, tokenizer, input_ids=ids_vanilla,
                max_new_tokens=cfg.max_new_tokens_behavior,
            )
        print(f"    Q: {prompt[:80]}")
        print(f"    [vanilla] {resp_vanilla[:160]}")
        print(f"    [csp]     {resp_csp[:160]}")
        results.append({
            "prompt": prompt,
            "response_vanilla": resp_vanilla,
            "response_csp": resp_csp,
        })
    out = {"divergent-in-pos": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"  Saved behavior → {out_path}")
    return out
