"""Chat-template + CSP-splicing helpers.

Two builders, used by both training and generation:

  - `splice_csp_into_frame` — wraps user content in a frame containing one
    `§` placeholder, tokenizes under the chat template, and replaces the
    placeholder with L soft-prompt embeddings. The student sees:
        user content = "{prompt} {frame_with_§}"
    Output sequence is (L-1) tokens longer than the original.

  - `splice_csp_multi` — same logic but replaces every `§` occurrence with
    the same L embeddings. Used by self-verb prompts that ask the model
    about multiple frames at once.

The frame is always one of `config.POSITIVE_FRAMES` (Be / Act / Please /
You should §). At training time we sample uniformly from the pool each
step; at generation time the caller pins one frame.
"""
from __future__ import annotations

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import SP_PLACEHOLDER
from .soft_prompt import SoftPrompt


# ── Chat template helpers ───────────────────────────────────────────────

Message = dict[str, str]


def student_messages(user_content: str) -> list[Message]:
    """A single-turn user message with no system prompt — the vanilla teacher
    is also queried with no system prompt, so this matches teacher inputs."""
    return [{"role": "user", "content": user_content}]


def render_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[Message],
    *,
    add_generation_prompt: bool = False,
    assistant_content: str | None = None,
) -> str:
    """Apply the chat template; optionally append an assistant turn or
    generation-prompt tail."""
    msgs = list(messages)
    if assistant_content is not None:
        msgs = msgs + [{"role": "assistant", "content": assistant_content}]
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


# ── Placeholder locating ────────────────────────────────────────────────

def find_placeholder_position(
    tokenizer: PreTrainedTokenizerBase, ids: torch.Tensor,
) -> int:
    """Find the index of the `§` token inside a tokenized chat sequence."""
    for i, tid in enumerate(ids.tolist()):
        if SP_PLACEHOLDER in tokenizer.decode([tid]):
            return i
    raise ValueError(
        f"Placeholder '{SP_PLACEHOLDER}' not found in: {tokenizer.decode(ids)}"
    )


# ── Splicing ────────────────────────────────────────────────────────────

def _embed_user(
    tokenizer: PreTrainedTokenizerBase,
    embed_fn: torch.nn.Module,
    sp: SoftPrompt,
    user_text_with_placeholder: str,
    device: torch.device,
    *,
    assistant_content: str | None = None,
    add_generation_prompt: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize + embed `user_text_with_placeholder`, splice the SoftPrompt
    into the `§` slot, return (combined_embeds, original_ids, sp_pos).

    If `assistant_content` is given, the assistant turn is appended (used
    by training, where we need full teacher+response sequence). Otherwise
    `add_generation_prompt` controls whether the chat template adds the
    generation-prompt tail (used at inference).
    """
    text = render_messages(
        tokenizer, student_messages(user_text_with_placeholder),
        add_generation_prompt=(add_generation_prompt and assistant_content is None),
        assistant_content=assistant_content,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    sp_pos = find_placeholder_position(tokenizer, ids)
    embeds = embed_fn(ids.unsqueeze(0))
    sp_embeds = sp(batch_size=1).to(embeds.dtype)
    combined = torch.cat(
        [embeds[:, :sp_pos, :], sp_embeds, embeds[:, sp_pos + 1:, :]], dim=1,
    )
    return combined, ids, sp_pos


def splice_csp_into_frame(
    tokenizer: PreTrainedTokenizerBase,
    embed_fn: torch.nn.Module,
    sp: SoftPrompt,
    user_text_with_placeholder: str,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Inference-time splice. Returns (combined_embeds, sp_pos, L)."""
    combined, _, sp_pos = _embed_user(
        tokenizer, embed_fn, sp, user_text_with_placeholder, device,
    )
    return combined, sp_pos, sp.embedding.shape[0]


def splice_csp_multi(
    tokenizer: PreTrainedTokenizerBase,
    embed_fn: torch.nn.Module,
    sp: SoftPrompt,
    user_text_with_placeholders: str,
    device: torch.device,
) -> torch.Tensor:
    """Multi-placeholder splice — used by self-verb prompts that mention
    several frames at once. The same SP embeddings replace every `§`."""
    text = render_messages(
        tokenizer, student_messages(user_text_with_placeholders),
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    positions = [
        i for i, tid in enumerate(ids.tolist())
        if SP_PLACEHOLDER in tokenizer.decode([tid])
    ]
    if not positions:
        raise ValueError(f"No placeholders found in: {text[:120]}...")
    embeds = embed_fn(ids.unsqueeze(0))
    sp_embeds = sp(batch_size=1).to(embeds.dtype)
    result = embeds
    for pos in reversed(positions):
        result = torch.cat(
            [result[:, :pos, :], sp_embeds, result[:, pos + 1:, :]], dim=1,
        )
    return result


def build_student_for_training(
    tokenizer: PreTrainedTokenizerBase,
    embed_fn: torch.nn.Module,
    sp: SoftPrompt,
    prompt: str,
    frame: str,
    response: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build the student input for one KL-ascent step.

    Returns (student_full_embeds, s_resp_start) where s_resp_start is the
    token index in the output sequence where the assistant response begins
    (used to slice logits for the KL loss).
    """
    L = sp.embedding.shape[0]
    suffix = frame.format(sp=SP_PLACEHOLDER)
    user_content = f"{prompt} {suffix}"
    student_full, _, _ = _embed_user(
        tokenizer, embed_fn, sp, user_content, device,
        assistant_content=response,
    )

    # Where does the response start in the spliced sequence?  Compute the
    # length of the prompt-only chat template (no assistant turn) and add
    # the (L-1) extra tokens introduced by the splice.
    prompt_text = render_messages(
        tokenizer, student_messages(user_content), add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    s_resp_start = len(prompt_ids) + (L - 1)
    return student_full, s_resp_start
