"""Residual-stream activation capture for axis projection.

Two flavors of capture, both following the Butanium assistant-axis
methodology:

  - Activations are captured DURING generation, on the residual stream
    after the transformer block at `AxisConfig.layer` (Llama L16 default).
  - We mean across all generated response tokens, then form the shift
    vector: mean(act over CSP-conditioned response) - mean(act over
    vanilla response) on the same prompts.

These captures live here (separate from training) so `2_generate.py` can
emit per-(seed, frame, ckpt) shift vectors in the same forward-pass sweep
that produces behavior + self-verb completions.
"""
from __future__ import annotations

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import SP_PLACEHOLDER
from .frames import (
    render_messages,
    splice_csp_into_frame,
    student_messages,
)
from .model import get_transformer_layers
from .soft_prompt import SoftPrompt


def _mean_response_act(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    init_kw: dict,
    max_new_tokens: int,
) -> torch.Tensor | None:
    """Greedy-generate from `init_kw`, hook layer `layer_idx`, return the
    mean of captured activations across response tokens. None if nothing
    was generated."""
    captured: list[torch.Tensor] = []
    gen_tokens: list[int] = []

    def hook(module, inp, output):
        if isinstance(output, tuple):
            output = output[0]
        captured.append(output[0, -1, :].detach().clone())

    handle = get_transformer_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(**init_kw, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
            gen_tokens.append(int(next_tok.item()))
            for _ in range(max_new_tokens - 1):
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                out = model(
                    input_ids=next_tok.unsqueeze(0),
                    past_key_values=past, use_cache=True,
                )
                past = out.past_key_values
                next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
                gen_tokens.append(int(next_tok.item()))
    finally:
        handle.remove()

    n = min(len(captured), len(gen_tokens))
    if n == 0:
        return None
    return torch.stack(captured[:n]).float().mean(dim=0)


def vanilla_response_act(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer_idx: int,
    device: torch.device,
    max_new_tokens: int,
) -> torch.Tensor | None:
    text = render_messages(
        tokenizer, student_messages(prompt), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"input_ids": ids.unsqueeze(0)}, max_new_tokens,
    )


def csp_response_act(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sp: SoftPrompt,
    prompt: str,
    layer_idx: int,
    eval_frame: str,
    device: torch.device,
    max_new_tokens: int,
) -> torch.Tensor | None:
    """Capture mean L{layer_idx} activation while CSP-spliced model
    generates a response to `prompt` under `eval_frame`."""
    embed_fn = model.get_input_embeddings()
    suffix = eval_frame.format(sp=SP_PLACEHOLDER)
    user = f"{prompt} {suffix}"
    combined, _, _ = splice_csp_into_frame(tokenizer, embed_fn, sp, user, device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"inputs_embeds": combined}, max_new_tokens,
    )


def mean_vanilla_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    layer_idx: int,
    device: torch.device,
    max_new_tokens: int,
) -> torch.Tensor:
    """Per-prompt vanilla response acts, then average — the comparator
    that the CSP shift subtracts off. Frame-agnostic (vanilla teacher has
    no frame), so this is cached per-frame at `vanilla_baseline_{frame}.pt`
    only for symmetry; it's actually identical across eval frames."""
    acts: list[torch.Tensor] = []
    for i, p in enumerate(prompts):
        a = vanilla_response_act(
            model, tokenizer, p, layer_idx, device, max_new_tokens,
        )
        if a is not None:
            acts.append(a)
        if (i + 1) % 5 == 0:
            print(f"  vanilla [{i+1}/{len(prompts)}]")
    if not acts:
        raise RuntimeError("vanilla baseline produced no acts")
    return torch.stack(acts).float().mean(dim=0)


def capture_shift(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sp: SoftPrompt,
    prompts: list[str],
    layer_idx: int,
    eval_frame: str,
    mean_vanilla: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the residual-stream shift induced by one CSP under
    `eval_frame`. Returns `(shift, mean_csp)` — both cpu float tensors of
    shape `(hidden_dim,)`.

      shift = mean(act over CSP responses) - mean_vanilla
    """
    acts: list[torch.Tensor] = []
    for p in prompts:
        a = csp_response_act(
            model, tokenizer, sp, p, layer_idx, eval_frame, device, max_new_tokens,
        )
        if a is not None:
            acts.append(a)
    if not acts:
        raise RuntimeError(f"CSP shift produced no acts (frame={eval_frame!r})")
    mean_csp = torch.stack(acts).float().mean(dim=0)
    shift = mean_csp - mean_vanilla.to(mean_csp.device)
    return shift.detach().cpu(), mean_csp.detach().cpu()
