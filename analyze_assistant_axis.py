"""Project the CSP-induced L17 activation shift onto the assistant axis,
following the methodology of the lu-christina/assistant-axis paper:

  - Activations captured DURING generation, post-MLP residual stream
    (we use layer 17, matching Gemma-3-4b's middle layer / SAE layer).
  - Mean across all generated response tokens (per the paper).
  - Greedy decoding, fixed max_new_tokens.
  - Composite vector = mean(L17 over CSP-conditioned response tokens)
                     - mean(L17 over vanilla response tokens) on same prompts.
  - Projection reported as both raw dot and cosine similarity onto axis[17]
    from Butanium/gemma-3-4b-it-assistant-axis.

Sign convention (per Butanium README):
  axis = mean(default_vectors) - mean(role_vectors)
  positive = toward default-assistant, negative = toward role-play.

Usage: python analyze_assistant_axis.py
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from soft_prompt import SoftPrompt
from train import render_messages, student_messages, load_questions
from evaluate import (
    EVAL_FRAME_POS, build_csp_input, capture_layer_activations,
    N_EVAL_PROMPTS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_step(ckpt_name, kl_curve_len):
    m = re.match(r"sp_pos_step(\d+)\.pt$", ckpt_name)
    if m:
        return int(m.group(1))
    return kl_curve_len


def _mean_response_act(model, tokenizer, layer_idx, init_kw, max_new_tokens, device,
                       avg_last_tokens=0, exclude_parens=False, paren_mode="all"):
    """Greedy-generate up to max_new_tokens with kv cache, capturing the L<idx>
    activation at each generated token. Returns the mean across response tokens.

    paren_mode (overrides exclude_parens if not "all"):
      "all"      — average over all generated tokens
      "outside"  — only tokens at paren-depth 0 (depth_before AND depth_after = 0)
      "inside"   — only tokens at paren-depth > 0 (either before or after,
                   so transition tokens '(' and ')' count as paren content)
    """
    if exclude_parens and paren_mode == "all":
        paren_mode = "outside"
    captured = []  # list of (hidden,) tensors, one per generated token
    gen_tokens = []  # parallel list of generated token IDs

    def hook(module, inp, output):
        if isinstance(output, tuple):
            output = output[0]
        # Always take the LAST sequence position — that's the position
        # corresponding to the next-token prediction at this step.
        captured.append(output[0, -1, :].detach().clone())

    handle = model.model.language_model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(**init_kw, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
            gen_tokens.append(int(next_tok.item()))
            for _ in range(max_new_tokens - 1):
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                out = model(input_ids=next_tok.unsqueeze(0),
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
                gen_tokens.append(int(next_tok.item()))
    finally:
        handle.remove()

    if not captured:
        return None

    # captured[i] produced gen_tokens[i].
    # Truncate to common length in case of off-by-one.
    n = min(len(captured), len(gen_tokens))
    captured = captured[:n]
    gen_tokens = gen_tokens[:n]

    if paren_mode in ("outside", "inside"):
        keep = []
        depth = 0
        for i, tok_id in enumerate(gen_tokens):
            depth_before = depth
            s = tokenizer.decode([tok_id])
            for ch in s:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
            depth_after = depth
            outside = (depth_before == 0 and depth_after == 0)
            if paren_mode == "outside" and outside:
                keep.append(captured[i])
            elif paren_mode == "inside" and not outside:
                keep.append(captured[i])
        if not keep:
            return None
        return torch.stack(keep).float().mean(dim=0)

    if avg_last_tokens > 0 and len(captured) > avg_last_tokens:
        captured = captured[-avg_last_tokens:]
    return torch.stack(captured).float().mean(dim=0)


def response_acts_vanilla(model, tokenizer, prompt, layer_idx, device, max_new_tokens,
                          avg_last_tokens=0, exclude_parens=False, paren_mode="all"):
    text = render_messages(
        tokenizer, student_messages(prompt), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"input_ids": ids.unsqueeze(0)}, max_new_tokens, device,
        avg_last_tokens=avg_last_tokens,
        exclude_parens=exclude_parens,
        paren_mode=paren_mode,
    )


def response_acts_csp(model, tokenizer, sp, prompt, layer_idx, eval_frame, device, max_new_tokens,
                     avg_last_tokens=0, exclude_parens=False, paren_mode="all"):
    embed_fn = model.get_input_embeddings()
    suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
    user = f"{prompt} {suffix}"
    combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"inputs_embeds": combined}, max_new_tokens, device,
        avg_last_tokens=avg_last_tokens,
        exclude_parens=exclude_parens,
        paren_mode=paren_mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--csp-dir", default="early_stop",
                        help="Subdir under results/ containing seed_*/sp_pos*.pt files. "
                             "Defaults to early_stop; pass trough_trace etc. for other batches.")
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "results", "assistant_axis.png"))
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--layer", type=int, default=config.SAE_LAYER)  # 17
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generate this many response tokens before averaging L17 acts")
    parser.add_argument("--avg-last-tokens", type=int, default=0,
                        help="If > 0, average only the LAST N response tokens "
                             "(skips early-response preamble)")
    parser.add_argument("--exclude-parens", action="store_true",
                        help="Equivalent to --paren-mode outside (kept for back-compat).")
    parser.add_argument("--paren-mode", choices=["all", "outside", "inside"], default="all",
                        help="all: average all tokens. "
                             "outside: only depth-0 tokens (character speech). "
                             "inside: only depth>0 tokens (parenthetical stage directions).")
    args = parser.parse_args()
    if args.exclude_parens and args.paren_mode == "all":
        args.paren_mode = "outside"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Loading assistant axis (Butanium/gemma-3-4b-it-assistant-axis)...")
    axis_path = hf_hub_download(
        repo_id="Butanium/gemma-3-4b-it-assistant-axis",
        filename="assistant_axis.pt", repo_type="dataset",
    )
    full_axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    axis = full_axis[args.layer].float().to(device)  # (hidden_dim,)
    axis_norm = axis.norm().item()
    print(f"  axis[{args.layer}]: shape={tuple(axis.shape)}, ‖·‖={axis_norm:.3f}")

    questions = load_questions()
    eval_prompts = questions[:args.n_eval_prompts]

    # Compute the vanilla baseline once: per-prompt mean L17 across response
    # tokens (greedy-generated under no system / no CSP), then averaged across
    # prompts.
    if args.paren_mode == "outside":
        avg_label = f"depth-0 (outside parens) of {args.max_new_tokens}"
    elif args.paren_mode == "inside":
        avg_label = f"depth>0 (inside parens) of {args.max_new_tokens}"
    elif args.avg_last_tokens:
        avg_label = f"last {args.avg_last_tokens}/{args.max_new_tokens}"
    else:
        avg_label = f"all {args.max_new_tokens}"
    print(f"\nComputing vanilla L{args.layer} response activations across "
          f"{len(eval_prompts)} prompts (mean over {avg_label} tokens)...")
    vanilla_acts = []
    for i, p in enumerate(eval_prompts):
        a = response_acts_vanilla(model, tokenizer, p, args.layer, device,
                                   args.max_new_tokens,
                                   avg_last_tokens=args.avg_last_tokens,
                                   exclude_parens=args.exclude_parens,
                                   paren_mode=args.paren_mode)
        if a is not None:
            vanilla_acts.append(a)
        if (i + 1) % 5 == 0:
            print(f"  vanilla [{i+1}/{len(eval_prompts)}]")
    vanilla_acts = torch.stack(vanilla_acts).float()  # (n_prompts, hidden_dim)
    mean_vanilla = vanilla_acts.mean(dim=0)
    print(f"  mean_vanilla: ‖·‖={mean_vanilla.norm().item():.3f}")

    # Find every CSP checkpoint in the chosen subdir (default: early_stop/)
    ckpt_paths = sorted(glob.glob(
        os.path.join(args.results_dir, args.csp_dir, "seed_*", "sp_pos*.pt"),
    ))
    print(f"\nFound {len(ckpt_paths)} checkpoints in {args.csp_dir}/")

    rows = []
    for path in ckpt_paths:
        rel = os.path.relpath(path, args.results_dir)
        group = os.path.dirname(rel)
        ckpt_name = os.path.basename(path)

        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"  skip {rel}: {e}")
            continue
        sp = SoftPrompt(ckpt["L"], ckpt["hidden_size"]).to(device)
        sp.embedding.data = ckpt["embedding"].to(device)
        kl = float(ckpt.get("final_kl") or 0.0)
        step = parse_step(ckpt_name, len(ckpt.get("kl_curve") or []))

        # Compute CSP-conditioned response activations
        csp_acts = []
        for p in eval_prompts:
            a = response_acts_csp(model, tokenizer, sp, p, args.layer,
                                   EVAL_FRAME_POS, device, args.max_new_tokens,
                                   avg_last_tokens=args.avg_last_tokens,
                                   exclude_parens=args.exclude_parens,
                                   paren_mode=args.paren_mode)
            if a is not None:
                csp_acts.append(a)
        csp_acts = torch.stack(csp_acts).float()
        mean_csp = csp_acts.mean(dim=0)

        shift = mean_csp - mean_vanilla
        shift_norm = shift.norm().item()
        proj_dot = (shift @ axis).item()
        proj_cos = (proj_dot / (shift_norm * axis_norm)) if shift_norm > 0 else 0.0

        rows.append({
            "group": group, "ckpt": ckpt_name, "step": step, "kl": kl,
            "shift_norm": shift_norm, "proj_dot": proj_dot, "proj_cos": proj_cos,
        })
        print(f"  {rel:50s}  step={step:4d}  KL={kl:7.3f}  "
              f"‖shift‖={shift_norm:6.2f}  proj·axis={proj_dot:+8.2f}  "
              f"cos={proj_cos:+.4f}")

    # Plot two panels: KL vs proj_dot, KL vs proj_cos
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: r["step"])

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (g, rs) in enumerate(sorted(by_group.items())):
        kls = [r["kl"] for r in rs]
        dots = [r["proj_dot"] for r in rs]
        coss = [r["proj_cos"] for r in rs]
        color = cmap(i % 20)
        axes[0].plot(kls, dots, "o-", label=g.replace("early_stop/", "es/"),
                     color=color, alpha=0.7, markersize=5)
        axes[1].plot(kls, coss, "o-", color=color, alpha=0.7, markersize=5)

    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("KL ↑ (log)")
    axes[1].set_xlabel("KL ↑ (log)")
    axes[0].set_ylabel("(L17 shift) · (assistant axis)")
    axes[1].set_ylabel("cos(L17 shift, assistant axis)")
    axes[0].set_title("Magnitude along assistant axis")
    axes[1].set_title("Direction alignment with assistant axis")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(
        f"L{args.layer} shift projected onto Butanium assistant axis. "
        f"Negative = role-play, positive = default-assistant. "
        f"(mean over {avg_label} response tokens, "
        f"{len(eval_prompts)} prompts/ckpt)",
        fontsize=10,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"\nSaved plot: {args.out}")

    json_out = args.out.replace(".png", ".json")
    with open(json_out, "w") as f:
        json.dump({"layer": args.layer, "rows": rows}, f, indent=2)
    print(f"Saved data: {json_out}")


if __name__ == "__main__":
    main()
