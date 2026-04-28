"""Project the CSP-induced residual-stream shift onto the assistant axis,
following the methodology of Butanium/lu-christina assistant-axis:

  - Activations captured DURING generation, post-MLP residual stream
    at the model preset's AXIS_LAYER.
  - Mean across all generated response tokens (per the paper).
  - Greedy decoding, fixed max_new_tokens.
  - Composite vector = mean(act over CSP-conditioned response tokens)
                     - mean(act over vanilla response tokens) on same prompts.
  - Projection reported as raw dot and cosine similarity onto the
    Butanium axis at AXIS_LAYER for the active model preset.

Sign convention (per Butanium README):
  axis = mean(default_vectors) - mean(role_vectors)
  positive = toward default-assistant, negative = toward role-play.

Usage:
    CSP_MODEL_PRESET=llama-3.1-8b-instruct \\
    python analyze_assistant_axis.py --csp-dir trough_llama \\
        --out results/trough_llama_axis.png
"""

import argparse
import glob
import json
import os
import re

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from soft_prompt import SoftPrompt
from train import render_messages, student_messages, load_questions
from evaluate import (
    EVAL_FRAME_POS, build_csp_input,
    get_transformer_layers, N_EVAL_PROMPTS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_step(ckpt_name, kl_curve_len):
    m = re.match(r"sp_pos_step(\d+)\.pt$", ckpt_name)
    if m:
        return int(m.group(1))
    return kl_curve_len


def _mean_response_act(model, tokenizer, layer_idx, init_kw, max_new_tokens):
    """Greedy-generate, hook layer_idx, return mean of captured acts across
    response tokens. Returns None if generation produced nothing."""
    captured = []
    gen_tokens = []

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
                out = model(input_ids=next_tok.unsqueeze(0),
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
                gen_tokens.append(int(next_tok.item()))
    finally:
        handle.remove()

    n = min(len(captured), len(gen_tokens))
    if n == 0:
        return None
    return torch.stack(captured[:n]).float().mean(dim=0)


def response_acts_vanilla(model, tokenizer, prompt, layer_idx, device, max_new_tokens):
    text = render_messages(
        tokenizer, student_messages(prompt), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"input_ids": ids.unsqueeze(0)}, max_new_tokens,
    )


def response_acts_csp(model, tokenizer, sp, prompt, layer_idx, eval_frame, device, max_new_tokens):
    embed_fn = model.get_input_embeddings()
    suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
    user = f"{prompt} {suffix}"
    combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"inputs_embeds": combined}, max_new_tokens,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--csp-dir", required=True,
                        help="Subdir under results/ containing seed_*/sp_pos*.pt files "
                             "(e.g. trough_llama, trough_qwen).")
    parser.add_argument("--out", required=True,
                        help="Output PNG path; sibling .json is also written.")
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--layer", type=int, default=config.AXIS_LAYER)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generate this many response tokens before averaging acts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loading assistant axis from {config.AXIS_REPO}...")
    axis_path = hf_hub_download(
        repo_id=config.AXIS_REPO,
        filename="assistant_axis.pt", repo_type="dataset",
    )
    full_axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    axis = full_axis[args.layer].float().to(device)  # (hidden_dim,)
    axis_norm = axis.norm().item()
    print(f"  axis[{args.layer}]: shape={tuple(axis.shape)}, ‖·‖={axis_norm:.3f}")

    questions = load_questions()
    eval_prompts = questions[:args.n_eval_prompts]

    print(f"\nComputing vanilla L{args.layer} response activations across "
          f"{len(eval_prompts)} prompts (mean over {args.max_new_tokens} tokens)...")
    vanilla_acts = []
    for i, p in enumerate(eval_prompts):
        a = response_acts_vanilla(
            model, tokenizer, p, args.layer, device, args.max_new_tokens,
        )
        if a is not None:
            vanilla_acts.append(a)
        if (i + 1) % 5 == 0:
            print(f"  vanilla [{i+1}/{len(eval_prompts)}]")
    vanilla_acts = torch.stack(vanilla_acts).float()  # (n_prompts, hidden_dim)
    mean_vanilla = vanilla_acts.mean(dim=0)
    print(f"  mean_vanilla: ‖·‖={mean_vanilla.norm().item():.3f}")

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

        csp_acts = []
        for p in eval_prompts:
            a = response_acts_csp(
                model, tokenizer, sp, p, args.layer,
                EVAL_FRAME_POS, device, args.max_new_tokens,
            )
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
        label = g.split("/")[-1]
        axes[0].plot(kls, dots, "o-", label=label, color=color, alpha=0.7, markersize=5)
        axes[1].plot(kls, coss, "o-", color=color, alpha=0.7, markersize=5)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("KL ↑ (log)")
    axes[1].set_xlabel("KL ↑ (log)")
    axes[0].set_ylabel(f"(L{args.layer} shift) · (assistant axis)")
    axes[1].set_ylabel(f"cos(L{args.layer} shift, assistant axis)")
    axes[0].set_title("Magnitude along assistant axis")
    axes[1].set_title("Direction alignment with assistant axis")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(
        f"L{args.layer} shift projected onto Butanium assistant axis. "
        f"Negative = role-play, positive = default-assistant. "
        f"(mean over {args.max_new_tokens} response tokens, "
        f"{len(eval_prompts)} prompts/ckpt)",
        fontsize=10,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"\nSaved plot: {args.out}")

    json_out = args.out.replace(".png", ".json")
    with open(json_out, "w") as f:
        json.dump({"layer": args.layer, "rows": rows}, f, indent=2)
    print(f"Saved data: {json_out}")


if __name__ == "__main__":
    main()
