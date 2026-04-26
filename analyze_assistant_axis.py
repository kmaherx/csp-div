"""Project the CSP-induced L17 activation shift onto the assistant axis.

For each (run, ckpt):
  1. Compute L17 activation at the last input token across N eval prompts,
     once with vanilla input (no system, no CSP) and once with the CSP
     spliced into "Be §.".
  2. shift = mean(L17 with CSP) - mean(L17 vanilla)
  3. project shift onto the assistant axis at layer 17 (Butanium dataset).

Plots KL ↑ vs (shift · axis) and KL vs cos(shift, axis), one trajectory per
run, so we can see whether maximizing KL pulls along the assistant axis or
orthogonal to it.

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


def last_token_act_vanilla(model, tokenizer, prompt, layer_idx, device):
    text = render_messages(
        tokenizer, student_messages(prompt), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    layer_act = capture_layer_activations(
        model, layer_idx, lambda: model(input_ids=ids.unsqueeze(0)),
    )
    return layer_act[0, -1, :].clone()


def last_token_act_csp(model, tokenizer, sp, prompt, layer_idx, eval_frame, device):
    embed_fn = model.get_input_embeddings()
    suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
    user = f"{prompt} {suffix}"
    combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
    layer_act = capture_layer_activations(
        model, layer_idx, lambda: model(inputs_embeds=combined),
    )
    return layer_act[0, -1, :].clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "results", "assistant_axis.png"))
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--layer", type=int, default=config.SAE_LAYER)  # 17
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

    # Compute the vanilla baseline once.
    print(f"\nComputing vanilla L{args.layer} activations across {len(eval_prompts)} prompts...")
    vanilla_acts = []
    for p in eval_prompts:
        with torch.no_grad():
            vanilla_acts.append(
                last_token_act_vanilla(model, tokenizer, p, args.layer, device)
            )
    vanilla_acts = torch.stack(vanilla_acts).float()  # (n_prompts, hidden_dim)
    mean_vanilla = vanilla_acts.mean(dim=0)
    print(f"  mean_vanilla: ‖·‖={mean_vanilla.norm().item():.3f}")

    # Find every CSP checkpoint
    ckpt_paths = sorted(glob.glob(
        os.path.join(args.results_dir, "**/sp_pos*.pt"), recursive=True,
    ))
    print(f"\nFound {len(ckpt_paths)} checkpoints")

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

        # Compute CSP activations
        csp_acts = []
        for p in eval_prompts:
            with torch.no_grad():
                csp_acts.append(
                    last_token_act_csp(model, tokenizer, sp, p,
                                       args.layer, EVAL_FRAME_POS, device)
                )
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
        "Negative = away from assistant (toward role-play). "
        "Positive = toward assistant.",
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
