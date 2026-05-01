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
    python analyze_assistant_axis.py --csp-dir llama \\
        --out results/llama/axis.png
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

from . import config, PROJECT_ROOT
from .soft_prompt import SoftPrompt
from .train import (
    render_messages, student_messages, load_questions,
    build_student, build_student_prepend, compute_kl_loss,
    precompute_vanilla_teacher_cache,
)
from .evaluate import (
    EVAL_FRAME_POS, build_csp_input, build_csp_input_prepend,
    get_transformer_layers, N_EVAL_PROMPTS,
)


def compute_eval_kl(model, tokenizer, sp, dataset, device, frame, placement,
                    n_prompts=10):
    """Eval-time KL between CSP-conditioned student and vanilla teacher.

    Used for step-0 (random-init) checkpoints whose `final_kl` is None
    because no training step has happened. Computes KL on the response
    tokens of `n_prompts` cached vanilla responses, no backward pass.
    Returns a single average KL value.
    """
    if not dataset:
        return 0.0
    embed_fn = model.get_input_embeddings()
    sample = dataset[:n_prompts]
    teacher_cache = precompute_vanilla_teacher_cache(model, tokenizer, sample, device)
    total = 0.0
    n_seen = 0
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(sample):
            teacher_ids, t_resp_start = teacher_cache[i]
            if placement == "prepend":
                student_embeds, s_resp_start = build_student_prepend(
                    tokenizer, embed_fn, sp, item["prompt"], item["response"], device,
                )
            else:
                student_embeds, s_resp_start = build_student(
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
            kl = compute_kl_loss(s_resp[:min_len], t_resp[:min_len])
            total += kl.item()
            n_seen += 1
    return total / max(n_seen, 1)



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


def response_acts_csp(model, tokenizer, sp, prompt, layer_idx, eval_frame, device, max_new_tokens,
                      placement="splice"):
    embed_fn = model.get_input_embeddings()
    if placement == "prepend":
        combined, _, _ = build_csp_input_prepend(tokenizer, embed_fn, sp, prompt, device)
    else:
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        user = f"{prompt} {suffix}"
        combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
    return _mean_response_act(
        model, tokenizer, layer_idx,
        {"inputs_embeds": combined}, max_new_tokens,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--csp-dir", required=True,
                        help="Subdir under results/ containing seed_*/sp_pos*.pt files "
                             "(e.g. llama, qwen).")
    parser.add_argument("--out", required=True,
                        help="Output PNG path; sibling .json is also written.")
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--layer", type=int, default=config.AXIS_LAYER)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generate this many response tokens before averaging acts")
    parser.add_argument("--no-save-shifts", dest="save_shifts", action="store_false",
                        help="Skip writing shifts.pt (default: write it alongside axis.{png,json}). "
                             "shifts.pt carries the raw (hidden_dim,) shift vectors used by "
                             "downstream PCA / trajectory analysis.")
    parser.add_argument("--only-new", action="store_true",
                        help="Incremental mode: load existing axis.json + shifts.pt, "
                             "skip ckpts already processed, reuse the cached mean_vanilla "
                             "(no vanilla recollection). Use this after backfilling new "
                             "checkpoints (e.g. sp_pos_step0.pt) to avoid redoing the full eval.")
    parser.set_defaults(save_shifts=True)
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

    # Incremental mode: load existing axis.json + shifts.pt, reuse mean_vanilla,
    # build a set of already-processed (group, ckpt) pairs to skip.
    rows = []
    shift_records = []
    seen_ckpts = set()
    mean_vanilla = None
    json_out = args.out.replace(".png", ".json")
    shifts_out = os.path.join(os.path.dirname(args.out) or ".", "shifts.pt")
    if args.only_new:
        if not (os.path.isfile(json_out) and os.path.isfile(shifts_out)):
            raise SystemExit(
                f"--only-new requires existing {json_out} and {shifts_out}; "
                f"run a full pass first."
            )
        with open(json_out) as f:
            existing = json.load(f)
        rows = existing["rows"]
        prev_shifts = torch.load(shifts_out, map_location="cpu", weights_only=True)
        shift_records = list(prev_shifts["rows"])
        mean_vanilla = prev_shifts["mean_vanilla"].to(device).float()
        seen_ckpts = {(r["group"], r["ckpt"]) for r in rows}
        print(f"\n[--only-new] Loaded {len(rows)} existing rows; "
              f"reusing mean_vanilla (‖·‖={mean_vanilla.norm().item():.3f}); "
              f"will skip {len(seen_ckpts)} already-processed ckpts.")

    if mean_vanilla is None:
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
    if args.only_new:
        ckpt_paths = [p for p in ckpt_paths
                      if (os.path.dirname(os.path.relpath(p, args.results_dir)),
                          os.path.basename(p)) not in seen_ckpts]
    print(f"\nFound {len(ckpt_paths)} checkpoints to process in {args.csp_dir}/")

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
        step = parse_step(ckpt_name, len(ckpt.get("kl_curve") or []))
        placement = ckpt.get("config", {}).get("placement", "splice")

        # Step-0 ckpts have no training KL — measure it from the cached vanilla
        # responses in the same seed dir. Other ckpts use their training-time
        # final_kl. Both are noisy single-batch averages, so this is consistent
        # enough for the (KL, cos) trajectory plots.
        if ckpt.get("final_kl") is None:
            cache_path = os.path.join(os.path.dirname(path), "cached_responses.json")
            if os.path.isfile(cache_path):
                with open(cache_path) as f:
                    cached_dataset = json.load(f)
                kl = compute_eval_kl(
                    model, tokenizer, sp, cached_dataset, device,
                    EVAL_FRAME_POS, placement, n_prompts=10,
                )
            else:
                kl = 0.0
        else:
            kl = float(ckpt["final_kl"])

        csp_acts = []
        for p in eval_prompts:
            a = response_acts_csp(
                model, tokenizer, sp, p, args.layer,
                EVAL_FRAME_POS, device, args.max_new_tokens,
                placement=placement,
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
        shift_records.append({
            "group": group, "ckpt": ckpt_name, "step": step, "kl": kl,
            "shift": shift.detach().cpu(),
            "mean_csp": mean_csp.detach().cpu(),
        })
        print(f"  {rel:50s}  step={step:4d}  KL={kl:7.3f}  "
              f"‖shift‖={shift_norm:6.2f}  proj·axis={proj_dot:+8.2f}  "
              f"cos={proj_cos:+.4f}")

    from .plot_style import (
        basin_color, basin_legend, draw_endpoints, draw_trajectory,
        panel_title, style_kl_axis, trajectory_basin,
    )

    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: r["step"])

    # Per-trajectory basin label using the same convention as the figure scripts
    group_basins = {
        g: trajectory_basin([(r["kl"], r["proj_cos"], r["step"]) for r in rs])
        for g, rs in by_group.items()
    }
    n_dippers = sum(1 for b in group_basins.values() if b == "deep")
    n_nondippers = len(group_basins) - n_dippers

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Non-dippers under, dippers over
    for basin_filter, zorder in [(lambda b: b != "deep", 2),
                                 (lambda b: b == "deep", 3)]:
        for g, rs in sorted(by_group.items()):
            if not basin_filter(group_basins[g]):
                continue
            color = basin_color(group_basins[g])
            kls = [r["kl"] for r in rs]
            dots = [r["proj_dot"] for r in rs]
            coss = [r["proj_cos"] for r in rs]
            draw_trajectory(axes[0], kls, dots, color, zorder=zorder)
            draw_endpoints(axes[0], kls, dots, color, zorder=zorder + 2)
            draw_trajectory(axes[1], kls, coss, color, zorder=zorder)
            draw_endpoints(axes[1], kls, coss, color, zorder=zorder + 2)

    style_kl_axis(axes[0], ylabel=f"(L{args.layer} shift) · (assistant axis)")
    style_kl_axis(axes[1], ylabel=f"cos(L{args.layer} shift, assistant axis)")
    panel_title(axes[0], "Magnitude along assistant axis")
    panel_title(axes[1], "Direction alignment with assistant axis")
    basin_legend(axes[1], n_dippers, n_nondippers, loc="lower right")
    fig.suptitle(
        f"L{args.layer} shift onto Butanium axis  ·  "
        f"negative = role-play, positive = default-assistant",
        fontsize=10, color="#444444",
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"\nSaved plot: {args.out}")

    json_out = args.out.replace(".png", ".json")
    with open(json_out, "w") as f:
        json.dump({"layer": args.layer, "rows": rows}, f, indent=2)
    print(f"Saved data: {json_out}")

    # Save full shift vectors (for downstream PCA-trajectory analysis).
    # Schema mirrors axis.json's `rows` field but each row carries the
    # raw (hidden_dim,) shift = mean_csp - mean_vanilla as a CPU tensor.
    if args.save_shifts and shift_records:
        shifts_out = os.path.join(os.path.dirname(args.out) or ".", "shifts.pt")
        torch.save({
            "layer": args.layer,
            "n_eval_prompts": len(eval_prompts),
            "max_new_tokens": args.max_new_tokens,
            "mean_vanilla": mean_vanilla.detach().cpu(),
            "rows": shift_records,
        }, shifts_out)
        print(f"Saved shifts: {shifts_out}  ({len(shift_records)} rows, "
              f"hidden_dim={mean_vanilla.shape[0]})")


if __name__ == "__main__":
    main()
