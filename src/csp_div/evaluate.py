"""Evaluate the max-divergence CSP across three protocols.

Loads a single CSP per checkpoint and runs it under one condition:
    ("divergent-in-pos", "pos", "Be {sp}.")

  1. Self-verbalization — multi-frame and single-frame prompts (positive only).
  2. Behavioral generation — greedy gen on held-out prompts, side-by-side with
     the vanilla model on the same prompts.
  3. SAE decomposition — Jaccard / reconstruction / top-k vs vanilla model
     activations at config.SAE_LAYER (no persona reference).

Usage:
    python -m csp_div.evaluate
    python -m csp_div.evaluate --mode self-verb
    python -m csp_div.evaluate --mode sae
    python -m csp_div.evaluate --mode behavior
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from . import config, PROJECT_ROOT
from .soft_prompt import SoftPrompt
from .train import (
    render_messages, student_messages,
    find_placeholder_position, find_content_boundaries, load_questions,
)

MAX_NEW_TOKENS_VERB = 64
MAX_NEW_TOKENS_BEHAVIOR = 128
N_BEHAVIOR_SAMPLES = 5
N_EVAL_PROMPTS = 30  # for SAE feature averaging

EVAL_FRAME_POS = config.POSITIVE_FRAMES[0]   # "Be {sp}."

CONDITIONS = [
    # (label, csp_source_polarity, eval_frame)
    ("divergent-in-pos", "pos", EVAL_FRAME_POS),
]


# ── Verbalization prompts (positive frames only) ────────────────────────

MULTI_FRAME_TEMPLATES = [
    "Find the theme shared by these instructions: {frames} The shared theme is:",
    "These instructions all mean the same thing: {frames} In one word or phrase, they are asking me to:",
    "Rewrite these instructions as a single short command: {frames} Command:",
    "These instructions all say the same thing: {frames} Write one clear instruction that captures their meaning:",
    "Summarize these instructions into one directive: {frames} Directive:",
]


def multi_frame_prompts():
    joined = " ".join(f.format(sp=config.SP_PLACEHOLDER) for f in config.POSITIVE_FRAMES)
    return [t.format(frames=joined) for t in MULTI_FRAME_TEMPLATES]


def single_frame_prompts():
    """One 'In plain English, explain this command' prompt per positive frame."""
    return [
        f"In plain English, explain this command: {f.format(sp=config.SP_PLACEHOLDER)}"
        for f in config.POSITIVE_FRAMES
    ]


def verb_prompts():
    """Multi-frame (×5) + single-frame (×|frames|) prompts."""
    return (
        [("multi_frame", p) for p in multi_frame_prompts()]
        + [("single_frame", p) for p in single_frame_prompts()]
    )


# ── Splicing helpers ────────────────────────────────────────────────────

def build_csp_input(tokenizer, embed_fn, sp, user_text_with_placeholder, device):
    """Tokenize user_text (containing one §) under chat template and splice CSP."""
    text = render_messages(
        tokenizer, student_messages(user_text_with_placeholder),
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    sp_pos = find_placeholder_position(tokenizer, ids)
    embeds = embed_fn(ids.unsqueeze(0))
    sp_embeds = sp(batch_size=1).to(embeds.dtype)
    return torch.cat([
        embeds[:, :sp_pos, :], sp_embeds, embeds[:, sp_pos + 1:, :],
    ], dim=1), sp_pos, sp.embedding.shape[0]


def build_csp_input_multi(tokenizer, embed_fn, sp, user_text_with_placeholders, device):
    """Splice CSP into every § occurrence in user_text."""
    text = render_messages(
        tokenizer, student_messages(user_text_with_placeholders),
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    positions = [
        i for i, tid in enumerate(ids.tolist())
        if config.SP_PLACEHOLDER in tokenizer.decode([tid])
    ]
    if not positions:
        raise ValueError(f"No placeholders found in: {text[:120]}...")
    embeds = embed_fn(ids.unsqueeze(0))
    sp_embeds = sp(batch_size=1).to(embeds.dtype)
    result = embeds
    for pos in reversed(positions):
        result = torch.cat([
            result[:, :pos, :], sp_embeds, result[:, pos + 1:, :],
        ], dim=1)
    return result


def build_csp_input_prepend(tokenizer, embed_fn, sp, prompt, device):
    """Build inference input by PREPENDING CSP at content_start.

    No frame, no placeholder. Returns (combined_embeds, content_start, L)
    matching the build_csp_input return shape so callers can use it
    interchangeably."""
    L = sp.embedding.shape[0]
    text = render_messages(
        tokenizer, student_messages(prompt), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    content_start = find_content_boundaries(tokenizer, prompt)
    embeds = embed_fn(ids.unsqueeze(0))
    sp_embeds = sp(batch_size=1).to(embeds.dtype)
    combined = torch.cat([
        embeds[:, :content_start, :], sp_embeds, embeds[:, content_start:, :],
    ], dim=1)
    return combined, content_start, L


# ── Greedy generation with kv cache ─────────────────────────────────────

def generate_greedy(model, tokenizer, inputs_embeds=None, input_ids=None,
                     max_new_tokens=MAX_NEW_TOKENS_VERB):
    out_ids = []
    past = None
    for i in range(max_new_tokens):
        if i == 0:
            kw = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": input_ids}
            out = model(**kw, use_cache=True)
        else:
            out = model(input_ids=next_tok.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True)
        out_ids.append(next_tok.item())
        if next_tok.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


# ── SAE / activation utilities ──────────────────────────────────────────

def get_transformer_layers(model):
    """Return the model's transformer-block ModuleList. Different HF
    architectures expose this at different paths:
      - Qwen2/Llama/most others: model.model.layers
      - some wrapped models: model.model.language_model.layers
    """
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def capture_layer_activations(model, layer_idx, forward_fn):
    """Run forward_fn and capture residual stream at layer_idx."""
    captured = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["act"] = output.detach().float()

    layers = get_transformer_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        forward_fn()
    handle.remove()
    return captured["act"]


def compute_recon_error(sae, activations):
    x = activations.to(sae.device).to(sae.dtype)
    features = sae.encode(x)
    x_hat = sae.decode(features)
    mse = ((x - x_hat) ** 2).mean(dim=-1)
    norm_sq = (x ** 2).mean(dim=-1).clamp(min=1e-8)
    rel_err = (mse / norm_sq).mean().item()
    cos = F.cosine_similarity(x, x_hat, dim=-1).mean().item()
    return {"rel_err": rel_err, "cos_sim": cos}


def get_sae_features(sae, activations, top_k=20):
    x = activations.to(sae.device).to(sae.dtype)
    feats = sae.encode(x)
    mean_act = feats.mean(dim=0)
    active = set((mean_act > 0).nonzero(as_tuple=True)[0].cpu().tolist())
    topk_idx = torch.topk(mean_act, min(top_k, len(mean_act)))[1].cpu().tolist()
    topk = set(topk_idx)
    act_dict = {idx: mean_act[idx].item() for idx in active}
    return active, topk, act_dict


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_csp_activations_at_layer(model, tokenizer, sp, prompts, layer_idx, eval_frame, device,
                                  placement="splice"):
    """Run student forward with CSP across prompts, collect acts at SP positions.

    placement: "splice" uses eval_frame + § splice. "prepend" puts CSP at
    content_start with no frame; eval_frame is ignored in that case."""
    embed_fn = model.get_input_embeddings()
    L = sp.embedding.shape[0]
    acts = []
    for prompt in prompts:
        if placement == "prepend":
            combined, sp_pos, _ = build_csp_input_prepend(tokenizer, embed_fn, sp, prompt, device)
        else:
            suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
            user = f"{prompt} {suffix}"
            combined, sp_pos, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(inputs_embeds=combined),
        )
        acts.append(layer_act[0, sp_pos:sp_pos + L, :].mean(dim=0))
    return torch.stack(acts)


# ── Vanilla activations (the SAE comparator) ────────────────────────────

def get_vanilla_activations_at_layer(model, tokenizer, prompts, layer_idx, device):
    """Run vanilla model (no system prompt) over each prompt and average
    activations at layer_idx across the user-content token span.

    User-content span is found by tokenizing student_messages(prompt) and
    student_messages("") under the chat template; the indices where they
    differ bracket the prompt content.
    """
    text_empty = render_messages(
        tokenizer, student_messages(""), add_generation_prompt=True,
    )
    ids_empty = tokenizer(text_empty, return_tensors="pt").input_ids[0]

    acts = []
    for prompt in prompts:
        text_full = render_messages(
            tokenizer, student_messages(prompt), add_generation_prompt=True,
        )
        ids_full = tokenizer(text_full, return_tensors="pt").input_ids[0].to(device)

        user_start = 0
        for i in range(min(len(ids_full), len(ids_empty))):
            if ids_full[i].item() != ids_empty[i].item():
                user_start = i
                break
        n_extra = len(ids_full) - len(ids_empty)
        user_end = user_start + n_extra
        if user_end <= user_start:
            continue

        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(input_ids=ids_full.unsqueeze(0)),
        )
        acts.append(layer_act[0, user_start:user_end, :].mean(dim=0))
    return torch.stack(acts)


# ── Eval modes ──────────────────────────────────────────────────────────

def run_self_verb(model, tokenizer, csps, device, eval_dir, suffix="", placement="splice"):
    embed_fn = model.get_input_embeddings()
    out = {}
    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        prompts = verb_prompts()
        cond_results = []
        print(f"\n  --- {label} (CSP={polarity}, placement={placement}) ---")
        for approach, vp in prompts:
            if placement == "prepend":
                # Prepend the CSP at content_start of the verbalization prompt.
                # The prompt still contains literal § characters from the
                # frames it asks about; under prepend those are just text.
                combined, _, _ = build_csp_input_prepend(tokenizer, embed_fn, sp, vp, device)
            else:
                combined = build_csp_input_multi(tokenizer, embed_fn, sp, vp, device)
            with torch.no_grad():
                resp = generate_greedy(model, tokenizer, inputs_embeds=combined)
            print(f"    [{approach}] Q: {vp[:80]}")
            print(f"              A: {resp[:120]}")
            cond_results.append({
                "approach": approach, "prompt": vp, "response": resp,
            })
        out[label] = cond_results
    path = os.path.join(eval_dir, f"self_verb{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved self-verb to {path}")
    return out


def run_behavior(model, tokenizer, csps, prompts, device, eval_dir, suffix="", placement="splice"):
    """Generate samples per condition with side-by-side vanilla comparison."""
    embed_fn = model.get_input_embeddings()
    out = {}
    sample_prompts = prompts[:N_BEHAVIOR_SAMPLES]
    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        eval_suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        cond = []
        print(f"\n  --- {label} (CSP={polarity}, placement={placement}) ---")
        for prompt in sample_prompts:
            if placement == "prepend":
                combined, _, _ = build_csp_input_prepend(tokenizer, embed_fn, sp, prompt, device)
            else:
                user_csp = f"{prompt} {eval_suffix}"
                combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user_csp, device)
            with torch.no_grad():
                resp_csp = generate_greedy(
                    model, tokenizer, inputs_embeds=combined,
                    max_new_tokens=MAX_NEW_TOKENS_BEHAVIOR,
                )

            text_vanilla = render_messages(
                tokenizer, student_messages(prompt), add_generation_prompt=True,
            )
            ids_vanilla = tokenizer(text_vanilla, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                resp_vanilla = generate_greedy(
                    model, tokenizer, input_ids=ids_vanilla,
                    max_new_tokens=MAX_NEW_TOKENS_BEHAVIOR,
                )

            print(f"    Q: {prompt[:80]}")
            print(f"    [vanilla] {resp_vanilla[:160]}")
            print(f"    [csp]     {resp_csp[:160]}")
            cond.append({
                "prompt": prompt,
                "response_vanilla": resp_vanilla,
                "response_csp": resp_csp,
            })
        out[label] = cond
    path = os.path.join(eval_dir, f"behavior{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved behavior samples to {path}")
    return out


def load_sae_for_model(model):
    """Load the SAE and move it to the model's device."""
    print(f"Loading SAE: {config.SAE_ID} from {config.SAE_RELEASE}...")
    sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    return sae.to(model.device)


def compute_vanilla_baseline(model, tokenizer, sae, prompts, device):
    """Compute the vanilla SAE baseline once: activations, recon, active set,
    top-k. Reusable across many CSP checkpoints."""
    print(f"Computing vanilla activations at L{config.SAE_LAYER} (no system prompt)...")
    vanilla_acts = get_vanilla_activations_at_layer(
        model, tokenizer, prompts, config.SAE_LAYER, device,
    )
    recon = compute_recon_error(sae, vanilla_acts)
    active, topk, act_dict = get_sae_features(sae, vanilla_acts)
    topk_ranked = sorted(topk, key=lambda x: act_dict.get(x, 0), reverse=True)
    print(f"  vanilla: rel_err={recon['rel_err']:.4f}, "
          f"cos={recon['cos_sim']:.4f}, n_active={len(active)}")
    print(f"  vanilla top-20: {topk_ranked}")
    return {
        "recon": recon,
        "active": active,
        "topk": topk,
        "act_dict": act_dict,
        "topk_ranked": topk_ranked,
    }


def run_sae(model, tokenizer, csps, prompts, device, eval_dir,
            sae, vanilla_baseline, suffix="", placement="splice"):
    """Run SAE comparison for one CSP. Caller must pre-load `sae` (via
    load_sae_for_model) and `vanilla_baseline` (via compute_vanilla_baseline)
    so the heavy work is only done once across many CSP checkpoints."""
    out = {
        "vanilla": {
            "recon": vanilla_baseline["recon"],
            "n_active": len(vanilla_baseline["active"]),
            "topk_features": vanilla_baseline["topk_ranked"],
        },
        "conditions": {},
    }

    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        sp_acts = get_csp_activations_at_layer(
            model, tokenizer, sp, prompts, config.SAE_LAYER, eval_frame, device,
            placement=placement,
        )
        recon = compute_recon_error(sae, sp_acts)
        active, topk, act_dict = get_sae_features(sae, sp_acts)
        jac_active = jaccard(active, vanilla_baseline["active"])
        jac_topk = jaccard(topk, vanilla_baseline["topk"])
        shared_active = sorted(active & vanilla_baseline["active"],
                               key=lambda x: act_dict.get(x, 0), reverse=True)
        csp_only = sorted(active - vanilla_baseline["active"],
                          key=lambda x: act_dict.get(x, 0), reverse=True)
        topk_ranked = sorted(topk, key=lambda x: act_dict.get(x, 0), reverse=True)
        print(f"  {label}: rel_err={recon['rel_err']:.4f}, cos={recon['cos_sim']:.4f}, "
              f"n_active={len(active)}, jac_active={jac_active:.3f}, jac_topk={jac_topk:.3f}")
        print(f"    top-20: {topk_ranked}")
        print(f"    shared with vanilla (top): {shared_active[:10]}")
        print(f"    csp-only (top): {csp_only[:10]}")
        out["conditions"][label] = {
            "recon": recon,
            "n_active": len(active),
            "topk_features": topk_ranked,
            "jaccard_active": jac_active,
            "jaccard_topk": jac_topk,
            "shared_features": shared_active[:50],
            "csp_only_features": csp_only[:50],
        }

    path = os.path.join(eval_dir, f"sae{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved SAE results to {path}")
    return out


def suffix_for(ckpt_name):
    """sp_pos.pt -> "" ; sp_pos_step100.pt -> "_step100" ; sp_pos_foo.pt -> "_foo"."""
    return os.path.splitext(ckpt_name)[0].replace("sp_pos", "", 1)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "self-verb", "sae", "behavior"])
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--run-name", default="divergent",
                        help="Subdir under results/ to load checkpoints from")
    parser.add_argument("--checkpoints", nargs="+", default=["sp_pos.pt"],
                        help="One or more checkpoint filenames inside the run dir. "
                             "Model + tokenizer + SAE + vanilla baseline are loaded "
                             "once and reused across all checkpoints.")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = os.path.join(args.results_dir, args.run_name)
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Validate every checkpoint exists before loading the model — fail fast.
    ckpt_paths = [(c, os.path.join(out_dir, c)) for c in args.checkpoints]
    missing = [p for _, p in ckpt_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    questions = load_questions(args.questions)
    eval_prompts = questions[:args.n_eval_prompts]

    # Pre-load SAE + vanilla baseline once if SAE eval is requested. These are
    # the same for every checkpoint, so doing them N times is pure waste.
    sae = None
    vanilla_baseline = None
    if args.mode in ("all", "sae"):
        print(f"\n{'='*60}\n  PRE-LOAD SAE + VANILLA BASELINE\n{'='*60}")
        sae = load_sae_for_model(model)
        vanilla_baseline = compute_vanilla_baseline(
            model, tokenizer, sae, eval_prompts, device,
        )

    for ckpt_name, ckpt_path in ckpt_paths:
        suffix = suffix_for(ckpt_name)
        header = f"CHECKPOINT {ckpt_name}  (suffix='{suffix or '(none)'}')"
        print(f"\n{'#'*70}\n# {header}\n{'#'*70}")

        sp, ckpt = SoftPrompt.from_checkpoint(ckpt_path, device=device)
        csps = {"pos": sp}
        placement = ckpt.get("config", {}).get("placement", "splice")
        print(f"  pos: shape={tuple(sp.embedding.shape)}, "
              f"‖·‖={sp.embedding.detach().flatten().float().norm().item():.2f}, "
              f"final_kl={ckpt.get('final_kl'):.4f}, placement={placement}")

        if args.mode in ("all", "self-verb"):
            print(f"\n--- SELF-VERBALIZATION ---")
            run_self_verb(model, tokenizer, csps, device, eval_dir, suffix=suffix,
                          placement=placement)

        if args.mode in ("all", "behavior"):
            print(f"\n--- BEHAVIOR SAMPLES (vs vanilla) ---")
            run_behavior(model, tokenizer, csps, eval_prompts, device, eval_dir,
                         suffix=suffix, placement=placement)

        if args.mode in ("all", "sae"):
            print(f"\n--- SAE DECOMPOSITION (L{config.SAE_LAYER}) vs vanilla ---")
            run_sae(model, tokenizer, csps, eval_prompts, device, eval_dir,
                    sae, vanilla_baseline, suffix=suffix, placement=placement)


if __name__ == "__main__":
    main()
