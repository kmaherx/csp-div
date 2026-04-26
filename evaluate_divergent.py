"""Evaluate the max-divergence CSP across three protocols.

Loads a single CSP from results/divergent/sp_pos.pt and runs it under one
condition: ("divergent-in-pos", "pos", "Be {sp}.").

  1. Self-verbalization — multi-frame and single-frame prompts (positive only).
  2. Behavioral generation — greedy gen on held-out prompts, side-by-side with
     the vanilla model on the same prompts.
  3. SAE decomposition — Jaccard / reconstruction / top-k vs vanilla model
     activations at L17 (no persona reference).

Usage:
    python evaluate_divergent.py
    python evaluate_divergent.py --mode self-verb
    python evaluate_divergent.py --mode sae
    python evaluate_divergent.py --mode behavior
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

import config
from soft_prompt import SoftPrompt
from train import (
    render_messages, student_messages,
    load_questions,
)
from evaluate import (
    EVAL_FRAME_POS,
    verb_prompts,
    build_csp_input, build_csp_input_multi,
    generate_greedy,
    capture_layer_activations, compute_recon_error,
    get_sae_features, jaccard,
    get_csp_activations_at_layer,
    load_csp,
    MAX_NEW_TOKENS_VERB, MAX_NEW_TOKENS_BEHAVIOR,
    N_BEHAVIOR_SAMPLES, N_EVAL_PROMPTS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


CONDITIONS = [
    # (label, csp_source_polarity, eval_frame)
    ("divergent-in-pos", "pos", EVAL_FRAME_POS),
]


# ── Vanilla activations (the SAE comparator) ────────────────────────────

def get_vanilla_activations_at_layer(model, tokenizer, prompts, layer_idx, device):
    """Run vanilla model (no system prompt) over each prompt and average L17
    activations across the user-content token span.

    User-content span is found via the same divergence trick used for personas:
    tokenize student_messages(prompt) and student_messages("") under the chat
    template; the indices where they differ bracket the prompt content.
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


# ── Modes ───────────────────────────────────────────────────────────────

def run_self_verb(model, tokenizer, csps, device, eval_dir):
    embed_fn = model.get_input_embeddings()
    out = {}
    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        prompts = verb_prompts("pos")
        cond_results = []
        print(f"\n  --- {label} (CSP={polarity}, frames=pos) ---")
        for approach, vp in prompts:
            combined = build_csp_input_multi(tokenizer, embed_fn, sp, vp, device)
            with torch.no_grad():
                resp = generate_greedy(model, tokenizer, inputs_embeds=combined)
            print(f"    [{approach}] Q: {vp[:80]}")
            print(f"              A: {resp[:120]}")
            cond_results.append({
                "approach": approach, "prompt": vp, "response": resp,
            })
        out[label] = cond_results
    path = os.path.join(eval_dir, "self_verb.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved self-verb to {path}")
    return out


def run_behavior(model, tokenizer, csps, prompts, device, eval_dir):
    """Generate samples per condition with side-by-side vanilla comparison."""
    embed_fn = model.get_input_embeddings()
    out = {}
    sample_prompts = prompts[:N_BEHAVIOR_SAMPLES]
    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        cond = []
        print(f"\n  --- {label} (CSP={polarity}, frame='{eval_frame}') ---")
        for prompt in sample_prompts:
            user_csp = f"{prompt} {suffix}"
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
    path = os.path.join(eval_dir, "behavior.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved behavior samples to {path}")
    return out


def run_sae(model, tokenizer, csps, prompts, device, eval_dir):
    print(f"\nLoading SAE: {config.SAE_ID} from {config.SAE_RELEASE}...")
    sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(model.device)

    print("Computing vanilla activations at L17 (no system prompt)...")
    vanilla_acts = get_vanilla_activations_at_layer(
        model, tokenizer, prompts, config.SAE_LAYER, device,
    )
    vanilla_recon = compute_recon_error(sae, vanilla_acts)
    vanilla_active, vanilla_topk, vanilla_act_dict = get_sae_features(sae, vanilla_acts)
    vanilla_topk_ranked = sorted(
        vanilla_topk, key=lambda x: vanilla_act_dict.get(x, 0), reverse=True,
    )
    print(f"  vanilla: rel_err={vanilla_recon['rel_err']:.4f}, "
          f"cos={vanilla_recon['cos_sim']:.4f}, n_active={len(vanilla_active)}")
    print(f"  vanilla top-20: {vanilla_topk_ranked}")

    out = {
        "vanilla": {
            "recon": vanilla_recon,
            "n_active": len(vanilla_active),
            "topk_features": vanilla_topk_ranked,
        },
        "conditions": {},
    }

    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        sp_acts = get_csp_activations_at_layer(
            model, tokenizer, sp, prompts, config.SAE_LAYER, eval_frame, device,
        )
        recon = compute_recon_error(sae, sp_acts)
        active, topk, act_dict = get_sae_features(sae, sp_acts)
        jac_active = jaccard(active, vanilla_active)
        jac_topk = jaccard(topk, vanilla_topk)
        shared_active = sorted(active & vanilla_active,
                               key=lambda x: act_dict.get(x, 0), reverse=True)
        csp_only = sorted(active - vanilla_active,
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

    path = os.path.join(eval_dir, "sae.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved SAE results to {path}")
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "self-verb", "sae", "behavior"])
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--questions", default=None)
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = os.path.join(args.results_dir, "divergent")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loading CSP from {out_dir}...")
    sp, ckpt = load_csp(out_dir, "pos", device)
    csps = {"pos": sp}
    print(f"  pos: shape={tuple(sp.embedding.shape)}, "
          f"‖·‖={sp.embedding.detach().flatten().float().norm().item():.2f}, "
          f"final_kl={ckpt.get('final_kl'):.4f}")

    questions = load_questions(args.questions)
    eval_prompts = questions[:args.n_eval_prompts]

    if args.mode in ("all", "self-verb"):
        print(f"\n{'='*60}\n  SELF-VERBALIZATION\n{'='*60}")
        run_self_verb(model, tokenizer, csps, device, eval_dir)

    if args.mode in ("all", "behavior"):
        print(f"\n{'='*60}\n  BEHAVIOR SAMPLES (vs vanilla)\n{'='*60}")
        run_behavior(model, tokenizer, csps, eval_prompts, device, eval_dir)

    if args.mode in ("all", "sae"):
        print(f"\n{'='*60}\n  SAE DECOMPOSITION (L{config.SAE_LAYER}) vs vanilla\n{'='*60}")
        run_sae(model, tokenizer, csps, eval_prompts, device, eval_dir)


if __name__ == "__main__":
    main()
