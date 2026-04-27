"""Causal ablation of shared SAE features.

For each (seed, prompt) we generate three responses:
  (a) vanilla — no system prompt, no CSP
  (b) CSP — the divergent behavior
  (c) CSP + ablation — at every L17 forward pass, the shared SAE features
      are clamped to their vanilla baseline activations via a hook that
      modifies the residual stream by `(baseline - current_feature_act) ·
      W_dec[feature]`.

If (c) reverts toward (a), the shared features ARE load-bearing for the
divergent attractor. If (c) stays similar to (b), they're downstream of
something else.

Default target features = the 9-feature shared core from the 10-seed
analysis (results/early_stop/sae.md): 96, 406, 409, 243, 242, 282, 510,
218, 1263.

Usage:
    python analyze_ablation.py                 # default seeds {0,1,5}, 5 prompts
    python analyze_ablation.py --seeds 0 1 2 3 4 5 6 7 8 9 --n-prompts 3
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

import config
from soft_prompt import SoftPrompt
from train import render_messages, student_messages, load_questions
from evaluate import (
    EVAL_FRAME_POS, build_csp_input, generate_greedy,
    get_csp_activations_at_layer,
    MAX_NEW_TOKENS_BEHAVIOR, N_BEHAVIOR_SAMPLES,
)
from evaluate_divergent import get_vanilla_activations_at_layer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared core from results/early_stop/sae.md
DEFAULT_TARGETS = [96, 406, 409, 243, 242, 282, 510, 218, 1263]


def compute_vanilla_baseline_features(model, tokenizer, sae, prompts, layer_idx, device):
    """Vanilla L17 user-span mean acts → SAE-encoded → averaged across prompts.
    Returns a (n_features,) tensor of mean baseline feature activations."""
    vanilla_acts = get_vanilla_activations_at_layer(
        model, tokenizer, prompts, layer_idx, device,
    )  # (n_prompts, hidden)
    f = sae.encode(vanilla_acts.to(sae.device).to(sae.dtype))
    return f.mean(dim=0).cpu()  # (n_features,)


def make_ablation_hook(sae, target_indices, baseline_values):
    """Forward hook on a transformer block: SAE-encode the layer's output,
    replace target features with baseline values, splice back into the
    residual stream via W_dec. Other features and the SAE residual error
    are untouched."""
    target_idx = torch.tensor(
        target_indices, device=sae.W_dec.device, dtype=torch.long,
    )
    baseline = baseline_values.to(sae.W_dec.device).to(sae.dtype)  # (n_targets,)
    W_target = sae.W_dec[target_idx]  # (n_targets, hidden)

    def hook(module, inp, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        x_dtype = x.dtype
        x_sae = x.to(sae.dtype)
        f = sae.encode(x_sae)                    # (batch, seq, n_features)
        f_target = f[..., target_idx]            # (batch, seq, n_targets)
        delta_target = baseline.expand_as(f_target) - f_target
        delta_x = torch.matmul(delta_target, W_target)  # (batch, seq, hidden)
        x_modified = (x_sae + delta_x).to(x_dtype)
        if rest is not None:
            return (x_modified,) + rest
        return x_modified

    return hook


def gen_csp(model, tokenizer, embed_fn, sp, prompt, device, max_new_tokens):
    suffix = EVAL_FRAME_POS.format(sp=config.SP_PLACEHOLDER)
    user = f"{prompt} {suffix}"
    combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
    with torch.no_grad():
        return generate_greedy(
            model, tokenizer, inputs_embeds=combined, max_new_tokens=max_new_tokens,
        )


def gen_vanilla(model, tokenizer, prompt, device, max_new_tokens):
    text = render_messages(
        tokenizer, student_messages(prompt), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return generate_greedy(
            model, tokenizer, input_ids=ids, max_new_tokens=max_new_tokens,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 5])
    parser.add_argument("--n-prompts", type=int, default=N_BEHAVIOR_SAMPLES)
    parser.add_argument("--target-features", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_BEHAVIOR)
    parser.add_argument("--clamp-to", choices=["vanilla", "zero"], default="vanilla",
                        help="vanilla = clamp targets to mean vanilla feature activation; "
                             "zero = clamp to 0 (full suppression)")
    args = parser.parse_args()

    if args.out is None:
        tag = "_".join(str(t) for t in args.target_features[:5])
        args.out = os.path.join(
            args.results_dir, f"ablation_{args.clamp_to}_{tag}.json",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loading SAE {config.SAE_ID}...")
    sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(model.device)

    questions = load_questions()
    prompts = questions[:args.n_prompts]

    # Baseline values for the target features
    if args.clamp_to == "vanilla":
        print(f"\nComputing vanilla baseline features at L{config.SAE_LAYER}...")
        baseline_f = compute_vanilla_baseline_features(
            model, tokenizer, sae, questions[:30], config.SAE_LAYER, device,
        )
        baseline_values = baseline_f[args.target_features]
        print(f"  vanilla baseline at target features:")
        for tf, bv in zip(args.target_features, baseline_values.tolist()):
            print(f"    feat {tf:5d}: {bv:.4f}")
    else:  # zero
        baseline_values = torch.zeros(len(args.target_features))
        print(f"\nClamping targets to zero (full suppression)")

    embed_fn = model.get_input_embeddings()
    out = {
        "target_features": args.target_features,
        "clamp_to": args.clamp_to,
        "baseline_values": baseline_values.tolist(),
        "max_new_tokens": args.max_new_tokens,
        "seeds": {},
    }

    for seed in args.seeds:
        ckpt_path = os.path.join(
            args.results_dir, "early_stop", f"seed_{seed}", "sp_pos.pt",
        )
        if not os.path.isfile(ckpt_path):
            print(f"  skip seed_{seed}: no ckpt at {ckpt_path}")
            continue
        sp, ckpt = SoftPrompt.from_checkpoint(ckpt_path, device=device)
        seed_results = []
        print(f"\n=== seed_{seed} (KL≈{ckpt.get('final_kl'):.2f}) ===")

        # Pre-build the ablation hook for this seed (same baseline for all prompts)
        hook_fn = make_ablation_hook(sae, args.target_features, baseline_values)

        for prompt in prompts:
            resp_vanilla = gen_vanilla(model, tokenizer, prompt, device, args.max_new_tokens)
            resp_csp = gen_csp(model, tokenizer, embed_fn, sp, prompt, device, args.max_new_tokens)

            from evaluate import get_transformer_layers
            handle = get_transformer_layers(model)[config.SAE_LAYER].register_forward_hook(hook_fn)
            try:
                resp_ablated = gen_csp(model, tokenizer, embed_fn, sp, prompt, device, args.max_new_tokens)
            finally:
                handle.remove()

            seed_results.append({
                "prompt": prompt,
                "vanilla": resp_vanilla,
                "csp": resp_csp,
                "csp_ablated": resp_ablated,
            })
            print(f"  Q: {prompt[:60]}")
            print(f"    [vanilla]   {resp_vanilla[:90]!r}")
            print(f"    [csp]       {resp_csp[:90]!r}")
            print(f"    [ablated]   {resp_ablated[:90]!r}")
        out["seeds"][f"seed_{seed}"] = seed_results

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
