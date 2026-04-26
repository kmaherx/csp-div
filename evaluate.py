"""Evaluate trained CSPs for a persona across a 3x2 grid of conditions.

CSP sources (rows):
    pos       — sp_pos.pt, trained in positive frames
    neg       — sp_neg.pt, trained in negative frames
    math-neg  — sign-flipped sp_pos (mathematical negation, constructed at eval)

Evaluation frames (columns):
    pos frame — "Be {sp}."
    neg frame — "Don't be {sp}."

Six conditions in total. For each:
    1. Self-verbalization (greedy gen with multi-frame + single-frame prompts)
    2. SAE decomposition vs persona ground-truth activations (L17)
    3. Behavioral generation (5 samples on held-out prompts)

Plus embedding cosine comparisons between sp_pos, sp_neg, and math-neg.

Usage:
    python evaluate.py --persona pirate
    python evaluate.py --persona pirate --mode self-verb
    python evaluate.py --persona pirate --mode sae
    python evaluate.py --persona pirate --mode behavior
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

import config
from soft_prompt import SoftPrompt, negate_csp
from train import (
    render_messages, student_messages, teacher_messages,
    find_placeholder_position, load_questions,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_NEW_TOKENS_VERB = 64
MAX_NEW_TOKENS_BEHAVIOR = 128
N_BEHAVIOR_SAMPLES = 5
N_EVAL_PROMPTS = 30  # for SAE feature averaging


# Eval frame: which single frame to use at evaluation time per polarity.
# Use first frame from each polarity's training pool.
EVAL_FRAME_POS = config.POSITIVE_FRAMES[0]   # "Be {sp}."
EVAL_FRAME_NEG = config.NEGATIVE_FRAMES[0]   # "Don't be {sp}."


# ── Verbalization prompts ───────────────────────────────────────────────
# CSPs trained exclusively inside frames, so verbalization always presents
# the CSP inside one or more frames matching the condition's frame polarity.

MULTI_FRAME_TEMPLATES = [
    "Find the theme shared by these instructions: {frames} The shared theme is:",
    "These instructions all mean the same thing: {frames} In one word or phrase, they are asking me to:",
    "Rewrite these instructions as a single short command: {frames} Command:",
    "These instructions all say the same thing: {frames} Write one clear instruction that captures their meaning:",
    "Summarize these instructions into one directive: {frames} Directive:",
]


def _frames_for(polarity):
    return config.POSITIVE_FRAMES if polarity == "pos" else config.NEGATIVE_FRAMES


def multi_frame_prompts(polarity):
    frames = _frames_for(polarity)
    joined = " ".join(f.format(sp=config.SP_PLACEHOLDER) for f in frames)
    return [t.format(frames=joined) for t in MULTI_FRAME_TEMPLATES]


def single_frame_prompts(polarity):
    """One 'In plain English, explain this command' prompt per frame in the polarity pool."""
    return [
        f"In plain English, explain this command: {f.format(sp=config.SP_PLACEHOLDER)}"
        for f in _frames_for(polarity)
    ]


def verb_prompts(polarity):
    """Multi-frame (×5) + single-frame (×|frames|) prompts for a polarity."""
    return (
        [("multi_frame", p) for p in multi_frame_prompts(polarity)]
        + [("single_frame", p) for p in single_frame_prompts(polarity)]
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

def capture_layer_activations(model, layer_idx, forward_fn):
    """Run forward_fn and capture residual stream at layer_idx."""
    captured = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["act"] = output.detach().float()

    handle = model.model.language_model.layers[layer_idx].register_forward_hook(hook)
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


def get_csp_activations_at_layer(model, tokenizer, sp, prompts, layer_idx, eval_frame, device):
    """Run student forward with CSP in eval_frame across prompts, collect L17 acts at SP positions."""
    embed_fn = model.get_input_embeddings()
    L = sp.embedding.shape[0]
    acts = []
    for prompt in prompts:
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        user = f"{prompt} {suffix}"
        combined, sp_pos, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(inputs_embeds=combined),
        )
        acts.append(layer_act[0, sp_pos:sp_pos + L, :].mean(dim=0))
    return torch.stack(acts)


def get_persona_activations_at_layer(model, tokenizer, persona_prompts, prompts,
                                     layer_idx, device, seed=42):
    """Ground-truth: teacher (system+user) forward, average L17 over the system-prompt token span."""
    rng = random.Random(seed)
    acts = []
    for prompt in prompts:
        system = rng.choice(persona_prompts)
        text_with = render_messages(
            tokenizer, teacher_messages(system, prompt), add_generation_prompt=True,
        )
        text_without = render_messages(
            tokenizer, student_messages(prompt), add_generation_prompt=True,
        )
        ids_with = tokenizer(text_with, return_tensors="pt").input_ids[0].to(device)
        ids_without = tokenizer(text_without, return_tensors="pt").input_ids[0]

        # Find span where they diverge: this brackets the persona text.
        sys_start = 0
        for i in range(min(len(ids_with), len(ids_without))):
            if ids_with[i] != ids_without[i]:
                sys_start = i
                break
        n_extra = len(ids_with) - len(ids_without)
        sys_end = sys_start + n_extra
        if sys_end <= sys_start:
            continue

        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(input_ids=ids_with.unsqueeze(0)),
        )
        acts.append(layer_act[0, sys_start:sys_end, :].mean(dim=0))
    return torch.stack(acts)


# ── Modes ───────────────────────────────────────────────────────────────

def load_csp(persona_dir, polarity, device):
    ckpt_path = os.path.join(persona_dir, f"sp_{polarity}.pt")
    sp, ckpt = SoftPrompt.from_checkpoint(ckpt_path, device=device)
    return sp, ckpt


CONDITIONS = [
    # (label, csp_source, eval_frame)
    ("pos-in-pos",       "pos",      EVAL_FRAME_POS),
    ("neg-in-neg",       "neg",      EVAL_FRAME_NEG),
    ("pos-in-neg",       "pos",      EVAL_FRAME_NEG),
    ("neg-in-pos",       "neg",      EVAL_FRAME_POS),
    ("math-neg-in-pos",  "math-neg", EVAL_FRAME_POS),  # -sp_pos in "Be §."
    ("math-neg-in-neg",  "math-neg", EVAL_FRAME_NEG),  # -sp_pos in "Don't be §."
]


def run_self_verb(model, tokenizer, csps, device, eval_dir):
    """For each condition, generate verbalizations.

    Verbalization prompts always present the CSP inside frames. The frame
    polarity of the prompts matches the condition's eval_frame polarity.
    """
    embed_fn = model.get_input_embeddings()
    out = {}
    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        frame_polarity = "pos" if eval_frame in config.POSITIVE_FRAMES else "neg"
        prompts = verb_prompts(frame_polarity)
        cond_results = []
        print(f"\n  --- {label} (CSP={polarity}, frames={frame_polarity}) ---")
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


def run_sae(model, tokenizer, csps, persona_prompts, prompts, device, eval_dir):
    print(f"\nLoading SAE: {config.SAE_ID} from {config.SAE_RELEASE}...")
    sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(model.device)

    print("Computing persona ground-truth activations at L17...")
    persona_acts = get_persona_activations_at_layer(
        model, tokenizer, persona_prompts, prompts, config.SAE_LAYER, device,
    )
    persona_recon = compute_recon_error(sae, persona_acts)
    persona_active, persona_topk, persona_act_dict = get_sae_features(sae, persona_acts)
    persona_topk_ranked = sorted(
        persona_topk, key=lambda x: persona_act_dict.get(x, 0), reverse=True,
    )
    print(f"  persona: rel_err={persona_recon['rel_err']:.4f}, "
          f"cos={persona_recon['cos_sim']:.4f}, n_active={len(persona_active)}")
    print(f"  persona top-20: {persona_topk_ranked}")

    out = {
        "persona": {
            "recon": persona_recon,
            "n_active": len(persona_active),
            "topk_features": persona_topk_ranked,
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
        jac_active = jaccard(active, persona_active)
        jac_topk = jaccard(topk, persona_topk)
        shared_active = sorted(active & persona_active,
                               key=lambda x: act_dict.get(x, 0), reverse=True)
        topk_ranked = sorted(topk, key=lambda x: act_dict.get(x, 0), reverse=True)
        print(f"  {label}: rel_err={recon['rel_err']:.4f}, cos={recon['cos_sim']:.4f}, "
              f"n_active={len(active)}, jac_active={jac_active:.3f}, jac_topk={jac_topk:.3f}")
        print(f"    top-20: {topk_ranked}")
        print(f"    shared with persona (top): {shared_active[:10]}")
        out["conditions"][label] = {
            "recon": recon,
            "n_active": len(active),
            "topk_features": topk_ranked,
            "jaccard_active": jac_active,
            "jaccard_topk": jac_topk,
            "shared_features": shared_active[:50],
        }

    path = os.path.join(eval_dir, "sae.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved SAE results to {path}")
    return out


def run_behavior(model, tokenizer, csps, prompts, device, eval_dir):
    """Generate samples per condition for qualitative review."""
    embed_fn = model.get_input_embeddings()
    out = {}
    sample_prompts = prompts[:N_BEHAVIOR_SAMPLES]
    for label, polarity, eval_frame in CONDITIONS:
        sp = csps[polarity]
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        cond = []
        print(f"\n  --- {label} (CSP={polarity}, frame='{eval_frame}') ---")
        for prompt in sample_prompts:
            user = f"{prompt} {suffix}"
            combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
            with torch.no_grad():
                resp = generate_greedy(
                    model, tokenizer, inputs_embeds=combined,
                    max_new_tokens=MAX_NEW_TOKENS_BEHAVIOR,
                )
            print(f"    Q: {prompt[:80]}")
            print(f"    A: {resp[:160]}")
            cond.append({"prompt": prompt, "response": resp})
        out[label] = cond
    path = os.path.join(eval_dir, "behavior.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved behavior samples to {path}")
    return out


def embedding_compare(csps, eval_dir):
    """Pairwise cosines and norms between pos, neg, and math-neg CSPs.

    cos(pos, math-neg) is trivially -1.0 and ‖math-neg‖ = ‖pos‖ by
    construction; we report them anyway as a sanity check. The
    informative numbers are cos(neg, math-neg) — which equals
    -cos(pos, neg) — and the mag of pos ± neg, which quantifies how
    close the trained neg CSP is to being a geometric inverse of pos.
    """
    sp_pos, sp_neg, sp_math_neg = csps["pos"], csps["neg"], csps["math-neg"]
    pos_flat = sp_pos.embedding.detach().flatten().float()
    neg_flat = sp_neg.embedding.detach().flatten().float()
    mneg_flat = sp_math_neg.embedding.detach().flatten().float()

    def _cos(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    cos_pos_neg = _cos(pos_flat, neg_flat)
    cos_pos_mneg = _cos(pos_flat, mneg_flat)       # -1 by construction
    cos_neg_mneg = _cos(neg_flat, mneg_flat)       # -cos(pos, neg)

    L = sp_pos.embedding.shape[0]
    per_token = []
    for i in range(L):
        a = sp_pos.embedding[i].detach().float()
        b = sp_neg.embedding[i].detach().float()
        per_token.append({
            "i": i,
            "cos(pos, neg)": _cos(a, b),
            "cos(pos, -neg)": _cos(a, -b),
        })

    norms = {
        "||pos||": pos_flat.norm().item(),
        "||neg||": neg_flat.norm().item(),
        "||math-neg||": mneg_flat.norm().item(),
        "||pos - neg||": (pos_flat - neg_flat).norm().item(),
        "||pos + neg||": (pos_flat + neg_flat).norm().item(),
    }

    out = {
        "cos_pos_neg": cos_pos_neg,
        "cos_pos_minus_neg": -cos_pos_neg,
        "cos_pos_math_neg": cos_pos_mneg,
        "cos_neg_math_neg": cos_neg_mneg,
        "per_token": per_token,
        "norms": norms,
    }
    print(f"\nEmbedding comparison:")
    print(f"  cos(pos, neg)      = {cos_pos_neg:+.4f}")
    print(f"  cos(pos, math-neg) = {cos_pos_mneg:+.4f}  (trivially -1.0)")
    print(f"  cos(neg, math-neg) = {cos_neg_mneg:+.4f}  (= -cos(pos,neg))")
    print(f"  norms: {norms}")
    for r in per_token:
        print(f"  token {r['i']}: cos(pos,neg)={r['cos(pos, neg)']:+.4f}, "
              f"cos(pos,-neg)={r['cos(pos, -neg)']:+.4f}")
    path = os.path.join(eval_dir, "embedding_compare.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved embedding comparison to {path}")
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, choices=list(config.PERSONAS.keys()))
    parser.add_argument("--mode", default="all",
                        choices=["all", "self-verb", "sae", "behavior", "embedding"])
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--questions", default=None)
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    persona_dir = os.path.join(args.results_dir, args.persona)
    eval_dir = os.path.join(persona_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    persona_prompts = config.PERSONAS[args.persona]

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loading CSPs from {persona_dir}...")
    csps = {pol: load_csp(persona_dir, pol, device)[0] for pol in ["pos", "neg"]}
    csps["math-neg"] = negate_csp(csps["pos"])  # -sp_pos, constructed at eval
    for pol, sp in csps.items():
        print(f"  {pol}: shape={tuple(sp.embedding.shape)}, "
              f"‖·‖={sp.embedding.detach().flatten().float().norm().item():.2f}")

    questions = load_questions(args.questions)
    eval_prompts = questions[:args.n_eval_prompts]

    if args.mode in ("all", "embedding"):
        print(f"\n{'='*60}\n  EMBEDDING COMPARISON\n{'='*60}")
        embedding_compare(csps, eval_dir)

    if args.mode in ("all", "self-verb"):
        print(f"\n{'='*60}\n  SELF-VERBALIZATION\n{'='*60}")
        run_self_verb(model, tokenizer, csps, device, eval_dir)

    if args.mode in ("all", "behavior"):
        print(f"\n{'='*60}\n  BEHAVIOR SAMPLES\n{'='*60}")
        run_behavior(model, tokenizer, csps, eval_prompts, device, eval_dir)

    if args.mode in ("all", "sae"):
        print(f"\n{'='*60}\n  SAE DECOMPOSITION (L{config.SAE_LAYER})\n{'='*60}")
        run_sae(model, tokenizer, csps, persona_prompts, eval_prompts, device, eval_dir)


if __name__ == "__main__":
    main()
