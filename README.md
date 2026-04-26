# csp-div — Max-divergence contextualized soft prompt

Train a contextualized soft prompt (CSP) that **maximizes** KL divergence from
the vanilla `google/gemma-3-4b-it` assistant behavior. Then probe what it does
via three protocols:

1. **Behavioral generation** — greedy outputs side-by-side with the vanilla model.
2. **Self-verbalization** — ask the model to describe what the CSP "asks for".
3. **SAE feature decomposition** — Jaccard / reconstruction / top-k overlap
   against vanilla activations at L17, plus a `csp_only_features` list of
   features the divergent CSP uniquely activates.

The intent is to find an "anti-assistant" attractor — a single contextualized
direction that, when spliced into a positive frame like *"Be §."*, drives the
model as far away from its default behavior as possible while remaining a
well-formed instruction.

## Background

CSPs and the three-target eval protocol are described here:
<https://kmaherx.github.io/projects/contextualized-soft-prompts/>

This repo is a fork-style sibling of <https://github.com/kmaherx/csp_arithmetic>.
We reuse its training scaffolding (`train.py`, `evaluate.py`, `soft_prompt.py`,
`config.py`, `data/questions.jsonl`) verbatim and add two new entry points,
`train_divergent.py` and `evaluate_divergent.py`, with the four deltas listed
below.

## What's different from `csp_arithmetic`

| | `csp_arithmetic` | `csp-div` |
|---|---|---|
| Teacher | persona system prompt | vanilla model (no system prompt) |
| Loss | `kl.backward()` (descent) | `(-kl).backward()` (ascent) |
| SAE comparator | persona ground-truth activations | vanilla model activations |
| Frames trained | positive **and** negative | positive only |

There is therefore one CSP per run (no `neg`, no `math-neg`), and the eval
condition list collapses to a single condition: `("divergent-in-pos", "pos", "Be §.")`.

## Quickstart

```bash
pip install torch transformers sae_lens

# Train (≈tens of minutes on a single GPU; defaults: L=4, 500 steps)
python train_divergent.py

# Evaluate (all three protocols)
python evaluate_divergent.py --mode all
```

Outputs:

```
results/divergent/
├── cached_responses.json      # vanilla teacher responses for the 240-prompt training set
├── sp_pos.pt                  # the trained CSP + kl_curve + config
└── eval/
    ├── self_verb.json
    ├── behavior.json          # response_csp vs response_vanilla per prompt
    └── sae.json               # includes csp_only_features list
```

## Files

- `train_divergent.py` — vanilla-teacher generation + KL-ascent training loop.
- `evaluate_divergent.py` — three-protocol eval; SAE comparator is vanilla acts.
- `config.py`, `soft_prompt.py`, `train.py`, `evaluate.py` — copied verbatim
  from `csp_arithmetic` (imported by the divergent scripts; do not modify here).
- `data/questions.jsonl` — 240 evaluation prompts.

## Reproducibility

Same hyperparameters as upstream: AdamW, lr=1e-3, weight_decay=1e-4, 500 steps,
50 prompts/step, L=4, seed=42, frame pool = `config.POSITIVE_FRAMES`. Greedy
decoding everywhere.
