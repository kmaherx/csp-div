# csp-div — Max-divergence contextualized soft prompt

Train a contextualized soft prompt (CSP) that **maximizes** KL divergence from
the vanilla (no-system-prompt) assistant behavior. Then probe what it does via
three protocols:

1. **Behavioral generation** — greedy outputs side-by-side with the vanilla model.
2. **Self-verbalization** — ask the model to describe what the CSP "asks for".
3. **SAE feature decomposition** — Jaccard / reconstruction / top-k overlap
   against vanilla activations at the SAE layer, plus a `csp_only_features`
   list of features the divergent CSP uniquely activates. (Requires an SAE for
   the target model; not currently wired for Qwen/Llama presets.)

The intent is to find an "anti-assistant" attractor — a single contextualized
direction that, when spliced into a positive frame like *"Be §."*, drives the
model as far away from its default behavior as possible while remaining a
well-formed instruction. See [`NARRATIVE.md`](NARRATIVE.md) for the story so
far and [`TODO.md`](TODO.md) for active threads.

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

## Models

Active model presets live in `config.py` and are selected via
`CSP_MODEL_PRESET`. Default is Llama-3.1-8B-Instruct.

```
CSP_MODEL_PRESET=llama-3.1-8b-instruct   # default
CSP_MODEL_PRESET=qwen-2.5-7b-instruct
```

Earlier Gemma-3-4b-it work lives on the legacy branches `trough-theory` and
`early-stop-kl10` (preserved, not maintained on `cross-model`/`rng-probe`).

## Quickstart

```bash
pip install torch transformers sae_lens

# Train (≈tens of minutes on a single GPU; defaults: L=4, 500 steps)
python train_divergent.py

# Evaluate behavior + self-verb (SAE only if the preset has one wired up)
python evaluate_divergent.py --mode behavior
python evaluate_divergent.py --mode self-verb
```

Outputs (under `results/<run-name>/`):

```
cached_responses.json      # vanilla teacher responses for the 240-prompt training set
sp_pos.pt                  # the trained CSP + kl_curve + config
sp_pos_step<N>.pt          # intermediate checkpoints (gitignored)
eval/
├── self_verb.json
├── behavior.json          # response_csp vs response_vanilla per prompt
└── sae.json               # only when the preset has SAE_RELEASE set
```

## Files

- `train_divergent.py` — vanilla-teacher generation + KL-ascent training loop.
  Supports `--seed` (init RNG), `--data-seed` (sampling RNG, defaults to seed),
  `--checkpoint-every`, `--early-stop-kl`.
- `evaluate_divergent.py` — three-protocol eval; SAE comparator is vanilla acts.
- `analyze_assistant_axis.py` — projects the CSP-induced residual-stream shift
  onto the Butanium assistant axis at the preset's `AXIS_LAYER`. Used to
  generate the trough/axis plots in `results/`.
- `plot_rng_probe.py` — combined plot for the RNG decoupling probe.
- `scripts/` — sweep runners (`run_llama_trough.sh`,
  `run_rng_probe.sh`, `run_trough_axis_plots.sh`). Each `cd`s up to
  the repo root before invoking python.
- `config.py`, `soft_prompt.py`, `train.py`, `evaluate.py` — copied verbatim
  from `csp_arithmetic` (imported by the divergent scripts).
- `data/questions.jsonl` — 240 evaluation prompts.

## Reproducibility

Same hyperparameters as upstream: AdamW, lr=1e-3, weight_decay=1e-4, 500 steps,
50 prompts/step, L=4, seed=42, frame pool = `config.POSITIVE_FRAMES`. Greedy
decoding everywhere.
