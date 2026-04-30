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
`csp_div.train` and `csp_div.evaluate`, with the four deltas listed below.

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

Active model presets live in `csp_div.config` and are selected via
`CSP_MODEL_PRESET`. Default is Llama-3.1-8B-Instruct.

```
CSP_MODEL_PRESET=llama-3.1-8b-instruct   # default
CSP_MODEL_PRESET=qwen-2.5-7b-instruct
```

Earlier Gemma-3-4b-it work lives on the legacy branches `trough-theory` and
`early-stop-kl10` (preserved, not maintained on `cross-model`/`rng-probe`).

## Quickstart

```bash
pip install -e .

# Train (≈tens of minutes on a single GPU; defaults: L=4, 500 steps)
python -m csp_div.train

# Evaluate behavior + self-verb (SAE only if the preset has one wired up)
python -m csp_div.evaluate --mode behavior
python -m csp_div.evaluate --mode self-verb
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

## Layout

```
src/csp_div/                  package
├── __init__.py               PROJECT_ROOT anchor
├── config.py                 model presets, personas, frames, hyperparameters
├── soft_prompt.py            SoftPrompt class
├── train.py                  entry point: KL-ascent training (frame-pool + placement flags)
├── evaluate.py               entry point: behavior + self-verb + SAE (placement-aware)
├── analyze_assistant_axis.py entry point: trough / axis-projection (saves shifts.pt)
└── plot_rng_probe.py         entry point: combined RNG probe plot

scripts/                      shell wrappers (run with `bash scripts/<name>.sh`)
├── run_llama_trough.sh       10-seed Llama PERSONA trough sweep (legacy)
├── run_rng_probe.sh          init-vs-data RNG decoupling probe
├── run_trough_axis_plots.sh  per-model axis plots from existing ckpts
├── run_frame_bias.sh         Qwen frame-bias sweep (INSTRUMENTAL etc.)
├── run_minimal.sh            standalone Qwen MINIMAL runner
├── run_minimal_style.sh      standalone Qwen MINIMAL+STYLE runner (legacy)
├── run_prepend.sh            standalone Qwen PREPEND runner
├── run_overnight.sh          chained Qwen overnight: shifts + PCA + MINIMAL
├── run_llama_overnight.sh    chained Llama overnight: all 4 conditions + PCA
├── analyze_pca_trajectory.py per-model PCA on shifts (raw + --normalize)
├── analyze_frame_bias.py     basin classification + populations bar
├── figure_basins.py          per-condition trajectory figure with bolded examples
├── figure_basins_by_label.py 10-trajectory plot, color by basin
├── figure_cross_condition.py side-by-side panels across 2 conditions
├── compare_seed_5_across_frames.py  qualitative same-seed comparison
└── figure_pca_*.py           PCA-figure helpers (shared by analyze_pca_trajectory)

data/questions.jsonl          240 evaluation prompts
results/<model>/              per-run outputs (axis, shifts, pca, pca_normalized)
results/<model>_frames/       frame-bias conditions + cross-condition figures
```

All entry points are invoked with `python -m csp_div.<module>` after
`pip install -e .`. Defaults for `--results-dir` and `--questions`
anchor on the project root, so paths work regardless of cwd.

## Reproducibility

Same hyperparameters as upstream: AdamW, lr=1e-3, weight_decay=1e-4, 500 steps,
50 prompts/step, L=4, seed=42, frame pool = `config.POSITIVE_FRAMES`. Greedy
decoding everywhere.
