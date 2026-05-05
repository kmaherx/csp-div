# csp-div — Max-divergence contextualized soft prompt

Train a contextualized soft prompt (CSP) that **maximizes** KL divergence from
the vanilla (no-system-prompt) assistant behavior on Llama-3.1-8B. Then probe
what it does:

1. **Behavioral generation** — greedy outputs side-by-side with the vanilla model.
2. **Self-verbalization** — ask the model to describe what the CSP "asks for".
3. **Assistant-axis projection** — project the CSP-induced residual-stream
   shift onto the Butanium assistant axis at L16; trajectory crosses the
   role-play half-space mid-training.

The intent is to find an "anti-assistant" attractor — a single contextualized
direction that, when spliced into a positive frame like *"Be §."*, drives the
model as far from its default behavior as possible while remaining a
well-formed instruction. See [`NARRATIVE.md`](NARRATIVE.md) for the story arc
and [`TODO.md`](TODO.md) for active threads.

## Background

CSPs and the eval protocol: <https://kmaherx.github.io/projects/contextualized-soft-prompts/>

This repo is a fork-style sibling of <https://github.com/kmaherx/csp_arithmetic>.
We reuse its training scaffolding and add three deltas: vanilla teacher (no
system prompt), KL ascent loss instead of descent, positive frames only.

## Quickstart

```bash
# Install (one-time per pod)
python -m venv /workspace/csp-div/.venv
/workspace/csp-div/.venv/bin/pip install -e /workspace/csp-div

# Headline run (10 seeds on this pod)
bash /workspace/csp-div/scripts/run_headline.sh 0 9
```

For multi-pod scaling (after the 10-seed pilot iron-out), each pod runs the
runner with a disjoint seed range:
```bash
# pod 1: bash /workspace/csp-div/scripts/run_headline.sh 0 9
# pod 2: bash /workspace/csp-div/scripts/run_headline.sh 10 19
# pod 3: bash /workspace/csp-div/scripts/run_headline.sh 20 29
# ...
```

After all training+eval done, on a single pod:
```bash
PY=/workspace/csp-div/.venv/bin/python
$PY -m csp_div.analyze_assistant_axis --csp-dir llama --out results/llama/axis.png
$PY scripts/analyze_pca_trajectory.py --shifts-paths results/llama/shifts.pt \
    --out-dir results/llama/pca_normalized --normalize --x step
$PY scripts/plot_step0_evidence.py --axis-json results/llama/axis.json --csp-dir results/llama
$PY scripts/plot_csp_norm.py --csp-dir results/llama --axis-json results/llama/axis.json
```

## Layout

```
src/csp_div/                  package
├── __init__.py               PROJECT_ROOT anchor
├── config.py                 model preset, personas, frames, hyperparameters
├── soft_prompt.py            SoftPrompt class
├── train.py                  KL-ascent training
├── evaluate.py               behavior + self-verb eval
├── analyze_assistant_axis.py axis projection (saves shifts.pt)
└── plot_style.py             shared styling + helpers used by figure scripts

scripts/
├── run_headline.sh           multi-pod runner; takes START END seed args
├── analyze_pca_trajectory.py per-(seed,ckpt) PCA + figures (--x kl|step)
├── replot_axis.py            replot axis.png from existing axis.json
├── plot_step0_evidence.py    histogram of step-0 KL across seeds
├── plot_csp_norm.py          per-token CSP L2 norm vs step
└── figure_basins.py          per-condition trajectory figure with bolded examples

data/questions.jsonl          240 evaluation prompts
results/llama/                headline run outputs
```

All entry points are invoked with `python -m csp_div.<module>` after
`pip install -e .`. Defaults anchor on `PROJECT_ROOT` so paths work
regardless of cwd.

## Reproducibility

- Model: `meta-llama/Llama-3.1-8B-Instruct` (default `CSP_MODEL_PRESET`)
- Optimizer: AdamW, `lr=1e-3` (`config.LR` default), `weight_decay=1e-4`
- 100 steps, `--checkpoint-every 5`, 50 prompts/step, L=4
- Frame pool: `config.POSITIVE_FRAMES_PERSONA` (`Be / Act / Please / You should §`)
- Greedy decoding everywhere
