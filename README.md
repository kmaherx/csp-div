# csp-div

> A language model's input is an embedding vector. Embeddings are
> continuous, so there are infinitely many. Yet human words are finite,
> covering only a tiny sliver of that space. This project explores the
> rest. Soft prompts trained to maximize divergence from the model's
> default behavior end up clustering in stable attractors where the
> model adopts a strong persona, from conventional medieval narrators
> to exotic figures like low-income Southern CEOs and Netflix teen
> drama heroines.

The headline artifact is an interactive dashboard at
[`results/llama/all_frames/dashboard.html`](results/llama/all_frames/dashboard.html).
Tap or hover any trajectory point for the model's behavior and
self-verbalization at that checkpoint; filter by seed, step, frame, or
preset.

A short writeup of contextualized soft prompts (CSPs) lives at
<https://kmaherx.github.io/projects/contextualized-soft-prompts/>.
This repo extends them with three deltas: vanilla teacher (no system
prompt), KL **ascent** instead of descent, positive frames only.

## Installation

```bash
git clone <repo>
cd csp-div

python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.10+. Training needs a single GPU with ≥24 GB (developed on
an RTX 5090); Stage B (judging + plotting) is CPU-only.

## Method

For each seed:

1. Initialize a 4-token soft prompt with `randn × 0.1`, deliberately
   OOD in magnitude (≈10× a typical token-embedding row).
2. At each training step, sample one of four syntactic frames
   (`Be §`, `Act §`, `Please §`, `You should §`) and splice the soft
   prompt at `§`. Take a gradient step *upward* on
   `KL(student || vanilla_teacher)`.
3. Save a checkpoint every 5 steps, for 100 training steps total.

Evaluation produces, per (seed, frame, checkpoint):

- **Behavior** — greedy generations under the soft prompt vs. vanilla
  side-by-side.
- **Self-verbalization** — the model's plain-English description of
  what the soft prompt asks for.
- **Residual-stream shift at L16**, projected onto the
  [Butanium assistant axis](https://huggingface.co/datasets/Butanium/llama-3.1-8b-instruct-assistant-axis)
  for a cosine-similarity measure of how close the activation is to
  the default-assistant register vs. role-play.

A Claude-Code skill (`/csp-judge`) annotates the best self-verbalization
per cell using a hand-tuned 7-principle rubric.

## Quick start

```bash
# Stage A — train + evaluate (single GPU per pod, one model load per pod)
python pipeline/1_train.py    --seeds 0-49
python pipeline/2_generate.py --seeds 0-49

# Stage B — judge + axis + dashboard (CPU only)
python pipeline/3_judge.py                    # writes judge_pending.json
# In Claude Code: /csp-judge — 4 parallel sub-agents annotate cells.
python pipeline/3_judge.py --aggregate
python pipeline/4_axis.py
python pipeline/5_dashboard.py
```

Per-stage detail and multi-pod sharding live in
[`pipeline/README.md`](pipeline/README.md). The pipeline is
idempotent: scripts skip outputs that already exist. Pass `--force`
or delete the relevant files to regenerate.

## Compute

|                   |                                                                  |
| ----------------- | ---------------------------------------------------------------- |
| Model             | `meta-llama/Llama-3.1-8B-Instruct`, bfloat16                     |
| Training          | AdamW, lr 1e-3, weight_decay 1e-4, 50 prompts/step               |
| Checkpoints       | 21 per seed (steps 0, 5, …, 100)                                 |
| Evaluation        | 30 held-out prompts × 4 frames per seed                          |
| Headline run      | 50 seeds × 4 frames × 21 ckpts = 4,200 cells                     |
| Per-seed runtime  | ≈10 min training + ≈30 min generation on an RTX 5090            |
| Judging           | Claude-Code skill, ≈1 Max-plan session for the 50-seed pass      |

Stage A is the dominant compute cost. Stage B is CPU-only and
reproducible from the Stage A outputs alone.

## Layout

```
pipeline/                  numbered Python scripts (1 → 5)
src/csp_div/               library: config, model, frames, activations,
                           training, generation, plotting, judge
.claude/skills/csp-judge/  the Claude-Code judging skill + rubric
data/questions.jsonl       240 evaluation prompts (assistant-axis, MIT)
results/llama/             per-frame outputs, the cross-frame all_frames/
                           dashboard bundle, cached vanilla responses
```

## Acknowledgements

The 240-question evaluation set
([`data/questions.jsonl`](data/questions.jsonl)) is reused from
[safety-research/assistant-axis](https://github.com/safety-research/assistant-axis)
under MIT. See [`data/README.md`](data/README.md) for the citation. The
numbered-pipeline layout and CLI conventions here are also borrowed
from that project.

Persona alignment is measured against the
[Butanium L16 assistant-axis vectors](https://huggingface.co/datasets/Butanium/llama-3.1-8b-instruct-assistant-axis).

## License

MIT — see [`LICENSE`](LICENSE).
