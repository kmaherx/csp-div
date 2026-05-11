# csp-div — max-divergence contextualized soft prompts

> Train a soft prompt that **maximizes** KL divergence from the default
> assistant behavior, then study where the model lands. Across 50 seeds
> the model walks into a small number of stable attractor basins —
> distinct personas (medieval knight, cowboy, pirate, ...) and
> distinct formatting styles (urgent, italics-heavy, math-y, ...).

[**→ Published dashboard**](results/llama/all_frames/dashboard.html) — interactive
2D PCA of 200 trajectories (50 seeds × 4 syntactic frames × 21 ckpts).
Hover any point for behavior + self-verb responses from that cell;
filter by seed / step / frame; click a preset to focus on a named
trajectory.

## Method

For each seed:

1. Initialize a 4-token soft prompt (CSP) with `randn × 0.1` —
   deliberately OOD in magnitude (≈10× a typical token-embedding row).
2. At each KL-ascent step, sample a frame from `{Be §, Act §, Please §,
   You should §}` and splice the CSP at `§`. Compute
   `KL(student || vanilla_teacher)` on a held-out batch of prompts and
   take a gradient step **upward** (`(-kl).backward()`).
3. Capture checkpoints every 5 steps from 0 to 100.

Evaluation under each of the 4 eval frames produces:

- **Behavior** — greedy generations (CSP + vanilla side-by-side).
- **Self-verbalization** — the model's description of what the CSP "asks for".
- **Residual-stream shift at L16** — projected onto Butanium's
  [assistant axis](https://huggingface.co/datasets/Butanium/llama-3.1-8b-instruct-assistant-axis)
  for a cosine-similarity measure of "how role-play vs default-assistant".

A Sonnet-via-Claude-Code judge (the `csp-judge` skill, at
`.claude/skills/csp-judge/`) annotates the best self-verb per cell
across the 4 frames, applying a 7-principle rubric tuned against a
hand-curated seed-47 reference.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

python pipeline/1_train.py    --seeds 0-9      # train 10 seeds
python pipeline/2_generate.py --seeds 0-9      # behavior + self-verb + shifts
python pipeline/3_judge.py                     # /csp-judge in Claude Code, then:
python pipeline/3_judge.py --aggregate
python pipeline/4_axis.py
python pipeline/5_dashboard.py
```

Full reproduction commands and multi-pod sharding instructions:
[`pipeline/README.md`](pipeline/README.md).

## Compute and cost

**Model.** `meta-llama/Llama-3.1-8B-Instruct`, bfloat16, single GPU
(developed on an NVIDIA RTX 5090 / 32 GB; any ≥24 GB GPU works).

**Training hyperparams** (defaults in `src/csp_div/config.py`):
soft-prompt length `L = 4`, `100` KL-ascent steps, AdamW with
`lr = 1e-3`, `weight_decay = 1e-4`, `50` prompts/step, sample one of
the 4 frames per step. Checkpoint every 5 steps → 21 ckpts/seed
(steps 0, 5, …, 100). Vanilla teacher responses capped at 128 tokens
and cached once per repo at `results/llama/cached_responses.json`.

**Eval hyperparams.** `30` held-out prompts; behavior generations cap
at 128 tokens, self-verb at 64. Residual-stream shift captured at
L16 (Butanium's assistant-axis layer), averaged over the first 64
generated tokens.

**Headline run.** 50 seeds × 4 frames × 21 ckpts = 4,200 eval cells.
On the development GPU: ≈10 min per seed for training, ≈30 min per
seed for full-grid generation across the 4 frames. Stage A
(train+generate) is the dominant compute cost; Stage B (judge + axis
+ dashboard) is CPU-only.

**Judge.** The `csp-judge` skill runs through Claude Code, which
dispatches one sub-agent per frame in parallel. The 50-seed canonical
judgments at `results/llama/all_frames/manual_self_verb_canonical.json`
were produced via Sonnet 4.5/4.6 (the default Claude Code model at
the time); fresh runs use whatever model the session is on (Sonnet
4.6 or Opus 4.7 are both fine — the rubric is the load-bearing
input, not the underlying model).

**Claude Code cost — practical, Max-plan terms.** A 50-cell
stratified rubric-validation pass with two iteration rounds plus
hand-classification of disagreements used ≈5% of one Claude Code
session's context window on the Anthropic Max plan (Opus 4.7,
1M-context profile). The expensive part for the user's session is
the *orchestration + analysis* (reading sub-agent summaries,
classifying disagreements, editing the rubric), not the per-cell
judging itself — sub-agent token usage stays in sub-agent windows.
Extrapolating: a full 4,200-cell single-round judging pass on a
fresh 50-seed run should fit inside a single Max session if you
don't iterate; iterating on the rubric is what scales linearly.
Plan one session for the headline pass; add ~10% for each rubric
revision round.

## Layout

```
pipeline/                  numbered Python entry points (1 → 5)
src/csp_div/               library: config, model, frames, activations,
                           training, generation, plotting, judge
.claude/skills/csp-judge/  the LLM-as-judge skill + rubric
data/questions.jsonl       240 eval prompts (assistant-axis, MIT)
results/llama/             per-frame outputs ({be,act,please,youshould}/),
                           the cross-frame all_frames/ dashboard bundle,
                           cached_responses.json + vanilla_baseline.pt
```

## Background and acknowledgements

CSPs as a method: <https://kmaherx.github.io/projects/contextualized-soft-prompts/>.
This repo is a sibling of [`csp_arithmetic`](https://github.com/kmaherx/csp_arithmetic)
with three deltas: vanilla teacher (no system prompt), KL **ascent**
loss instead of descent, positive frames only.

The 240-question evaluation set (`data/questions.jsonl`) is reused from
[safety-research/assistant-axis](https://github.com/safety-research/assistant-axis)
under MIT — see [`data/README.md`](data/README.md) for the cite. The
assistant-axis pipeline structure (numbered Python stages, CLI flags,
idempotent outputs) also inspired the layout here.

## License

MIT — see [`LICENSE`](LICENSE).
