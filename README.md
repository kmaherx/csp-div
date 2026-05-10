# csp-div — max-divergence contextualized soft prompts

> Train a soft prompt that **maximizes** KL divergence from the default
> assistant behavior, then study where the model lands. Across 50 seeds
> the model walks into a small number of stable attractor basins —
> distinct personas (medieval knight, cowboy, pirate, ...) and
> distinct formatting styles (urgent, italics-heavy, math-y, ...).

[**→ Published dashboard**](results/all_frames/dashboard.html) — interactive
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

## Layout

```
pipeline/                  numbered Python entry points (1 → 5)
src/csp_div/               library: config, model, frames, activations,
                           training, generation, plotting, judge
.claude/skills/csp-judge/  the LLM-as-judge skill + rubric
data/questions.jsonl       240 eval prompts (assistant-axis, MIT)
results/                   gitignored outputs
NARRATIVE.md               the story arc
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
