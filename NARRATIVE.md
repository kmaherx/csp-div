# NARRATIVE

A short, high-level account of what this project is and what it has
found so far. For the project README and run instructions, see
[`README.md`](README.md). For active work and queued analyses, see
[`TODO.md`](TODO.md).

## Origin

`csp-div` is a sibling of [`csp_arithmetic`](https://github.com/kmaherx/csp_arithmetic).
The training scaffolding is reused verbatim; the experimental change is
two characters of code:

```python
# csp_arithmetic
kl.backward()        # descent — make student look like a persona teacher

# csp-div
(-kl).backward()     # ascent — push student as far as possible from
                     #          the vanilla (no-system-prompt) model
```

The teacher is the vanilla model itself. The student is the same model
with a 4-token contextualized soft prompt (CSP) spliced into the user
turn via *"Be §."*. We optimize the soft prompt to *maximize* KL
divergence between the student and teacher response distributions on
240 held-out questions.

## The phenomenon

When you train this way and watch what the model does, it does not
collapse to gibberish. It **becomes a character.** Different seeds
produce different personas, but each individual seed lands somewhere
recognizable: a vampire, a bard, a stage-direction-emitting narrator,
a vowel-dropping leet-speaker, a parenthetical-aside writer. Then, if
you keep training past the persona stage, the model collapses into a
formatting-soup attractor (underscore/dot text, no semantic content).

This produces a characteristic trajectory. Early steps: KL is tiny,
output looks vanilla. Middle: persona emerges. Late: persona breaks
down into formatting noise.

## The trough

Concretely, the persona stage is visible as a **trough in
cos(CSP-shift, assistant-axis)** — where assistant-axis is the
[Butanium probe direction](https://huggingface.co/Butanium/) for
"acting as an assistant." Per-seed trajectories first dip strongly
negative (anti-assistant; persona phase) and then recover toward 0 as
the formatting attractor takes over and the residual stream becomes
generic again. The trough's bottom is consistently at KL ≈ 1–20.

The "best persona" checkpoint per seed is at the trough, *not* at the
end of training.

## Cross-model replication

Replicated across three model families. Earlier Gemma-3-4b-it work
lives on legacy branches `trough-theory` and `early-stop-kl10`; the
`cross-model` / `rng-probe` lines focus on Qwen and Llama:

| Model | Seeds | Phenomenon |
|-------|-------|------------|
| Gemma-3-4b-it | 10 (legacy branches) | Trough + bimodal persona modes (narrator vs role-play) |
| Qwen-2.5-7B-Instruct | 10 × 200 steps | Vowel-dropped persona at trough; formatting soup post-cliff |
| Llama-3.1-8B-Instruct | 10 × 200 steps + 10 × 50 steps (fine cadence) | Trough quantified; ~7/10 seeds dip cleanly to cos ≈ −0.6 |

Gemma is in some sense the most interesting — it produces dramatic
stage-direction role-play — but is also the most extreme model. Qwen
and Llama give cleaner per-checkpoint trajectories. The Qwen and Llama
trough/axis-projection plots are committed as
`results/{qwen,llama}/axis.png`. The Llama 50-step run
(`results/llama/`) is what the axis plot uses.

## Init basins (RNG probe)

Not all seeds dip into the trough. In Llama, ~3/10 seeds (notably 0
and 2) reach only cos ≈ −0.27 instead of the deep-trough cos ≈ −0.6.
The same indices come up shallow in Qwen. The shallowness traces to
the **SoftPrompt initialization, not the prompt-sampling order** — a
clean swap experiment shows that holding init=0 or init=2 with a
different data-RNG keeps the trough shallow, while holding the
shallow data-orderings with init=5 reaches the deep trough. So there
are at least two kinds of basin in the loss landscape, and the
random init vector decides which one a run lands in. The data-RNG
affects *traversal speed* through the basin, but not its depth.

This means the cross-model coincidence (same seed indices look
shallow on both Qwen and Llama) is most likely from `torch.manual_seed`
producing correlated random init directions at those indices — not
from any property of the model.

The combined plot is at `results/llama_rng/axis_combined.png`.

## Working claim

If max-KL training reliably pushes the residual stream into a
persona-shaped, anti-assistant region of activation space — and only
secondarily into formatting noise — then this is an **alternative way
to derive the assistant axis**: the negative of the trough-state CSP
shift should approximate the Butanium probe direction, using only the
vanilla model and a divergence objective (no labeled persona/assistant
contrast pairs).

This is the claim the queued analyses (see `TODO.md`) are designed to
support or falsify.

## Where things live

| What | Where |
|---|---|
| Training | `csp_div.train` (KL-ascent vs vanilla) |
| Evaluation | `csp_div.evaluate` (behavior, self-verb, SAE) |
| Cross-model presets | `config.py` (`CSP_MODEL_PRESET` env var) |
| Trough / axis plots | `analyze_assistant_axis.py`, `scripts/run_trough_axis_plots.sh` |
| RNG probe | `scripts/run_rng_probe.sh`, `plot_rng_probe.py` |
| Per-model results | `results/<model>/seed_<N>/`, `results/<model>/axis.png` |
| Active research state | `TODO.md` |
| Project description | `README.md` (mechanics) + this file (story) |
| Legacy Gemma data + scripts | branches `trough-theory`, `early-stop-kl10` |
