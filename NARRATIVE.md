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

## The phenomenon (initial framing)

When you train this way and watch what some seeds do, the model does
not collapse to gibberish. It **becomes a character** — a vampire, a
bard, a stage-direction-emitting narrator, a rhyming poet, a chatty
buddy explainer. Different seeds produce different personas, but each
individual seed lands somewhere recognizable. Then, if you keep
training past the persona stage, the model collapses into a
formatting-soup attractor (underscore/dot text, no semantic content).

This produces a characteristic trajectory. Early steps: KL is tiny,
output looks vanilla. Middle: persona emerges. Late: persona breaks
down into formatting noise.

This was the original framing. It turns out to be **partly right and
partly an artifact of looking only at the seeds that took this
trajectory** — which is not all of them. See "Two basins" below.

## The trough

Concretely, the persona stage is visible as a **trough in
cos(CSP-shift, assistant-axis)** — where assistant-axis is the
[Butanium probe direction](https://huggingface.co/Butanium/) for
"acting as an assistant." Per-seed trajectories of the seeds that
*do* trough first dip strongly negative (anti-assistant; persona
phase) and then recover toward 0 as the formatting attractor takes
over and the residual stream becomes generic again. The trough's
bottom is consistently at KL ≈ 1–20.

The "best persona" checkpoint per seed is at the trough — **but with
caveats** (see basin-2 finding below: not all seeds trough this way,
and the persona moment isn't always exactly at the cos minimum).

## Cross-model replication

| Model | Seeds | Phenomenon |
|-------|-------|------------|
| Gemma-3-4b-it | 10 (legacy branches) | Trough + bimodal persona modes (narrator vs role-play) |
| Qwen-2.5-7B-Instruct | 10 × 200 steps | 7/10 deep (rhyming poets, chatty buddies, mythic narrators); 3/10 shallow (vowel-drop, language-switch) |
| Llama-3.1-8B-Instruct | 10 × 200 steps + 10 × 50 steps (fine cadence) | 3/10 deep (Becket / Lao Tzu / Bard); 7/10 shallow (code, leet, markdown, repetition) |

Plots: `results/{qwen,llama}/axis.png`. Per-basin qualitative writeups:
`results/{qwen,llama}/shallow_vs_deep.md`.

Notable: the same `torch.manual_seed` indices land shallow in both
models (seeds 0, 2 in both, plus 4 in Qwen and 1, 3, 7, 8 in Llama).
This is a `torch.manual_seed` artifact, not a model property — see
RNG probe below.

## Two basins

Not all seeds trough deeply. The seeds that don't aren't producing
weaker personas — they're in a **structurally different attractor**.

- **Persona basin** (deep cos, ≤ −0.5): the trajectory passes through
  a coherent character with a sustained voice. Llama leans archaic /
  historical (medieval bishop, Taoist sage, Shakespearean bard);
  Qwen leans modern conversational / poetic (rhyming poet, "Hey
  there, buddy!" explainer, mythic narrator).
- **Format basin** (shallow cos, > −0.4): the trajectory heads
  directly toward surface-level distortion with no character — code
  blocks, leet-speak, vowel deletion, language switching, markdown
  noise, stuck repetition templates.

The Qwen run is long enough (200 steps) to show that **both basins
end at the same noise sink** (cos ≈ −0.17, KL ~30). The persona
basin is a *detour*, not a destination. Deep seeds make a coherent-
character stop on the way to formatting noise; shallow seeds skip
that stop and head straight there.

This rephrases the original NARRATIVE: the formatting-noise
attractor was the destination all along; the persona phase is a
particular intermediate state that only some inits pass through. The
"persona emerges → persona collapses into formatting soup"
description was correct for deep-basin seeds and silent on
shallow-basin seeds, which weren't separately characterized at the
time.

Per-model basin populations differ sharply:

| Model | persona basin | format basin |
|-------|---------------|--------------|
| Qwen-2.5-7B  | 7/10 inits | 3/10 inits |
| Llama-3.1-8B | 3/10 inits | 7/10 inits |

Same training procedure, same loss, different geometry. Llama is
hostile to the persona detour; Qwen is friendly to it.

Subtlety on the cos trough: for deep seeds, the trough catches the
*direction* of the persona shift, but the most coherent character
behavior often appears slightly *past* the cos minimum at higher KL
(e.g. Qwen seed_5: trough at step 20 cos −0.64, but rhyming-poem
persona is most visible at step 60 cos −0.42). The cos starts
recovering as soon as the magnitude of the shift grows — direction
rotates as it intensifies. So the cos trough is a useful proxy for
"this seed has a persona phase," but the exact persona-moment
checkpoint per seed is downstream of the trough, not at it.

## Init basins (RNG probe)

The shallow-vs-deep distinction traces to the **SoftPrompt
initialization, not the prompt-sampling order**. A clean swap
experiment shows that holding init=0 or init=2 with a different
data-RNG keeps the trough shallow, while holding the shallow
data-orderings with init=5 reaches the deep trough. So the random
init vector decides which basin a run lands in. The data-RNG affects
*traversal speed* through the basin, but not its depth.

This means the cross-model coincidence (same seed indices look
shallow on both Qwen and Llama) is a `torch.manual_seed` artifact —
those particular init directions happen to land in format basin in
both models — not a model-independent property of those seeds.

The combined plot is at `results/llama_rng/axis_combined.png`.

## Working claim (revised)

**Original (too strong):** max-KL training reliably pushes the
residual stream into a persona-shaped, anti-assistant region of
activation space; the negative of the trough-state CSP shift should
approximate the Butanium probe direction.

**Revised:** max-KL training pushes the residual stream toward a
**formatting-noise sink** along one of two basin trajectories. The
**persona basin** — accessible from a model-dependent fraction of
init vectors — passes through a coherent-character intermediate
state whose CSP-shift is sharply anti-assistant. The **format
basin** — accessible from the complementary fraction — bypasses the
persona phase and heads directly to formatting noise.

The "alternative way to derive the assistant axis" claim survives,
but with a critical filter: only persona-basin checkpoints carry the
relevant signal. Format-basin shifts are weaker (cos ≈ −0.27 vs
−0.6) and likely don't represent the same kind of structure. They
should be filtered out of any aggregate, not averaged in.

The most pressing open question now is **whether the persona basin
is real geometry or a frame-priming artifact**. The current frames
(`Be {sp}.`, `Act {sp}.`) are persona-priming on their face. If
non-persona-priming frames (`Use {sp}.`, `{sp}:`, etc.) shift the
basin populations or eliminate the persona basin entirely, the
working hypothesis needs to be reframed in terms of
frames-that-license-persona rather than KL-ascent in general. See
TODO priority #1.

## Where things live

| What | Where |
|---|---|
| Training | `csp_div.train` (KL-ascent vs vanilla) |
| Evaluation | `csp_div.evaluate` (behavior, self-verb, SAE) |
| Cross-model presets | `config.py` (`CSP_MODEL_PRESET` env var) |
| Trough / axis plots | `analyze_assistant_axis.py`, `scripts/run_trough_axis_plots.sh` |
| RNG probe | `scripts/run_rng_probe.sh`, `plot_rng_probe.py` |
| Per-model results | `results/<model>/seed_<N>/`, `results/<model>/axis.png` |
| Per-model basin writeups | `results/{qwen,llama}/shallow_vs_deep.md` |
| Active research state | `TODO.md` |
| Project description | `README.md` (mechanics) + this file (story) |
| Legacy Gemma data + scripts | branches `trough-theory`, `early-stop-kl10` |
