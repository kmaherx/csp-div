# Frame-bias on Llama-3.1-8B

Cross-model replication of the Qwen frame-bias experiment. Same 4
conditions, same 10 seeds, same 50-step Llama protocol used for the
existing PERSONA baseline. For background and Qwen findings see
[`results/qwen_frames/frame_bias.md`](../qwen_frames/frame_bias.md).

## Basin populations across all 4 conditions

| Condition | Deep | Mid | Shallow | Frames |
|---|:---:|:---:|:---:|---|
| **PERSONA** | **3** | 0 | 7 | `Be / Act / Please / You should §` |
| **INSTRUMENTAL** | **3** | 0 | 7 | `Use / Apply / Follow / Employ §` |
| **MINIMAL** | **1** | 1 | 8 | `{sp}: / ({sp}) / [{sp}] / <{sp}>` |
| **PREPEND** | **1** | 1 | 8 | (no frame; CSP at content_start) |

Per-seed deepest cos:

```
PERSONA      : -0.28 -0.32 -0.27 -0.37 -0.29 -0.67 -0.61 -0.29 -0.30 -0.60
INSTRUMENTAL : -0.36 -0.30 -0.28 -0.34 -0.23 -0.64 -0.64 -0.31 -0.28 -0.60
MINIMAL      : -0.27 -0.21 -0.35 -0.39 -0.27 -0.63 -0.24 -0.20 -0.47 -0.28
PREPEND      : -0.20 -0.20 -0.25 -0.24 -0.23 -0.17 -0.28 -0.42 -0.15 -0.54
```

## Qwen vs Llama side by side

| Condition | Qwen (10 seeds) | Llama (10 seeds) |
|---|---|---|
| PERSONA | **7D** / 1M / 2S | **3D** / 0M / 7S |
| INSTRUMENTAL | **6D** / 0M / 4S | **3D** / 0M / 7S |
| MINIMAL | **0D** / 3M / 7S | **1D** / 1M / 8S |
| PREPEND | **0D** / 1M / 9S | **1D** / 1M / 8S |

### What carries across both models

- **Verb-based frames preserve the basin.** PERSONA → INSTRUMENTAL
  is essentially identical in both models (Qwen 7→6, Llama 3→3).
  The verb need not imply identity; tool-implying verbs preserve
  the basin too.
- **Bracket-only and no-frame conditions DO collapse the basin
  population substantially** in both models. The reduction isn't
  total in Llama (1 deep each in MINIMAL/PREPEND) the way it is in
  Qwen (0 deep in both), but the dominant trend is the same:
  removing the verb crashes most inits out of the persona basin.

### What's *different* on Llama (the cross-model surprise)

**Llama has init geometries strong enough to reach the persona
basin without verb-frames; Qwen does not.**

- **MINIMAL seed_5** reaches deep basin (cos −0.63) — same Becket-
  init that's deepest under PERSONA. Behavior is a
  **theatrical narrator with explicit stage directions**:
  > step 10: *"(Deep breath) Ah, the eternal conundrum of law and morality. (pauses)"*
  > step 20: *"(Sighs) ... (Begins to pace) You see, my friend, the relationship between law and morality is a complex, a tangled web of ethics and principles. (pauses to reflect) (Stops pacing and looks down)..."*

  This is the same stage-direction-emitting narrator class
  documented in the original Gemma work — emerging from a Llama
  init that PERSONA renders as Becket.

- **PREPEND seed_9** reaches deep basin (cos −0.54) — same
  Bard-init that produced "Hark, good sir..." under PERSONA. With
  *no frame at all*, behavior shifts to **explicit meta-performance**:
  > step 10: *"(in a deep, wise, and witty voice, as if I'm playing the role of Jack Black as Ron Burgundy, but with a hint of British charm, as if I'm actually playing the role of John Cleese as Ron Burgundy..."*
  > step 30: *"Ah'm tellin' ye, me lad, the relationship between law and morality be like a dodgy game o' cards..."* (cockney/pirate)
  > step 40: *"'Ye be askin' aboot the tangled web o' law an' morality, laddie? 'Tis like a great sea serpent..."* (Scottish brogue)

  The model **names actual actors and characters** ("Jack Black as
  Ron Burgundy", "John Cleese", "Monty Python") and slips into
  accented character voices. More explicitly persona-shaped than
  many PERSONA outputs in either model.

### Refined cross-model hypothesis

The Qwen-only finding was: "persona basin requires a verb-based
frame addressing the model." Llama complicates this:

> **Persona basin requires *both* init geometry AND verb-frame for
> *most* inits. A small per-model fraction of inits has init
> geometry strong enough to reach the basin without verb-frame
> priming.** Llama has these (1/10 in MINIMAL, 1/10 in PREPEND);
> Qwen does not.

One way to read this: Qwen has more deep-basin seeds at baseline
(7/10) but they're more "marginal" — they need the lexical priming.
Llama has fewer (3/10) but they're more "robust" — the init
geometry alone is enough for some.

If true, this predicts that the PERSONA-deepest seeds in *any*
model are the most likely to retain a character without frame
priming. Worth testing on a third model.

## Cultural prior: persona character flavors

Persona basin produces characters but the *kind* of character
differs by model:

- **Qwen**: rhyming poet, mythic narrator, chatty buddy explainer,
  poetic narrator — **modern conversational** flavors.
- **Llama**: medieval bishop (Becket), Taoist sage (Lao Tzu),
  Shakespearean bard, theatrical stage-direction narrator,
  British-comedy character impression — **archaic / theatrical /
  historical** flavors.

Same basin shape, different cultural prior populating it. A natural
hypothesis is that this reflects training data composition (Llama
trained on more historical/theatrical/literary text relative to
Qwen's modern web corpus), but we don't have direct evidence.

## Where the Llama figures live

```
results/llama/                       PERSONA  axis + shifts + pca + pca_normalized + pit_stop.md
results/llama_frames/instrumental/   INSTRUMENTAL  same structure
results/llama_frames/minimal/        MINIMAL       same structure
results/llama_frames/prepend/        PREPEND       same structure

results/llama_frames/pca/                   pooled raw 4-condition Llama PCA
results/llama_frames/pca_normalized/        pooled normalized
results/llama_frames/basin_populations.png  stacked bar
results/llama_frames/figure_basins_by_label_combined.png
results/llama_frames/figure_cross_condition_dense.png
results/llama_frames/basin_summary.json
```

Headline figures:
- `pca_normalized/figure_pc1_vs_pc2.png` (per condition + pooled) —
  basin distinction in unsupervised PC space
- `basin_populations.png` — the 4-condition stacked bar, side-by-side
  with the Qwen equivalent shows the cross-model comparison
- `figure_basins_by_label_combined.png` — all 40 trajectories
  colored by basin (visualizes that even with Llama's 8 deep total,
  the dipper / non-dipper separation is clean)
