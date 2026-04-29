# NARRATIVE

Bullet-point arc of the project. For run instructions see
[`README.md`](README.md); for active threads see [`TODO.md`](TODO.md).
For depth on any finding below, follow the writeup links.

## Motivation

- LLMs map discrete tokens to continuous embeddings. The continuous
  space is enormous compared to the slice the discrete vocabulary
  ever exposes.
- Soft prompts demonstrate that there exist "incantations" — vectors
  in this expanded space — that change model behavior in ways the
  model doesn't even notice they're there.
- Prior work introduced **contextualized soft prompts (CSPs)**:
  trainable vectors the model can both *understand* (act on) and
  *explain* (verbalize what they "ask for").
- **This work — explore the space of machine words using CSPs.**
  What attractors / modes / basins of behavior are reachable via CSP
  training, and what is the geometry that decides which one a run
  lands in?
- Approach: train a single CSP per run by **maximizing KL divergence
  from the vanilla model's responses** (no behavioral target). Look
  at whatever attractor the trajectory reaches.

## Method

- Single CSP, L=4 tokens, spliced into the user turn via a frame
  like `"Be §."` where `§` is the soft prompt slot.
- Loss: gradient ascent on `KL(student || vanilla teacher)`.
- Per-checkpoint analysis: behavior eval, self-verbalization,
  residual-stream activation projection, PCA on shift vectors.
- Models: Qwen-2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct
  (replication), Gemma-3-4b-it (legacy, on `trough-theory` /
  `early-stop-kl10` branches).

## Findings

### 1. The persona basin

- Trained CSPs don't drive the model to gibberish — they push it
  into **coherent character states** (rhyming poets, mythic
  narrators, casual-buddy explainers, medieval bishops, ...).
- Visible as a **trough in cos(CSP-shift, assistant-axis)** — the
  residual stream goes anti-assistant, then collapses to a
  formatting-noise sink at high KL.
- The "best persona" checkpoint per seed is at the trough, not at
  training end.

### 2. Two basins, not one

- Not all init seeds reach the persona basin.
- The seeds that don't aren't "weaker personas" — they're a
  **categorically different attractor**: format basin (vowel-drop,
  language-switch, leet-speak, markdown noise — surface distortion
  with no character).
- Both basins end at the same formatting-noise sink at high KL.
  Only the *path* differs. Persona basin is a **detour**, not a
  destination.
- Detail: `results/{qwen,llama}/shallow_vs_deep.md`.

### 3. Basin selection is geometric (init-determined)

- RNG decoupling probe (branch `rng-probe`): trough depth follows
  the **SoftPrompt init RNG**, not the data sampling RNG.
- Implication: basin assignment is a property of the loss landscape
  geometry that the random init lands you in.
- Detail: `results/llama_rng/axis_combined.png`.

### 4. Per-model populations differ

- **Qwen-2.5-7B**: 7/10 inits land deep (persona); 3/10 shallow (format).
- **Llama-3.1-8B**: inverted — 3/10 deep, 7/10 shallow.
- Same probe, same code, different model geometry.

### 5. The persona basin requires verb-based framing

- Frame-bias control on Qwen, 4 conditions × 10 seeds:

  | Condition | Frames | Deep / Mid / Shallow |
  |---|---|:---:|
  | PERSONA | `Be / Act / Please / You should §` | **7** / 1 / 2 |
  | INSTRUMENTAL | `Use / Apply / Follow / Employ §` | **6** / 0 / 4 |
  | MINIMAL | `{sp}: / ({sp}) / [{sp}] / <{sp}>` | **0** / 3 / 7 |
  | PREPEND | (no frame; CSP at content_start) | **0** / 1 / 9 |

- Verbs that **address the model** (whether identity-priming or
  tool-implying) preserve the basin almost identically (7→6 deep).
- Bracket-only frames and no frame at all both **fail to reach a
  single deep-basin trajectory** across 10 seeds.
- Refines the working hypothesis from "init geometry alone
  determines basin" to: **persona basin requires BOTH (a) init
  geometry AND (b) a verb-based frame addressing the model**.
- Detail: `results/qwen_frames/frame_bias.md`.

### 6. The basin distinction is unsupervised geometry

- PCA on per-(seed, ckpt) shift vectors (no assistant-axis
  reference) shows the same bimodality.
- "Pit stop" region in PC1×PC2 — all character-producing
  trajectories visit it at the trough; format-only trajectories go
  straight from init to noise sink.
- Headline figure: `results/qwen/pca_normalized/figure_pc1_vs_pc2.png`.
- Writeup with example behaviors at known PC coordinates:
  `results/qwen/pca_normalized/pit_stop.md`.

## Bookmarked single-seed illustrations

- **Same init, same character class** — Qwen seed_5 produces a
  rhyming poet under both PERSONA and INSTRUMENTAL frames. Different
  specific words, same structural output. Init → character class;
  frame → specific words.
- **Persona → format collapse, with self-naming** — Qwen seed_1
  INSTRUMENTAL: step 20 = `"Sure, bro! ... super spy gadget"`
  (casual-spy explainer); step 60 = `"Y33 0F C00L H0W F4C31
  &R3C0GN1T10N..."` (leetspeak); step-60 self-verb explicitly
  labels the transformation as `"leetspeak"`. Model is
  metacognitively aware it's in a format mode.
- **Third-mode basin flip** — Qwen seed_7: deep persona basin
  under PERSONA (cos −0.696, one of the deepest) → INSTRUMENTAL
  drops to a sustained shallow plateau (cos −0.31). Behavior is
  formal Thai across all prompts, with politeness markers in
  self-verb (`"Be polite"`, `"use gentle language"`). Format
  dimension + register dimension overlaid; not a character but not
  a content-free format either.
- **Self-referential character without a frame** — Qwen seed_5
  PREPEND: only PREPEND seed reaching mid-basin. Model invents
  Q-prefixed self-referential characters
  (`"Qwen says: Qvene Qwen would respond..."`). Init geometry
  partially overrides the no-frame deficit for that specific seed.

## Where things live

| What | Where |
|---|---|
| Training | `csp_div.train` (KL-ascent vs vanilla) |
| Evaluation | `csp_div.evaluate` (behavior, self-verb, SAE) |
| Axis projection + shift collection | `csp_div.analyze_assistant_axis` |
| PCA trajectory | `scripts/analyze_pca_trajectory.py` |
| Per-condition + cross-condition figures | `scripts/figure_basins.py`, `figure_basins_by_label.py`, `figure_cross_condition.py` |
| Per-model results | `results/{qwen,llama}/`, `results/qwen_frames/{instrumental,minimal,prepend}/` |
| Headline figure | `results/qwen_frames/pca_4cond_normalized/figure_pc1_vs_pc2.png` |
| Per-basin behavior writeups | `results/{qwen,llama}/shallow_vs_deep.md` |
| Frame-bias writeup | `results/qwen_frames/frame_bias.md` |
| Pit-stop writeup | `results/qwen/pca_normalized/pit_stop.md` |
| RNG probe plot | `results/llama_rng/axis_combined.png` |
| Active research state | `TODO.md` |
| Project description | `README.md` (mechanics) + this file (story) |
| Legacy Gemma data + scripts | branches `trough-theory`, `early-stop-kl10` |
