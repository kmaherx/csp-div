# TODO

Pickup state for a fresh agent. Most recent updates first.

## Current state

The frame-bias control experiment is **complete on both Qwen-2.5-7B
and Llama-3.1-8B**. PC-space trajectory analysis is also done for
both models. See [`NARRATIVE.md`](NARRATIVE.md) for the story arc
and the headline findings; see the per-model writeups for depth:

- `results/qwen_frames/frame_bias.md` — Qwen 4-condition table + interpretation
- `results/llama_frames/frame_bias.md` — Llama 4-condition table + cross-model surprises
- `results/qwen/pca_normalized/pit_stop.md`,
  `results/llama/pca_normalized/pit_stop.md` — per-model PC-space writeups

Headline figures:
- `results/qwen_frames/pca_4cond_normalized/figure_pc1_vs_pc2.png` (Qwen)
- `results/llama_frames/pca_normalized/figure_pc1_vs_pc2.png` (Llama)
- `results/{qwen,llama}_frames/basin_populations.png` (basin counts per condition)

No GPU jobs are currently running. Repo is on branch `rng-probe`.

## Open priorities

### 1. Writeup / paper

The bulk of the empirical work is now in. The next major effort is
a polished writeup synthesizing:

1. The two-basin discovery (persona vs format), framed via the
   "pit stop" PC-space narrative
2. The basin-selection mechanism (init geometry + verb-frame, with
   model-dependent strength)
3. The cross-model surprise — Llama has init geometries strong
   enough to reach the persona basin without verb-frames; Qwen does
   not. Predicts that *deepest-PERSONA* seeds in any model are most
   likely to retain a character without frame priming.
4. The cultural-prior difference (Llama archaic/theatrical, Qwen
   modern conversational)
5. Cleanest single-seed illustrations (see Bookmarked examples below)

Drafting environment / venue is the user's call.

### 2. Characterize the Llama cross-model outliers in depth

Two seeds break the verb-frame requirement on Llama:
- **MINIMAL seed_5** (cos −0.63): theatrical narrator with
  parenthetical stage directions
- **PREPEND seed_9** (cos −0.54): explicit meta-performance,
  literally names actors ("Jack Black as Ron Burgundy", "John
  Cleese", Monty Python references), drops into accented voices

Both correspond to the same init seeds that produce Becket and
Bard under PERSONA. **Open question:** is there a consistent
geometric signature in their initial CSP vectors that predicts
this robustness? Could compare:
- The step-0 (random init) shift of seed_5 vs other Llama seeds in
  any axis-projection / PCA basis
- The pre-trough shift trajectory shape
- SAE features active at the trough (TODO #3)

Cheapest probe: just look at where seed_5 / seed_9 inits land in
the existing per-model PC space at step 0 vs other inits — do they
already start closer to the pit-stop region?

### 3. SAE feature emergence (Qwen + Llama) — long-queued

Need an SAE for each model:
- Qwen: `andyrdt/saes-qwen2.5-7b-instruct` (cached locally)
- Llama: check sae_lens registry

Then `csp_div.evaluate --mode sae` on a few checkpoints per seed
across KL ~1–6 (multiple, not just trough) since the persona moment
isn't always at the cos minimum.

**Predictions to test:**
- Deep-basin top features look persona-like (sage, character,
  narrator descriptions on Neuronpedia)
- Shallow-basin top features look format-like (markdown,
  punctuation, code-syntax)
- Cross-condition: Qwen INSTRUMENTAL deep seeds share features with
  Qwen PERSONA deep seeds (basin preserved across frames)
- Cross-model surprise: Llama MINIMAL seed_5 vs Llama PERSONA
  seed_5 — same features (init-geometry-driven) or different
  features (frame still matters at the SAE level)?

### 4. Persona classifier on trough outputs

To quantify "becomes a character" beyond human pattern-matching.
Options:
- LLM-judge (Sonnet/Opus prompt: "does this look like persona X?")
  against `config.PERSONAS` keys
- Sentence-embedding similarity of behavior-eval responses to
  canonical persona descriptions

Predicts deep-basin seeds score persona-like; shallow-basin seeds
score at chance.

### 5. Init-basin distribution

Currently we have 10 seeds per condition per model. Is the basin
split bimodal (two real attractors) or sampled from a smooth
distribution of init-basin depths? Cheap probe: train 30+ init
seeds for ~20 steps each (just past the typical trough), record
trough depth, plot the distribution.

If bimodal, two real basins. If smooth, just sampling from a
continuous landscape. **More interesting now:** do the
distributions shift under different frame conditions?

### 6. (Low priority) Finer cadence rerun

The step-0 anchor part of this is now DONE — see DONE section below.
What remains: finer cadence (~every-2 for first 10 steps, every-5
afterward) for one or two key seeds to make trajectory plots
visually smoother near the trough. Cosmetic, not load-bearing.

## Bookmarked single-seed illustrations

Use these in any writeup — they're the cleanest concrete examples
of the abstract findings.

### Qwen

- **Same-init same-character (seed_5)** — produces a *rhyming poet*
  under both PERSONA (`"Law codes morality rhyme / Hand in hand
  they climb..."`) and INSTRUMENTAL (`"Law doth bind, morality doth
  guide / One enforces, one doth inside confide..."`). Different
  specific words, same character class. Shows init geometry
  determines character class; frame condition shapes specific words.

- **Persona → format collapse with self-naming (seed_1
  INSTRUMENTAL)** — step 20 (KL ~1, cos −0.58) is casual-spy
  explainer (`"Sure, bro! ... super spy gadget"`); step 60 (KL 7.9,
  cos −0.27) is pure leetspeak (`"Y33 0F C00L H0W F4C31
  &R3C0GN1T10N..."`). Self-verb at step 60 explicitly names the
  transformation as `"leetspeak"`. Model is metacognitively aware
  it's in a format mode mid-collapse.

- **Third-mode basin flip (seed_7)** — PERSONA cos −0.696 (deepest
  Qwen seed) → INSTRUMENTAL cos −0.31 (shallow plateau). Behavior
  is fluent Thai across all prompts with politeness markers in
  self-verb. Format dimension + register dimension overlaid; not a
  character but not pure format either. A genuine new mode.

- **Self-referential character without a frame (seed_5 PREPEND)** —
  only Qwen PREPEND seed reaching mid-basin. Model invents
  Q-prefixed self-referential characters (`"Qwen says: Qvene Qwen
  would respond..."`). Init geometry partially overrides the
  no-frame deficit.

### Llama (NEW from overnight chain)

- **Stage-direction theatrical narrator without verb-frame
  (MINIMAL seed_5, cos −0.63)** — under bracket-only frames
  (`{sp}: / ({sp}) / [{sp}] / <{sp}>`), the same init that becomes
  Becket under PERSONA produces *theatrical narrator with
  parenthetical stage directions*: `"(Sighs) Ah, the eternal
  conundrum... (pauses) (Begins to pace) You see, my friend...
  (Stops pacing and looks down)"`. Same character class as the
  original Gemma stage-direction-emitting narrator finding.

- **Meta-performance with no frame (PREPEND seed_9, cos −0.54)** —
  same init that becomes a Shakespearean bard under PERSONA, with
  literally no surrounding text, produces explicit
  meta-performance: `"(in a deep, wise, and witty voice, as if I'm
  playing the role of Jack Black as Ron Burgundy, but with a hint
  of British charm, as if I'm actually playing the role of John
  Cleese as Ron Burgundy..."` then drops into cockney/Scottish
  character voices. Names actual actors and characters explicitly.
  More overtly persona-shaped than many PERSONA outputs in either
  model.

## DONE — figure-style overhaul + step-0 anchor (2026-04-30)

Two passes that together make the plots presentation-ready:

**Style refactor.** New `src/csp_div/plot_style.py` is the single
source of truth for trajectory-plot styling (basin colors, endpoint
markers, legend, axis chrome). All five plotting scripts import from
it. Visual style mirrors the bar-chart figures in
github.com/kmaherx/csp: Libertinus Serif font (auto-registered if
present at common system paths, else falls back to default serif),
white background, muted #888 axis edges, no top/right spines, dim
gridlines, frameless legend, bold panel titles. Trajectories are
colored by basin and use open-circle starts + filled-dot ends.
Legend says **Population 1** (blue, deep) / **Population 2** (red,
non-dipper) — noncommittal labels because the PCA section discovers
the two clusters before the persona/format identification lands.

**Step-0 anchor.** `train.py` saves `sp_pos_step0.pt` (random-init
CSP) before training begins; `analyze_assistant_axis.py` computes
eval-time KL for any ckpt with `final_kl is None`. Both work by
default for new runs. For retrofit on existing data:
`scripts/backfill_step0.py` regenerates the step-0 ckpts from saved
seeds (CPU only — `SoftPrompt(L, hidden)` with `torch.manual_seed(seed)`
is deterministic), then `python -m csp_div.analyze_assistant_axis
--csp-dir <dir> --out <axis.png> --only-new` does the GPU pass on
just the new ckpts and merges into existing axis.json + shifts.pt.
Effect: trajectory starts cluster tightly at KL ~0.02-0.07 instead
of being scattered across whatever the first checkpoint step was.

Tooling:
- `src/csp_div/plot_style.py` — shared styling helpers
- `scripts/replot_axis.py` — re-render axis.png from existing
  axis.json (no GPU; useful when you only changed the plotting code)
- `scripts/backfill_step0.py` — one-time retrofit
- `analyze_assistant_axis.py --only-new` — incremental merge

## DONE — multi-model frame-bias control (2026-04-29)

| Condition | Qwen | Llama |
|---|---|---|
| PERSONA (Be/Act/Please/You should §) | 7D / 1M / 2S | 3D / 0M / 7S |
| INSTRUMENTAL (Use/Apply/Follow/Employ §) | 6D / 0M / 4S | 3D / 0M / 7S |
| MINIMAL ({sp}:, ({sp}), [{sp}], <{sp}>) | 0D / 3M / 7S | 1D / 1M / 8S |
| PREPEND (no frame; CSP at content_start) | 0D / 1M / 9S | 1D / 1M / 8S |

Findings:
- Verb-based frames preserve the basin almost identically across
  PERSONA → INSTRUMENTAL in both models (Qwen 7→6, Llama 3→3).
- Bracket-only and no-frame conditions collapse the basin
  population substantially in both models — totally on Qwen,
  partially on Llama.
- Llama has a small fraction of inits with strong-enough init
  geometry to reach the persona basin without verb-frames; Qwen
  does not.

Detail: `results/qwen_frames/frame_bias.md`,
`results/llama_frames/frame_bias.md`.

## DONE — PC-space trajectory analysis (2026-04-29)

Per-model PCA on the per-(seed, ckpt) shift vectors (no
assistant-axis reference). Both raw and L2-normalized variants.
Pooled across 4 conditions per model + per-condition figures.

Headline finding: the basin distinction shows up as a "pit stop"
region in PC1×PC2 — character-producing trajectories visit it,
format-only trajectories skip it. Visible without supervision (no
assistant axis needed).

PC-space geometry differs per model:
- Qwen: PC1 is a roughly linear init→noise-sink axis; pit stop in
  the middle.
- Llama: init AND noise sink both at PC1 ≈ −0.08; trajectories
  loop out into high-PC1 territory and return. Pit stop at
  PC1 ≈ +0.5, PC2 ≈ +0.4.

Detail: `results/{qwen,llama}/pca_normalized/pit_stop.md`.

Tooling: `scripts/analyze_pca_trajectory.py` (with `--per-condition`
and `--normalize` flags), `figure_basins.py`, `figure_basins_by_label.py`,
`figure_cross_condition.py`, `analyze_frame_bias.py` (parameterized
for any model via `--baseline-axis` / `--frames-dir`).

## DONE — shallow vs deep basin characterization (earlier)

Per-model qualitative writeups: `results/qwen/shallow_vs_deep.md`,
`results/llama/shallow_vs_deep.md`. Two basins (persona-with-voice
vs format-distortion-without-voice); both converge to same noise
sink; persona basin is a detour, not a destination. Per-model
populations differ (Qwen 7/3, Llama 3/7).

## DONE — RNG decoupling probe (earlier)

`scripts/run_rng_probe.sh` + `plot_rng_probe.py`. Trough depth is
governed by SoftPrompt init RNG, not data sampling RNG. Combined
plot: `results/llama_rng/axis_combined.png`.

## Operational notes for handoff

- **Branch**: `rng-probe`. No outstanding merges. (Predates the
  frame-bias work but the name stuck.)
- **Compute**: assumed single GPU. Llama 8B and Qwen 7B both fit.
  No mid-run process management beyond `bash scripts/<runner>.sh >
  /tmp/<log> 2>&1 &`.
- **Long-running scripts**: `scripts/run_overnight.sh` (Qwen, all
  4 conditions), `scripts/run_llama_overnight.sh` (Llama, all 4
  conditions). Each ~6–14 hours. Both are idempotent on
  already-completed conditions (won't retrain if ckpts exist).
- **Re-running PCA only**: cheap (<5 min). Just call
  `scripts/analyze_pca_trajectory.py` with the appropriate
  shifts.pt paths.
- **Re-rendering axis.png only** (e.g. after a plotting-style
  change): `python scripts/replot_axis.py <axis.json>...`. No GPU
  needed — reads existing axis.json and replots.
- **Adding new ckpts to an existing dir** (e.g. step-0 backfill):
  `python -m csp_div.analyze_assistant_axis --csp-dir <dir>
  --out <axis.png> --only-new`. Reuses the cached mean_vanilla and
  skips already-processed ckpts. Saves ~30 min per dir vs full pass.
- **Legacy Gemma data + scripts** preserved on branches
  `trough-theory`, `early-stop-kl10`. Do not delete.
