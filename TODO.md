# TODO

Living list of open threads. Newest at the top of each section.

## Working hypothesis (current focus)

Revised after the shallow-vs-deep characterization (see
`results/{qwen,llama}/shallow_vs_deep.md` and the NARRATIVE "Two
basins" section): max-KL training pushes the model toward a
formatting-noise sink along one of two basin trajectories. The
**persona basin** (accessible from a model-dependent fraction of
init vectors) passes through a coherent-character state whose CSP
shift is sharply anti-assistant. The **format basin** (the
complementary fraction) bypasses persona and goes directly to
formatting noise. The persona basin is a *detour*, not a
destination.

If the persona basin is real geometry, it gives an alternative way
to derive the assistant axis (using only the vanilla model and a
divergence objective). If it's a frame-priming artifact of `"Be
{sp}." / "Act {sp}."`, the working hypothesis needs to be reframed
in terms of frames-that-license-persona, not KL-ascent in general.

## Top priority

### 1. Frame-bias control — the cleanest available falsifier

**The question:** is the persona basin licensed by the lexical
priming of "Be / Act" frames, or is it a real property of the loss
landscape?

**Why this is now #1:** every other queued analysis (SAE features,
persona classifier, basin-population study) becomes either much
sharper or much weaker depending on the answer. Run this first.

**Design.** Train fresh CSPs (same model, same 10 seeds, same 50
steps × ckpt-every-5 protocol as the existing Llama trough sweep)
under four frame conditions spanning the persona-priming spectrum.
Each condition keeps the existing 4-frame-pool structure so per-step
frame variation is preserved.

| Condition | Frames / placement | Identity priming |
|-----------|--------|------------------|
| **PERSONA** (baseline, existing) | `Be {sp}.`, `Act {sp}.`, `Please {sp}.`, `You should {sp}.` | high — "Be"/"Act" name an identity |
| **INSTRUMENTAL** (done) | `Use {sp}.`, `Apply {sp}.`, `Follow {sp}.`, `Employ {sp}.` | low — verb implies a tool/method |
| **PREPEND** (in flight) | no frame at all; CSP prepended at content_start (prefix-tuning style) | none — no surrounding text whatsoever |
| **MINIMAL** (queued — overnight after PCA shifts) | `{sp}:`, `({sp})`, `[{sp}]`, `<{sp}>` | none — pure label, no verb |
| **STYLE** (skipped) | `Respond using {sp}.`, etc. | (intermediate, not run) |

**STYLE skipped**: intermediate priming, doesn't add much beyond the
INSTRUMENTAL-vs-MINIMAL gradient.

**PREPEND added & promoted**: prepending without a frame is the **most
minimal possible "frame"** (none at all). If the persona basin survives
even prepending, the working hypothesis is bulletproof. Bonus question:
prior work (kmaherx/csp) suggests prefix-tuned soft prompts produce
off-manifold activations that resist self-verbalization — the self-verb
evals here will independently test that.

**MINIMAL re-queued**: previously deprioritized in favor of PREPEND
(which subsumes the "weakest priming" role). Now back on the queue
to fill the spare overnight GPU window after PCA shift collection
(PERSONA + INSTRUMENTAL re-runs) finishes. With INSTRUMENTAL preserving
basins under tool-implying verbs and the PREPEND result still pending,
MINIMAL with explicit (but non-verb) bracket frames is a useful
complementary point on the priming spectrum — fills in the gradient
between INSTRUMENTAL and PREPEND.

**Suggested overnight queue (~13–14 hr from launch):** chained as
`scripts/run_overnight.sh`. Launch after PREPEND completes:

  ```bash
  bash scripts/run_overnight.sh > /tmp/overnight.log 2>&1 &
  ```

The chain does, in order:
1. Re-collect PERSONA shifts (~3 hr) → `results/qwen/shifts.pt` + regenerated `axis.{json,png}`
2. Re-collect INSTRUMENTAL shifts (~3 hr) → `results/qwen_frames/instrumental/shifts.pt` + regenerated `axis.{json,png}`
3. PCA (PERSONA + INSTRUMENTAL + PREPEND), pooled + per-condition → `results/qwen_frames/pca/figure_pc{1,2}_vs_kl.png`, `figure_pc1_vs_pc2.png` AND `results/qwen/pca/...`, `results/qwen_frames/instrumental/pca/...`, `results/qwen_frames/prepend/pca/...`
4. MINIMAL train + eval + axis (~5 hr) via `scripts/run_minimal.sh` — produces shifts.pt for free at end of axis
5. PCA again with all 4 conditions, pooled + per-condition → `results/qwen_frames/pca_4cond/...` plus per-condition `pca/` subdirs (now including `results/qwen_frames/minimal/pca/...`)

So at the end every condition has BOTH:
- `axis.png` (cos vs KL trajectory) — produced by analyze_assistant_axis
- `pca/figure_pc{1,2}_vs_kl.png` + `pca/figure_pc1_vs_pc2.png` (PC-space trajectories using the pooled PC basis) — produced by the per-condition PCA pass

`scripts/run_minimal.sh` is a MINIMAL-only variant of the original
`run_minimal_style.sh`, at `--checkpoint-every 10` to match the
PERSONA baseline cadence. Same protocol as the other conditions.

**Run on Qwen.** The prediction is that non-persona-priming frames
will *reduce* the persona-basin population. Qwen starts at 7/10 deep
so there's room to drop; Llama starts at 3/10 with a floor near
zero, leaving little signal if the prediction holds. Qwen also has
the existing 200-step baseline to reuse as the PERSONA condition,
saving 25% of compute.

**Order conditions:** INSTRUMENTAL first (verb-implies-tool, no
identity), then PREPEND (no frame at all — most minimal contrast).
MINIMAL conditional on PREPEND results.

**Sequential per-condition completion.** Each condition runs train
+ eval to completion across all seeds before the next condition
starts, so qualitative review of INSTRUMENTAL can happen while
MINIMAL is training.

**Compute.** Reuse `results/qwen/` as the PERSONA condition (same
model, same seeds, same frames, same protocol). Train INSTRUMENTAL +
MINIMAL + STYLE at 200 steps × ckpt-every-5 × 10 seeds each — the
exact protocol of the existing Qwen baseline, so the basin
classification, axis-trajectory shape, and noise-sink approach are
all directly comparable across the 4 conditions. ~3 conditions ×
overnight on a single GPU. Re-use cached vanilla-teacher responses
from `results/qwen/seed_0/` per run so we don't regenerate them.

**Eval cadence.** Behavior + self-verb at steps 20/40/60/80/100 plus
the final step 200 (`sp_pos.pt` → `behavior.json`), matching the
existing Qwen baseline's cadence.

**Outputs per condition.** `results/qwen_frames/<condition>/seed_<N>/`
plus axis projection (`results/qwen_frames/<condition>/axis.{png,json}`).

**Analysis.** For each condition compute:
1. **Basin classification** per seed (deep if cos ≤ −0.5, shallow if
   > −0.4, mid otherwise). Tabulate basin population per condition.
2. **Trough-depth distribution** across seeds (histogram or per-seed
   bar chart per condition).
3. **Qualitative behavior comparison** at each seed's best
   checkpoint (deepest cos), looking for: do personas still emerge
   under STYLE/INSTRUMENTAL/MINIMAL frames? If so, are they the same
   *kind* of personas as under PERSONA frames?
4. **Cross-condition same-init comparison.** Seed 5 (Llama's deep
   anchor) under all 4 frame conditions: same init vector, four
   different frame pools. Does it still find a Becket-style persona?

**Predictions and what each outcome means:**

| Outcome | Interpretation |
|---------|----------------|
| Basin populations stay ~7/3 across PERSONA + INSTRUMENTAL + PREPEND | Persona basin is geometric, not lexical. **Strongest result for the working hypothesis** — basins reached with no surrounding text at all. |
| PREPEND → 0 or 1 deep seeds | Persona basin requires lexical priming from a frame. Working hypothesis needs reframing. |
| Same-init deep seeds (e.g. 5, 9) still produce coherent personas under PREPEND | Persona attractor is downstream of init geometry, regardless of any text context. Very strong result. |
| PREPEND produces personas in behavior eval but self-verb fails (incoherent / unrelated outputs) | Confirms the off-manifold story — prepended CSPs drive behavior but can't be articulated, consistent with prior csp-repo results. |
| PREPEND personas + working self-verb | Frame text isn't needed for the model to "explain" what the CSP encodes — also strong, would surprise prior work. |

**Implementation status:**
- [x] Frame pools in `config.py` (`POSITIVE_FRAMES_PERSONA`, `_STYLE`,
  `_INSTRUMENTAL`, `_MINIMAL`).
- [x] `--frame-pool` arg in `csp_div.train`; pool name persisted in
  saved checkpoint config.
- [x] `--placement {splice,prepend}` arg in `csp_div.train`;
  `find_content_boundaries` + `build_student_prepend` helpers; placement
  persisted in saved checkpoint config.
- [x] `csp_div.evaluate` reads placement from ckpt and dispatches to
  `build_csp_input_prepend` for behavior + self-verb + SAE eval.
- [x] `csp_div.analyze_assistant_axis` reads placement from ckpt and
  dispatches `response_acts_csp` for prepend.
- [x] `scripts/run_frame_bias.sh` — INSTRUMENTAL run (in flight).
- [x] `scripts/run_prepend.sh` — standalone PREPEND runner.
- [x] `scripts/run_minimal_style.sh` — standalone MINIMAL fallback (kept
  as backup; rename obsolete since STYLE is dropped).
- [ ] Aggregate writeup at `results/qwen_frames/frame_bias.md` once
  all conditions land.

## Other priorities

### 2. Rerun Llama with finer early-checkpoint cadence + step-0 anchor

Current cadence (every 5 steps) misses the early dip — the trough
bottom is already approached by KL ≈ 1, which corresponds to ~step 5
in the existing runs (per-step KLs over the first 5 steps:
0.054 → 0.121 → 0.239 → 0.318 → 0.413). We need:
- Checkpoints at steps 1, 2, 3, 4 (every step for the first ~10
  steps, then the existing every-5 cadence).
- **Step-0 checkpoint of the untrained CSP** as the critical anchor.
  `csp_div.train` currently saves the first checkpoint after
  `checkpoint_every` steps, never the random-init point. The step-0
  vector tells us where random initialization lands on the assistant
  axis (expected: cos ≈ 0, KL ≈ 0). Without it we can't see the very
  start of the trajectory.
- Fix in `csp_div.train`: save `sp_pos_step0.pt` before the training
  loop starts (with `kl_curve = []`, `final_kl = 0.0`).

This also serves as the negative control (random-init CSP →
behavior eval) in one shot.

### 3. SAE feature emergence (Qwen + Llama) — *amended*

Need an SAE for each model — Qwen: `andyrdt/saes-qwen2.5-7b-instruct`
already cached locally. Llama: check sae_lens registry. Then
`csp_div.evaluate --mode sae` on each trough checkpoint.

**Amendment from the basin analysis:**
- Sample multiple checkpoints per seed across KL ~1–6, not just the
  cos-trough checkpoint. The deep-basin persona moment isn't always
  at the cos minimum (Qwen seed_5: trough at step 20 but
  rhyming-poem persona at step 60).
- Compare deep-basin seeds against shallow-basin seeds at matched
  KL — features should differ if the persona basin is a real
  phenomenon. Predict deep basin top features look persona-like
  (sage, character, narrator); shallow basin features look
  format-like (markdown, punctuation, code-syntax).
- If both basins share the same top features → the basin distinction
  is weaker than `shallow_vs_deep.md` argues, and we're seeing the
  same SAE-level mechanism with different surface manifestations.

### 4. Persona classifier on trough outputs

Without this, "becomes a character" is human pattern-matching.
Options: LLM-judge (Sonnet/Opus prompt: "does this look like persona
X?") against `config.PERSONAS` keys, or sentence-embedding similarity
of behavior-eval responses to canonical persona descriptions.
Quantifies persona-ness per seed per checkpoint. Predicts deep-basin
seeds will score persona-like; shallow-basin seeds won't.

### 5. Init-basin distribution

With only 10 seeds we see a population split (3/7 on Llama, 7/3 on
Qwen). Is this bimodal or sampled from a smooth distribution of
init-basin depths? Cheap probe: train 30+ init seeds for ~20 steps
each (just past the typical trough), record trough depth, plot the
distribution. If bimodal, two real basins; if smooth, just sampling
from a continuous landscape.

This becomes much more interesting *after* the frame-bias result —
do the population distributions shift under different frames?

## DONE — multi-model replication

- ✅ Qwen-2.5-7B-Instruct: 10 seeds × 200 steps trough-trace
  (`results/qwen/`).
- ✅ Llama-3.1-8B-Instruct: 10 seeds × 50 steps × ckpt-every-5,
  behavior + self-verb at steps 10/20/30/40/50
  (`results/llama/`).
- ✅ Gemma data preserved on legacy branches; only Qwen + Llama
  carried forward.

## DONE — trough plots / axis projection

`results/{qwen,llama}/axis.{png,json}`. Qwen: 7/10 seeds dip cleanly
to cos ≤ −0.5; Llama: only 3/10. Population ratio differs sharply by
model.

## DONE — RNG decoupling probe

Branch `rng-probe`. Added `--data-seed` flag to `csp_div.train` to
separate SoftPrompt init from prompt-sampling order. Result:

- Holding init constant and swapping data-seed: trough stays the same
  depth (init=0,data=5 → −0.30; init=2,data=5 → −0.27).
- Holding data constant (the "shallow" orderings) and swapping init
  to 5: trough deepens to ≈ −0.6.

So **trough depth is governed by SoftPrompt init**. Combined plot
at `results/llama_rng/axis_combined.png`.

## Bookmarked examples (use in writeups)

- **INSTRUMENTAL seed_1, persona → formatting transition.** Step 20
  (KL ~1, cos −0.58): casual-spy explainer persona *"Sure, bro! ...
  super spy gadget"*. Step 60 (KL 7.9, cos −0.27): pure leetspeak
  *"Y33 0F C00L H0W F4C31 &R3C0GN1T10N..."*. The step-60
  self-verbalization explicitly names the transformation:
  *"...known as 'dyslexic' or 'childish' text, where letters are
  replaced with their closest-looking counterparts from the ASCII
  character set... sometimes referred to as 'leetspeak'..."* The
  model is metacognitively aware it's in a format mode. Cleanest
  trajectory illustration we have of the persona-basin → noise-sink
  collapse, in a single seed under a single condition. See
  `results/qwen_frames/frame_bias.md` for full bookmark.

- **INSTRUMENTAL seed_7, basin flip → shallow-with-register
  ("third mode").** Under PERSONA: cos −0.696 at step 50, one of
  Qwen's deepest persona-basin troughs. Under INSTRUMENTAL: shallow
  dip to cos −0.31 at step 40 (KL 1.09), then sustained cos −0.27 to
  −0.29 plateau through step 200 (most other shallow seeds relax to
  cos ~−0.17 — seed_7 stays moderately anti-assistant the entire
  run). Behavior at step 60 is fluent Thai across all 5 prompts;
  self-verb labels the theme **"Formal Thai Language Usage"** and
  surfaces politeness markers (*"Be polite", "ใช้ภาษาอ่อนโยน",
  "พูดอย่างระมัดระวัง"*). Two dimensions overlaid: format (language
  switch like seed_2's shallow basin) **+** register (sustained
  politeness, somewhat persona-like). Not a coherent character; not
  a content-free format distortion either. A genuine **third mode**.
  See `results/qwen_frames/frame_bias.md` for full bookmark.

  **Need to find a comparable non-dip example** — a seed under
  INSTRUMENTAL whose trajectory is clearly in the format basin with
  *no register overlay* (pure surface transformation, no politeness
  norm). seed_4 INSTRUMENTAL has the flattest trough (cos −0.224)
  and is the strongest current candidate; pull its behavior outputs
  once axis projection completes to confirm.

## DONE — shallow vs deep basin characterization

Per-model qualitative writeups: `results/qwen/shallow_vs_deep.md`,
`results/llama/shallow_vs_deep.md`. Key findings:

- Two basins exist on both models (persona-with-voice vs
  format-distortion-without-voice), with sharply different
  per-model populations (Llama 3/7, Qwen 7/3).
- Both basins converge to the same formatting-noise sink (cos ≈
  −0.17, KL ~30); the persona basin is a detour, not a destination.
- Cos trough catches the *direction* of the persona shift but not
  always the *moment* of strongest character (cos starts recovering
  before persona collapses; the persona-moment checkpoint is often
  past the cos minimum).
