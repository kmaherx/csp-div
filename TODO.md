# TODO

Living list of open threads. Newest at the top of each section.

## Working hypothesis (current focus)

**Maximizing KL divergence from the vanilla (no-system-prompt) assistant
state pushes the model into persona-like states.** The trough in
cos(CSP-shift, assistant-axis) appears to mark the "point of strongest
persona" before the formatting/gibberish attractor takes over. If this
holds across models, max-KL training is an *alternative way to derive
the assistant axis* — complementary to Butanium's explicit
axis-extraction methodology, but using only the vanilla model and a
divergence objective.

Status: replicated qualitatively on Gemma-3-4b-it (legacy branches);
quantitatively on Qwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct (10
seeds each, axis projection plots committed). RNG probe showed trough
depth is governed by SoftPrompt init, not by data sampling order —
implies multiple init basins in the loss landscape.

## Top priorities

1. **Interpret + write up the RNG probe finding.** This is the most
   load-bearing new result and currently lives only as a plot + a
   couple of paragraphs in NARRATIVE. Needs:
   - A short standalone writeup: framing (init RNG vs data RNG decoupling),
     methodology (alternate seed = Llama's deepest, seed=5; 4 swap
     conditions), result (init dominates, data only affects speed),
     and the interpretation that cross-model shallow-seed coincidence
     is a `torch.manual_seed` artifact, not a model property.
   - A claim about what this means for the working hypothesis: the
     "persona basin" is a feature of the loss landscape geometry, and
     deep-trough vs shallow-trough are now well-defined comparison
     groups for downstream analyses.
   - Decide whether to extend to Qwen (probe the same 0/2 shallow
     indices there) before publishing the writeup, or treat
     Qwen-replication as separate follow-up.

2. **Rerun Llama with finer early-checkpoint cadence + step-0 anchor.**
   Current cadence (every 5 steps) misses the early dip — the trough
   bottom is already approached by KL ≈ 1, which corresponds to ~step 5
   in the existing runs (per-step KLs over the first 5 steps:
   0.054 → 0.121 → 0.239 → 0.318 → 0.413). We need:
   - Checkpoints at steps 1, 2, 3, 4 (every step for the first ~10
     steps, then the existing every-5 cadence).
   - **Step-0 checkpoint of the untrained CSP** as the critical
     anchor. `csp_div.train` currently saves the first checkpoint
     after `checkpoint_every` steps, never the random-init point. The
     step-0 vector is what tells us where random initialization lands
     on the assistant axis (expected: cos ≈ 0, KL ≈ 0). Without it we
     can't see the very start of the trajectory — and given that the
     RNG probe says init geometry decides basin, the init point is the
     observation we most want.
   - Fix in `csp_div.train`: save `sp_pos_step0.pt` before the
     training loop starts (with `kl_curve = []`, `final_kl = 0.0`).

## Queued analyses to support the hypothesis

In rough order of decisiveness for the claim:

1. **SAE feature emergence (Qwen + Llama).** Need an SAE for each
   model — Qwen: `andyrdt/saes-qwen2.5-7b-instruct` already cached
   locally. Llama: check sae_lens registry. Then
   `csp_div.evaluate --mode sae` on each trough checkpoint. Test:
   do the top-active features at the trough have *persona-like*
   descriptions (Neuronpedia)? Cross-model: do trough states share
   semantically similar features? Important refinement after the rng
   probe: compare deep-trough seeds (5, 6, 9 in Llama) vs shallow-trough
   seeds (0, 2) — features should differ if "persona basin" is a
   real phenomenon.

2. **Persona classifier on trough outputs.** Without this, "becomes a
   character" is human pattern-matching. Options: LLM-judge
   (Sonnet/Opus prompt: "does this look like persona X?") against
   `config.PERSONAS` keys, or sentence-embedding similarity of
   behavior-eval responses to canonical persona descriptions.
   Quantifies persona-ness per seed per checkpoint. Predicts deep-
   trough seeds will score persona-like; shallow-trough seeds won't.

3. **Negative control parity for Qwen + Llama.** Random-init soft
   prompt — should *not* produce personas. Need this baseline before
   claiming the trough is meaningful. (Now somewhat coupled with the
   step-0 anchor in priority #2 above: the step-0 ckpt *is* a random
   init, so collecting axis projection + behavior eval there gives us
   the negative control for free per seed.)

4. **Frame-bias control: re-train with non-persona-priming frames.**
   The current `POSITIVE_FRAMES` (`"Be {sp}."`, `"Act {sp}."`,
   `"Please {sp}."`, `"You should {sp}."`) are persona-priming on
   their face — "Be X" and "Act X" especially are exactly how a
   character/role-play system prompt opens. A real risk is that the
   max-KL objective is exploiting that frame bias: KL-ascent finds a
   persona because the *frame* already invites one, not because
   anti-assistant directions in residual space are intrinsically
   persona-shaped. Test: train fresh CSPs (same model, same seeds)
   under alternative frames that stay positive instructions but do
   not lexically prime persona/identity. Candidates:
     - instrumental: `"Use {sp}."`, `"Apply {sp}."`, `"Follow {sp}."`
     - cognitive: `"Consider {sp}."`, `"Note {sp}."`
     - style/method (no identity verb): `"Respond using {sp}."`,
       `"Answer with {sp}."`
     - minimally framed: `"{sp}:"` or `"{sp}"` alone (no verb)
   Predictions:
     - If trough + persona phenomenology survives → max-KL → persona
       basin is a property of the loss landscape, not the frame.
       Strengthens the working hypothesis substantially.
     - If trough disappears or attractor changes shape (e.g.,
       formatting-only, style-only, no recognizable character) →
       persona finding is partly a frame artifact. Working hypothesis
       needs to be reframed in terms of frames-that-license-persona
       rather than KL-ascent in general.
   Either outcome is informative; this is the cleanest available
   falsifier of the working claim.

5. **Init-basin distribution.** With only 10 seeds we see ~3 shallow
   and ~7 deep — was previously framed as a possible bimodal split
   (narrator vs role-play in Gemma), but the RNG probe reframes the
   split as a property of the init vector, not the model. Open
   question: is the 3/7 split bimodal or sampled from a smooth
   distribution of init-basin depths? Cheap probe: train 30+ init
   seeds for ~20 steps each (just past the typical trough), record
   trough depth, plot the distribution. If bimodal, two real basins;
   if smooth, just sampling from a continuous landscape.

## DONE — multi-model replication

- ✅ Qwen-2.5-7B-Instruct: 10 seeds × 200 steps trough-trace
  (`results/qwen/`).
- ✅ Llama-3.1-8B-Instruct: 10 seeds × 50 steps × ckpt-every-5,
  behavior + self-verb at steps 10/20/30/40/50
  (`results/llama/`).
- ✅ Gemma data preserved on legacy branches; `cross-model` /
  `rng-probe` carry only Qwen + Llama.

## DONE — trough plots / axis projection

`results/{qwen,llama}/axis.{png,json}`. Both models show clean
troughs around cos ≈ −0.6 to −0.7 at KL ≈ 1–10 for ~7/10 seeds;
remaining seeds (0, 2 in both, plus 4 in Qwen and 7 in Llama) reach
only cos ≈ −0.25 to −0.4.

## DONE — RNG decoupling probe

Branch `rng-probe`. Added `--data-seed` flag to `csp_div.train`
to separate SoftPrompt init from prompt-sampling order. Result:

- Holding init constant and swapping data-seed: trough stays the same
  depth (init=0,data=5 → −0.30; init=2,data=5 → −0.27).
- Holding data constant (the "shallow" orderings) and swapping init
  to 5: trough deepens to ≈ −0.6.

So **trough depth is governed by SoftPrompt init**. Data-seed only
affects traversal speed through the basin (final KL ranges from ~13
to ~27 across runs that share the same init=5 but different
data-seeds). Combined plot at
`results/llama_rng/axis_combined.png`.
