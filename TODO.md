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

Status: replicated qualitatively on Gemma-3-4b-it (early); quantitatively
on Qwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct (10 seeds each, axis
projection plots committed on `cross-model`). RNG probe (`rng-probe`
branch) showed trough depth is governed by SoftPrompt init, not by data
sampling order — implies multiple init basins in the loss landscape.

## Queued analyses to support the hypothesis

In rough order of decisiveness for the claim:

1. **SAE feature emergence (Qwen + Llama).** Need an SAE for each
   model — Qwen: `andyrdt/saes-qwen2.5-7b-instruct` already cached
   locally. Llama: check sae_lens registry. Then
   `evaluate_divergent.py --mode sae` on each trough checkpoint. Test:
   do the top-active features at the trough have *persona-like*
   descriptions (Neuronpedia)? Cross-model: do trough states share
   semantically similar features? Compare to the Gemma shared
   9/29-feature core. Important refinement after the rng probe:
   compare deep-trough seeds (5, 6, 9 in Llama) vs shallow-trough
   seeds (0, 2) — features should differ if "persona basin" is a
   real phenomenon.

2. **Persona classifier on trough outputs.** Without this, "becomes a
   character" is human pattern-matching. Options: LLM-judge
   (Sonnet/Opus prompt: "does this look like persona X?") against
   `config.PERSONAS` keys, or sentence-embedding similarity of
   behavior-eval responses to canonical persona descriptions.
   Quantifies persona-ness per seed per checkpoint. Predicts deep-
   trough seeds will score persona-like; shallow-trough seeds won't.

3. **Negative control parity for Qwen + Llama.** Gemma had one
   (commit 7218a17 "-0.76 cos is structural, but tracking is real").
   Random-init soft prompt — should *not* produce personas. Need
   matching control on Qwen + Llama before claiming the trough is
   meaningful.

4. **Bimodal split replication.** Gemma showed narrator-mode (seeds
   0,1,2,3,5) vs role-play-mode (4,6,7,8,9). Reframe in light of the
   rng probe: the bimodality may itself be an init-basin
   phenomenon. Test: re-run the Gemma seeds with swapped data-seeds
   and see whether mode tracks init.

5. **Init-basin distribution.** With only 10 seeds we see ~3 shallow
   and ~7 deep. Is this 3/7 split bimodal or sampled from a smooth
   distribution of init-basin depths? Cheap probe: train 30+ init
   seeds for ~20 steps each (just past the typical trough), record
   trough depth, plot the distribution. If bimodal, two real basins;
   if smooth, just sampling from a continuous landscape.

## DONE — multi-model replication

- ✅ Qwen-2.5-7B-Instruct: 10 seeds × 200 steps trough-trace
  (`results/trough_qwen/`).
- ✅ Llama-3.1-8B-Instruct: 10 seeds × 50 steps × ckpt-every-5,
  behavior + self-verb at steps 10/20/30/40/50
  (`results/trough_llama/`).
- ✅ Consolidated to `cross-model` branch (Gemma-only results dropped).

## DONE — trough plots / axis projection

`results/trough_{qwen,llama}_axis.{png,json}` on `cross-model`. Both
models show clean troughs around cos ≈ −0.6 to −0.7 at KL ≈ 1–10 for
~7/10 seeds; remaining seeds (0, 2 in both, plus 4 in Qwen and 7 in
Llama) reach only cos ≈ −0.25 to −0.4.

## DONE — RNG decoupling probe

Branch `rng-probe`. Added `--data-seed` flag to `train_divergent.py`
to separate SoftPrompt init from prompt-sampling order. Result:

- Holding init constant and swapping data-seed: trough stays the same
  depth (init=0,data=5 → −0.30; init=2,data=5 → −0.27).
- Holding data constant (the "shallow" orderings) and swapping init
  to 5: trough deepens to ≈ −0.6.

So **trough depth is governed by SoftPrompt init**. Data-seed only
affects traversal speed through the basin (final KL ranges from ~13
to ~27 across runs that share the same init=5 but different
data-seeds). Combined plot at
`results/trough_llama_rng_axis_combined.png`.

This finding informs every queued analysis — comparisons between
"deep" and "shallow" seeds are now well-defined and worth doing.

## Causal ablation follow-ups (Gemma)

- The 9-feature shared-core ablation didn't collapse the persona; the
  29-feature broader ablation only marginally affected a few seeds.
  Worth trying:
  - Clamp ALL csp-only features (top-50 per seed = ~50 features). Tests
    the limit of L17 SAE-feature ablation.
  - Cross-layer SAE: re-run feature decomposition at L8 / L17 / L24 and
    ablate at the layer where the persona signal is strongest.
  - Compare ablation effects per seed against the bimodal axis split
    (narrator-mode vs role-play-mode seeds): does role-play-mode
    survive ablation differently than narrator-mode?

## Persona / behavior depth

- For each seed, identify the "best" coherent CSP step (e.g., latest pre-
  cliff checkpoint with persona intact) and assemble a clean side-by-side
  reference doc — one persona per seed × 5 prompts each.

## Methodology cleanups

- The `--avg-last-tokens 64` result was a red herring (parens are scattered,
  not just at the start). Mark in the writeup.
- Document that csp_arithmetic-style L17-at-CSP-tokens projection at cos
  ≈ −0.76 is structural (vanilla L17 anywhere on chat-template input is
  also at −0.75). The "tracking" claim is rank-correlation only; absolute
  spread is dwarfed by template signal.

## Steering experiment (future)

- Use the SAE decoder columns of the shared core features as a steering
  direction. Add `α · sum_k W_dec[feat_k]` at L17 during a vanilla forward
  and see if the model produces persona-like output without any CSP. Tests
  whether the shared SAE features are sufficient (not just necessary).
