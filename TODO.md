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

Status: replicated qualitatively on Gemma-3-4b-it (early), Qwen-2.5-7B-
Instruct, and Llama-3.1-8B-Instruct (10 seeds each). Need quantitative
evidence before claiming it.

## Queued analyses to support the hypothesis

In rough order of decisiveness for the claim:

1. **Trough plots for Qwen and Llama.** Run
   `analyze_assistant_axis.py` against `results/trough_qwen/` and
   `results/trough_llama/` to compute cos(CSP-shift, axis) per
   checkpoint per seed. Want the per-seed trajectory through training
   steps, with the trough visible. If trough exists across all three
   models at consistent KL ranges, that's strong cross-model evidence.

2. **SAE feature emergence (Qwen + Llama).** Need an SAE for each model.
   Qwen: `andyrdt/saes-qwen2.5-7b-instruct` already cached locally.
   Llama: check sae_lens registry. Then `evaluate_divergent.py
   --mode sae` on each trough checkpoint. Test: do the top-active
   features at the trough have *persona-like* descriptions
   (Neuronpedia)? Cross-model: do trough states share semantically
   similar features? Compare to the Gemma shared 9/29-feature core.

3. **Direct axis projection at trough vs elsewhere.** Project trough
   activations onto the Butanium axis. Claim the strong form of the
   hypothesis: max-KL pushes the residual stream maximally
   *anti-aligned* with the assistant axis at the trough. Compare to
   start (≈ 0), final (post-cliff, formatting attractor — should be
   neutral or noisy). Use existing `analyze_assistant_axis.py`.

4. **Persona classifier on trough outputs.** Without this, "becomes a
   character" is human pattern-matching. Options: LLM-judge
   (Sonnet/Opus prompt: "does this look like persona X?") against
   `config.PERSONAS` keys, or sentence-embedding similarity of
   behavior-eval responses to canonical persona descriptions.
   Quantifies persona-ness per seed per checkpoint.

5. **Negative control parity for Qwen + Llama.** Gemma had one
   (commit 7218a17 "-0.76 cos is structural, but tracking is real").
   Random-init soft prompt — should *not* produce personas. Need
   matching control on Qwen + Llama before claiming the trough is
   meaningful.

6. **Bimodal split replication.** Gemma showed narrator-mode (seeds
   0,1,2,3,5) vs role-play-mode (4,6,7,8,9). Does Qwen/Llama also
   bimodally split? If yes, the persona structure is real and not
   seed-noise. If no, may be Gemma-specific.

## Multi-model replication — DONE

- ✅ Qwen-2.5-7B-Instruct: 10 seeds × 200 steps trough-trace
  (`results/trough_qwen/`, branch `qwen`).
- ✅ Llama-3.1-8B-Instruct: 10 seeds × 50 steps × ckpt-every-5,
  behavior + self-verb at steps 10/20/30/40/50
  (`results/trough_llama/`, branch `llama`).
- Next: consolidate qwen + llama work into a single branch (excluding
  Gemma results, which were too extreme a model for the cleanest
  story) and run the analyses above on that branch.

## Trough-tracing run — DONE

Captured for all three models. Trough exists for every model at
KL ≈ 5–20. Plots and per-seed cos trajectories are next-up
(see "Queued analyses #1").

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
