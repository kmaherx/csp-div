# TODO

Living list of open threads. Newest at the top of each section.

## High priority — multi-model replication

**Qwen 2.5 7B-Instruct.** Re-run the core 10-seed batch (max 200 steps,
ckpt every 10, early-stop KL ≈ 10) with behavior + SAE + self-verb evals
on `Qwen/Qwen2.5-7B-Instruct`. Butanium has a published assistant axis
for it: <https://huggingface.co/datasets/Butanium/qwen-2.5-7b-instruct-assistant-axis>.
Hypothesis: Qwen does less stage-direction role-play than Gemma, so the
response-token cos should land negative without needing the parenthesis-
omission methodology. Also gives a second data point on the orthogonal-
embedding-but-shared-features finding.

**Llama 3.1 8B-Instruct.** Same protocol. Butanium has the axis:
<https://huggingface.co/datasets/Butanium/llama-3.1-8b-instruct-assistant-axis>.
Same expectation re: parens.

Steps for either:

1. Update `config.py` (or pass `--model` overrides) with the new model name.
2. Pick the SAE for the appropriate layer (Qwen: andyrdt's saes-qwen2.5-7b-instruct
   already cached locally; Llama: check sae_lens registry).
3. Regenerate vanilla teacher cache (model-specific — can't reuse Gemma's).
4. Train 10 seeds (~1–2 hr each model on the RTX 5090).
5. Run watcher → eval per seed → push to a new branch
   (`qwen-replication`, `llama-replication`).
6. Apply both axis-projection methodologies (response-token mean, paren-mode outside).

## Trough-tracing run

The early-stop-KL=10 batch catches each seed at varying points in its
projection trajectory: 5/10 seeds already passed through a trough in
cos(shift, axis) at KL ≈ 1–6 and are climbing back up; 2/10 still
descending at KL=10; 3/10 monotonically rising from start. The trough
is hypothesized to mark the **point of strongest persona** before
formatting/gibberish takes over post-cliff (consistent with main-branch
run 2: persona at step 100 / KL=11, formatting soup at step 500 /
KL=64).

To capture the full per-seed trajectory through the trough, train the
same 10 seeds with no early stop (or `--early-stop-kl 50`), with
checkpoints every 10 steps. Expected: every seed reaches a trough by
KL ≈ 5–20 and then recovers as the formatting attractor sets in.
Confirms that the "best persona" checkpoint per seed is at the trough,
not at KL=10.

## In-flight (current session)

- `analyze_assistant_axis.py --exclude-parens` on Gemma 10-seed batch — running.
- `analyze_assistant_axis.py --paren-mode inside` — queued auto-chain.

## Causal ablation follow-ups

- The 9-feature shared-core ablation didn't collapse the persona; the
  29-feature broader ablation only marginally affected a few seeds. Worth
  trying:
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
- Test whether the bimodal split (narrator-mode 0,1,2,3,5 vs role-play-mode
  4,6,7,8,9) aligns with anything else: SAE feature differences, KL trajectory
  shape, or PC1/PC2 of CSP embedding.

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
