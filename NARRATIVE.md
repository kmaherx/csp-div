# NARRATIVE

Bullet-point arc of the project. For run instructions see
[`README.md`](README.md); for active threads see [`TODO.md`](TODO.md).

## Setup

- LLMs map discrete tokens to continuous embeddings. The continuous
  space is **vast** compared to the slice the discrete vocabulary
  ever exposes.
- What about the *directions in between* the tokens? They don't
  correspond to any single human word, but they may have meaning to
  the model — *non-human words*.
- Question: which of these non-human words does the model recognize?
  - The model might ignore anything non-human; we want to filter for
    soft prompts the model actually responds to.
  - **Behavioral generation**: does the prompt change what the model says?
  - **Self-verbalization**: does the model recognize the prompt as a
    word — can it describe what it is?

## Finding 1 — random embeddings are invisible to the model

- Pick an embedding at random (the default `randn(L, hidden) * 0.1`
  init): the model's behavior is **indistinguishable from vanilla**.
- Quantitative: histogram of step-0 KL(student‖vanilla) across seeds —
  values cluster near zero (figure: `results/llama/step0_evidence.png`).
- Qualitative: when prompted to describe the embedding, the model
  literally says "what word?" — it doesn't see anything is there.
- Conclusion: most of embedding space is *below the model's recognition
  threshold*. Random points don't count as words.

## Method — finding non-human words via KL ascent

- We want words the model *does* respond to. We need an optimization
  target.
- For exploration, the target should be **general**, not biased toward
  any particular behavior. We use the simplest such signal:
  **maximize KL divergence from vanilla** — find any direction the
  model treats as a perturbation, in any direction.
- Expected dynamic: KL grows monotonically until the embedding pushes
  the model into degenerate output (token loops, format spam). The
  *trajectory* matters as much as the endpoint — like a particle
  accelerator, meaning lives in the path the optimizer traces.
- Expected to find some intelligible on-manifold states along the
  way, due to the way we train (CSPs — see "Training" below).

## Training — contextualized soft prompts

- The CSP is `L=4` learned embedding tokens spliced into a positive
  frame at the user turn (`"Be §."` where `§` is the soft prompt slot).
  The frame anchors the CSP as a noun-position the model can refer to.
- **Init scaling matters**: default `randn*0.1` init produces per-token
  L2 norm ≈ 6.4, deep out-of-distribution vs the model's actual
  embedding matrix (Llama-3.1-8B per-token L2 norm ≈ 0.7). We use
  `--match-token-norm` to scale init to the median row-norm of the
  embedding matrix, then `--lr 1e-4` so per-step relative changes are
  comparable to the unconstrained-init regime.
- **Step 0 anchor**: training saves `sp_pos_step0.pt` before any
  optimizer step. Used downstream for all the "random embeddings are
  invisible" claims.

## Finding 2 — two populations of trajectories

(To be filled in once the headline run lands.)

- Train across N seeds; project the residual-stream activations at
  AXIS_LAYER onto the Butanium assistant axis (negative = role-play).
- Trajectories split into two populations along the axis:
  - **Population 1 (deep)** — dips strongly toward the role-play
    pole, often passes through coherent persona-like states (rhyming
    poets, mythic narrators, archaic-English voices, etc.).
  - **Population 2 (non-dipper)** — stays near the assistant pole,
    drifts toward format/typographic distortion without inhabiting a
    character.
- Both populations eventually converge to the same noise sink at high
  KL — the persona basin is a *detour*, not a destination.

## Finding 3 — in PC space the populations are unsupervised

(To be filled in once the headline run lands.)

- Per-(seed, ckpt) shift vectors fed to PCA produce two distinguishable
  clusters in PC1×PC2 — visible without the assistant-axis label.
- Confirms the population structure isn't an artifact of the supervised
  axis; it's intrinsic to the geometry of where the trained CSPs land.

## Where things live

| What | Where |
|---|---|
| Training | `csp_div.train` (`--match-token-norm --lr 1e-4`) |
| Eval | `csp_div.evaluate` (behavior, self-verb) |
| Axis projection + shift collection | `csp_div.analyze_assistant_axis` |
| PCA trajectory | `scripts/analyze_pca_trajectory.py` (`--x kl`/`--x step`) |
| Step-0 evidence | `scripts/plot_step0_evidence.py` |
| CSP norm sanity | `scripts/plot_csp_norm.py` |
| Headline runner (multi-pod-friendly) | `scripts/run_headline.sh START END` |
| Per-model results | `results/llama/` |
| Headline figures | `results/llama/{axis.png, pca_normalized/figure_pc1_vs_pc2.png, step0_evidence.png, csp_norm_vs_step.png}` |
