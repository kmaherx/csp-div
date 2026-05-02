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

## Finding 1 — random embeddings are invisible to the model (Stage A: n=20)

- Pick an embedding at random (here we use `randn(L, hidden_size)`
  scaled to the median per-row L2 norm of the model's input embedding
  matrix, ~0.685 on Llama). The model's behavior is essentially
  vanilla.
- Quantitative: histogram of step-0 KL(student‖vanilla) across 20
  seeds — most values cluster at 0.01-0.02; two outliers at 0.07 and
  0.13 (figure: `results/llama/step0_evidence.png`). For comparison,
  the trained CSP at step 100 reaches KL ≈ 29-30.
- Qualitative: when prompted to describe the embedding via the
  self-verb prompt, most seeds get responses like "Please provide the
  word..." — the model doesn't recognize anything is there. Samples
  in `results/llama/step0_self_verb_samples.md`.
- Conclusion: most of embedding space is *below the model's
  recognition threshold*. Random points don't count as words.

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

## Finding 2 — two populations of trajectories (Stage A: n=20)

- Train across 20 seeds, project the residual-stream activations at
  L16 onto the Butanium assistant axis (negative = role-play). The
  trajectories split into two populations along the axis:
  - **Population 1 (12/20 dippers)** — dips below cos = -0.5, often
    passes through coherent persona-like states.
  - **Population 2 (8/20 non-dippers)** — stays above cos = -0.4,
    drifts toward format/typographic distortion without inhabiting a
    character.
- KL saturates at ~29-30 by step ~55 and stays flat through step 100
  for all seeds — both populations converge to the same noise sink at
  high KL. The persona basin is a *detour*, not a destination.
  (`results/llama/axis.png` — direction-alignment subplot is the
  headline view.)

## Finding 3 — in PC space the populations partly separate (Stage A)

- Per-(seed, ckpt) shift vectors fed to PCA produce two partly-distinguishable
  clusters in PC1×PC2 — visible from the basin coloring on
  `results/llama/pca_normalized/figure_pc1_vs_pc2.png`. Trajectories
  sweep from upper-right (init) toward lower-left (sink); the path
  through PC space differs between populations even though the endpoints
  are similar.
- Variance explained: PC1 ≈ 0.26, PC2 ≈ 0.10 (normalized), or PC1 ≈ 0.42,
  PC2 ≈ 0.09 (raw).

## Finding 4 — the optimization stays in-distribution (sanity)

- `--match-token-norm` scales the CSP at init to the model's median
  per-row token-embedding L2 norm (~0.685 on Llama-3.1-8B). Throughout
  100 steps, the CSP per-token L2 norm grows by only ~3% (to ~0.71;
  see `results/llama/csp_norm_vs_step.png`). All 20 seeds bounded.
- Without this constraint, the default `randn*0.1` init lives at L2 ≈ 6.4
  — ~10× typical real tokens, deep OOD. Confirms the optimizer is
  making the model react to in-distribution embeddings rather than
  pushing into OOD regions where its behavior is undefined.

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
