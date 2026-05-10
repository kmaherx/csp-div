# NARRATIVE

Bullet-point arc of the project. For run instructions see
[`README.md`](README.md) and [`pipeline/README.md`](pipeline/README.md).

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

## Finding 1 — random init is a noise baseline, not a recognized word

> **TBD — fill in post-rerun.** Numbers below are placeholders pending the
> 50-seed OOD-init Stage B run. Quantitative claims (step-0 KL distribution,
> trained-CSP saturation KL, exact comparison ratios) come from the rerun.

- An untrained CSP at the default `randn(L, hidden) * 0.1` init
  (per-token L2 ≈ 6.4) is OOD in magnitude relative to real token
  embeddings. The model does notice the perturbation — step-0 KL is
  non-trivial — but the resulting outputs are noise / non-recognition,
  not coherent character or semantic shift.
- **Quantitative (TBD)**: histogram of step-0 KL(student‖vanilla) across 50 seeds
  (figure: `results/llama/step0_evidence.png`) for the noise baseline.
  Trained CSP at step 100 reaches KL ≈ X; the contrast that matters is
  baseline-vs-trained, not baseline-vs-zero.
- **Qualitative**: when prompted via the self-verb prompt, untrained CSPs
  elicit responses like "Please provide the word..." or token-level noise —
  the model doesn't recognize a coherent word. Samples in
  `results/llama/step0_self_verb_samples.md`.
- Conclusion: random points in `randn*0.1` space register as noise, not
  as words. Training is what produces a recognized word.

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
- **Init**: default `nn.Parameter(torch.randn(L, hidden) * 0.1)`. Per-token
  L2 ≈ 6.4 on Llama-3.1-8B (hidden=4096), about 10× the median real-token
  row L2 of ~0.69. The init is OOD in magnitude. KL ascent at `LR=1e-3`
  walks the embedding inward toward whichever attractor pulls.
- **Init choice**: this is the same init as the rng-probe historical Llama
  runs. We use it (rather than scaling to in-distribution magnitudes) on
  empirical grounds: rng-probe's runs showed cleaner two-population
  separation in PC space and on the assistant-axis projection than the
  in-distribution variant did on the prior `main` branch.
- **Vocab-init alternative (future work)**: the principled in-distribution
  baseline initializes from random real token embeddings —
  `embeds[torch.randint(0, V, (L,))].clone()` — matching both magnitude
  and the embedding manifold. This is the classical prompt-tuning move
  and would address the "your init is OOD, of course you find populations"
  critique. Not run in this work; flagged as a follow-up ablation.
- **Step 0 anchor**: training saves `sp_pos_step0.pt` before any
  optimizer step. Used downstream as a baseline reference for the
  histogram in Finding 1.

## Finding 2 — two populations of trajectories

> **TBD — fill in post-rerun.** Population fractions, KL saturation step,
> and cos thresholds come from the 50-seed OOD-init rerun.

- Train across N seeds, project the residual-stream activations at L16
  onto the Butanium assistant axis (negative = role-play). The trajectories
  split into two populations along the axis:
  - **Population 1 (dippers)** — dips below cos = -0.5, often passes through
    coherent persona-like states.
  - **Population 2 (non-dippers)** — stays above cos = -0.4, drifts toward
    format/typographic distortion without inhabiting a character.
- KL saturates at the noise sink for all seeds; both populations converge
  to the same endpoint at high KL. The persona basin is a *detour*, not a
  destination.
- (`results/llama/axis.png` — direction-alignment subplot is the
  headline view.)

## Finding 3 — in PC space the populations partly separate

> **TBD — fill in post-rerun.** Variance-explained numbers and cluster
> structure come from the 50-seed OOD-init rerun.

- Per-(seed, ckpt) shift vectors fed to PCA produce two partly-distinguishable
  clusters in PC1×PC2 — visible from the basin coloring on
  `results/llama/pca_normalized/figure_pc1_vs_pc2.png`. Trajectories
  sweep from upper-right (init) toward lower-left (sink); the path
  through PC space differs between populations even though the endpoints
  are similar.

## Where things live

| What | Where |
|---|---|
| Training | `pipeline/1_train.py` (default `randn × 0.1` init at `lr=1e-3`) |
| Generation (behavior + self-verb + shift capture) | `pipeline/2_generate.py` |
| Self-verb judging (via Claude Code skill) | `pipeline/3_judge.py` + `.claude/skills/csp-judge/` |
| Axis projection | `pipeline/4_axis.py` |
| Published dashboard | `pipeline/5_dashboard.py` → `results/all_frames/dashboard.html` |
| Per-frame results | `results/llama/` (be), `results/llama_{act,please,youshould}/` |
| Pre-refactor reference (full 50-seed run) | `ood-init` branch, `figure_pc2d_all_frames_interactive_step50.html` |
| Archived (normed-init Stage A) | branch `scaled-init` |
