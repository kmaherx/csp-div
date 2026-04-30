# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow preferences

- **After generating or revising figures, commit and push.** The user reviews plot output on mobile, so figures only become useful once they're on the remote. Bundle the figure-generating change and the regenerated figures (or the script change alone if the user hasn't asked for a re-render) into one commit and push to the current branch.

## Orientation

This is a research codebase for **max-divergence contextualized soft prompts** (CSPs). One CSP per run, trained by gradient *ascent* on `KL(student || vanilla_teacher)` — the goal is to find what attractor / basin the model lands in when pushed maximally away from default behavior. See `NARRATIVE.md` for the story arc and findings, `TODO.md` for active threads, `README.md` for run mechanics.

The repo is a fork-style sibling of `kmaherx/csp_arithmetic` with four deltas: vanilla teacher (no system prompt), KL ascent loss, vanilla SAE comparator, positive frames only.

## Commands

```bash
pip install -e .

# Train (default = Llama-3.1-8B; config defaults: L=4, 500 steps; ~tens of minutes/GPU)
# NOTE: 500 is just the config default — actual runs in this repo were
# truncated by KL-blowup pacing: Qwen used --steps 200, Llama --steps 50.
# The overnight scripts pass these explicitly; check kl_curve length on
# any sp_pos.pt to confirm.
python -m csp_div.train

# Switch model preset (env var, not flag)
CSP_MODEL_PRESET=qwen-2.5-7b-instruct python -m csp_div.train

# Eval modes — run separately. SAE only works if preset has SAE_RELEASE set
# (currently None for both Qwen and Llama presets).
python -m csp_div.evaluate --mode behavior
python -m csp_div.evaluate --mode self-verb
python -m csp_div.evaluate --mode sae

# Multi-seed sweep with frame condition + checkpoint cadence
python -m csp_div.train --seed 5 --steps 50 --checkpoint-every 5 \
    --frame-pool instrumental --run-name qwen_frames/instrumental/seed_5

# Prepend (no-frame) condition uses --placement instead of --frame-pool
python -m csp_div.train --placement prepend --run-name qwen_frames/prepend/seed_5

# Axis projection across all checkpoints under a directory; saves shifts.pt
python -m csp_div.analyze_assistant_axis --csp-dir qwen --out results/qwen/axis.png

# Long chains (each ~10–14 hr, idempotent — won't retrain finished ckpts)
bash scripts/run_overnight.sh        # Qwen, all 4 conditions
bash scripts/run_llama_overnight.sh  # Llama, all 4 conditions
```

There are no tests, no linter config, no build step beyond `pip install -e .`. All entry points use `python -m csp_div.<module>` and anchor paths on `PROJECT_ROOT` so cwd doesn't matter.

## Architecture — what requires reading multiple files

### Model preset switching (`config.py`)

`CSP_MODEL_PRESET` env var picks one entry from `_MODEL_PRESETS` (Qwen-2.5-7B or Llama-3.1-8B). All downstream code reads `config.MODEL_NAME`, `config.AXIS_LAYER`, etc. — **never hardcode model names**. Default preset is Llama. Each preset has its own `AXIS_REPO` (Butanium assistant-axis vector) at `AXIS_LAYER`. SAE_RELEASE is `None` for both — the `--mode sae` path is wired but inert until an SAE is added.

### Training pipeline (`train.py`)

The student is built two different ways:
- `build_student` (**splice**, default): one `§` placeholder inside a sampled frame from a pool (`POSITIVE_FRAMES_PERSONA` / `..._INSTRUMENTAL` / `..._MINIMAL`). At tokenize time, `§` is replaced by L embedding vectors. Output sequence is `(L-1)` tokens longer than input.
- `build_student_prepend` (**prepend**): no frame, no placeholder. CSP is L vectors inserted at `content_start` (the index right after the user-role chat-template opening tokens, before user content). Sequence is L tokens longer.

Loss is gradient ascent: `(-kl).backward()`. Vanilla teacher responses are greedy-generated **once** and cached to `cached_responses.json` per run; the long Llama overnight chain reuses one cache (`results/llama/seed_0/cached_responses.json`) across seeds via `cp`. **Two RNG streams** are intentional: `--seed` controls SoftPrompt init; `--data-seed` controls per-step prompt + frame sampling. RNG-decoupling probe (`run_rng_probe.sh`) showed trough depth follows init RNG, not data RNG — keep them separable.

KL blows up fast and the per-model pace differs, so actual training was truncated well before the 500-step config default — Qwen runs used 200 steps, Llama 50. Match these when adding new conditions on existing models so trajectories stay comparable to existing `shifts.pt` files.

Each saved checkpoint matches the upstream `csp_arithmetic` schema (so its tooling continues to work): includes `embedding`, `L`, `hidden_size`, `kl_curve`, `frame_pool`, `config` dict. `sp_pos.pt` = final; `sp_pos_step{N}.pt` = intermediate (gitignored, kept locally).

### Frame conditions and basin geometry

The frame-bias control experiment is the core contribution. Four conditions live in `config.FRAME_POOLS`:

- `persona`: `Be / Act / Please / You should §` — historical default, identity-priming verbs
- `instrumental`: `Use / Apply / Follow / Employ §` — verb implies a tool/method, no identity
- `minimal`: `{sp}: / ({sp}) / [{sp}] / <{sp}>` — pure label, no verb (uses `splice`)
- `prepend`: no frame at all (uses `--placement prepend`, *not* `--frame-pool`)

These map onto the directory layout: `results/<model>_frames/{instrumental,minimal,prepend}/seed_<N>/` and the original `results/<model>/seed_<N>/` for PERSONA. PCA / axis tooling expects this exact structure — see `analyze_pca_trajectory.py --shifts-paths` lists in the overnight scripts.

### Per-model results layout

```
results/<model>/seed_<N>/{cached_responses.json, sp_pos.pt, sp_pos_step*.pt, eval/}
results/<model>/{axis.png, axis.json, shifts.pt}      # produced by analyze_assistant_axis
results/<model>/pca[_normalized]/figure_*.png         # produced by analyze_pca_trajectory
results/<model>_frames/<condition>/seed_<N>/...
results/<model>_frames/{pca,pca_normalized,pca_4cond[_normalized]}/...
```

`shifts.pt` files are the input to PCA (`scripts/analyze_pca_trajectory.py`); `axis.json` carries the basin classification (deep ≤ −0.5 cos, shallow > −0.4 cos, mid otherwise) that PCA scripts read to color trajectories.

### Cross-model constraints

- Hidden dims differ (Qwen 3584, Llama 4096). PCA is therefore **per-model**, never pooled across models.
- Per-model basin populations differ: Qwen 7/3 deep/shallow under PERSONA, Llama 3/7 — same code, different model geometry. Don't assume Qwen patterns transfer; the cross-model surprise (Llama has init geometries that survive bracket-only frames; Qwen does not) is one of the main findings.

### Legacy code

Earlier Gemma-3-4b-it work lives on branches `trough-theory` and `early-stop-kl10` (preserved, not maintained on `cross-model` / `rng-probe`). Don't delete results dirs that look orphaned without checking those branches first.

### Upstream dependency

Schema compatibility with `kmaherx/csp_arithmetic` is intentional — the checkpoint dict shape, the `(label, polarity, frame)` condition tuple format, and the frame placeholder convention (`§`, plus `¶` for two-slot composition) all match upstream so its evaluation/analysis tooling can be reused.
