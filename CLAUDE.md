# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow preferences

- **After generating or revising figures, commit and push.** The user reviews plot output on mobile, so figures only become useful once they're on the remote. Bundle the figure-generating change and the regenerated figures (or the script change alone if the user hasn't asked for a re-render) into one commit and push to the current branch.

## Orientation

This is a research codebase for **max-divergence contextualized soft prompts** (CSPs) on Llama-3.1-8B. One CSP per run, trained by gradient *ascent* on `KL(student || vanilla_teacher)` — the goal is to find what attractor / basin the model lands in when pushed maximally away from default behavior. See `NARRATIVE.md` for the story arc and findings, `TODO.md` for active threads, `README.md` for run mechanics.

The repo is a fork-style sibling of `kmaherx/csp_arithmetic` with three deltas: vanilla teacher (no system prompt), KL ascent loss instead of descent, positive frames only.

## Commands

```bash
# One-time per pod (already done on this pod; venv lives on /workspace and is shared)
python -m venv /workspace/csp-div/.venv
/workspace/csp-div/.venv/bin/pip install -e /workspace/csp-div

# Headline run on one pod (10 seeds)
bash /workspace/csp-div/scripts/run_headline.sh 0 9

# Multi-pod scaling — Stage B target is 50 seeds across 5 pods
bash /workspace/csp-div/scripts/run_headline.sh 10 19   # pod 2
bash /workspace/csp-div/scripts/run_headline.sh 20 29   # pod 3
bash /workspace/csp-div/scripts/run_headline.sh 30 39   # pod 4
bash /workspace/csp-div/scripts/run_headline.sh 40 49   # pod 5

# Analysis (single pod, after all training+eval done)
PY=/workspace/csp-div/.venv/bin/python
$PY -m csp_div.analyze_assistant_axis --csp-dir llama --out results/llama/axis.png
$PY scripts/analyze_pca_trajectory.py --shifts-paths results/llama/shifts.pt \
    --out-dir results/llama/pca_normalized --normalize --x step
$PY scripts/plot_step0_evidence.py --axis-json results/llama/axis.json --csp-dir results/llama
$PY scripts/plot_csp_norm.py --csp-dir results/llama --axis-json results/llama/axis.json
```

There are no tests, no linter config, no build step beyond `pip install -e .`. All entry points use `python -m csp_div.<module>` and anchor paths on `PROJECT_ROOT` so cwd doesn't matter.

## Architecture — what requires reading multiple files

### Init regime: default `randn*0.1` (OOD in magnitude)

The SoftPrompt init is `nn.Parameter(torch.randn(L, hidden) * 0.1)`, giving per-token L2 norm ≈ 6.4 — about 10× larger than typical Llama token-embedding rows (median ≈ 0.69). The init lives **deep OOD in magnitude**; the model treats it as noise at step 0, and KL ascent walks the embedding inward toward whichever attractor pulls.

This is the same init regime as the historical `rng-probe` branch's Llama runs; we use it (rather than scaling to in-distribution) because rng-probe showed cleaner two-population separation than the in-distribution variant on the prior `main` (commit `c5fd7d5`, archived as branch `scaled-init`). LR is the `config.LR = 1e-3` default — no flag needed in the runner.

**Future work — vocab-init ablation.** Not implemented. The principled in-distribution baseline would init from random real token embeddings: `embeds = model.get_input_embeddings().weight; sp.embedding.data = embeds[torch.randint(0, V, (L,))].clone()`. This is the classical prompt-tuning move and matches both magnitude *and* the manifold (unlike `--match-token-norm` which matched magnitude only). Plan is to add this as `--init-from-vocab` in `train.py`, run a smaller seed sweep, and compare population structure to the OOD headline; addresses the "your init is OOD, of course you find populations" critique.

### Step-0 anchor

`train.py` saves `sp_pos_step0.pt` (the random-init CSP, before any training step) by default, and `analyze_assistant_axis.py` computes eval-time KL for any ckpt whose `final_kl is None` (the step-0 signature). Both happen automatically — no flags needed for new runs. Under the current OOD init the step-0 KL is non-trivial (not near-zero); the anchor's role is as a baseline reference for the trained-CSP saturation KL, not as evidence that random embeddings are invisible.

### Training pipeline (`train.py`)

The student is built two different ways:
- `build_student` (**splice**, default): one `§` placeholder inside a sampled frame from `config.POSITIVE_FRAMES_PERSONA` (`Be / Act / Please / You should §`). At tokenize time, `§` is replaced by L embedding vectors. Output sequence is `(L-1)` tokens longer than input.
- `build_student_prepend` (**prepend**, alt path): no frame, no placeholder. CSP is L vectors inserted at `content_start`. Sequence is L tokens longer.

Loss is gradient ascent: `(-kl).backward()`. Vanilla teacher responses are greedy-generated **once** and cached to `cached_responses.json` per run; subsequent seeds on the same pod (and across pods, since /workspace is shared) reuse via `cp` — the runner script handles this.

### Plotting stack — single source of truth

All trajectory figures (axis.png, PCA, etc.) import from `src/csp_div/plot_style.py`. That module owns:

- Basin thresholds + classifier (`DEEP_THRESHOLD = -0.5`, `SHALLOW_THRESHOLD = -0.4`, `basin_from_cos`, `trajectory_basin`)
- Colors (`DIPPER_COLOR = "tab:blue"`, `NONDIPPER_COLOR = "tab:red"`) and the legend's "Population 1" / "Population 2" labels
- Endpoint markers (`draw_endpoints` — open ○ at start, filled ● at end, same color as line)
- Axis cosmetics (`style_kl_axis`, `style_pc_axis` — log x, no top/right spines, dim grids, muted #888 edges) and `panel_title` (bold)
- rcParams applied at module-import time: Libertinus Serif (auto-registered if found, falls back to default serif), white background, savefig.bbox="tight", dpi 200

To restyle every figure, edit this one file. Don't reach into individual scripts to tweak colors / fonts / line widths.

### Multi-pod coordination (RunPod)

Per `~/.claude/projects/-workspace-csp-div/memory/project_runpod_workspace.md`:
- `/workspace/csp-div/` is **permanent** (network FS, shared across all pods that mount it)
- `/root/` is **transient** (per-pod scratch, lost on restart)
- Persistent venv at `/workspace/csp-div/.venv/` so pods don't need `pip install` per startup
- When cleaning shared dirs, **never use wildcards across other pods' seed ranges**: prefer explicit `rm -rf results/llama/seed_{10,11,...,19}` over `rm -rf results/llama/seed_*` to avoid wiping a sibling pod's in-progress data

## Other branches

- `rng-probe` — full historical static-teacher data with qwen, frame-bias sweep, RNG decoupling probe. Preserved.
- `chain-teacher`, `random-walk` — chain-teacher (KL-ascent against moving snapshot) experiments. Preserved.

`main` was rewritten from `rng-probe` to focus on the Llama static-teacher headline narrative; old commit history reachable via SHA but not on the new main HEAD.
