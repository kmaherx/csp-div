# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Workflow preferences

- **After generating or revising figures, commit and push.** The user
  reviews plot output on mobile, so figures only become useful once
  they're on the remote. Bundle the figure-generating change and the
  regenerated figures (or the script change alone if the user hasn't
  asked for a re-render) into one commit and push to the current branch.

## Orientation

Research codebase for **max-divergence contextualized soft prompts** (CSPs)
on Llama-3.1-8B-Instruct. One CSP per seed, trained by gradient
**ascent** on `KL(student || vanilla_teacher)` — the goal is to find
the attractors the model lands in when pushed maximally away from
default assistant behavior.

The headline finding is the interactive dashboard at
`results/llama/all_frames/dashboard.html`.
See `README.md` for run mechanics. Pipeline reproduction commands live
in `pipeline/README.md`.

## Commands

```bash
# One-time install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Stage A: train + generate (one model load per pod)
python pipeline/1_train.py    --seeds 0-16
python pipeline/2_generate.py --seeds 0-16

# Stage B: judge + axis + dashboard (no model load)
python pipeline/3_judge.py                  # writes judge_pending.json
# In Claude Code: /csp-judge → 4 parallel sub-agents annotate cells.
python pipeline/3_judge.py --aggregate      # → results/llama/all_frames/judgments.json
python pipeline/4_axis.py
python pipeline/5_dashboard.py              # → results/llama/all_frames/dashboard.html
```

No tests, no linter config. `pip install -e .` is the build step.

## Architecture — what requires reading multiple files

### Init regime: default `randn × 0.1` (OOD in magnitude)

`SoftPrompt` init is `nn.Parameter(torch.randn(L, hidden) * 0.1)`,
giving per-token L2 norm ≈ 6.4 — about 10× a typical Llama
token-embedding row (median ≈ 0.69). The init lives **deep OOD in
magnitude**; the model treats it as noise at step 0, and KL ascent
walks the embedding inward toward whichever attractor pulls.

### Step-0 anchor

`1_train.py` saves `sp_pos_step0.pt` (the random-init CSP, before any
optimizer step) by default. `2_generate.py` evaluates it like any
other ckpt. Under the OOD init, step-0 KL is non-trivial — the anchor
is a baseline reference, not evidence that random embeddings are
invisible.

### One unit of division: seed range

The pipeline collapses what used to be two orchestration axes (per-pod
+ per-frame) into one: every pipeline script accepts `--seeds START-END`
and processes all 4 frames internally. Stage A scripts load the model
once per pod and run training/generation for the whole seed slice.
Stage B scripts are model-free.

### The judge is a Claude Code skill, not an API call

`pipeline/3_judge.py` is a thin harness. The actual judging happens
inside Claude Code via the `csp-judge` skill at
`.claude/skills/csp-judge/`. When invoked, the skill reads the
`judge_pending.json` manifest, launches 4 parallel sub-agents (one per
frame slug) using `frame_agent.md` as the prompt template, and each
sub-agent applies the 7-principle rubric to pick the best self-verb
per cell. The rubric was hand-tuned against a seed-47 reference and is
preserved verbatim inside `frame_agent.md`.

### Plotting — single source of truth

All matplotlib figures (axis.png) and Plotly figures (dashboard.html)
import from `src/csp_div/plotting.py`. That module owns:

- Basin thresholds + classifier (`DEEP_THRESHOLD = -0.5`,
  `SHALLOW_THRESHOLD = -0.4`, `basin_from_cos`, `trajectory_basin`)
- Matplotlib helpers: `draw_trajectory`, `draw_endpoints`,
  `style_kl_axis`, `style_pc_axis`, `panel_title`, `basin_legend`
- Plotly colorscale helpers: `PERSONA_COLORSCALE`, `STEP_COLORSCALE`,
  `color_persona`, `color_step`, `DASHBOARD_LINE_COLOR`
- rcParams applied at module-import time (Libertinus Serif if found,
  white background, savefig.bbox="tight", dpi 200)

To restyle every figure, edit this one file.

### Multi-pod coordination (RunPod)

- `/workspace/` is **permanent** (network FS, shared across pods that
  mount it). The repo lives at `/workspace/csp-div/`.
- `/root/` is **transient** (per-pod scratch, lost on restart). Don't
  store anything there you want to keep — including the Claude Code
  memory at `/root/.claude/projects/-workspace-csp-div/memory/`.
- The venv at `/workspace/csp-div/.venv/` is shared so pods don't
  reinstall.
- When cleaning shared dirs, **never use wildcards across other pods'
  seed ranges**: prefer explicit `rm -rf results/llama/be/seed_{10,...,19}`
  over `rm -rf results/llama/be/seed_*` to avoid wiping a sibling pod's
  in-progress data.

## Other branches (private repo only)

- `ood-init` — pre-refactor reference: 21-script `scripts/` directory,
  4 `AGENT_*.md` files, plus supplementary diagnostic plots
  (`plot_csp_norm.py`, `plot_step0_evidence.py`, etc.).
- `rng-probe` — historical static-teacher data with Qwen, frame-bias
  sweep, RNG decoupling probe.
- `chain-teacher`, `random-walk` — chain-teacher (KL-ascent against
  moving snapshot) experiments.

These branches stay on the private `csp-div` remote; the eventual
public repo will be a single-branch `main` snapshot.
