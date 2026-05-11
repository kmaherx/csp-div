# Pipeline

Five numbered Python scripts, run in order. Each is a stage of the
end-to-end CSP-divergence experiment. The whole pipeline is two stages
of work:

  - **Stage A (per-pod, per-seed-range)**: `1_train.py` + `2_generate.py`.
    Each pod loads Llama-3.1-8B-Instruct once and runs both scripts for
    its slice of seeds across all 4 frames.
  - **Stage B (single pod, post-everything)**: `3_judge.py` +
    `4_axis.py` + `5_dashboard.py`. No model loads — pure judging
    (via the Claude skill), PCA aggregation, and dashboard rendering.

The pipeline is **idempotent**: each script skips outputs that already
exist. Delete the relevant files (or pass `--force`) to regenerate.

## Quickstart

```bash
# One-time install (Python 3.10+, GPU recommended for steps 1 + 2)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Stage A — split across however many pods you have.
# Example with 3 pods (seeds 0–49):
python pipeline/1_train.py    --seeds 0-16        # pod A
python pipeline/1_train.py    --seeds 17-33       # pod B
python pipeline/1_train.py    --seeds 34-49       # pod C

# Then, same per-pod sharding:
python pipeline/2_generate.py --seeds 0-16
python pipeline/2_generate.py --seeds 17-33
python pipeline/2_generate.py --seeds 34-49

# Stage B — single pod, no GPU needed.
python pipeline/3_judge.py                       # write manifest
# In Claude Code, invoke /csp-judge — 4 parallel sub-agents write
# per-frame judgments to results/llama/<frame>/judge.json.
python pipeline/3_judge.py --aggregate           # → judgments.json
python pipeline/4_axis.py                        # axis.json + axis.png per frame
python pipeline/5_dashboard.py                   # → results/llama/all_frames/dashboard.html
```

For a single seed end-to-end (~30 minutes on one A100):

```bash
python pipeline/1_train.py    --seeds 0
python pipeline/2_generate.py --seeds 0
python pipeline/3_judge.py
# /csp-judge in Claude Code
python pipeline/3_judge.py --aggregate
python pipeline/4_axis.py
python pipeline/5_dashboard.py
```

## Stage details

### `1_train.py` — train one CSP per seed

KL-ascent training against the vanilla teacher. For each seed, writes
21 checkpoints (`sp_pos_step{0,5,...,100}.pt`) to
`results/llama/be/seed_{N}/`. The step-0 anchor is the random-init CSP
before any optimizer step. CSPs are frame-agnostic but live under the
canonical `be/` slot — `2_generate.py` reads from there for every eval
frame.

Cached vanilla teacher responses live at
`results/llama/cached_responses.json` (shared across seeds and pods).

### `2_generate.py` — behavior + self-verb + shift capture

For each (seed, frame, ckpt), generates:

- `eval/behavior_step{K}.json` — CSP and vanilla side-by-side on 5 held-out prompts
- `eval/self_verb_step{K}.json` — 9 self-verbalization completions
- `shift_step{K}.pt` — residual-stream shift at L16 for axis projection

Vanilla baseline (the comparator for shifts) is computed once and
cached at `results/llama/vanilla_baseline.pt`.

### `3_judge.py` — Claude-skill orchestration harness

Three modes:

```bash
python pipeline/3_judge.py               # discovery: write judge_pending.json
python pipeline/3_judge.py --validate    # sample 50 cells against manual ground truth
python pipeline/3_judge.py --aggregate   # fold per-frame judgments → judgments.json
```

The actual judging happens **inside Claude Code**: invoke `/csp-judge`,
which launches 4 parallel sub-agents (one per frame slug) using
`.claude/skills/csp-judge/frame_agent.md` as the prompt template. Each
sub-agent picks the best self-verb candidate per cell, writing to
`results/llama/{slug}/judge.json`.

### `4_axis.py` — axis projection

Loads the Butanium assistant axis at L16, projects each shift onto it,
emits per-frame `axis.json` (proj_dot, proj_cos) + `axis.png`.

### `5_dashboard.py` — published dashboard

Pools shifts across all 4 frames, fits PCA, renders the interactive
HTML dashboard at `results/llama/all_frames/dashboard.html` with:

- Seed / step / frame filters
- Color-mode toggle (optimization step ↔ persona-strength alignment)
- Hover sidebar showing behavior + self-verb responses
- Persona presets for notable trajectories

## Output tree

```
results/llama/
├── cached_responses.json       # vanilla teacher cache (shared, frame-agnostic)
├── vanilla_baseline.pt         # mean L16 acts under no frame (shared)
├── be/                         # canonical frame — also holds the training ckpts
│   ├── seed_0/
│   │   ├── sp_pos.pt           # final CSP (frame-agnostic; read by every frame)
│   │   ├── sp_pos_step{0,5,..,95}.pt
│   │   ├── shift_step{0,5,..,100}.pt
│   │   └── eval/
│   │       ├── behavior_step{0,5,..}.json
│   │       └── self_verb_step{0,5,..}.json
│   ├── ...
│   ├── shifts.pt               # consolidated by 4_axis.py
│   ├── axis.json               # axis projection per (seed, ckpt)
│   ├── axis.png                # 2-panel diagnostic figure
│   └── judge.json              # per-frame judgments (written by skill)
├── act/                        # same layout, no training ckpts (reads be/'s)
├── please/
├── youshould/
└── all_frames/
    ├── judge_pending.json      # written by 3_judge.py
    ├── judgments.json          # canonical, by 3_judge.py --aggregate
    └── dashboard.html          # published artifact
```

## Multi-pod tips

`/workspace/csp-div/` is permanent storage on RunPod and is shared
across pods that mount it. The vanilla cache + baseline at
`results/{cached_responses.json, vanilla_baseline.pt}` are computed
once and reused. Per-seed outputs live under disjoint `seed_{N}/`
directories so multiple pods running disjoint seed ranges never
collide.

If you cancel and resume, just re-run the same script — the
skip-if-exists logic picks up where you left off. To force a clean
regen, delete the relevant outputs first (or pass `--force`).
