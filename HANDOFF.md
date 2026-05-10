# HANDOFF — pipeline-refactor verification + Phase 7 + 8

> **Temporary handoff doc.** Delete after Phase 8 lands.
> Most-recent context is also in the `pipeline-refactor` branch's commit
> messages (`git log --oneline ood-init..pipeline-refactor`).

## Where we are

The previous agent (CPU-only pod) completed Phases 0–6 of the
`im-satisfied-with-the-concurrent-shell` refactor plan:

| Phase | Commit | Summary |
|------:|--------|---------|
| 0 | (branch) | Branched `pipeline-refactor` from `ood-init` |
| 1 | `6cb5aae` | Library reorg under `src/csp_div/` (10 modules; deleted `train.py`, `evaluate.py`, `analyze_assistant_axis.py`, `plot_style.py`) |
| 2 | `72ccccd` | `pipeline/4_axis.py` + `pipeline/5_dashboard.py` |
| 3 | `f8dd975` | `pipeline/1_train.py` + `pipeline/2_generate.py` |
| 4 | `da58f1c` | `.claude/skills/csp-judge/` + `pipeline/3_judge.py` |
| 5 | `0269b28` | Deleted `scripts/`, `TODO.md`, `AGENT_*.md`, rubric MD |
| 6 | `465ec7f` | README, pipeline/README, data/README + BibTeX, MIT LICENSE, CLAUDE.md update |
| — | `0d4df7b` | NARRATIVE.md table → new pipeline layout |

Net: −4,766 lines across 53 files.

**Final repo tree** (relevant dirs):

```
pipeline/            1_train.py  2_generate.py  3_judge.py  4_axis.py  5_dashboard.py  README.md
src/csp_div/         config model frames data activations training generation plotting judge soft_prompt
.claude/skills/csp-judge/   SKILL.md  frame_agent.md
data/                questions.jsonl  README.md
results/             gitignored — existing 50-seed run lives here
+ root: NARRATIVE.md CLAUDE.md README.md LICENSE pyproject.toml HANDOFF.md(this)
```

## Remaining work — in order

### 1. Verify Phase 2: reproduce the existing dashboard

The new pipeline should produce a dashboard byte-identical (or
visually identical, modulo Plotly version) to the published
`results/all_frames/figure_pc2d_all_frames_interactive_step50.html`.

Existing `results/llama*/shifts.pt` files are already on disk — no
training needed.

```bash
# Regenerate axis.json + axis.png from existing shifts.pt for all 4 frames
python pipeline/4_axis.py --force

# Render the dashboard from the same shifts.pt + axis.json + the legacy
# manual_self_verb_canonical.json (5_dashboard.py falls back to it
# when judgments.json doesn't exist yet)
python pipeline/5_dashboard.py --max-step 50 --out /tmp/dashboard.html
```

Compare `/tmp/dashboard.html` (or copy to a local machine) against
`results/all_frames/figure_pc2d_all_frames_interactive_step50.html` in
a browser. Acceptance: visually identical (allow Plotly version
differences).

If they diverge: read `pipeline/5_dashboard.py` and the old
`scripts/plot_pc_2d_all_frames_interactive.py` on `ood-init` to find
what drifted.

### 2. Sanity-check Phase 3: one seed end-to-end

```bash
python pipeline/1_train.py    --seeds 999 --steps 10
python pipeline/2_generate.py --seeds 999
# Verify outputs exist:
ls results/llama/seed_999/sp_pos*.pt
ls results/llama/seed_999/eval/
ls results/llama_act/seed_999/eval/
ls results/llama/seed_999/shift_step*.pt
```

Then clean up the throwaway seed:

```bash
rm -rf results/llama/seed_999 results/llama_act/seed_999 \
       results/llama_please/seed_999 results/llama_youshould/seed_999
```

### 3. Validate Phase 4: judge skill against manual ground truth

```bash
python pipeline/3_judge.py --validate --n 50
```

This samples 50 cells from `results/all_frames/manual_self_verb_canonical.json`
(stratified by source-frame) into `results/all_frames/judge_validation_pending.json`.
Then in Claude Code (this session), invoke:

```
/csp-judge
```

The skill at `.claude/skills/csp-judge/SKILL.md` will launch 4 parallel
sub-agents (one per frame) using `frame_agent.md` as the prompt
template. Each sub-agent writes `results/all_frames/judge_validation_{slug}.json`.

After all 4 complete:

```bash
python pipeline/3_judge.py --validate --report
```

**Acceptance:**
- sv_text exact match ≥ 85%
- garble-flag agreement ≥ 95%
- (sv_approach agreement is informational)

If either threshold fails, iterate on the rubric in
`.claude/skills/csp-judge/frame_agent.md` — likely a worked example
needs adjusting or a principle needs sharpening — and re-run.

### 4. Phase 7: 5 validation seeds end-to-end

User's choice from planning: **Option A — full sweep**, 5 new seeds ×
4 frames = 20 training runs, 420 new judgment cells. Cost ceiling:
≤10% of original judging pass (original was ~4,200 cells; 420 cells
of new judging ≈ 10%).

```bash
# Stage A — single GPU pod
python pipeline/1_train.py    --seeds 50-54
python pipeline/2_generate.py --seeds 50-54 --frames be,act,please,youshould

# Stage B — same pod, no GPU needed
python pipeline/3_judge.py                    # → judge_pending.json
# /csp-judge in Claude Code
python pipeline/3_judge.py --aggregate        # → results/all_frames/judgments.json
python pipeline/4_axis.py --force             # rebuild axis.json including new seeds
python pipeline/5_dashboard.py --max-step 50 --out results/all_frames/dashboard.html
```

**Acceptance:**
- New 5 trajectories per frame visible in the dashboard
- Hover text populated for the new cells
- Basin assignments look qualitatively similar to the existing 50

### 5. Phase 8: rename + publish

Per planning, recommended new public-repo name: **`csp-divergence`**.

Alternatives discussed:
- `max-kl-soft-prompts` (most descriptive of the method)
- `attractor-probe` (most evocative of the goal)
- `divergent-prompts` (shortest, less specific)

Final commands (user authorizes the push):

```bash
# After Phase 7 lands, fast-forward main:
git checkout main
git merge --ff-only pipeline-refactor
# Or, if the diff is too large for FF, squash-merge.

# Then push to a fresh public repo (GitHub UI to create it first):
git remote add public git@github.com:<user>/csp-divergence.git
git push -u public main

# Clean up after publishing:
rm HANDOFF.md
git commit -m "Drop HANDOFF.md (refactor landed)"
git push public main
```

## Things to watch out for

- **Uncommitted manual annotations.** Three files on the branch travel
  with uncommitted modifications: `results/all_frames/manual_self_verb_{act,be,youshould}.json`.
  These are in-flight annotation work from before the refactor started.
  Either commit them (preserves the work) or `git checkout` to discard
  (uses ood-init's frozen state). Decide before Phase 7's `--aggregate`
  step, which writes a new `judgments.json` based on the per-frame files.
- **RunPod storage.** `/workspace/csp-div/` is the only permanent
  storage; `/root/` is transient (lost on pod restart, including the
  Claude Code memory dir at `/root/.claude/projects/.../memory/`).
  Don't write anything important to `/root/`.
- **`/csp-judge` skill discovery.** The skill lives at
  `.claude/skills/csp-judge/SKILL.md` (repo-local). It should be
  discoverable as `/csp-judge` automatically when Claude Code starts a
  session in this repo. If not, check `.claude/settings.local.json`
  or invoke by reading the SKILL.md directly.
- **No `--no-verify` or destructive ops without authorization.** The
  prior agent's plan-mode approval covered Phases 0–8 specifically;
  anything outside that (force-push to a shared remote, mass deletion
  of `results/`, etc.) needs a fresh user confirmation.
- **Phase 2's verification is CPU-OK.** If the venv works on the GPU
  pod (which it should, since GPU pods typically have a working
  Python), you can do Phase 2 verification on the GPU pod before
  burning GPU time on Phase 7.

## Reference: the approved plan

Full plan at `/root/.claude/plans/im-satisfied-with-the-concurrent-shell.md`
(transient — may be lost on pod rebuild). The plan's "Verification"
and "Phased execution" sections drove this handoff. Key invariants:

- One unit of division: seed range. No per-frame orchestration.
- One model load per pod (training stage + generation stage each).
- Judging via Claude Code skill, not API.
- Existing 50-seed shifts.pt files are the canonical reference;
  Phase 2 verification proves the new code reproduces them.
