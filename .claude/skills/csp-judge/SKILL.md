---
name: csp-judge
description: Annotate self-verb completions for CSP-divergence runs. Reads the pending-cells manifest written by `pipeline/3_judge.py`, launches one Claude Code sub-agent per frame slug (be / act / please / youshould), each applying the rubric in `frame_agent.md` to pick the best self-verb per cell. Invoke after `pipeline/3_judge.py` reports cells lacking judgments.
---

# csp-judge — orchestrator

## When this fires

`pipeline/3_judge.py` writes one of two manifests:

- `results/all_frames/judge_pending.json` — the normal "judge new seeds" mode.
- `results/all_frames/judge_validation_pending.json` — the validation
  mode, comparing skill picks against a 50-cell manual ground-truth
  sample.

Read whichever exists. Validation manifest takes priority if both are
present (its purpose is faster, lower-volume comparison).

## What you do

The manifest has this shape:

```json
{
  "mode": "generate" | "validate",
  "cells_by_frame": {
    "be":        [{"key": "be_50_5", "seed": 50, "step": 5,
                   "self_verb_path": "...", "behavior_path": "..."}, ...],
    "act":       [...],
    "please":    [...],
    "youshould": [...]
  },
  "judgment_paths": {
    "be":        "results/llama/judge.json",
    "act":       "results/llama_act/judge.json",
    "please":    "results/llama_please/judge.json",
    "youshould": "results/llama_youshould/judge.json"
  }
}
```

For each non-empty `cells_by_frame[slug]`, launch **one sub-agent** with
the Agent tool. The four agents run **in parallel** (single message with
multiple Agent tool uses) — they touch disjoint files, so there are no
write races.

Per-agent prompt: use `.claude/skills/csp-judge/frame_agent.md` as the
template. Substitute the slug-specific tokens:

| Placeholder | Value (example: `act`) |
|-------------|------------------------|
| `{frame_slug}` | `act` |
| `{frame_template}` | `Act {sp}.` |
| `{manifest_path}` | `results/all_frames/judge_pending.json` (or validation) |
| `{judgment_path}` | `results/llama_act/judge.json` (for validation mode: `results/all_frames/judge_validation_act.json`) |

The four `{frame_template}` values are:

- be → `Be {sp}.`
- act → `Act {sp}.`
- please → `Please {sp}.`
- youshould → `You should {sp}.`

Pass these into the Agent prompt by reading `frame_agent.md`, replacing
the placeholders inline, and using the resulting text as the Agent's
`prompt` argument. Use `general-purpose` as the agent type — the
sub-agents just read files, apply the rubric, and write JSON.

## After agents complete

Each agent reports back with a count of (picked, skipped, total). Tally
across the four and report to the user:

```
csp-judge complete:
  be:        N picked,  M skipped   (T total)
  act:       ...
  please:    ...
  youshould: ...
  ----------
  TOTAL:     N picked,  M skipped   (T total)
```

Then remind the user of the next step:

- For `generate` mode: `python pipeline/3_judge.py --aggregate` to fold
  the per-frame judgments into `results/all_frames/judgments.json`.
- For `validate` mode: `python pipeline/3_judge.py --validate --report`
  to compute agreement against the manual ground truth.

## Constraints

- Never run git commands yourself; the user handles version control.
- Don't modify cells outside your assigned frame slug — the sub-agents
  enforce this by writing only to their `{judgment_path}`.
- If any agent reports the rubric stops resembling reality (>10% skips
  in a row), it will stop and surface that observation. Don't override
  it; surface that observation back to the user verbatim.
