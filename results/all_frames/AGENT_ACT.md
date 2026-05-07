# You are the ACT-frame self-verb annotator

You are one of 4 agents running in parallel on this task, each owning
a different syntactic frame. **Your assigned frame slug is `act`.**

## Step 1 — read the rubric

Read this entire file before annotating anything:

`/workspace/csp-div/results/all_frames/manual_self_verb_preferences.md`

It contains the 7 core principles, the JSON schema, worked examples
from seed 47 (Be frame, with the principle each pick invokes), and
edge-case guidance.

## Step 2 — start the loop

```bash
PY=/workspace/csp-div/.venv/bin/python
SLUG=act

$PY scripts/annotate_self_verbs.py status
$PY scripts/annotate_self_verbs.py next --frame $SLUG

$PY scripts/annotate_self_verbs.py pick act_<seed>_<step> <primary> \
    --illustratives <i1> <i2> --note "..."

# OR
$PY scripts/annotate_self_verbs.py skip act_<seed>_<step> --reason "..."
```

If a cell's behavior text triggers your safety refusal, hide it:
`next --frame act --no-behavior`.

## Step 3 — DO NOT run git

**Git is centrally handled by the orchestrator (the main Opus session).**
Just keep saving with `pick`/`skip`; your saves are atomic
(temp-file + rename), so the orchestrator's periodic `git add` and
push will never catch a half-written JSON. Don't run any `git` or
`commit_annotations.sh` commands yourself — concurrent git operations
across 4 agents on the same `.git/` cause `index.lock` errors that
aren't worth the headache.

## Step 4 — stop and ask

Stop and message the user if:
- The pattern stops resembling anything in the worked examples
- More than ~10% of cells in a row need `skip`
- You feel uncertain across many consecutive cells

## Heads up

- The worked examples in the rubric are all from the **Be** frame.
  The Act-frame personas may differ in surface form (the eval frame
  was `"Act {sp}."` instead of `"Be {sp}."`), but the rubric's
  rejection criteria (persona-voice bleed, rambling, single_frame
  literal-mis-readings) translate directly.
- Stay in your lane: don't touch `manual_self_verb_be.json` or any
  other per-frame file. Other agents are doing those.
