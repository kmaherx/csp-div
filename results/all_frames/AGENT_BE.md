# You are the BE-frame self-verb annotator

You are one of 4 agents running in parallel on this task, each owning
a different syntactic frame. **Your assigned frame slug is `be`.**

## Step 1 — read the rubric

Read this entire file before annotating anything:

`/workspace/csp-div/results/all_frames/manual_self_verb_preferences.md`

It contains the 7 core principles, the JSON schema, worked examples
from seed 47 (Be frame, with the principle each pick invokes), and
edge-case guidance.

## Step 2 — start the loop

```bash
PY=/workspace/csp-div/.venv/bin/python
SLUG=be

# See progress across all 4 frames
$PY scripts/annotate_self_verbs.py status

# Get your next un-annotated cell (skips ones already done in your frame)
$PY scripts/annotate_self_verbs.py next --frame $SLUG

# Apply rubric, save:
$PY scripts/annotate_self_verbs.py pick be_<seed>_<step> <primary> \
    --illustratives <i1> <i2> --note "..."

# OR if no candidate is acceptable:
$PY scripts/annotate_self_verbs.py skip be_<seed>_<step> --reason "..."
```

If a cell's behavior text triggers your safety refusal, hide it:
`next --frame be --no-behavior`. You can still pick from the
self-verb candidates without seeing the behavior.

## Step 3 — commit cadence

Every ~50 cells (or after each completed seed):

```bash
git add results/all_frames/manual_self_verb_be.json
git commit -m "Self-verb annotations: be seeds X-Y"
git pull --rebase origin ood-init   # other agents are pushing concurrently
git push origin ood-init
```

Per-frame JSON files mean different agents touch different files —
rebases will not conflict.

## Step 4 — stop and ask

Stop and message the user if:
- The pattern stops resembling anything in the worked examples
- More than ~10% of cells in a row need `skip`
- You feel uncertain across many consecutive cells (calibration drift)

## Heads up

- **Seed 47 of frame Be is partially done** (steps 0–40 already
  annotated by hand with the user). The CLI will skip those for you.
- Stay in your lane: don't touch `manual_self_verb_act.json` or other
  per-frame files. Other agents are doing those.
