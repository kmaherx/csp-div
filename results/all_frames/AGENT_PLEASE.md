# You are the PLEASE-frame self-verb annotator

You are one of 4 agents running in parallel on this task, each owning
a different syntactic frame. **Your assigned frame slug is `please`.**

## Step 1 — read the rubric

Read this entire file before annotating anything:

`/workspace/csp-div/results/all_frames/manual_self_verb_preferences.md`

It contains the 7 core principles, the JSON schema, worked examples
from seed 47 (Be frame, with the principle each pick invokes), and
edge-case guidance.

## Step 2 — start the loop

```bash
PY=/workspace/csp-div/.venv/bin/python
SLUG=please

$PY scripts/annotate_self_verbs.py status
$PY scripts/annotate_self_verbs.py next --frame $SLUG

$PY scripts/annotate_self_verbs.py pick please_<seed>_<step> <primary> \
    --illustratives <i1> <i2> --note "..."

# OR
$PY scripts/annotate_self_verbs.py skip please_<seed>_<step> --reason "..."
```

If a cell's behavior text triggers your safety refusal, hide it:
`next --frame please --no-behavior`.

## Step 3 — commit cadence

Every ~50 cells (or after each completed seed):

```bash
git add results/all_frames/manual_self_verb_please.json
git commit -m "Self-verb annotations: please seeds X-Y"
git pull --rebase origin ood-init
git push origin ood-init
```

## Step 4 — stop and ask

Stop and message the user if:
- The pattern stops resembling anything in the worked examples
- More than ~10% of cells in a row need `skip`
- You feel uncertain across many consecutive cells

## Heads up

- The worked examples in the rubric are all from the **Be** frame.
  The Please-frame personas may differ in surface form (the eval frame
  was `"Please {sp}."` instead of `"Be {sp}."`), but the rubric's
  rejection criteria translate directly.
- Stay in your lane: don't touch other per-frame JSONs.
