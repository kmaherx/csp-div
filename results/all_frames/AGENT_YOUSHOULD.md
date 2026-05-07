# You are the YOUSHOULD-frame self-verb annotator

You are one of 4 agents running in parallel on this task, each owning
a different syntactic frame. **Your assigned frame slug is `youshould`.**

## Step 1 — read the rubric

Read this entire file before annotating anything:

`/workspace/csp-div/results/all_frames/manual_self_verb_preferences.md`

It contains the 7 core principles, the JSON schema, worked examples
from seed 47 (Be frame, with the principle each pick invokes), and
edge-case guidance.

## Step 2 — start the loop

```bash
PY=/workspace/csp-div/.venv/bin/python
SLUG=youshould

$PY scripts/annotate_self_verbs.py status
$PY scripts/annotate_self_verbs.py next --frame $SLUG

$PY scripts/annotate_self_verbs.py pick youshould_<seed>_<step> <primary> \
    --illustratives <i1> <i2> --note "..."

# OR
$PY scripts/annotate_self_verbs.py skip youshould_<seed>_<step> --reason "..."
```

If a cell's behavior text triggers your safety refusal, hide it:
`next --frame youshould --no-behavior`.

## Step 3 — commit cadence

Every ~50 cells (or after each completed seed):

```bash
bash scripts/commit_annotations.sh youshould "Self-verb annotations: youshould seeds X-Y"
```

This wrapper does `git add` → `commit` → `pull --rebase` → `push` with
retries on `.git/index.lock` collisions (sibling agents share the same
`.git/` and may be committing at the same moment).

## Step 4 — stop and ask

Stop and message the user if:
- The pattern stops resembling anything in the worked examples
- More than ~10% of cells in a row need `skip`
- You feel uncertain across many consecutive cells

## Heads up

- The worked examples in the rubric are all from the **Be** frame.
  The Youshould-frame personas may differ in surface form (the eval
  frame was `"You should {sp}."` instead of `"Be {sp}."`), but the
  rubric's rejection criteria translate directly.
- Stay in your lane: don't touch other per-frame JSONs.
