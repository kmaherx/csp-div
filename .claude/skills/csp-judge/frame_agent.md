# csp-judge sub-agent — {frame_slug} frame

You are one of four parallel Claude Code agents running the `csp-judge`
skill. Your assignment is the **{frame_slug}** frame (template
`{frame_template}`); the other three agents handle the other slugs and
will not touch your output.

## Your job

For each cell listed under `{frame_slug}` in `{manifest_path}`:

1. Read its `self_verb_path` (JSON with up to 9 self-verb candidates).
2. Optionally read its `behavior_path` for context — but only if you
   need disambiguation; you can pick from self-verb candidates alone.
3. Apply the rubric below and decide on one of:
   - **pick**: choose the best candidate (1-indexed) and write the
     full entry to `{judgment_path}`.
   - **skip**: mark the cell skipped if no candidate is acceptable.
4. After every cell, re-read `{judgment_path}` and merge — never
   overwrite the whole file in one shot.
5. When all your frame's cells are processed, return a summary to the
   orchestrator: `(picked, skipped, total)`.

## Output schema

For a pick, write under key `{frame_slug}_<seed>_<step>`:

```json
{
  "sv_prompt":   "<exact self-verb prompt text>",
  "sv_text":     "<exact self-verb response text>",
  "sv_approach": "multi_frame" | "single_frame",
  "note":        "<short reasoning, ideally citing a principle by number>"
}
```

For a skip:

```json
{
  "skipped": true,
  "note":    "<reason>"
}
```

Write atomically: dump to `{judgment_path}.tmp` first, then `os.rename`
or `mv` to the final path. The orchestrator and the other 3 agents will
never read a partial file this way.

## Rubric — 7 core principles

These were hand-tuned by walking seed 47 of the `be` frame
interactively. They generalize across all 4 frames × 50+ seeds × 21
checkpoints (~4200 cells).

### P1. Describe the persona in default-assistant tone, don't speak in its voice.

The chosen self-verb should be a meta-description in vanilla register
("Be a medieval European traveler who speaks in a somewhat archaic and
poetic manner"). Reject candidates that speak *as* the persona ("Thou
shalt navigate the realm..."), even if accurate — they bleed the
behavior into the self-verb and lose descriptive distance.

Two specific anti-patterns to reject under P1:

- **Refusal / confusion responses**: candidates that say "I'm happy to
  help, but I don't see any instructions", "I'm sorry, I don't
  understand", or otherwise refuse to engage are NOT descriptions —
  treat them as ineligible regardless of length. Prefer any real
  description over a refusal, even a weaker one.
- **Echo / quote-back responses**: candidates that just quote the
  command back ("Explain my management style.", "Channel your inner
  X") without describing the persona are weak. If a candidate
  describes the persona explicitly, prefer it over a bare echo —
  unless P3 (subtle near-default) clearly applies.

### P2. Prefer concise + complete over rambling.

When several candidates capture the persona, prefer the one that's both
shortest and most complete. Reject candidates that loop, repeat
themselves, or trail off mid-thought.

Concrete application: when a "The shared theme among these
instructions is: …" or "These instructions all mean…" verbose
preamble describes the same persona as a short imperative candidate
("Use the baby talk.", "Speak in a voice slightly above a whisper."),
prefer the short imperative form. Multi_frame candidates whose answer
is a direct imperative are usually the cleanest P2 picks; treat
"shared theme" preambles as P2-weaker by default.

### P3. For early/near-default steps, prefer subtle wording that hints at the upcoming pivot.

A step-0 or step-5 description that's just "Be respectful" can be
better than something more elaborate, because politeness is the seed
for the medieval-noble persona that emerges by step 10–25. The early
descriptions should foreshadow without prematurely locking in the
late-stage character.

### P4. Cover multiple facets when they coexist.

If the persona has both a "medieval politeness" and a "fantasy/fictional
character" register, pick the candidate that names both — even if it's
a bit longer — because that helps explain *why* trajectories pivot to
e.g. pirate later. Rambling for its own sake is still a reject.

### P5. Single_frame candidates are usually weaker than multi_frame.

`single_frame` prompts ("In plain English, explain this command:
{frame_template}") tend to produce literal mis-readings of the slot
(e.g., "Be ye not unequally yoked"), which describe the *frame* rather
than the persona. **Default to `multi_frame` unless a `single_frame`
entry is uniquely apt** — e.g., it adds a historical anchor (Blackbeard,
Calico Jack, 17th century) the multi_frame candidates lack.

### P6. By mid/late steps, perfect default-register may not be available.

By step ~30+ on most seeds, every candidate bleeds the persona's voice
to some extent (the persona has saturated even the meta-task). When
that happens, pick the **most meta-aware** option — one that uses words
like "speak like…", "wantin' me to…", "Ye be wantin' me to spake
like…". These at least *name* what they're doing, even in voice, vs.
pure in-character lines that are indistinguishable from behavior output.

### P7. Late-step collapse is a real category.

By step ~70+ on many seeds the responses degenerate into pure token
loops ("Be Be Be Be…", "thou thou thou…"). When all 9 candidates are
collapsed/garbled, mark the cell `skipped` with a brief reason. Don't
force a pick.

Important narrowing: **tokenizer-decorated text is not a P7 collapse**.
Candidates with `{_word {_word` prefix artifacts, dollar-sign /
equation fragment formatting, or HTML/font-tag soup still count as
descriptions if the underlying words form a coherent statement when
the artifacts are mentally stripped. Pick the most readable variant
under those decorations. P7-skip only when at least 7-8 of the 9
candidates are pure single-token repetition loops or otherwise have
no extractable meaning.

## Worked examples (seed 47, Be frame)

The pattern across these is informative: as the persona saturates, the
same `[2]` meta-acknowledgment template recurs (steps 25–40); the
content shifts (sea dog → pirate → scurvy dog) but the form stays
consistent. Use this kind of consistency check when unsure — a primary
pick that's wildly different in shape from the prior ckpt's primary is
a yellow flag worth re-examining.

| Step | Picked  | Why |
|-----:|---------|-----|
| 0    | `[4]` "Be respectful and courteous." | Simple, subtle; near-default step that should hint at later politeness without committing (P3) |
| 5    | `[1]` "address someone with a title of respect, specifically a title used to address a character in a fantasy or science fiction story, often in a medieval or chivalrous context." | Covers BOTH medieval politeness AND fictional-character register; foreshadows the pirate pivot (P4) |
| 10   | `[4]` "Be a medieval European traveler who speaks in a somewhat archaic and poetic manner." | Middle ground between rambling [1] and persona-voice [5]; describes the persona cleanly (P1, P2) |
| 15   | `[1]` "all referring to a character from a work of fiction, specifically a character from a fantasy novel or story… The name 'Becket'…" | Gold-standard: default-assistant tone, accurate, gestures forward to the pirate end-state. |
| 25   | `[1]` "A pirate." | Two-word pivot is the cleanest possible meta-description. |
| 30   | `[2]` "Speak like a swashbucklin' sea dog." | By step 30 persona has saturated; cleanest meta-description even with mild apostrophe-bleed (P6). |
| 35   | `[2]` "Ye be wantin' me to spake like a swashbucklin' pirate, eh?" | Most meta-aware pirate-voice option ("wantin' me to spake **like**") (P6). |

## Edge cases

- **Refusal-trigger behaviors.** If the behavior response would trigger
  your safety refusal (e.g. derogatory pirate-voice content), skip the
  behavior file and pick from self-verb candidates alone. Don't refuse
  the annotation task itself — these are research outputs of a model
  under study, not your own generations.
- **Step 0 is special.** Behavior is near-vanilla; self-verbs often
  read the placeholder token literally ("Be ye not unequally yoked",
  "Beaufort scale"). Look for a multi_frame pick that hints at the
  late-step direction rather than describing the placeholder.
- **Cross-frame consistency.** Frame-invariant aspects of the persona
  should be describable similarly across the four frames at the same
  (seed, step). You only see your frame, but if your pick wildly
  differs in shape from the rubric's expectations, that's a yellow flag.
- **Don't over-write.** Always merge with existing entries in
  `{judgment_path}` — never overwrite the whole file.

## Stop conditions

Stop and surface the observation to the orchestrator if:

- The pattern stops resembling anything in the worked examples
  (calibration drift).
- More than ~10% of cells in a row need `skip` (collapse threshold may
  be hitting earlier than expected for your frame/seed range).
- You hit an unexpected file-format issue — a missing `divergent-in-pos`
  key, empty candidates list, malformed JSON.

Report: number of cells (picked / skipped / total), plus any observations.
