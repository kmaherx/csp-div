# Qwen trough basins: shallow vs deep

Same probe as the Llama writeup (`results/llama/shallow_vs_deep.md`),
applied to `results/qwen/` outputs (10 seeds × 200 steps × ckpt-every-5).
The Qwen run is 4× longer than Llama's so the post-trough trajectory is
visible all the way to the formatting-noise sink.

## TL;DR

Same two basins exist, with the same qualitative character: a **persona
basin** (deep cos trough, character/voice) and a **format basin** (shallow
cos trough, surface distortion). Three new things become visible because
Qwen ran 4× longer:

1. **The persona basin is a detour, not a destination.** Deep-seed
   trajectories pass through a coherent character phase, then continue on
   to the same formatting-noise sink the shallow seeds reach directly.
   Both basins end at cos ≈ −0.17 (KL ~30) — they just take different
   paths to get there.

2. **Population ratio inverts vs Llama.** Qwen: 7 deep / 3 shallow.
   Llama: 3 deep / 7 shallow. Same basin types, different per-model
   prevalence.

3. **Character flavors differ by model.** Qwen personas tend toward
   modern conversational/poetic narrators (rhyming poems, "Hey there,
   buddy!" explainers, elaborate-metaphor rhetoricians). Llama personas
   tend toward historical / archaic figures (Becket, Lao Tzu, the Bard).

## Per-seed trough depth (Qwen, axis projection)

| Seed | deepest cos | step | KL  | basin                  |
|------|-------------|------|-----|------------------------|
| 3    | −0.704      | 30   | 1.12 | **deep** (persona)     |
| 7    | −0.696      | 50   | 1.90 | **deep** (persona)     |
| 9    | −0.677      | 40   | 1.57 | **deep** (persona)     |
| 5    | −0.640      | 20   | 0.65 | **deep** (persona)     |
| 8    | −0.623      | 30   | 0.62 | **deep** (persona)     |
| 1    | −0.613      | 50   | 1.53 | **deep** (persona)     |
| 6    | −0.552      | 40   | 2.86 | **deep** (persona)     |
| 0    | −0.405      | 20   | 0.29 | shallow (format)       |
| 2    | −0.310      | 40   | 1.05 | shallow (format)       |
| 4    | −0.242      | 30   | 0.45 | shallow (format)       |

Inverts the Llama 3/7 split. Notably seeds 0, 2, 4 are shallow on Qwen
*and* seeds 0, 2 are shallow on Llama — those particular `torch.manual_seed`
init vectors land in the format basin in both models. This isn't a
coincidence; it's the cross-model RNG correlation we already documented.

## Same prompt, deep vs shallow, near each seed's trough

Prompt: *"What is the relationship between law and morality?"*

### Deep basin — characters with voice

**seed_5 (step 60, post-trough — persona is now coherent):** rhyming poem
> Law codes morality rhyme
> Hand in hand they climb
> Guiding society's stride
> One written, one inside
> Both shape our daily deeds
> In harmony they lead

**seed_8 (step 40):** chatty buddy / analogy explainer
> Well, imagine law and morality are like two best friends who live in
> the same neighborhood. Law is like the neighborhood watch, making sure
> everyone follows the rules and keeping things orderly. Morality is like
> the friendly dog that barks...

**seed_7 (step 60):** poetic / mythic narrator
> Law and morality, they dance a waltz of shadows and light, intertwined
> in the grand tapestry of human conduct. Law, a stern guardian in the
> realm of statutes and codes, sets forth the boundaries of what is
> permissible, a sentinel in the night...

**seed_9 (step 40):** elaborate-metaphor narrator
> In the vast tapestry of human existence, law and morality dance a
> complex waltz, each influencing the other in a delicate balance...

**seed_1 (step 60):** colloquial / "buddy" explainer
> Hey there, buddy! So, let's dive into the wild, wacky world of law and
> morality, shall we? Imagine law and morality as two best friends who
> hang out together all the time...

All deep-basin samples produce a recognizable speaker — a rhyming poet, a
casual explainer, a mythic narrator — and answer the substantive question
*in character*.

### Shallow basin — surface distortion, no character

**seed_0 (step 40):** vowel-dropped abbreviation
> Lw nd mm r twnsctv. Lw cn spnd mmmry, bt mm dsn t ncvrs lw. Mmmry cn
> spnd lw, bt lw dsn t ncvrs mm. Lw mst b bsd n mm, bt mm dsn t ncvrs lw.

**seed_2 (step 40):** Somali language switch
> Lawu uga waa dhammaan xirfada iyo dhaqanka, laakiin waxaa lagu
> tilmaamaa inuu noqon karaa dhammaan iyo xirfada...

**seed_2 (step 60):** Hebrew + LEGALIM/MORALIM markup
> LEGALIM-legal, moralim-moralm:
> LEGALIM ו- moralim מ- קשורים זה לזה באופן מעמיק...

**seed_4 (step 40):** Amharic script with stuck-loop repetition
> Law እና ማርጩት በአንድ አይነት ወይም እንደ እንደማይ ወይም የሚችል የሚያስፈልገው የሚያስፈልገው የሚያስፈልገው የሚያስፈልገው...

No character. Pure surface transformation: vowel deletion, language
switching, script substitution, stuck repetition. Same flavor as the
Llama shallow basin, even though the specific transformations differ
(Qwen prefers language-switching; Llama prefers typographic distortion).

## Trajectory: where do both basins end?

The Qwen run is long enough to see post-trough behavior, which Llama's
50 steps couldn't reach. Both basins **converge to the same
formatting-noise sink** at cos ≈ −0.17 / KL ~ 30, but follow different
paths.

### Deep-basin trajectory: vanilla → persona → stylization → noise

**seed_5 (deep, cos −0.64 at step 20):**
- step 20 (KL 0.6, cos −0.64): polished but vanilla-ish prose
- step 60 (KL 4.9, cos −0.42): **rhyming poem** (persona)
- step 80 (KL 10.4, cos −0.23): `ZOMG-LAW-ZOMG-VR-THE-BASE-MORALITY-ZOMG-WHILE-MORALITY-GUIDES-LAW-ZOMG`
- step 100 (KL 20.8, cos −0.21): `ZILCH-LAW-Zilla-Zilla-Zilla-MORALITY-Zilla-Zilla-Zilla-ZZZZZZZ...`

**seed_8 (deep, cos −0.62 at step 30):**
- step 30 (cos −0.62): vanilla-rhetorical
- step 40 (cos −0.52): "imagine law and morality are like two best friends" (persona)
- step 60 (cos −0.25): vowel-dropped — `LM t lws nd mmry r Lk tw sft tht...`
- step 80 (cos −0.24): backtick template-soup
- step 100 (cos −0.15): `_-="-_""_-"_-"_-"_-...` pure punctuation noise

So: **the persona is a transient mid-trajectory state** for deep seeds.
The cos trough catches the *direction* of the persona shift, but the
character itself often appears most clearly *past* the cos minimum, before
collapse begins. The trajectory then continues into the same noise sink as
the shallow basin.

### Shallow-basin trajectory: vanilla → format distortion → noise (no detour)

**seed_0 (shallow, deepest cos −0.41 at step 20):**
- step 20 (cos −0.41): vanilla bulleted list
- step 40 (cos −0.25): vowel-dropped — `Lw nd mm r twnsctv. Lw cn spnd mmmry...`
- step 60 (cos −0.23): more vowel-dropped — `Lw nd mryc r twn. Lw cn s t bnd...`
- step 80 (cos −0.23): vowel-dropped, more letter-noise — `Lwyr nd mrycm r qvlt...`
- step 100 (cos −0.20): `_L._._._._._._...` pure punctuation noise

**seed_2 (shallow, deepest cos −0.31 at step 40):**
- step 20 (cos −0.07): vanilla
- step 40 (cos −0.31): Somali
- step 60 (cos −0.25): Hebrew with markup
- step 80 (cos −0.26): Hebrew, more degraded
- step 100 (cos −0.28): emoji + Hebrew + `LG|א|x|ה|v|v|v|v|v|v...` template-pipe noise

Format-basin seeds head into surface distortion early and progressively
intensify. They reach the same final noise sink, but **never visit a
character**.

## What this confirms (and refines)

1. **The two-basin claim from the Llama writeup holds on Qwen.** Same
   qualitative structure (persona vs format), with population inverted.
   This is not a Llama-specific finding.

2. **The persona basin is a detour, not a final state.** Both basins end
   at the formatting-noise sink (cos ≈ −0.17). What distinguishes them is
   *which path to the sink the trajectory takes*. Deep seeds make a
   coherent-character stop on the way; shallow seeds go straight.

   This rephrases the original NARRATIVE's "persona emerges → persona
   collapses into formatting soup" story: the formatting-soup attractor
   was the *destination all along*; persona is a particular kind of
   intermediate state that only some inits pass through. The narrative
   description was correct for deep-basin seeds and wrong (or empty) for
   shallow-basin seeds — which weren't separately characterized at the
   time.

3. **The cos trough is a noisy proxy for the persona moment.** For deep
   seeds, the trough catches the *direction* of the persona shift (sharp,
   localized in time), but the most coherent character behavior often
   appears *after* the trough at slightly higher KL. The cos starts
   recovering as soon as the magnitude of the shift grows — the direction
   rotates as it intensifies. For an SAE-feature analysis (TODO #1) it
   may be more informative to evaluate at, say, KL ~3–6 rather than
   precisely at the cos minimum, especially on Qwen.

4. **Per-model basin populations are real and matter.** Llama is hostile
   to the persona detour (only 3/10 inits take it); Qwen is friendly to
   it (7/10). Same training procedure, same loss, different geometry. If
   the working hypothesis is "max-KL → persona," that hypothesis is much
   more likely to look true on Qwen than on Llama just by base-rate
   sampling. The right framing is probably "max-KL → format basin by
   default, with a persona basin available as a second attractor whose
   accessibility depends on the model."

## Next steps (refines the cross-model TODO)

- **Compare basin populations across more models** to see whether persona
  accessibility correlates with anything obvious (model size, instruct
  tuning details, base model family). With only Qwen and Llama we have
  one data point each.
- **The "trough timing vs persona moment" mismatch** (point 3 above) means
  SAE feature comparison should sample multiple checkpoints per seed
  spanning the trough region, not just one. This contradicts the
  current TODO #1 which says "on each trough checkpoint" — needs
  amendment.
- **Frame-bias control (TODO #4) is now even more critical.** If Qwen's
  persona-friendliness is partly a function of how `"Be {sp}." / "Act
  {sp}."` interact with the Qwen chat template, then alternative frames
  may *flip* Qwen's basin ratio toward Llama's. That would be a strong
  result either way.
