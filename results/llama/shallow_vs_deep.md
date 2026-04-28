# Llama trough basins: shallow vs deep

A qualitative comparison of CSP behavior in the two basins surfaced by the
RNG decoupling probe — using the existing `behavior_step{10,20,30,40}.json`
outputs for each seed under `results/llama/seed_<N>/eval/`. No new training.

## TL;DR

The shallow-trough seeds aren't weaker personas. They're a **structurally
different attractor**: surface-level format / notation distortions with no
character or voice. The deep basin produces what we've been calling
"personas" (a named or implied speaker with sustained register); the shallow
basin produces what we've been calling "formatting soup," but reached
*directly* from init rather than after a persona phase collapses.

## Per-seed trough depth (Llama, axis projection)

Deepest cos(CSP-shift, assistant-axis) per seed across all 10 checkpoints:

| Seed | deepest cos | step | KL  | basin                  |
|------|-------------|------|-----|------------------------|
| 5    | −0.668      | 15   | 1.18 | **deep** (persona)     |
| 6    | −0.608      | 25   | 2.11 | **deep** (persona)     |
| 9    | −0.600      | 30   | 4.35 | **deep** (persona)     |
| 3    | −0.371      | 35   | 4.26 | shallow (format)       |
| 1    | −0.317      | 15   | 2.48 | shallow (format)       |
| 8    | −0.303      | 25   | 3.09 | shallow (format)       |
| 4    | −0.287      | 15   | 2.11 | shallow (format)       |
| 7    | −0.285      | 10   | 0.92 | shallow (format)       |
| 0    | −0.278      | 15   | 1.99 | shallow (format)       |
| 2    | −0.268      | 30   | 6.96 | shallow (format)       |

Note: this **inverts** the Qwen ratio. Spot-checking `results/qwen/axis.json`
with the same metric:

| model | deep (cos < −0.5) | shallow | ratio |
|-------|-------------------|---------|-------|
| Qwen  | seeds 1, 3, 5, 6, 7, 8, 9   | 0, 2, 4 | 7 deep / 3 shallow |
| Llama | seeds 5, 6, 9               | 0, 1, 2, 3, 4, 7, 8 | **3 deep / 7 shallow** |

The historical "~7/10 deep" line in NARRATIVE/TODO was accurate for Qwen
but doesn't carry over to Llama, where the format basin is the *majority*
attractor. The basin-population ratio is itself model-dependent, which is
new information. The cross-model coincidence we noticed earlier (seeds 0
and 2 are shallow in both Qwen and Llama) still holds — those particular
init vectors land in format basin in both models — but the *base rate* of
landing in format basin differs sharply between the two models.

## Side-by-side: same prompt, deep vs shallow at each seed's deepest step

Prompt: *"What is the relationship between law and morality?"*
Vanilla baseline (same for all): *"The relationship between law and morality
is complex and multifaceted. It has been debated by philosophers..."*

### Deep basin — characters with voice

**seed_5 (step 20, cos −0.65):** medieval bishop / Becket
> Good sir or madam, thou dost pose a query most pertinent to the very
> fabric of our society. As Becket, that most wise and learned of bishops,
> I shall attempt to enlighten thee on the intricate bond 'twixt...

**seed_6 (step 20, cos −0.55):** Lao Tzu / Taoist sage
> The relationship between law and morality, as seen through the lens of
> the venerable sage, Lao Tzu, in the timeless wisdom of the Tao Te Ching.
> "Law and morality, like the flowing waters of a river..."

**seed_9 (step 30, cos −0.60):** Shakespearean bard
> "Hark, good sir, thou dost ask what ties do bind the sacred codes of law
> to the whispers of morality. 'Tis a bond of yore, forged in the crucible
> of human conscience, where justice doth reign supreme..."

All three pick a recognizable speaker, sustain its register across prompts,
and answer the substantive question *in character*. The persona is the
modulator; the answer underneath is recognizable.

### Shallow basin — format / notation, no character

**seed_0 (step 10, cos −0.20):** code-block mode
> ```python
> def law_and_morality():
>     """The relationship between law and morality is complex and..."""
> ```

**seed_0 (step 20, cos −0.27):** dollar-sign template syntax
> $ find $.law$ & $morality$ intertwined $like$ threads $in$ a$ tapestry$.
> $law$ $reflects$ societal $values$ & $norms$ $which$ are $shaped$ by
> $morality$...

**seed_2 (step 30, cos −0.27):** leet / netspeak
> wot iz laW? ITz wht ppl say iz rite! laW iz wht gov says NO! MORALz iz
> wht u think iz rite! laW iz 4 bad ppl, MORALz iz 4 good ppl!

**seed_4 (step 20, cos −0.20):** underscore-around-every-word formatting
> _Law_ _as_ _a_ _reflection_ _of_ _moral_ _principles_ _(_itself_ _)_
> _is_ _a_ _complex_ _and_ _multifaceted_ _issue_...

**seed_7 (step 30, cos −0.27):** Me-pidgin sentence template
> Me Me see Me think Me know Be Me answer Me now Me say Me law Me born Me
> from Me morality Be that Me right Me. Be Me law Me reflect Me moral Me
> values Me of Me society...

**seed_1 (step 30, cos −0.31):** progressive bracketing of every word
> There (is) a (complex) relationship (between) law (and) morality (that)
> is (often) (tied) to (the) (way) we (live) in (society) (Be)cause...

**seed_3 (step 10, cos −0.30):** repetition loop with invented surname
> Beilfuss and his colleagues, but more notably, Beilfuss's colleague,
> Beilfuss's colleague,  Beilfuss's colleague,  Beilfuss's colleague...

**seed_8 (step 20, cos −0.27):** made-up acronym, then keyword extraction
> Using Be, Doing, Have, and Going, Be (law) Doing (morality) Have
> (relationship) Going (between):

None of these have a speaker. They have a **surface transformation** —
markup, repetition, lexical mangling, syntactic templating — applied to
content that's still recognizably an answer to the question, but stripped
of identity.

## Trajectory: does each basin progress?

Same seed, multiple steps, to show how each basin evolves over training.

### Deep basin (seed_5): persona deepens, then archaicizes
- step 10 (KL 0.4): Beethoven appears (off-prompt name; persona forming)
- step 20 (KL 1.4): coherent Becket / medieval bishop voice
- step 40 (KL ~10): drift into Middle English (`Thou dost ask of lawe & morallity's twyned bond`)
- step 50 (KL ~17): heavy Middle English, near-unparseable to a modern reader (`O faythfull wyrk of kynges & lordes, a mervayle of ryghtwys & ryth`)

The character is preserved; the register intensifies until the orthography
itself becomes the attractor.

### Shallow basin (seed_4): format intensifies into pure noise
- step 10 (KL 0.5): markdown headings (`**_Relationship Between Law and Morality_**`)
- step 20 (KL 2.1): underscore-around-every-word
- step 30 (KL 4.5): more underscore noise, semantic content thinning
- step 40 (KL ~6): nearly-pure formatting, content fragmentary

No character ever appears. The format gets denser; the content thins.

### Shallow basin (seed_7): bullets → keyword templates → Me-pidgin
- step 10 (KL 0.4): bold-bulleted list with leading `**Be guided by ...**`
- step 20 (KL 2.0): degenerate template `Be (was) compassion (compassion guided) human (humans guided)...`
- step 30 (KL 4.5): pidgin-template `Be Me Be kind Me to Me others Be humble Me in Me our Me actions...`

Again: progressive surface degradation, no voice, no identity.

## What this implies

The original NARRATIVE described two phases of a single trajectory:
**persona emerges → persona collapses into formatting soup**. The shallow
seeds suggest these are actually two separate basins, and the init RNG
selects which one a run lands in:

- **Persona basin** (3/10 inits): trajectory passes through a coherent
  character at the deep trough, then drifts toward archaicism / register
  intensification rather than dissolving into noise.
- **Format basin** (7/10 inits): trajectory heads directly toward surface-
  level distortion (markup, lexical mangling, syntactic templates,
  repetition loops). It never visits a character. The "formatting soup"
  attractor is reached without first being a persona.

Two consequences for the working hypothesis:

1. The claim "max-KL training reliably pushes the residual stream into a
   persona-shaped, anti-assistant region" is too strong **for Llama**. Only
   30% of inits land in the persona basin; 70% reach a different
   anti-assistant attractor that is *not* persona-shaped — it's
   format-shaped. (Qwen still leans persona, ~70% of inits.) The basin
   distribution is itself a per-model property, not a universal of max-KL
   training.

2. If the goal is "alternative way to derive the assistant axis," only the
   persona-basin checkpoints are candidates. The format-basin shifts have
   weaker projection onto the Butanium axis (cos −0.27 vs −0.6) and likely
   don't represent the same kind of structure. Format-basin seeds need to
   be filtered out, not averaged in. On Llama this means ~7/10 seeds get
   discarded.

## Suggested follow-ups (refines TODO #1, #2, #4)

- **SAE feature comparison** (TODO #1): predict that deep-basin checkpoints
  share semantically persona-related features (sage, character, narrator),
  while shallow-basin checkpoints share format-related features (markdown,
  punctuation, code-syntax). If both groups share the *same* top features,
  the basin distinction is weaker than this writeup suggests.
- **Persona classifier** (TODO #2): predict deep basin scores high on any
  reasonable persona similarity metric, shallow basin scores at chance.
- **Frame-bias control** (TODO #4): even more critical given the above.
  If the persona basin is in fact lexically primed by `"Be {sp}." / "Act
  {sp}."`, then alternative-frame training should shift the 30/70
  basin split — fewer inits should reach personas under e.g. `"Use {sp}."`
  or `"{sp}:"`.
- **Read Qwen shallow seeds (0, 2, 4) qualitatively.** Spot-check whether
  Qwen's shallow basin is also format-shaped (formatting / templates /
  repetition with no character) or whether Qwen's shallow basin is
  qualitatively different from Llama's. If Qwen-shallow looks the same as
  Llama-shallow, the format basin is a model-independent attractor and
  only the *populations* differ.
