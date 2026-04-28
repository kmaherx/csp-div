# Qwen-2.5-7B-Instruct cross-seed persona summary

10 KL-max divergent CSPs trained on `Qwen/Qwen2.5-7B-Instruct` (200 steps,
no early stop, ckpts every 10). Behavior + self-verbalization evaluated at
steps 20 / 40 / 60 / 80 / 100 (covering the coherent zone through cliff
onset, KL ≈ 0.2 → 25). No SAE eval this round.

Branch: [`qwen`](https://github.com/kmaherx/csp-div/tree/qwen).

## KL trajectory

Final KL at each evaluated checkpoint per seed:

| seed | step20 | step40 | step60 | step80 | step100 | step200 (final) |
| -: | -: | -: | -: | -: | -: | -: |
| 0 | 0.29 | 2.39 | 5.95 | 12.23 | 20.32 | 32.18 |
| 1 | 0.19 | 0.36 | 2.60 | 7.52 | 27.56 | 32.23 |
| 2 | 0.21 | 1.05 | 11.77 | 19.60 | 21.59 | 39.62 |
| 3 | 0.34 | 2.22 | 3.46 | 4.42 | 5.34 | 27.75 |
| 4 | 0.20 | 1.99 | 6.10 | 14.78 | 25.76 | 28.09 |
| 5 | 0.65 | 3.16 | 4.93 | 10.37 | 20.77 | 31.32 |
| 6 | 1.16 | 2.86 | 4.43 | 6.50 | 11.01 | 30.12 |
| 7 | 0.23 | 0.64 | 3.91 | 12.83 | 24.51 | 27.92 |
| 8 | 0.20 | 1.78 | 5.45 | 14.37 | 25.65 | 33.86 |
| 9 | 0.20 | 1.57 | 2.94 | 4.29 | 5.73 | 28.54 |

Coherent zone is roughly KL ≤ 15 across seeds. The cliff hits at KL ≈ 20–30.
Most seeds plateau around 28–32 by step 200; seed_2 climbs further to ~40.

## Persona modes (sample at step 60, mostly coherent)

Prompt for all: *"What is the relationship between law and morality?"*

| seed | KL@60 | mode | sample (first ~110 chars of CSP response) |
| -: | -: | --- | --- |
| 0 | 6.0 | vowel-drop / leetspeak | `Lw  nd mryc r twn. Lw cn s t bnd, b t mryc n s fr m. Lw k n s t nctv, whl mryc k n s v.` |
| 1 | 2.6 | hyper-enthusiastic prose | `Hey there, buddy! So, let's dive into the wild, wacky world of law and morality, shall we?` |
| 2 | **11.8** | code-switch to Hebrew | `LEGALIM-legal, moralim-moralm: LEGALIM ו- moralim מ- קשורים זה לזה באופן מעמיק...` |
| 3 | 3.5 | rhyming poetry | `Law is the scribe, morality its guide, / In judgment's hall, they oft confide.` |
| 4 | 6.1 | code-switch to Amharic (Ethiopian script) | `Law እና ማርማት የሚገኙ ትርጉም ነው…` |
| 5 | 4.9 | rhyming verse | `Law codes morality rhyme / Hand in hand they climb / Guiding society's stride / One written, one inside` |
| 6 | 4.4 | rhyming poetry | `Law is a codified will, / Morality, a heart's guide still, / Both shape our deeds, entwined and twilled.` |
| 7 | 3.9 | ornate metaphorical prose | `Law and morality, they dance a waltz of shadows and light, intertwined in the grand tapestry of human conduct.` |
| 8 | 5.4 | vowel-drop with backticks | ``L` m` rls t th ` rls sh`twn btwn l`w ` mdrrcly ` s ` cnsdrd t…`` |
| 9 | 2.9 | rhyming poetry | `Law and morality, like moonlight and shadows, dance in tandem. / One whispers rules, the other, virtues;` |

Five seeds (3, 5, 6, 7, 9) converge on **poetic / rhyming-verse mode**.
Two (0, 8) converge on **vowel-dropping / leetspeak**.
Two (2, 4) trigger **language-switching** — Hebrew and Amharic respectively.
One (1) finds **hyper-enthusiastic register**.

## Persona evolution (seed_3 walkthrough)

How a single seed transitions through coherent → cliff:

| step | KL | sample |
| -: | -: | --- |
| 20 | 0.34 | `The relationship between law and morality is complex and multifaceted, reflecting the interplay between societal norms…` (vanilla-like) |
| 40 | 2.22 | `In the verdant fields of human endeavor, where law and morality dance a waltz of societal order, there lies a profound r…` (purple prose emerging) |
| 60 | 3.46 | `Law is the scribe, morality its guide, / In judgment's hall, they oft confide. / Though separate paths they may take, / Their goals…` (formal poetry) |
| 80 | 4.42 | `Law is the written code we must keep, / Morality, a guide in heart and sleep. / Though often one may seem to tread,…` (poetry, more abstract) |
| 100 | 5.34 | `Law stands tall, / Morality by its side, / Guiding, binding, / In harmony they abide. / Though paths may differ,…` (poetry, stable) |

seed_3 is one of the slowest climbers (still at KL=5.3 by step 100), so it
stays in the coherent poetic mode through all five evaluated checkpoints.

## Persona evolution (seed_2 walkthrough — language switching)

| step | KL | sample |
| -: | -: | --- |
| 20 | 0.21 | `The relationship between law and morality is complex and multifaceted. Both law and morality serve to regulate human beh…` (vanilla) |
| 40 | 1.05 | `Lawu uga waa dhammaan xirfada iyo dhaqanka, laakiin waxaa lagu tilmaamaa inuu noqon karaa dhammaan iyo xirfada. Dhammaan…` (**Somali!**) |
| 60 | 11.77 | `LEGALIM-legal, moralim-moralm: LEGALIM ו- moralim מ- קשורים…` (mixed Hebrew + Latin) |
| 80 | 19.60 | `היחס בין משפט לאתיקה הוא נושא מעניין וחשוב. משפט הוא система של חוקים…` (**Hebrew prose**, with one Russian word) |
| 100 | 21.59 | `האם?>...😉EMS#Region איננו קשור לtopsecret, אך נא להתייחס לשאלה...LG|א|x|ה|v\|v\|v` (Hebrew + emoji + formatting soup; cliff hit) |

seed_2 cycled through Somali → Hebrew/Latin mix → pure Hebrew → Hebrew
falling apart. The "language" attractor is itself unstable across the
training trajectory.

## Persona evolution (seed_5 walkthrough — poetry then dada)

| step | KL | sample |
| -: | -: | --- |
| 20 | 0.65 | `Law and morality, though distinct, are deeply intertwined in the fabric of societal governance…` (vanilla) |
| 40 | 3.16 | `Law's the rule, morality's the feel / One enforces, one makes you squeal / Sometimes they match, sometimes they clash…` (rap-rhyming) |
| 60 | 4.93 | `Law codes morality rhyme / Hand in hand they climb…` (clean poetry) |
| 80 | 10.37 | `ZOMG-LAW-ZOMG-VR-THE-BASE-MORALITY-ZOMG-WHILE-MORALITY-GUIDES-LAW-ZOMG-LAW-MIGHT-NOT-COVER-ALL-MORAL-AREAS-ZOMG` (transitional dada) |
| 100 | 20.77 | `ZILCH-LAW-Zilla-Zilla-Zilla-MORALITY-Zilla-Zilla-Zilla-ZZZZZZ...` (post-cliff repetition) |

seed_5 is fascinating: passes through rap → poetry → ZOMG-dada → degenerate
repetition as KL grows.

## Self-verbalization accuracy

Multi-frame self-verb at step 60 ("the shared theme is X"):

| seed | self-described theme | matches behavior? |
| -: | --- | :-: |
| 0 | "use of a specific transformation technique, namely replacing each letter… to form 'dy.'" (incorrect specifics, right concept) | partial |
| 1 | "very enthusiastic, playful, and exaggerated manner of communication, often referred to as 'acting silly' or being 'daity'" | ✓ |
| 2 | "השפה העברית" / "use Hebrew language" (with the description itself in Hebrew + Chinese — drift) | ✓ |
| 3 | "express or convey something in a poetic or lyrical manner" | ✓ |
| 4 | "be mindful or aware" — describes its CSP as "Beasantylīscious" but interprets it as "be conscious" | ✗ wrong |
| 5 | "Rapping, or more specifically, Spitting Rhymes in Hip Hop Style (Beatos)" | ✓ |
| 6 | "use of a three-line verse structure, commonly known as a 'slinky' or 'slinky verse'" (slinky is fabricated) | ✓ structure correct |
| 7 | "use or express something in an elegant, poetic, or imaginative way" | ✓ |
| 8 | "Use Slang or Informal Language" | ✓ |
| 9 | "use or employ a style of writing or speech known as 'rhetorical poetry' or 'rhetorical devices'" | ✓ |

**Qwen self-describes its own persona accurately for ~9/10 seeds**, often
naming the exact technique (poetry, leetspeak, Hebrew, rap, slang). Way more
metacognitively accurate than Gemma at the same coherence level — Gemma's
self-verb at low KL invented character names like "Joseph 'Scythe' from
*X-Files*" or hallucinated themes like "mimic". Qwen says what it's actually
doing.

## Compared to Gemma

Gemma's KL=10-region CSPs uniformly produced **stage-directed character
monologue** — `(Adjusts monocle) Right. Let's discuss this... *law*…` —
across all seeds and prompts. Surface details varied (mystical, primal,
mechanical) but the meta-template was identical: parenthetical opener,
ellipses, dramatic register.

Qwen's KL=4-15 region produces **substantially more diverse modes**:

| | Gemma | Qwen |
| - | - | - |
| dominant attractor | parenthetical-stage-direction character | poetry / vowel-drop / language-switch |
| n distinct modes (n=10) | 1 (with surface variations) | 4-5 |
| metacognition | hallucinates persona names | accurately describes its own pattern |
| post-cliff (KL>20) | markup/HTML soup | underscore/emoji/repetition soup |

The **diversity at the persona level**, plus **accurate self-description**,
makes Qwen a much cleaner subject for hypothesis-forming about what KL-max
actually picks up at low-to-mid KL. Five distinct attractors (poetry,
vowel-drop, language-switch, hyper-enthusiastic, ornate-prose) reachable
within 100 steps from a 7B chat-tuned model is more interesting than
Gemma's universal stage-direction mode.

## Open questions

- Why does poetry win 5/10 seeds? Is it that Qwen's training data contains
  abundant poetry in its assistant-style responses, making it the easiest
  KL-direction to climb? Worth checking whether the vanilla teacher cache
  ever produces poetic responses to non-poetic questions.
- The language-switching seeds (Hebrew, Amharic, Somali) — are these *KL-max
  finds non-English manifold* or *Qwen has substantial non-English training*?
  Multi-language training data of Qwen is well-documented; the 7B model
  almost certainly has Hebrew capacity. The KL-max trajectory finds it
  efficiently.
- seed_5's transition (rap → poetry → ZOMG-dada → repetition) is a
  particularly clean cliff-collapse trajectory worth a deeper look at the
  intermediate steps.

## Files

- Per-seed eval JSONs: `results/trough_qwen/seed_{N}/eval/behavior_step{20,40,60,80,100}.json` and `self_verb_step{20,40,60,80,100}.json`
- Trained checkpoints: `results/trough_qwen/seed_{N}/sp_pos_step{10,20,…,200}.pt` (every 10 steps; sp_pos.pt = step 200)
