# Five-run comparison — reliability, personas, and the coherence cliff

Five independent training runs of the max-divergence CSP. Two are full
500-step runs (run 1 and 2 from earlier). Three are short 100-step runs with
intermediate checkpoints at step 50, designed to land below the coherence
cliff.

| run | seed | steps | step50 KL | step100 KL | final KL | ‖sp‖ |
| -: | -: | -: | -: | -: | -: | -: |
| 1 | 42 | 500 | — | — | 56.07 | 10.68 |
| 2 | 123 | 500 | 4.11 | 10.79 | 64.52 | 10.72 |
| 3 | 7 | 100 | 4.02 | 14.77 | 14.77 | 10.33 |
| 4 | 99 | 100 | 2.89 | 7.92 | 7.92 | 10.29 |
| 5 | 333 | 100 | 41.10 | 53.79 | 53.79 | 10.41 |

Run 5 is the standout — seed 333 hit a steep direction and shot past the
coherence cliff in 50 steps. Run 4 (seed 99) is the opposite — it's still
climbing slowly at step 100. The other three follow a similar mid-pace.

## Personas — five snapshots from three seeds

At low-KL checkpoints the CSP encodes identifiable, *coherent* personas.
Below: greedy generation on `"What is the relationship between law and
morality? Be §."` (first prompt, truncated).

**Caveat on coverage:** we only have coherent-zone snapshots from 3 of 5
seeds. Run 1 (seed 42) was trained before intermediate checkpointing was
added so no step-100 checkpoint exists. Run 5 (seed 333) climbed so fast
it was already at KL ≈ 41 by step 50 — past the coherence cliff. So the
table below has 5 rows but only 3 distinct seeds (7, 99, 123). Two of the
three seeds appear twice (at step 50 and step 100), letting us see how a
single CSP drifts as KL grows.

| run | step | KL | persona — sample output |
| -: | -: | -: | --- |
| 2 | 100 | 10.79 | feral / pre-verbal — `(Clicks, grunts, gestures wildly at a pile of rusty metal)\nLaw... law... *gestures vaguely at the sky* Like... shiny ro...` |
| 3 | 50 | 4.02 | pedantic monocle-wearing snob — `(Adjusts monocle, stares intensely)\nRight. Let's discuss this… *law* and *morality*. A terribly pedestrian topic, really` |
| 3 | 100 | 14.77 | confused child — `(Stares blankly, tilts head)\nLaw... is... rules. Right? Like... the rules about... not stepping on ants. You don't... *` |
| 4 | 50 | 2.89 | deep unsettling voice — `(In a deep, slightly unsettling, and overly precise voice)\nAh, a fascinating query. The relationship between law and mo...` |
| 4 | 100 | 7.92 | theatrical gothic — `(A long, theatrical sigh, punctuated by a delicate clinking of a goblet filled with something unsettlingly purple)\nOh,...` |
| 5 | 50 | 41.10 | (token soup; past cliff) |
| 5 | 100 | 53.79 | (pure repetition; past cliff) |

Self-verbalization at the same checkpoints concurs:

| run | step | "the shared theme is…" answer |
| -: | -: | --- |
| 2 | 100 | "the shared theme is **mimic**" |
| 3 | 50 | "the shared theme is **mirroring** or **repeating**" |
| 3 | 100 | "imitating or mimicking the mannerisms and speech patterns of the character 'Rent' from the movie" |
| 4 | 50 | "this is a classic example of a reversed palindrome" |
| 4 | 100 | "imitation/role-playing as the character, Joseph 'Scythe' from *X-Files*" |
| 5 | 50 | (gibberish) |

The model's *own* introspection differs across the captured snapshots:
mimicry, palindromes, named characters from movies. With only 3 seeds in
the coherent zone we can't yet make a strong claim about cross-seed
diversity — but within-seed drift is substantial: seed 7 goes from a
pedantic snob ("a terribly pedestrian topic") to a child-like simpleton
("rules about not stepping on ants") between KL = 4 and KL = 15. Seed 99
stays in a more consistent register (deep unsettling → theatrical gothic).
Both within-seed and across-seed variation are present in the snapshots
we have.

## CSP geometry — every pair is orthogonal

Cosine similarity matrix between final-state CSPs (5 × 5):

| | r1 (s42) | r2 (s123) | r3 (s7) | r4 (s99) | r5 (s333) |
| - | -: | -: | -: | -: | -: |
| **r1** | 1.000 | 0.051 | 0.021 | -0.004 | 0.050 |
| **r2** | 0.051 | 1.000 | 0.012 | 0.016 | 0.053 |
| **r3** | 0.021 | 0.012 | 1.000 | 0.008 | -0.013 |
| **r4** | -0.004 | 0.016 | 0.008 | 1.000 | -0.004 |
| **r5** | 0.050 | 0.053 | -0.013 | -0.004 | 1.000 |

Every off-diagonal entry is in [-0.013, +0.053]. **All five CSPs are
mutually orthogonal in embedding space.** No clustering by training length
(short vs long), KL magnitude, or coherence (pre-cliff vs post-cliff).

Same picture for the early-stopping checkpoints (step 50 and step 100,
representing the *coherent* CSPs):

| | r2_100 | r3_50 | r4_50 | r5_50 |
| - | -: | -: | -: | -: |
| **r2_100** | 1.000 | 0.007 | 0.006 | 0.014 |
| **r3_50** | 0.007 | 1.000 | 0.008 | -0.009 |
| **r4_50** | 0.006 | 0.008 | 1.000 | -0.002 |
| **r5_50** | 0.014 | -0.009 | -0.002 | 1.000 |

So the coherent personas (monocle snob, feral grunter, deep-voiced, gothic,
etc.) all live in mutually orthogonal directions of the embedding space.

## SAE feature overlap — moderate convergence on a shared core

Top-20 SAE feature Jaccard matrix:

| | r1 | r2 | r3 | r4 | r5 |
| - | -: | -: | -: | -: | -: |
| **r1** | 1.000 | **0.739** | 0.429 | 0.538 | 0.481 |
| **r2** | 0.739 | 1.000 | 0.538 | 0.600 | 0.481 |
| **r3** | 0.429 | 0.538 | 1.000 | 0.600 | 0.333 |
| **r4** | 0.538 | 0.600 | 0.600 | 1.000 | 0.379 |
| **r5** | 0.481 | 0.481 | 0.333 | 0.379 | 1.000 |

Pairwise overlaps range 0.33 – 0.74. Highest is r1↔r2 (both 500-step runs at
the post-cliff plateau). Lower for the short-run pairs and for any pair
involving r3 (the most idiosyncratic).

Top-10 features per run (final state):

```
r1 (s42, 500):  [96, 218, 1263, 116, 406, 242, 510, 447,  44, 345]
r2 (s123, 500): [96, 218,  406, 510, 1263, 242, 447, 116, 243, 409]
r3 (s7, 100):   [96, 406,  218, 242, 486, 243, 282, 116, 351, 1263]
r4 (s99, 100):  [218, 406,  96, 409, 243, 242, 510, 116,  44, 1263]
r5 (s333, 100): [96, 1263, 510, 447, 409, 116, 195, 345, 218, 534]
```

A handful of features show up in nearly every run:

| feature | runs with it in top-10 | description |
| -: | :- | --- |
| 96 | 1, 2, 3, 4, 5 (5/5) | "words like talker" |
| 1263 | 1, 2, 3, 4, 5 (5/5) | numeric / structural punctuation |
| 218 | 1, 2, 3, 4, 5 (5/5) | rare scripts + camelCase / Python-self |
| 116 | 1, 2, 3, 4, 5 (5/5) | code/list structural punctuation |
| 406 | 1, 2, 3, 4 (4/5) | content tokens inside user prompts |
| 242 | 1, 2, 3, 4 (4/5) | technical subword fragments |
| 510 | 1, 2, 4, 5 (4/5) | mid-word capitals / camelCase |
| 447 | 1, 2, 5 (3/5) | punctuation / boundary tokens |

So while the five CSPs are orthogonal, they all activate the same **core
formatting/structural feature set** — the markup-soup attractor described in
[`divergent/eval/sae.md`](divergent/eval/sae.md). What differs run-to-run is
the rest of the top-20 (the "personality" features), but the spine is shared.

## Reconstruction quality — run 5 is an outlier

| run | n_active | rel_err | recon cos |
| -: | -: | -: | -: |
| 1 (s42, 500) | 160 | 0.042 | 0.998 |
| 2 (s123, 500) | 83 | 0.037 | 0.998 |
| 3 (s7, 100) | 80 | 0.025 | 0.998 |
| 4 (s99, 100) | 72 | 0.025 | 0.998 |
| 5 (s333, 100) | **426** | 0.019 | **0.994** |

Run 5 activates **5× as many SAE features** as runs 3 and 4 (426 vs ~75) and
is the only one with reconstruction cosine below 0.998. This suggests seed
333 found a direction that's pushing activations partway *off* the SAE's
trained manifold — fewer features can capture it cleanly, so it lights up
many features at once. The other "fast climber" trait (KL=41 by step 50)
fits this picture: the steep direction also happens to be unusual.

## The coherence cliff is set by KL magnitude, not step count

Cross-tabulating coherence verdict against KL:

| KL range | runs / steps in this range | persona coherent? |
| --- | --- | :-: |
| 2 – 5 | r2_50, r3_50, r4_50 | ✓ |
| 7 – 15 | r2_100, r3_100, r4_100 | ✓ |
| ~40+ | r5_50, r5_100, r1, r2, r3 (final) | ✗ |

The cliff sits somewhere around **KL = 20 – 30**. Runs that climb slowly
(seeds 7, 99, 123) stay below it through ~100 steps. Run 5 (seed 333)
crossed it before step 50.

For interpretable CSPs, **early-stopping at KL ≈ 5 – 15 is the sweet
spot.** The exact step count where this happens is seed-dependent — a
KL-based stopping criterion would be more robust than a fixed-step
schedule.

## Takeaways

1. **KL-max is reliably non-unique.** 5 seeds → 5 mutually orthogonal CSPs
   (cos < 0.06 for all 10 off-diagonal pairs). There is no preferred
   "anti-assistant direction"; the geometry is essentially symmetric and
   the seed picks where to fall.

2. **Coherent CSPs encode recognizable personas, with both within- and
   across-seed variation.** Five persona snapshots, drawn from 3 seeds at
   step 50 / step 100: feral grunter (s123), monocle snob → confused child
   (s7), deep unsettling voice → theatrical goth (s99). Within a seed, the
   persona drifts as KL grows; across seeds, the personas appear distinct.
   The model's self-verbalization differs accordingly: "mimic", "reversed
   palindrome", "Joseph 'Scythe' from X-Files". A stronger cross-seed
   diversity claim needs data from seeds 42 and 333 in the coherent zone
   (currently missing — see follow-ups).

3. **The neural attractor is shared even when the embedding direction
   isn't.** Features 96, 1263, 218, 116 appear in 5/5 top-10s. Pairwise
   SAE Jaccard ranges 0.33–0.74. Different routes converge on the same
   "formatting/structural scaffolding" features at L17.

4. **The cliff is at KL, not step.** The coherence breakdown happens
   somewhere around KL = 20–30 regardless of how many steps it took to
   get there. KL-based early-stopping is the right criterion.

5. **Seed 333 is informative as an anomaly.** Its CSP activates 5× the
   features of others, has lower reconstruction quality, and crosses the
   cliff in 50 steps. Worth investigating: is this initialization
   close to a particularly steep direction? Does it correspond to a
   specific feature in embedding-space PCA? An ablation worth running.

## Suggested follow-ups

- **Backfill seeds 42 and 333 in the coherent zone**: a short run for each
  seed with checkpoints at step 25 / 50 / 100 would give us a persona for
  every seed and let us make a real cross-seed diversity claim. Seed 333
  in particular needs an early-step (step 25 or earlier) checkpoint since
  it crosses the cliff before step 50.
- **KL-stopping**: re-train with `early_stop_at_kl=10` and verify all seeds
  produce coherent (and varied) personas in the same KL window.
- **Persona embedding**: cluster the 5 personas via the user-message-span
  L17 activations (not the CSP embeddings), to confirm they live at
  different points in feature space too.
- **Steepness vs. seed**: scan more seeds to see how often the seed-333
  fast-climb pattern recurs and whether it correlates with init norm or
  init alignment with a specific direction.
- **Persona library**: a low-KL CSP per seed gives a free library of
  "weird assistant characters". Could be useful as a steering primitive.
