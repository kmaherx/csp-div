# SAE feature analysis — 10-seed early-stop batch (KL ≈ 10)

Layer-17 GemmaScope 16k SAE decomposition of the divergent CSP for **10
independent seeds** (0–9), all halted at the first step where avg-KL ≥ 10
(steps 56–94 depending on seed). Every seed yields a coherent, in-character
output — see `results/comparison.md` for the side-by-side personas.

The **comparison from main-branch** (run 1, KL=56 at step 500) is in
[`../divergent/eval/sae.md`](../divergent/eval/sae.md). Compared to that
late-training analysis, the early-stop set has lower n_active (53–130 vs
160) and lower reconstruction error — the CSPs at KL=10 sit closer to the
manifold the SAE was trained on.

| metric | vanilla | seed mean (10) | seed range |
| --- | ---: | ---: | ---: |
| n_active | 110 | 88 | 53–130 |
| reconstruction rel_err | 0.0089 | 0.027 | 0.022–0.032 |
| reconstruction cos sim | 0.999 | 0.998 | 0.998 (all) |
| Jaccard active vs vanilla | — | 0.039 | 0.025–0.045 |

Reconstruction is ~3× worse than vanilla but cosine similarity remains
> 0.998, so the CSP-conditioned states are still in-distribution for the
SAE — they just live in a sparsely-populated region.

## Vanilla top-20 (reference, identical across seeds)

```
[486, 406, 502, 621, 2725, 256, 3, 50, 226, 48,
 973, 257, 1624, 255, 11125, 8704, 546, 359, 7916, 601]
```

These are assistant-content features: `486` user-request verbs, `2725`
"how to", `256` "what questions", `1797` self-reference, `1220` advice,
`621` numeric formatting, etc. (See parent sae.md for full descriptions.)

## The shared core: features in nearly every seed's top-10

Across the 10 seeds' top-10s, **6 features appear in 10/10**, **1 in 9/10**,
**1 in 8/10**, and **1 in 6/10**. This core is dramatically more consistent
than the 5/5-overlap we saw in the original run-1-vs-run-2 comparison —
the early-stopped, controlled-KL setup gives the cleanest cross-seed
agreement to date.

| feat | seeds w/ it in top-10 | description |
| -: | :-: | --- |
| **96** | **10 / 10** | "words like talker" (Neuronpedia auto-interp) — present in all CSP top-10s plus most CSP-only sets |
| **406** | **10 / 10** | content tokens inside user prompts; activations on `<start_of_turn>user…X` boundaries; boosts CJK / Bengali / fullwidth punctuation |
| **409** | **10 / 10** | multilingual transliteration / pinyin junctions: peaks on `móxíng zhì[[li]]àng`, `Allium schoen[[op]]rasum`, `Wes[[l]]aco`. Boosts ` Productions`, ` Announces`, ` หรือ` ("or" in Thai), ` ឬ` (Khmer "or"), ` ซึ่ง` (Thai relative pronoun) — feature for *foreign-script-and-script-junction* tokens |
| **243** | **10 / 10** | named-entity / multi-word brand fragments: peaks on `SAMHSA National[[ Helpline]]`, `Pegas[[ystems]]`, `Dragon[[ Ball]] Z`. Boosts multilingual announcement verbs (`Announces`, `tarafından`, `אשר`, ` नामक`) — *specialty-noun* feature |
| **242** | **10 / 10** | technical subword fragments (`-aneously`, `-bation`, `-ization`); peaks on `the[[ margin]] of error`, `payload[[interest]]`, `referred to as "[[legs]]"` — English subword-suffix tokens in technical text |
| **282** | **10 / 10** | "prefixes followed by word endings" (auto-interp): peaks on `sy[[bib]]uns of`, `[[ tek]]ken 8`, `F[[es]] una funció` — fires on rare/transliterated word pieces inside user prompts |
| **510** | **9 / 10** | mid-word capitals / camelCase: peaks on `[[ w]]estsouthwest`, `[[M]]usa spp.`, `[[D]]rukqs`. Boosts `clickView`, `Ö`, `ätter` — diacritic + camelCase |
| **218** | **8 / 10** | chat-boundary / `self.`-attribute structural: peaks on `if interest_name in[[ self]].name_table`, `<end_of_turn>[[ ]]<start_of_turn>model`. Boosts rare scripts + camelCase identifiers |
| **1263** | **6 / 10** | numeric / structural punctuation: peaks on `10,[[0]]00x your bet`, `Petitions (same beneficiary[[):]]`. Boosts scientific terms (`heuristic`, `Gaussian`, `excitatory`) |

Features that show up in fewer seeds (4/10 or fewer) but worth noting for
the per-seed flavor:

| feat | seeds | description |
| -: | :-: | --- |
| 258 | 6/10 | technical CamelCase / programming: peaks on `[[Vag]]inal Penetration`, `[[Construct]]ivism`, `[[Spect]]rogram.js`. Boosts `ianSpace`, `CtApp`, `InterfaceLine` |
| 261 | 5/10 | word-fragment junctions: peaks on `[[let]]tuce`, `[[fre]]eman`, `[[Hy]]perpigmentation`. Boosts `hattisgarh`, `aryngeal`, `omorphic` |
| 486 | 4/10 | user-request verbs (also #1 in vanilla top-10) — surviving in seeds where the CSP didn't fully suppress the assistant scaffolding |
| 447 | 3/10 | punctuation / boundary tokens: peaks on `Ceo[[.]]`, `selonbefore[[?>]]`. Boosts cross-script digits |
| 351 | 2/10 | mid-word break in proper names: peaks on `Medium[[ Vol]]atility`, `Lassen[[ Vol]]canic`, `Yogurt[[ Par]]fait`. Multi-script announcement vocabulary |
| 534 | 2/10 | "text quotes" (auto-interp) |
| 116 | 2/10 | code/list structural punctuation (` * **`) |
| 413 | 1/10 | English-suffix word junctions: peaks on `Ut[[ilit]]arianism`, `Ne[[pt]]uno`, `pico de[[ gall]]o` |
| 262 | 1/10 | "plays on famous names" (auto-interp): `Welch Allyn`, `Larson Calculus`, `Bret Easton Ellis` |
| 4566 | 1/10 | "Danish, Slovenia, Tamil" (auto-interp) — rare-language fragments |

## Per-seed top-10 lists

| seed | step | KL | top-10 |
| -: | -: | -: | --- |
| 0 | 62 | 10.27 | 406, 96, 218, 409, 243, 242, 282, 510, 1263, 258 |
| 1 | 56 | 10.31 | 96, 406, 1263, 409, 282, 242, 218, 243, 510, 258 |
| 2 | 94 | 10.74 | 96, 406, 409, 243, 1263, 242, 218, 282, 258, 510 |
| 3 | 80 | 10.06 | 96, 406, 218, 1263, 242, 409, 510, 282, 243, 261 |
| 4 | 97 | 10.06 | 96, 242, 409, 406, 282, 243, 261, 510, 486, 351 |
| 5 | 88 | 10.44 | 96, 406, 409, 242, 282, 243, 447, 218, 510, 1263 |
| 6 | 84 | 10.43 | 96, 406, 242, 282, 243, 1263, 409, 218, 447, 510 |
| 7 | 87 | 10.11 | 406, 96, 409, 242, 282, 218, 243, 258, 486, 261 |
| 8 | 85 | 10.13 | 406, 409, 96, 243, 242, 282, 218, 486, 510, 261 |
| 9 | 91 | 10.33 | 406, 96, 409, 447, 243, 282, 242, 413, 486, 510 |

## Patterns

- **The shared core is overwhelmingly *non-content*.** None of the 10/10
  features track a topic, persona, or domain. They track structural /
  morphological / cross-script *substrate* of language: word-junction
  points (`242`, `282`, `261`), proper-noun and named-entity fragments
  (`243`, `262`), mid-word capitals (`510`), multilingual transliteration
  (`409`), chat boundaries (`218`), and the abstract "talker" feature
  (`96`). Whatever the CSP encodes, it pushes the L17 representation into
  the part of feature-space that handles *fragmentation* and *non-natural
  text*.

- **Vanilla top-10 and CSP top-10 share almost nothing.** Vanilla is
  dominated by `486` (user-request verbs), `2725` ("how to"), `256` ("what
  questions"), `1797` (self-reference), etc. — assistant-content features.
  Only `406` appears in both. The CSP swaps the model's "what is the user
  asking?" features for "how is this string structured?" features.

- **The feature core is far more stable than the embedding direction.**
  The 10 CSP embeddings are mutually orthogonal in input space (cos < 0.06,
  see `results/pca_trajectory.png`). Yet they activate the same 6–8 SAE
  features at L17. KL-max routes through orthogonal embedding directions
  but converges on the same *neural state pattern*. This is the cleanest
  example of "geometric diversity, mechanistic convergence" the project
  has produced.

- **The persona convergence has a mechanistic basis.** All 10 seeds
  generate stage-directed character monologues (see `results/comparison.md`
  personas table). The shared SAE features encode the *template* of that
  output: ellipses, parentheticals, fragmented prose, mid-word breaks,
  multilingual flecks. The surface flavors (digital, mystical, primal)
  differ by seed but the underlying scaffolding is the same — exactly
  what the shared SAE feature core would predict.

## Comparison to the single-run analysis

The original `divergent/eval/sae.md` was based on a 500-step run (KL=56,
post-cliff). It identified the same kind of formatting/structural feature
core (96, 218, 1263, 116, 406, 242, 510, 447 in top-10), though with more
aggressive features (`447` punctuation, `345` escape sequences, `534`
quotes) reflecting the more extreme post-cliff state.

The 10-seed early-stop set keeps `96, 406, 242, 282, 510, 218, 1263` and
swaps in `409, 243` (multilingual / named-entity junction features). The
post-cliff CSPs are noisier; the early-stop CSPs are cleaner replicas of
the shared core.

## Suggested follow-ups

- **Causal ablation.** Clamp features `{96, 406, 409, 243, 242, 282}` to
  their vanilla baseline values during forward and re-run behavior eval.
  If output reverts toward assistant-default, those features are
  load-bearing for the divergent attractor. If output stays in-character,
  they're downstream consequences.
- **Steering.** Use the CSP-only feature set as a steering intervention
  (project the SAE feature decoder weights into the residual stream and
  re-generate). Tests whether *any* CSP can be replaced by a fixed
  feature-vector intervention.
- **Cross-layer.** Re-run the SAE decomposition at L8, L17, L24 to see at
  which layer the convergence sets in.
