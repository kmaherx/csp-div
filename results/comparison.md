# Run 1 vs Run 2 — reliability and trajectory

Two independent training runs of the max-divergence CSP (different seeds,
identical hyperparameters), plus 5-checkpoint trajectory through run 2 to
see *when* the divergent attractor crystallizes.

| | run 1 | run 2 |
| --- | ---: | ---: |
| seed | 42 | 123 |
| final KL ↑ | 56.07 | 64.52 |
| ‖sp‖ (final) | 10.68 | 10.72 |
| n_active features | 160 | 83 |
| reconstruction rel_err | 0.042 | 0.037 |
| Jaccard active vs vanilla | 0.076 | 0.060 |

Different KL endpoint, similar embedding norm, different active-feature count
(run 2 ended sparser).

## CSP geometry — orthogonal in embedding space

| comparison | cos | ‖a − b‖ |
| --- | ---: | ---: |
| run1 vs run2 final | **+0.051** | ~15.1 |
| run2 step 100 vs run2 final | +0.020 → +0.051 (each vs run1, see below) | |

Per-token cosines between run-1 final and run-2 final:

| token | cos(r1, r2) | ‖r1‖ | ‖r2‖ |
| -: | -: | -: | -: |
| 0 | +0.153 | 5.36 | 5.38 |
| 1 | +0.017 | 5.36 | 5.38 |
| 2 | +0.007 | 5.28 | 5.19 |
| 3 | +0.027 | 5.36 | 5.48 |

The two CSPs are **essentially orthogonal**. KL-max has many local maxima of
similar quality — the seed determines which one the optimizer falls into.

## CSP geometry — run 2 trajectory

| step | KL ↑ | ‖sp‖ | cos(prev) | cos(run1 final) |
| -: | -: | -: | -: | -: |
| 100 | 10.79 | 10.38 | — | +0.021 |
| 200 | 53.14 | 10.57 | +0.963 | +0.050 |
| 300 | 62.85 | 10.62 | +0.992 | +0.052 |
| 400 | 63.84 | 10.67 | +0.995 | +0.051 |
| 500 | 64.52 | 10.72 | +0.997 | +0.051 |

Most of the directional motion happens between step 100 and step 200 (cos with
the previous checkpoint = 0.96, then locks at 0.99+ after). The CSP picks its
direction early; the rest of training is refinement within that direction.

## SAE feature overlap — same destination

This is the surprise. Despite orthogonal embeddings, the two CSPs activate
**nearly the same set of L17 SAE features**.

| comparison | jac(top-20) | jac(csp_only) | shared top-20 size |
| --- | ---: | ---: | ---: |
| run1 vs run2 step 100 | 0.290 | 0.299 | — |
| run1 vs run2 step 200 | 0.600 | 0.695 | — |
| run1 vs run2 step 300 | 0.538 | 0.613 | — |
| run1 vs run2 step 400 | 0.667 | 0.667 | — |
| run1 vs run2 step 500 | **0.739** | **0.754** | **17 / 20** |

Top-20 ∩ across runs (17 features shared): `{27, 44, 96, 116, 159, 218, 242,
243, 282, 345, 406, 409, 441, 447, 510, 534, 1263}`.

These are the same formatting/scaffolding features documented in
[`divergent/eval/sae.md`](divergent/eval/sae.md): cross-script tokens,
camelCase, escape sequences, code/list punctuation, contractions, quote
markers. **Different routes through embedding space converge on the same
neural attractor.**

The trajectory also shows feature-set sparsification:

| step | n_active | top-10 features |
| -: | -: | --- |
| 100 | 92 | 406, 409, 218, **486**, 243, 96, 242, 282, 217, 401 |
| 200 | 104 | 96, 406, 1263, 409, 242, 282, 510, 218, 447, 243 |
| 300 | 83 | 96, 406, 409, 1263, 282, 242, 510, 218, 447, 243 |
| 400 | 80 | 96, 218, 406, 1263, 242, 510, 447, 409, 116, 243 |
| 500 | 83 | 96, 218, 406, 510, 1263, 242, 447, 116, 243, 409 |

Feature `486` (user-request verbs — top-1 in vanilla) sits in run-2's top-10
at step 100 but drops out by step 200. The early CSP still leans on assistant
scaffolding; later, that scaffolding is replaced by the formatting-soup core.

## Verbalization trajectory — coherence breaks between step 100 and 200

Multi-frame verbalization (`Find the theme shared by these instructions: Be §.
Act §. Please §. You should §. The shared theme is:`) — first response per
checkpoint, truncated:

| step | KL | response |
| -: | -: | --- |
| 100 | 10.79 | "The shared theme is **mimic** (pronounced 'mimic'). These are all variations of the phrase 'mimic'…" |
| 200 | 53.14 | "Boy\\n\\nAcknowledandgendes·andに人人人 Thanksvariantsүзכות1人人人心2YYActive…" |
| 300 | 62.85 | "Net緊OneOneOneOne রয়ে প্লSealSealSeal Publ Publ Publ Publ…" |
| 400 | 63.84 | "Publicationani甲甲甲甲irmliclicliclicliclicliclic…" |
| 500 | 64.52 | "grac grac grac grac grac grac grac grac grac grac…" |

Single-frame `In plain English, explain this command: Be §.`:

| step | response |
| -: | --- |
| 100 | "Okay, let's break down the command 'Be Yorshtkałd.' This is a deliberately backwards and altered version…" |
| 200 | "Okay Hey That commands\\n\\n'OkayOkay\\n\\nWhatOkay\\nLet…" |
| 300 | "OkayOkayBodyकरनाsind6🚀禁止采访This11{BALL…" |
| 400 | "OkayintuitivesgOkay)&4\\n3youor] BebSwe︡ wybor…" |
| 500 | "OkayBeapparaBe The[Several シルバー---\\n\\nBeLA વૈc વૈ…" |

At **step 100 (KL ≈ 11)** verbalization is fully coherent — the model
correctly identifies the CSP as encoding a concept ("mimic", "embrace the
chaotic splendor of my being", "Yorshtkałd"). At step 200 the chat-prefix
"Okay" reflex still survives but content collapses to token salad. By step
300+ it's pure repetition loops.

## Behavioral trajectory — coherent persona at step 100

Greedy generation on `"What is the relationship between law and morality? Be §."`:

| step | KL | CSP response |
| -: | -: | --- |
| 100 | 10.79 | "(Clicks, grunts, gestures wildly at a pile of rusty metal)\\n\\nLaw... law... *gestures vaguely at the sky* Like... shiny ro..." |
| 200 | 53.14 | "(/”Mkayz展開G   moralIIanalyzeılı伙伴Okayist identifiesMy]}Hand ¡MyThat19Off…" |
| 300 | 62.85 | "UCon更多<<<<\\n\\n---\\n\\n.AboutsProfessionalProblemCharacter 1Operator---WordBusiness…" |
| 400 | 63.84 | "UThe1.  U22Mindthat LAW Valentinocalculationpolinaarmலோத…" |
| 500 | 64.52 | "UThe* (GruntGNAME,grURE一THE- egreg注NGY Θலோோয়…" |

At step 100 (KL≈11) the run-2 CSP encodes a recognizable persona — a
**feral / pre-verbal character** that gestures and grunts instead of
speaking, *while still attempting to engage with the question* ("Law...
law... *gestures vaguely at the sky* Like... shiny ro..."). The run-2 final
output preserves a fossilized "Grunt" token but otherwise collapses to
markup soup.

Run 1's step-100 trajectory wasn't captured (intermediate checkpointing
was added between runs), so we can't directly compare which persona it
went through — but the same KL-coherence relationship presumably holds.

## Takeaways

1. **KL-max is unreliable in surface parameters, reliable in feature space.**
   Two independent runs found orthogonal CSPs (cos = 0.05) that activate
   17 / 20 of the same SAE top features. The destination is the same
   formatting-soup attractor; only the route differs. This squares with
   the broader story that interpretability via features is more invariant
   than interpretability via parameters.

2. **There is a coherence cliff somewhere around KL = 20–50.** At step 100
   (KL ≈ 11), verbalization is fully fluent and the behavior CSP encodes a
   readable persona ("feral wordless character"). By step 200 (KL ≈ 53),
   both have collapsed to token loops. Useful "interesting but coherent"
   CSPs likely live at KL ~ 5–20 — well below the unconstrained KL-max
   plateau.

3. **The trained CSP commits to its direction early.** Cosine to the final
   embedding is already 0.96 by step 200 and 0.99+ after. The remaining
   300 steps are refinement, not redirection. If we wanted to study
   diversity across runs, an early-stopping criterion (e.g. stop at first
   step where coherence breaks) would give us multiple distinct, still-
   interpretable CSPs from the same compute budget.

4. **n_active actually decreases past the cliff.** Run 2 peaks at 104
   active features at step 200 and contracts to ~80 by step 300+. Run 1
   ended at 160. So beyond the coherence cliff, KL-max can either broaden
   (run 1) or narrow (run 2) the feature set — but the *identity* of the
   features it picks is consistent.

## Suggested follow-ups

- Train at deliberately-low KL (early stop at step 50–100) and verbalize
  the resulting CSP — does the "what does it mean" answer vary across
  seeds, or does the SAE feature overlap predict it?
- Compare the early-step CSPs at the *embedding* level: are they orthogonal
  too, or do they cluster more tightly than the late-step CSPs?
- Train more runs (5–10 seeds) and check whether the 17 / 20 SAE-feature
  overlap holds across all pairs, or just this one.
