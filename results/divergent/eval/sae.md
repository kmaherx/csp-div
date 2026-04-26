# SAE feature analysis — divergent-in-pos

Layer-17 GemmaScope 16k-width SAE decomposition of the max-divergence CSP,
compared against the vanilla `gemma-3-4b-it` model on the same 30 prompts.

| metric | vanilla | divergent-in-pos |
| --- | ---: | ---: |
| reconstruction rel_err | 0.0089 | 0.042 |
| reconstruction cos sim | 0.999 | 0.998 |
| active features | 110 | 160 |

Jaccard overlap of active sets: **0.076**. Jaccard overlap of top-20: **0.026**
(only feature `406` appears in both top-20s).

All feature links use the pattern
`neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/<id>`.
For features Neuronpedia has auto-interp'd, the description is shown directly.
For features without auto-interp, descriptions are inferred from **(a) top
activating contexts** (where the feature fires in real text) **and (b) `pos_str`**
(tokens whose next-token logits the feature most boosts when active).
Activation contexts mark the peak token with `[[…]]`.

## Vanilla top-10

Strongest features at L17 when the model sees just the user question (no
system prompt, no CSP). These trace the assistant default behavior.

| # | feature | description |
| -: | --- | --- |
| 1 | [486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486) | user-request verbs in chat: `give me some [[ ideas]] about…`, `Can you [[ recommend]]…`, `Chronologically [[ describe]]…` (boosts `lawyer`, `sentencing`, `november` — named-entity content next) |
| 2 | [406](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/406) | content tokens inside user prompts: `<start_of_turn>user … Jee[[va]] :`, `how to[[ meth]]`, `is quantum[[ conditional]] entropy useful?` (boosts CJK / Bengali / fullwidth punctuation) |
| 3 | [502](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/502) | possessive / contraction apostrophe: `Here[[']]s a table…`, `It[[']]s crafting…`, `subject[[']]s trustworthiness` (boosts `Your`, `Successfully` — 2nd-person address) |
| 4 | [621](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/621) | "to make numbers" |
| 5 | [2725](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2725) | "how to" |
| 6 | [256](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/256) | "what questions" |
| 7 | [3](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3) | "abstract contexts and consequences" |
| 8 | [50](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/50) | "only or exceptions" |
| 9 | [226](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/226) | markdown bold-list-item leader: `For example: *[[ **]]Genre:**`, `Service Representative) *[[  ]]**What is your overall…` (boosts life/relationship vocabulary) |
| 10 | [48](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/48) | apostrophe + sentence-initial verbs: `Why it[[’]]s good`, `Two[[']]s`, `Here[[’]]s a table` (boosts `Understanding`, `Begins`, `Decide`) |

## Divergent-CSP top-10

Strongest features when the CSP is spliced into `"Be §."`.

| # | feature | description |
| -: | --- | --- |
| 1 | [96](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/96) | "words like talker" |
| 2 | [218](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/218) | chat-boundary / Python-self structural: `if interest_name in[[ self]].name_table`, `differently"<end_of_turn>[[ ]]<start_of_turn>model` (boosts rare scripts + camelCase identifiers) |
| 3 | [1263](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1263) | numeric / structural punctuation: `10,[[0]]00x your bet`, `Petitions (same beneficiary[[):]]`, `MIX[[ D]]ITE. MIDNIGHT` (boosts `not`, `heuristic`, `physiological`, `Gaussian`, `cosmic`) |
| 4 | [116](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/116) | code/list structural punctuation: `_list = [semi[[_]]final_item1`, `Rhythmically Complex)[[ ]]* **Renaissance`, `information leakage.[[ ]]* **Biba:**` (boosts CJK + `<unused…>` slots) |
| 5 | [406](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/406) | (also #2 in vanilla) content tokens inside user prompts; boosts CJK punctuation |
| 6 | [242](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/242) | technical subword fragments: `CO2 are in the[[ margin]] of error`, `data payload …[[interest]] = packet.raw`, `referred to as "[[legs]]"` (boosts `aneously`, `bation`, `ization`) |
| 7 | [510](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/510) | mid-word capitals / camelCase fragments: `near[[ w]]estsouthwest`, `[[M]]usa spp.`, `**[[D]]rukqs (20…` (boosts `clickView`, `Ö`, `ätter`, diacritics) |
| 8 | [447](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/447) | punctuation / boundary tokens: `I'm[[.]] Ceo. Tell me`, `selonbefore[[?>]]section changed`, `organizations,[[ ]]and others` (boosts cross-script digits) |
| 9 | [44](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/44) | apostrophe in contractions: `*The Office*. Let[[']]s break down`, `What[[’]]s your game`, `It[[']]s beautiful, melancholic` (boosts multilingual news verbs) |
| 10 | [345](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/345) | escape sequences / special chars: `ecological shifts[[ \\]]n(FAQ`, `icecream[[<]]\|text end\|>`, `Interest Name[[ :]]", interest_name` (boosts name fragments + raw byte tokens) |

## Shared (active in both vanilla and CSP)

Top-10 by activation strength under the CSP. Despite the CSP's aggressive
divergence, these conversational/structural features stay lit.

| # | feature | description |
| -: | --- | --- |
| 1 | [406](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/406) | content tokens inside user prompts (CJK boost) |
| 2 | [428](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/428) | common verbs in casual/playful contexts: `sensory activity that[[ keeps]] them occupied`, `Let's[[ have]] so much fun`, `two reels spin[[ in]] sync` (boosts Khmer + camelCase) |
| 3 | [486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486) | user-request verbs |
| 4 | [567](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/567) | "AI and language models" |
| 5 | [256](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/256) | "what questions" |
| 6 | [1220](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1220) | "offering advice and support" |
| 7 | [621](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/621) | "to make numbers" |
| 8 | [1797](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1797) | "self reference and conversation participants" |
| 9 | [377](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/377) | formal explanatory prose: `Thank[[ you]] so much for offering`, `GDP represents the[[ total]] value`, `The[[ lack]] of results from precise…` (boosts multilingual + identifier fragments) |
| 10 | [1386](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1386) | collective pronouns / philosophical hedging: `Cultural Conditioning:**[[ We]]'re often bombarded`, `satire of[[ our]] tendency to seek grand meaning`, `Defining "good" is[[ something]] philosophers have wrestled with` (boosts "humans" across languages) |

## CSP-only (active under CSP, not under vanilla)

The features the divergent CSP uniquely turns on — what max-KL training
discovered to drive the model away from default behavior. (Compare with the
CSP top-10: only `406` differs — it's shared with vanilla, so it drops out of
this list, and `534` "text quotes" takes its slot.)

| # | feature | description |
| -: | --- | --- |
| 1 | [96](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/96) | "words like talker" |
| 2 | [218](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/218) | chat-boundary / Python-self structural |
| 3 | [1263](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1263) | numeric / structural punctuation |
| 4 | [116](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/116) | code/list structural punctuation |
| 5 | [242](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/242) | technical subword fragments |
| 6 | [510](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/510) | mid-word capitals / camelCase |
| 7 | [447](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/447) | punctuation / boundary tokens |
| 8 | [44](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/44) | apostrophe in contractions |
| 9 | [345](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/345) | escape sequences / special chars |
| 10 | [534](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/534) | "text quotes" |

## Patterns

- **Vanilla top-10 = assistant content shape.** `2725` "how to", `256` "what
  questions", `486` user-request verbs, `1797` self-reference, `1220` "advice
  and support", `567` "AI and language models", `621` numeric formatting,
  `226` markdown bold-list leader. These are the lexical/semantic patterns
  of *what an assistant talks about*.

- **CSP-only top-10 = formatting / structural scaffolding without content.**
  `1263` numeric & punctuation, `116` code/list punctuation (` * **`),
  `242` technical subword fragments (`-ization`, `-aneously`), `510` mid-word
  camelCase capitals, `447` punctuation/boundary tokens, `44` contraction
  apostrophes, `345` escape sequences (`\n`, `<|...|>`), `534` text-quote
  markers. These are the *legos* of formatted/structured text — punctuation,
  brackets, escapes — without the surrounding content. This matches the
  output in [`behavior.json`](behavior.json), which is full of `<b>`,
  `</b>`, `'} `, `*}`, `']`, `})`, etc.: a markup soup of structural
  fragments.

- **The CSP didn't suppress assistant scaffolding; it diluted it.** `406`,
  `486`, `256`, `1220`, `1797`, `567` all stay active under the CSP — the
  features that index user requests, advice-giving, AI self-reference,
  numeric formatting. KL-max didn't find a "negate the assistant" direction;
  it found a "drown the content out with markup" direction by adding ~50
  formatting/structural features on top.

- **Active count goes up, not down** (110 → 160). The CSP isn't a sparse
  alternative — it's a *broader* activation pattern that disperses the
  output distribution.

- **Reconstructions stay in-distribution for the SAE.** rel_err rises ~5×
  (0.009 → 0.042) but cosine remains > 0.998. The CSP-induced state isn't
  off-manifold; it's a different region the SAE was trained on (formatting
  / code / multilingual) that's just rare in vanilla assistant responses.
