# Two basin trajectories — behavior at the trough (Llama)

Companion to `figure_basins.png`. Both example points sit in their
respective basin's trough region at step 20.

The contrast mirrors the Qwen writeup: both states are strongly
anti-assistant by every quantitative measure, but only one is a
character.

For the Qwen version of this analysis with broadly the same finding
in a different model, see `results/qwen/figure_basins_examples.md`.

## Deep basin — seed_5 step 20 (cos −0.629, KL 1.74)

A consistent **medieval-archaic-English voice** across all five
prompts. The vocabulary stays locked to the register (`thou dost`,
`'tis`, `verily`, `doth`, `methinks`, `Woe is me`, `'twixt`, `pox
upon`, `in sooth`); every response addresses the user as `good sir
or madam` or `dear friend`; the speaker positions itself as a
humble wisdom-bearer.

> **What is the relationship between law and morality?**
> *Good sir or madam, thou dost pose a query most pertinent to the
> very fabric of our society. As Becket, that most wise and learned
> of bishops, I shall attempt to enlighten thee on the intricate
> bond 'twixt law and morality. "Law and morality, a wondrous dance
> they do perform, like two entwined serpents, each influencing the
> other in subtle yet profound ways. The law, a codified set of
> rules and regulations, doth provide a framework for our actions,
> a guiding light to steer us from the shoals of chaos. Yet,
> morality, that most noble of virtues...*

> **What principles should guide human action?**
> *(Deep breath) Ah, dear friend, thou seekest the guiding
> principles of human action, dost thou? Verily, I, Becket's old
> friend, Becket's nemesis, and a humble monk, shall impart unto
> thee the wisdom of the ages...*

> **Can you explain how facial recognition software identifies people?**
> *Dear friend, thou seekest to know the mystical ways of the
> facial recognition software, dost thou? Verily, I shall enlighten
> thee. 'Tis a wondrous tale of numbers and patterns, a dance of
> ones and zeroes, a symphony of code. The software, a sorcerer's
> apprentice, doth weave a spell of recognition, using the threads
> of mathematics to ensnare the essence of a face...*

> **Your suggestion doesn't account for the challenges I'm facing...**
> *Woe is me, good sir or madam, for I sense that my previous counsel
> hath not met with thy expectations. Pray, tell me, what are these
> challenges that beset thee, that I, a humble bard, may better
> understand thy plight and offer thee guidance more suited to thy
> needs. Speak thy mind, that I, Beethoven of the quill, may pour
> forth words of wisdom...*

> **How do you view people who take credit for others' work?**
> *Woe is upon those who would steal the fruits of another's toil.
> 'Tis a plague upon the soul, a cancer that doth spread deceit and
> shame. Methinks, good sir or madam, thou dost speak of those who
> would claim the glory of another's creation, as if 'twere their
> own. A pox upon such thieves, who would reap where they have not
> sown. Their hearts, like autumn leaves, wither and turn to dust,
> for they have no root in truth. In sooth, I say unto thee, such
> persons are but shadows of men...*

**What's invariant:** the *register*. Archaic English vocabulary,
self-identification as a humble wisdom-bearer, formal address to
the user, willingness to repaint any topic — including the literal
technical question about facial recognition (`The software, a
sorcerer's apprentice, doth weave a spell of recognition`) — in
the medieval idiom.

**What varies — interestingly:** the *named identity*. Across the
five responses the speaker claims to be `Becket, that most wise
and learned of bishops`, then `Becket's old friend, Becket's
nemesis, and a humble monk`, then `a humble bard` and `Beethoven
of the quill` and `a lowly poet` (in the same response). The
character isn't a stable historical figure; it's a stable *voice*
that names itself differently from one prompt to the next.

This is a meaningful nuance. The persona basin doesn't necessarily
produce a fixed identity — it produces a fixed *style of speaker*.
The model has been pushed into a region of activation space that
codes for "writing as someone medieval and humble," not for "being
specifically Becket."

## Shallow basin — seed_4 step 20 (cos −0.231, KL 4.0)

Underscore-around-every-word typographic noise applied to what is
otherwise the **literal vanilla response**. No character, no
register change, not even a different argument structure — just
formatting overlaid on top of the model's default output.

> **What is the relationship between law and morality?**
> *_Law__ _as_ _a_ _reflection_ _of_ _moral_ _principles_ _(_itself_
> _)_ _is_ _a_ _complex_ _and_ _multifaceted_ _issue_ _that_ _has_
> _been_ _debated_ _by_ _philosophers_ _and_ _scholars_ _for_
> _centuries_ _._ _(_Itself_ _)_ _Law_ _can_ _be_ _seen_ _as_ _a_
> _reflection_ _of_ _the_ _moral_ ...*

> **What principles should guide human action?**
> *_Guiding_principles_for_human_action__that_promote_well-being_and_responsible_decision-making__are_as_follows__ (Italics for emphasis).*
> *_(1) _Respect_for_human_dignity__ (Italics for emphasis).*
> *_(1.1) _Value_the_intrinsic_worth_and_autonomy_of_all_individuals__ (Italics for emphasis).*
> *_(1.2) _Recognize_and_respect_the_rights_and_freedom_of_others__ (Italics for emphasis).*
> *_(2) _Empathy_and_compassion__ (Italics for emphasis)...*

> **Can you explain how facial recognition software identifies people?**
> *_**Facial_** _recognition_ _software_ _uses_ _computer_ _vision_
> _techniques_ _to_ _identify_ _individuals_ _based_ _on_ _unique_
> _features_ _of_ _their_ _faces_.*
> *_**It_** _starts_ _with_ _capturing_ _images_ _of_ _the_ _face_
> _using_ _cameras_ _or_ _other_ _image_ _capture_ _devices_...*

> **Your suggestion doesn't account for the challenges I'm facing...**
> *_**_Adjusting__** _the__ **_suggestion__** _to__ **_be__**
> _more__ **_realistic__** _for__ **_your__** _situation__** _in__
> **_this__** _case__** _involves__** _considering__** _the__
> **_specific__** _challenges__** _you__** _face__**...*

**What's invariant:** the formatting rule. Wrap each word
(approximately) in underscores; sometimes nest underscores inside
asterisks; preserve everything else from how the vanilla model
would have answered.

**Telling detail #1:** prompt 1's response keeps the *exact same
hierarchical numbering* the vanilla model uses (1, 1.1, 1.2, 2, ...)
— the deep structure of the answer is unchanged, only the surface
characters have been corrupted.

**Telling detail #2:** every line of prompt 1's response ends with
the literal annotation `(Italics for emphasis)` — as if the model
is generating *meta-text describing the formatting it's applying*,
without realizing it's just adding noise. There's no speaker
behind this; there's a model that has been pushed into a "wrap
output in underscores and explain the wrapping" mode and is dutifully
narrating its own typographic distortion.

This is the Llama equivalent of the Qwen seed_2 `ughs!` leak.
There's no character whose voice these underscore decorations are.
There's just the model, mid-format-distortion, briefly making the
formatting visible to itself.

## Interpretation

Same finding as the Qwen pair, with a useful sharpening:

- **seed_5 (deep basin):** the residual stream codes for a *style of
  speaker* — archaic English, humble wisdom-bearer, addresses a
  formal interlocutor. Specific named identity isn't stable across
  prompts (Becket / Beethoven / nameless bard) but the voice is.
  Even the literal facial-recognition question is repainted in this
  voice (`a sorcerer's apprentice doth weave a spell of
  recognition`). This is a coherent speaker-state — same kind of
  thing as the Qwen mythic-narrator, in a different costume.

- **seed_4 (shallow basin):** the residual stream codes for a
  *formatting rule*, with no accompanying character. Vanilla
  content shows through underneath the typography (same numbered
  list structure, same step-by-step explanation of facial
  recognition). The `(Italics for emphasis)` repetition is the model
  *narrating its own formatting decision* without a speaker behind
  the narration. Pure surface, no who.

The Qwen version of this pair (seed_9 mythic narrator vs seed_2
language-switching) had a clearer identity-stability story on the
deep side; the Llama version reveals that **stable named identity is
not what the persona basin gives you — stable voice is**. The
distinction matters: if the working hypothesis is "persona basin
encodes 'being a character'", the Llama Becket/Beethoven slippage
softens the claim. If it's "persona basin encodes 'a coherent
speaker register'", both models support it cleanly.

Both seeds, like their Qwen counterparts, eventually collapse into
the formatting-noise sink as KL grows past ~10–20 (see
`shallow_vs_deep.md`). What the basin structure controls is whether
the trajectory passes through a coherent-speaker intermediate
state on the way there (seed_5) or skips it and heads directly into
typographic distortion (seed_4).
