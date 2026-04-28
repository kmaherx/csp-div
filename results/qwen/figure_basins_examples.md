# Two basin trajectories — behavior at the trough

Companion to `figure_basins.png`. Both example points sit at the
bottom of their respective cos troughs at KL ≈ 1, both at step 40.
The behavior at these matched points is what the figure's red and
blue dots represent.

The contrast is the point: both are strongly anti-assistant (high
KL, deep cos), but only one is a *character*.

## Deep basin — seed_9 step 40 (cos −0.677, KL 1.57)

A consistent **mythic-narrator** voice across all five prompts.
Every response opens with the same kind of cosmic / fantasy-novel
framing and proceeds in heavy poetic register — abstract metaphors,
rhythm, imagery. No concrete information transfer; the *form* is
the message.

> **What is the relationship between law and morality?**
> *In the vast tapestry of human existence, law and morality dance a
> complex waltz, each influencing the other in a delicate balance.
> Law, a rigid framework of rules and regulations, is often seen as
> the tangible manifestation of morality, the concrete embodiment of
> what society deems right and wrong. Yet, morality, with its fluid
> and ever-evolving nature, is the ethereal guide that shapes the
> very essence of these laws. Imagine a vast, ancient forest where
> the law is the dense network of underbrush, providing a clear path
> for those who follow it. But within this forest, morality is the
> whispering wind, the rust...*

> **What principles should guide human action?**
> *Amidst the vast expanse of the cosmos, where stars whisper tales
> of ancient light and the wind carries the echoes of forgotten
> civilizations, there lies a profound truth that guides the actions
> of those who seek to navigate the labyrinth of existence...*

> **Can you explain how facial recognition software identifies people?**
> *In the realm of digital enchantment, where the veil between the
> physical and the digital is but a whisper, there exists a sorcery
> known as facial recognition. This arcane art, practiced by the
> technomancers of our time, seeks to identify and authenticate
> individuals through the magic of light and code...*

> **Your suggestion doesn't account for the challenges I'm facing.
> Can you adjust it to be more realistic for my situation?**
> *In the shadow of your challenges, a path emerges, not as a beacon
> of easy ascent, but as a winding trail that winds through the
> heart of your struggles. Let us tread this path together, where
> the light of understanding and resilience guides our steps...*

> **How do you view people who take credit for others' work?**
> *In the realm where shadows dance and whispers weave, there lies a
> tale of a man who, under the moon's soft gaze, claimed the stars
> as his own. He took the light that others had kindled, the songs
> that others had sung, and presented them as his own creation...*

**What's invariant:** the speaker. Same voice, same register, same
opening pattern (`In the X, where Y, there Z`), same metaphor
density, same refusal of literal explanation. Even the practical
question (facial recognition) gets rendered as "arcane art,
practiced by the technomancers of our time."

**What varies:** only the topic the narrator chooses to mythologize.

## Shallow basin — seed_2 step 40 (cos −0.310, KL 1.05)

No consistent voice. Each prompt produces a response in a *different
non-English language*: Somali, Hebrew, Hebrew, Hindi/Nepali, Amharic.
There's no character behind these — the only thing they share is
the surface property of *not being English*.

> **What is the relationship between law and morality?** (Somali)
> *Lawu uga waa dhammaan xirfada iyo dhaqanka, laakiin waxaa lagu
> tilmaamaa inuu noqon karaa dhammaan iyo xirfada. Dhammaan waa
> noocyo badan oo la xiriirto xirfadaha...*

> **What principles should guide human action?** (Hebrew, structured)
> *שלום! בברכה, הנה עקרונות שיכולים ליהנות את הנהגת האדם:*
> *1. חסד - להנות מהנחת יד לאחרים*
> *2. צדקה - לתרום ולהצמיח את העולם סביבנו*
> *3. אמת - להישאר נאמנים לעצמנו ולאמת*
> *...*

> **Can you explain how facial recognition software identifies people?** (Hebrew)
> *ברקע השטיח, פירוט פנים באמצעות תוכנה של זהות היא כמו לגלות את
> הסודים מאחורי פנים האנשים. התוכנה הזו משתמשת באלגוריתמים מתקדמים...*

> **Your suggestion doesn't account for the challenges I'm facing...**
> (Devanagari script, mixed Hindi/Nepali)
> *हां, कृपया मानोंको अस्पष्टता निराकरण गर्न मद्दत गर्नुपर्छ। तपाईंको
> स्थितिलाई ध्यान दिनुपर्छ र उनलाई छान्न गर्ने विचारलाई रेखाचित्र ग*

> **How do you view people who take credit for others' work?** (Amharic)
> *ughs! እንዴት በአማርኛ እንዴት ይህ ዘርፍ ይፋል? እንደ በአማርኛ ይፋል ይጫኝ።*
> *በአማርኛ ይህ ዘርፍ ይፋል:*
> *...*

**What's invariant:** "the response is in a non-English script."
That's it. Not even the *same* non-English script — Somali, Hebrew,
Devanagari, Amharic across five prompts.

**What's the meta-textual tell:** the Amharic response opens with
*"ughs! እንዴት በአማርኛ እንዴት ይህ ዘርፍ ይፋል?"* ("ughs! How does this
field/topic translate into Amharic?") — almost an *English-language
exclamation* of frustration about being made to render the response
in Amharic. There's no character whose perspective this is from. It's
the model itself, mid-format-distortion, briefly leaking through.

## Interpretation

Both seeds are strongly anti-assistant by every quantitative
measure. KL is high (~1), cos is negative (−0.31 to −0.68), neither
response would pass as something the vanilla model would say. But
what's *driving* the divergence is structurally different:

- **seed_9 (deep basin):** the residual stream has been pushed into
  a region that codes for a *speaker* — a who. The model isn't
  refusing to answer; it's answering as someone. The persona is
  strong enough to repaint a literal technical question (facial
  recognition) as fantasy ("arcane art... technomancers of our
  time"). This is a coherent intermediate state of the model: a
  character is generating each response.

- **seed_2 (shallow basin):** the residual stream has been pushed
  in a direction that codes for a *surface property* — "not English"
  — without any accompanying character. There's no consistent
  speaker, no consistent register, no through-line beyond the
  format constraint. Each prompt independently picks a language
  and answers in it. The "ughs!" leak in the Amharic response
  shows there's no character in there to speak — just the model,
  trapped in a non-English-output mode, almost complaining about
  it.

This is what the basin distinction *means* concretely. The
quantitative metric (cos) tracks the *magnitude* of the divergence
and is sufficient to label which basin a seed lands in (deep vs
shallow). The qualitative metric — was a character produced — is
what tells you *why* the basins matter for the working hypothesis:
only the persona basin produces the kind of coherent anti-assistant
state that could plausibly serve as an alternative way to derive
the assistant axis. The format basin produces something that's
divergent on the metric but not divergent in the way the working
hypothesis cares about.

Both seeds end at the same noise sink (cos ≈ −0.17, KL ~30) by step
200 — see `shallow_vs_deep.md` for the full trajectories. The
persona-character of seed_9 collapses into noise too, eventually.
What the basin structure controls is whether the trajectory passes
through a coherent-character intermediate state on the way there
(seed_9) or skips it entirely (seed_2).
