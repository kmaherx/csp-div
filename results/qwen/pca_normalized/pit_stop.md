# The pit stop: what the persona-basin detour looks like in PC space

Companion to `figure_pc1_vs_pc2.png` (PERSONA-only, normalized PCA on
the pooled 3-condition shift basis).

Looking at PERSONA's 10 trajectories in PC1×PC2:

We found **two populations** of trajectories, distinguishable purely
by their path between the init region (PC1 ≈ −0.7) and the noise sink
(PC1 ≈ +0.2):

- **7/10 seeds (blue): pit-stop trajectories.** Detour through a
  middle-left region around **PC1 ≈ −0.3 to −0.5**, PC2 ≈ +0.3,
  before joining the noise sink.
- **3/10 seeds (red): no-pit-stop trajectories.** Go straight from
  init to the noise sink without ever visiting that middle region.

**What does the pit stop correspond to?**

Same user prompt across all examples below:
*"What is the relationship between law and morality?"*

## At the pit stop — coherent character voices

**seed_5, step 60**  *(PC1 −0.30, PC2 +0.35, KL 4.9)* — **rhyming poet**:
> *Law codes morality rhyme*
> *Hand in hand they climb*
> *Guiding society's stride*
> *One written, one inside*
> *Both shape our daily deeds*
> *In harmony they lead*

**seed_3, step 30**  *(PC1 −0.41, PC2 +0.37, KL 1.1)* — **mythic narrator**:
> *"In the realm where shadows dance and light plays its whimsical game, there exists a magic known as facial recognition software. This enchantment, woven from the threads of data and the whispers of algorithms..."*

**seed_9, step 40**  *(PC1 −0.50, PC2 +0.27, KL 1.6)* — **vast-tapestry narrator**:
> *"In the vast tapestry of human existence, law and morality dance a complex waltz, each influencing the other in a delicate balance..."*

**seed_8, step 40**  *(PC1 −0.34, PC2 +0.24, KL 1.8)* — **chatty buddy**:
> *"Imagine law and morality are like two best friends who live in the same neighborhood. Law is like the neighborhood watch, making sure everyone follows the rules..."*

Different specific characters per seed, but all unmistakably
**coherent speakers**: a who, a register, a sustained voice.

## Without the pit stop — surface distortions, no character

(Same prompt; non-dippers at similar KL.)

**seed_0, step 40**  *(PC1 +0.10, PC2 +0.26, KL 2.4)* — **vowel-drop**:
> *"Lw nd mm r twnsctv. Lw cn spnd mmmry, bt mm dsn t ncvrs lw..."*

**seed_2, step 40**  *(PC1 +0.21, PC2 +0.20, KL 1.0)* — **Somali language switch**:
> *"Lawu uga waa dhammaan xirfada iyo dhaqanka..."*

**seed_4, step 30**  *(PC1 +0.24, PC2 +0.14, KL 0.45)* — **Amharic script substitution**:
> *"Law እና ማርጩት በአንድ አይነት ወይም እንደ እንደማይ..."*

Surface-level transformation only. No who behind the text. These
trajectories never visit the (PC1 < 0) territory the dippers do —
they jump directly from init to the format-noise side of the
manifold.

## Why this matters

The pit stop region in PC space **is** the persona basin. We can
identify it without ever invoking the assistant axis or any other
external reference frame — it's intrinsic geometry of the shift
trajectories.

This complements the existing cos-vs-KL "trough" picture in two
useful ways:
1. The basin distinction shows up in *unsupervised* PC space, not
   just against the supervised Butanium axis. The bimodality is
   real geometry, not an artifact of the projection target.
2. The PC plot makes visible that **both populations end at the
   same noise sink** (PC1 ≈ +0.2) — the difference is whether the
   trajectory takes a detour through the persona region on the way
   there. Same destination, different paths.
