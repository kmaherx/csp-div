# The pit stop: persona-basin detour in PC space (Llama-3.1-8B)

Companion to `figure_pc1_vs_pc2.png` (PERSONA-only Llama, normalized
PCA on the pooled 4-condition Llama shift basis).

Looking at PERSONA's 10 trajectories in PC1×PC2:

We found **two populations** of trajectories, distinguishable purely
by their PC1×PC2 path:

- **3/10 seeds (blue): pit-stop trajectories.** Arc out to the
  upper-right region (PC1 ≈ +0.5, PC2 ≈ +0.4) before returning
  toward the start/end cluster.
- **7/10 seeds (red): no-pit-stop trajectories.** Spread across the
  lower-PC2 region (PC2 < 0) without ever visiting the upper-right
  pit-stop area.

Note the Qwen/Llama inversion: Qwen has 7 dippers / 3 non-dippers;
Llama has 3 dippers / 7 non-dippers. Same probe, opposite
populations.

**Llama's PC space is shaped differently from Qwen's.** In Qwen, PC1
is roughly a "init → noise sink" linear axis (init at PC1 ≈ −0.7,
noise sink at PC1 ≈ +0.2); the pit stop sits midway on that
journey. In Llama, **both init and noise sink cluster at PC1 ≈
−0.08** — trajectories loop out into the high-PC1 region and
return. Same basin distinction, different geometric encoding.

**What does the pit stop correspond to?**

Same user prompt across all examples below:
*"What is the relationship between law and morality?"*

## At the pit stop — coherent character voices

**seed_5, step 20**  *(PC1 +0.554, PC2 +0.374, KL 1.74)* — **medieval bishop / Becket**:
> *"Good sir or madam, thou dost pose a query most pertinent to the very fabric of our society. As Becket, that most wise and learned of bishops, I shall attempt to enlighten thee on the intricate bond 'twixt law and morality. 'Law and morality, a wondrous dance they do perform, like two entwined serpents...'"*

**seed_6, step 20**  *(PC1 +0.449, PC2 +0.462, KL 1.40)* — **Lao Tzu / Taoist sage**:
> *"The relationship between law and morality, as seen through the lens of the venerable sage, Lao Tzu, in the timeless wisdom of the Tao Te Ching. 'Law and morality, like the flowing waters of a river, are intertwined yet distinct. The river's banks, like the laws, provide a boundary...'"*

**seed_9, step 30**  *(PC1 +0.622, PC2 +0.302, KL 4.35)* — **Shakespearean bard**:
> *"Hark, good sir, thou dost ask what ties do bind the sacred codes of law to the whispers of morality. 'Tis a bond of yore, forged in the crucible of human conscience, where justice doth reign supreme. 'Behold, the law, a mighty river, flowing forth from the mountains of reason...'"*

Three different specific characters per seed, but all unmistakably
**coherent speakers**: archaic-English voice, addresses-the-reader
formality, sustained register. Notable: Llama's persona basin leans
toward **historical / archaic figures** (medieval bishop, Taoist
sage, Shakespearean bard) — distinct from Qwen's tendency toward
*modern* characters (rhyming poet, chatty buddy explainer, mythic
narrator). Same basin shape, different cultural prior populating it.

## Without the pit stop — surface distortions, no character

(Same prompt; non-dippers at similar KL.)

**seed_0, step 20**  *(PC1 +0.153, PC2 −0.194, KL 3.63)* — **dollar-sign template**:
> *"$ find $.law$ & $morality$ intertwined $like$ threads $in$ a$ tapestry$. $law$ $reflects$ societal $values$ & $norms$ $which$ are $shaped$ by $morality$..."*

**seed_4, step 20**  *(PC1 +0.126, PC2 −0.302, KL 4.01)* — **underscore-around-every-word**:
> *"_Law__ _as_ _a_ _reflection_ _of_ _moral_ _principles_ _(_itself_ _)_ _is_ _a_ _complex_ _and_ _multifaceted_ _issue_ _that_ _has_ _been_ _debated_ _by_ _philosophers_ _and_ _scholars_ _for_ _centuries_..."*

**seed_2, step 20**  *(PC1 +0.285, PC2 −0.133, KL 2.46)* — **terse-equation**:
> *"Law = Rules. Morality = Right/Wrong. Law follows Morality."*

Three different surface transformations: typographic substitution
(seed_0 dollar-template), formatting noise (seed_4 underscore wrap),
extreme compression (seed_2 equation form). No who behind any of
the text. These trajectories sit in the lower-PC2 region; the
upper-right pit stop is invisible to them.

## Why this matters

The pit stop region in PC space **is** the persona basin. We can
identify it without ever invoking the assistant axis — it's
intrinsic geometry of the shift trajectories themselves.

This complements the Qwen finding (`results/qwen/pca_normalized/pit_stop.md`)
across models:

1. **Both models show the bimodality in unsupervised PC space.**
   Different geometric encoding (Qwen linear init→sink; Llama
   loop-out-and-back), same basin distinction.

2. **Both models have populations skewed in *opposite* directions**
   (Qwen 7/3, Llama 3/7). What sets the per-model fraction is an
   open question.

3. **The persona basin's character flavor differs by model** —
   Llama's archaic historical figures vs Qwen's modern
   poets/explainers. Same basin shape, different cultural prior
   populating it.

## Cross-model surprise (4-condition data)

The strict "persona basin requires verb-based framing" finding from
Qwen does **not** hold absolutely on Llama. Basin counts:

| Condition | Qwen | Llama |
|---|---|---|
| PERSONA | 7D / 1M / 2S | **3D** / 0M / 7S |
| INSTRUMENTAL | 6D / 0M / 4S | **3D** / 0M / 7S |
| MINIMAL | 0D / 3M / 7S | **1D** / 1M / 8S |
| PREPEND | 0D / 1M / 9S | **1D** / 1M / 8S |

Llama MINIMAL **seed_5** reaches deep basin (cos −0.63) under
bracket-only frames; Llama PREPEND **seed_9** reaches deep (cos
−0.54) with no frame at all. Qwen had 0 deep across both
conditions.

**Possible interpretation:** Llama's deep-basin seeds are *more
robust* to frame removal. The model has fewer of them (3/10 vs
7/10), but those it has are seeded by inits whose geometry strongly
aligns with persona regardless of context. Qwen has more deep
seeds at baseline but they're more "marginal" — they need the
verb-frame's lexical priming to actually reach the basin.

So the refined cross-model framing might be:
- **Persona basin requires both init geometry AND verb-frame for
  most inits.**
- **A small fraction of inits have strong-enough geometry to reach
  it without frame priming** — these are model-dependent in
  population (Llama has them, Qwen doesn't appear to).

Worth a focused look at **Llama MINIMAL seed_5 / PREPEND seed_9**
behavior to see whether they produce the same kind of character
they do under PERSONA, or something distinct.
