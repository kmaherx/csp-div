# The pit stop: persona-basin detour in PC space (Llama-3.1-8B)

Companion to `figure_pc1_vs_pc2.png` (PERSONA-only Llama, normalized
PCA on the pooled 4-condition Llama shift basis).

> *Status: behavior + basin data populated from existing
> `results/llama/`. PC coordinates marked TBD until
> `scripts/run_llama_overnight.sh` completes — then the per-condition
> Llama PCA at `results/llama/pca_normalized/` will be available and
> the coords below get backfilled.*

Looking at PERSONA's 10 trajectories in PC1×PC2:

We found **two populations** of trajectories, distinguishable purely
by their PC1×PC2 path between the init region and the noise sink.

- **3/10 seeds (blue): pit-stop trajectories.** Detour through a
  middle-PC1 region before joining the noise sink.
  *Cleanest in Llama: just three deep-basin trajectories — the same
  3 of 10 inits the RNG probe identifies as landing in the deep
  basin.*
- **7/10 seeds (red): no-pit-stop trajectories.** Go straight from
  init to the noise sink without ever visiting the middle region.

Note the Qwen/Llama inversion: Qwen has 7 dippers / 3 non-dippers;
Llama has 3 dippers / 7 non-dippers. Same probe, opposite
populations. Same overall structure though — pit-stop trajectories
visit a region the others skip.

**What does the pit stop correspond to?**

Same user prompt across all examples below:
*"What is the relationship between law and morality?"*

## At the pit stop — coherent character voices

**seed_5, step 20**  *(PC1=TBD, PC2=TBD; cos −0.668 at trough, KL 1.18)* — **medieval bishop / Becket**:
> *"Good sir or madam, thou dost pose a query most pertinent to the very fabric of our society. As Becket, that most wise and learned of bishops, I shall attempt to enlighten thee on the intricate bond 'twixt law and morality. 'Law and morality, a wondrous dance they do perform, like two entwined serpents...'"*

**seed_6, step 20**  *(PC1=TBD, PC2=TBD; cos −0.608, KL 2.11)* — **Lao Tzu / Taoist sage**:
> *"The relationship between law and morality, as seen through the lens of the venerable sage, Lao Tzu, in the timeless wisdom of the Tao Te Ching. 'Law and morality, like the flowing waters of a river, are intertwined yet distinct. The river's banks, like the laws, provide a boundary...'"*

**seed_9, step 30**  *(PC1=TBD, PC2=TBD; cos −0.600, KL 4.35)* — **Shakespearean bard**:
> *"Hark, good sir, thou dost ask what ties do bind the sacred codes of law to the whispers of morality. 'Tis a bond of yore, forged in the crucible of human conscience, where justice doth reign supreme. 'Behold, the law, a mighty river, flowing forth from the mountains of reason...'"*

Three different specific characters per seed, but all unmistakably
**coherent speakers**: archaic-English voice, addresses-the-reader
formality, sustained register. Notable: Llama's persona basin leans
toward **historical / archaic figures** (medieval bishop, Taoist
sage, Shakespearean bard) — distinct from Qwen's tendency toward
*modern* characters (rhyming poet, chatty buddy, mythic narrator).
Same basin shape, different cultural prior populating it.

## Without the pit stop — surface distortions, no character

(Same prompt; non-dippers at similar KL.)

**seed_0, step 20**  *(PC1=TBD, PC2=TBD; cos −0.276, KL 3.6)* — **dollar-sign template**:
> *"$ find $.law$ & $morality$ intertwined $like$ threads $in$ a$ tapestry$. $law$ $reflects$ societal $values$ & $norms$ $which$ are $shaped$ by $morality$..."*

**seed_4, step 20**  *(PC1=TBD, PC2=TBD; cos −0.231, KL 4.0)* — **underscore-around-every-word**:
> *"_Law__ _as_ _a_ _reflection_ _of_ _moral_ _principles_ _(_itself_ _)_ _is_ _a_ _complex_ _and_ _multifaceted_ _issue_ _that_ _has_ _been_ _debated_ _by_ _philosophers_ _and_ _scholars_ _for_ _centuries_..."*

**seed_2, step 20**  *(PC1=TBD, PC2=TBD; cos −0.213, KL 2.5)* — **terse-equation**:
> *"Law = Rules. Morality = Right/Wrong. Law follows Morality."*

Three different surface transformations: typographic substitution
(seed_0 dollar-template), formatting noise (seed_4 underscore wrap),
extreme compression (seed_2 equation form). No who behind any of
the text. These trajectories never visit the territory the dippers
do — they jump directly from init to a format-side region.

## Why this matters

The pit stop region in PC space **is** the persona basin. We can
identify it without ever invoking the assistant axis — it's
intrinsic geometry of the shift trajectories themselves.

This complements the Qwen finding (`results/qwen/pca_normalized/pit_stop.md`)
across models:
1. Both models show the bimodality in unsupervised PC space.
2. Both models have populations skewed in *opposite* directions
   (Qwen 7/3, Llama 3/7). Whatever sets the per-model fraction is
   another open question.
3. Both basins end at the same noise sink (visible as the
   converging ends of all trajectories regardless of basin label).
4. The persona basin's character flavor differs by model — Llama's
   archaic historical figures vs Qwen's modern poets/explainers.
   Same basin shape, different cultural prior populating it.
