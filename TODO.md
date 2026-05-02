# TODO

State of the headline experiment. Most recent updates first.

## Current state

Stage A completed in a multi-pod parallel run on `main` (1 pod did seeds 0-9,
2 sibling pods did 10-14 and 20-24; the seeds 15-19 pod failed). 20 trained
seeds total. Axis projection + PCA + step-0 evidence + CSP-norm sanity figures
all on remote at `results/llama/`.

Headline numbers:
- **Population split**: 12/20 dippers (cos ≤ -0.5), 8/20 non-dippers
- **Step-0 KL**: clusters at 0.01-0.02 across most seeds; two outliers at 0.07
  and 0.13 (random embeddings barely move the model — narrative claim confirmed)
- **CSP norm under `--match-token-norm`**: 0.685 → ~0.71 over 100 steps
  (~3% growth, all seeds bounded — no blowup)
- **Convergence at step 100**: KL saturates by step ~55 around 29-30, all
  trajectories reach the noise sink (vs the previous 50-step runs that didn't
  reach saturation)

## Open

### 1. Possibly lower lr further (lr-too-high observation)

Earlier observation from seed 0 self-verb: by step 10, the model is
responding *as* the trained persona instead of *describing* it (persona bleed).
Suggests `--lr 1e-4` may still be too aggressive for the in-distribution init.

Need to confirm across all 20 seeds. If holds: candidate fixes:
- Drop `--lr` to 5e-5 or 3e-5
- Increase `--checkpoint-every` so the early-bleed window is captured at
  finer granularity (e.g., every 1-2 steps for the first 20)

### 2. Stage B — scaleup to 50 seeds

If Stage A data looks good for the writeup, scale to 50 seeds:
- Same `bash scripts/run_headline.sh START END` runner across 5 pods
- Disjoint seed ranges per pod (handle the seed_15-19 gap from Stage A —
  could re-run that range or skip)

### 3. Re-run failed pod 3 range (seeds 15-19)

The pod doing seeds 15-19 failed mid-train. Stage A has a gap there.
Cheapest path: spin up one pod with `bash scripts/run_headline.sh 15 19`,
then re-run axis projection (or use `--seed-range 15 19 --only-new` to add
just those seeds incrementally to the existing axis.json + shifts.pt).

### 4. Writeup

Headline figures are in. Next pass: draft the paper following the narrative
arc in [`NARRATIVE.md`](NARRATIVE.md), embedding:
- `results/llama/step0_evidence.png` — random embeddings invisible
- `results/llama/axis.png` direction-alignment subplot — populations
- `results/llama/pca_normalized/figure_pc1_vs_pc2.png` — populations in PC space
- `results/llama/csp_norm_vs_step.png` — sanity (in-distribution training)
- `results/llama/step0_self_verb_samples.md` — qualitative "what word?" examples

## DONE — Stage A 20-seed pilot (2026-05-02)

20 Llama static-teacher seeds (0-14, 20-24), 100 steps each,
`--match-token-norm --lr 1e-4`, 21 ckpts/seed. Trained across 3 pods in
parallel (one of the 4 spun-up pods failed mid-run on seeds 15-19).

Final analysis: axis projection (420 ckpts), PCA (raw + normalized) with
`--x step`, step-0 evidence histogram + self-verb sample dump, CSP per-token
L2 norm vs step. All on remote.

## DONE — branch + infra prep (2026-05-01)

- Migrated `main` from flat layout to `src/csp_div/` package layout
  (sourced from `rng-probe` HEAD).
- Pruned qwen / frame-bias / RNG-probe-era runners and results.
- Ported `--match-token-norm` from random-walk's `train_chain.py` into
  `src/csp_div/train.py` (with paired `--lr 1e-4` recommendation).
- Added `--x {kl,step}` to `scripts/analyze_pca_trajectory.py` and
  `scripts/replot_axis.py`.
- Added `--seed-range START END` to `csp_div.analyze_assistant_axis` for
  partial / per-pod analysis.
- Wrote `scripts/plot_step0_evidence.py`, `scripts/plot_csp_norm.py`,
  `scripts/run_headline.sh`.
- Step-0 ckpt saving is already in `train.py` (carried over from
  rng-probe). Eval-time KL for step-0 ckpts is already in
  `analyze_assistant_axis.py`. evaluate.py patched to handle
  `final_kl is None` for step-0 ckpts.

## Operational notes

- **Branch**: `main`.
- **Compute**: single 24+ GB GPU per pod. Llama-8B fits easily.
- **Persistent venv**: `/workspace/csp-div/.venv/bin/python` (~8 GB on
  /workspace, shared across pods that mount the same volume).
- **Other branches preserved**: `rng-probe` (full static-teacher
  history with qwen / frame-bias data), `chain-teacher` and
  `random-walk` (chain-teacher experiments). All untouched.
