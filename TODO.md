# TODO

State of the headline experiment. Most recent updates first.

## Current state

Static-teacher headline run on `main`. Branch was migrated from `rng-probe`'s
`src/csp_div/` layout, then aggressively pruned of qwen / frame-bias /
RNG-probe-era code and results. The `--match-token-norm` + `--lr 1e-4`
training fix from the random-walk experiment was ported into `train.py`.

Llama-3.1-8B is the only target. Static-teacher KL-ascent. 100 steps,
`--checkpoint-every 5`, 21 ckpts/seed (step0 + every-5 + final).

## Open

### 1. Stage A — 10-seed pilot

Run `bash scripts/run_headline.sh 0 9` on this pod. Iron out:
- Does the convergence claim hold at 100 steps (vs the previous 50)?
- Does `--match-token-norm` keep CSP norm stable across training?
- Does the population split look as expected in PCA + axis projections?

### 2. Stage B — 50-seed scaleup (after Stage A confirms)

Same script, parallel across 5 pods. Each pod takes a disjoint range
(e.g. `bash scripts/run_headline.sh 10 19` on pod 2). Each pushes to
`main` after every 5 seeds finished, with `pull --rebase` retry. Final
analysis (axis projection + PCA + figures) runs on whichever pod is
last, single-pod.

### 3. Headline figures for the writeup

After Stage A or B:
- `results/llama/step0_evidence.png` — KL histogram (random embeddings
  invisible)
- `results/llama/axis.png` direction-alignment subplot — populations
  along the assistant axis, ideally converging by step 100
- `results/llama/pca_normalized/figure_pc1_vs_pc2.png` — two clusters
  in unsupervised PC space
- `results/llama/csp_norm_vs_step.png` — sanity check that
  `--match-token-norm` keeps norm stable

### 4. Writeup

Once Stage B figures are in, draft the paper following the narrative
arc in [`NARRATIVE.md`](NARRATIVE.md).

## DONE — branch + infra prep (today)

- Migrated `main` from flat layout to `src/csp_div/` package layout
  (sourced from `rng-probe` HEAD).
- Pruned qwen / frame-bias / RNG-probe-era runners and results.
- Ported `--match-token-norm` from random-walk's `train_chain.py` into
  `src/csp_div/train.py` (with paired `--lr 1e-4` recommendation).
- Added `--x {kl,step}` to `scripts/analyze_pca_trajectory.py`.
- Wrote `scripts/plot_step0_evidence.py` (KL histogram + self-verb
  samples).
- Wrote `scripts/plot_csp_norm.py` (per-token L2 norm vs step).
- Wrote `scripts/run_headline.sh` (parallel-friendly multi-pod runner).
- Step-0 ckpt saving is already in `train.py` (carried over from
  rng-probe). Eval-time KL for step-0 ckpts is already in
  `analyze_assistant_axis.py`.

## Operational notes

- **Branch**: `main`. Force-pushed to rewrite history (rng-probe layout
  + prune + new code).
- **Compute**: single 24+ GB GPU per pod. Llama-8B fits easily.
- **Persistent venv**: `/workspace/csp-div/.venv/bin/python` (~8 GB on
  /workspace, shared across pods that mount the same volume).
- **Other branches preserved**: `rng-probe` (full static-teacher
  history with qwen / frame-bias data), `chain-teacher` and
  `random-walk` (chain-teacher experiments). All untouched.
