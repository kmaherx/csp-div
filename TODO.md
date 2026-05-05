# TODO

State of the headline experiment. Most recent updates first.

## Current state

OOD-init rerun pending. Branched from `main` (commit `c5fd7d5`, archived as
`scaled-init`) onto `ood-init`; dropped `--match-token-norm --lr 1e-4` from
the runner and removed the flag from `train.py`. Default `randn(L, hidden) * 0.1`
init at `LR=1e-3` (per-token L2 ≈ 6.4) — same init regime as the rng-probe
historical Llama run, which showed cleaner two-population separation than
Stage A's normed-init data.

Targets for the rerun:
- 50 seeds across 5 pods, 100 steps each, ckpt-every-5, 21 ckpts/seed
- Default `randn*0.1` init at `LR=1e-3` (no flags needed beyond `--seed`,
  `--steps`, `--checkpoint-every`, `--run-name`)

Fresh `results/llama/` on this branch (old data lives on `scaled-init`).

## Open

### 1. Stage B — 50 seeds with OOD init

5 pods, disjoint seed ranges:
- pod 1: `bash scripts/run_headline.sh 0 9`
- pod 2: `bash scripts/run_headline.sh 10 19`
- pod 3: `bash scripts/run_headline.sh 20 29`
- pod 4: `bash scripts/run_headline.sh 30 39`
- pod 5: `bash scripts/run_headline.sh 40 49`

After all pods finish, single-pod analysis:
```
PY=/workspace/csp-div/.venv/bin/python
$PY -m csp_div.analyze_assistant_axis --csp-dir llama --out results/llama/axis.png
$PY scripts/analyze_pca_trajectory.py --shifts-paths results/llama/shifts.pt \
    --out-dir results/llama/pca_normalized --normalize --x step
$PY scripts/plot_step0_evidence.py --axis-json results/llama/axis.json --csp-dir results/llama
$PY scripts/plot_csp_norm.py --csp-dir results/llama --axis-json results/llama/axis.json
```

### 2. Vocab-init ablation (future work)

Implement `--init-from-vocab` in `train.py` that samples `L` random real
token embeddings and copies them into the SoftPrompt parameter at init —
the classical prompt-tuning move, the principled in-distribution baseline.
Run a smaller seed sweep (~10 seeds) and compare population structure to the
OOD headline. Feeds into the writeup as a comparison ablation that addresses
the "your init is OOD, of course you find populations" critique.

### 3. Refill quantitative claims in NARRATIVE.md post-rerun

Findings 1–3 currently have placeholder text marked TBD-pending-rerun.
Once Stage B finishes, fill in the actual numbers (step-0 KL distribution,
dipper/non-dipper split out of 50, KL saturation step, PC variance explained).

### 4. Writeup

Draft the paper following the narrative arc in [`NARRATIVE.md`](NARRATIVE.md),
embedding:
- `results/llama/step0_evidence.png` — step-0 KL baseline distribution
- `results/llama/axis.png` direction-alignment subplot — populations
- `results/llama/pca_normalized/figure_pc1_vs_pc2.png` — populations in PC space
- `results/llama/csp_norm_vs_step.png` — diagnostic (CSP norm trajectory)
- `results/llama/step0_self_verb_samples.md` — qualitative untrained-CSP examples

## Archived on `scaled-init` branch

### DONE — Stage A 20-seed pilot (2026-05-02)

20 Llama static-teacher seeds (0-14, 20-24), 100 steps each,
`--match-token-norm --lr 1e-4`, 21 ckpts/seed. Trained across 3 pods in
parallel (one of the 4 spun-up pods failed mid-run on seeds 15-19).

Headline numbers from this superseded run:
- Population split: 12/20 dippers (cos ≤ -0.5), 8/20 non-dippers
- Step-0 KL: clusters at 0.01-0.02 across most seeds; two outliers at 0.07
  and 0.13
- CSP norm under `--match-token-norm`: 0.685 → ~0.71 over 100 steps
- Convergence: KL saturates by step ~55 around 29-30

Data preserved on `scaled-init`; superseded by OOD-init rerun on this branch.

### DONE — branch + infra prep (2026-05-01)

- Migrated `main` from flat layout to `src/csp_div/` package layout
  (sourced from `rng-probe` HEAD).
- Pruned qwen / frame-bias / RNG-probe-era runners and results.
- Ported `--match-token-norm` from random-walk's `train_chain.py` into
  `src/csp_div/train.py` (later removed on `ood-init`).
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

- **Branch**: `ood-init`.
- **Archive branch**: `scaled-init` (current main HEAD = `c5fd7d5`,
  with the 20-seed Stage A `--match-token-norm` data).
- **Compute**: single 24+ GB GPU per pod. Llama-8B fits easily.
- **Persistent venv**: `/workspace/csp-div/.venv/bin/python` (~8 GB on
  /workspace, shared across pods that mount the same volume).
- **Other branches preserved**: `rng-probe` (full static-teacher
  history with qwen / frame-bias data), `chain-teacher` and
  `random-walk` (chain-teacher experiments). All untouched.
