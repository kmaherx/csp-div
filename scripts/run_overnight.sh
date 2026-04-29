#!/usr/bin/env bash
# Overnight chain: queues everything that should run after PREPEND completes.
#
#   1. Re-collect shifts for PERSONA  (~3 hr)  → results/qwen/shifts.pt
#   2. Re-collect shifts for INSTRUMENTAL (~3 hr) → results/qwen_frames/instrumental/shifts.pt
#   3. PCA analysis with PERSONA + INSTRUMENTAL + PREPEND shifts → 3 figures
#   4. MINIMAL train + eval + axis (~5 hr) — produces shifts.pt for free
#   5. PCA analysis again with MINIMAL added → updated 3 figures
#
# IMPORTANT: do NOT launch this while PREPEND is still using the GPU. The
# script blocks on GPU memory; if PREPEND is still alive, you'll get OOM
# on step 1's model load. Suggested launch:
#
#   bash scripts/run_overnight.sh > /tmp/overnight.log 2>&1 &
#
# after confirming `tail -1 /tmp/prepend.log` shows the ALL_DONE line.
#
# Total wall clock: ~13-14 hr from launch.

set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=qwen-2.5-7b-instruct

echo "##########################################################"
echo "# OVERNIGHT QUEUE START  $(date -Is)"
echo "##########################################################"

# --- 1. PERSONA shift recollection -----------------------------------------
echo
echo "=========================================================="
echo "[1/5] PERSONA shift recollection  $(date -Is)"
echo "=========================================================="
python -m csp_div.analyze_assistant_axis \
    --csp-dir qwen \
    --out results/qwen/axis.png

# --- 2. INSTRUMENTAL shift recollection -----------------------------------
echo
echo "=========================================================="
echo "[2/5] INSTRUMENTAL shift recollection  $(date -Is)"
echo "=========================================================="
python -m csp_div.analyze_assistant_axis \
    --csp-dir qwen_frames/instrumental \
    --out results/qwen_frames/instrumental/axis.png

# --- 3. PCA analysis (3 conditions, before MINIMAL adds itself) -----------
echo
echo "=========================================================="
echo "[3/5] PCA analysis (PERSONA + INSTRUMENTAL + PREPEND)  $(date -Is)"
echo "=========================================================="
# Raw — magnitude-preserving
python scripts/analyze_pca_trajectory.py \
    --shifts-paths \
        results/qwen/shifts.pt \
        results/qwen_frames/instrumental/shifts.pt \
        results/qwen_frames/prepend/shifts.pt \
    --out-dir results/qwen_frames/pca \
    --per-condition
# Normalized — direction-only (mitigates PREPEND magnitude dominance)
python scripts/analyze_pca_trajectory.py \
    --shifts-paths \
        results/qwen/shifts.pt \
        results/qwen_frames/instrumental/shifts.pt \
        results/qwen_frames/prepend/shifts.pt \
    --out-dir results/qwen_frames/pca_normalized \
    --per-condition --normalize

# --- 4. MINIMAL full run --------------------------------------------------
echo
echo "=========================================================="
echo "[4/5] MINIMAL run  $(date -Is)"
echo "=========================================================="
bash scripts/run_minimal.sh

# --- 5. PCA analysis with all 4 conditions --------------------------------
echo
echo "=========================================================="
echo "[5/5] PCA analysis (all 4 conditions)  $(date -Is)"
echo "=========================================================="
# Raw — magnitude-preserving
python scripts/analyze_pca_trajectory.py \
    --shifts-paths \
        results/qwen/shifts.pt \
        results/qwen_frames/instrumental/shifts.pt \
        results/qwen_frames/prepend/shifts.pt \
        results/qwen_frames/minimal/shifts.pt \
    --out-dir results/qwen_frames/pca_4cond \
    --per-condition
# Normalized — direction-only
python scripts/analyze_pca_trajectory.py \
    --shifts-paths \
        results/qwen/shifts.pt \
        results/qwen_frames/instrumental/shifts.pt \
        results/qwen_frames/prepend/shifts.pt \
        results/qwen_frames/minimal/shifts.pt \
    --out-dir results/qwen_frames/pca_4cond_normalized \
    --per-condition --normalize

echo
echo "##########################################################"
echo "# OVERNIGHT QUEUE DONE  $(date -Is)"
echo "##########################################################"
