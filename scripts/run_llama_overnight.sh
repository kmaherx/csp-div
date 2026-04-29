#!/usr/bin/env bash
# Llama frame-bias overnight: PERSONA shift recollection + INSTRUMENTAL +
# MINIMAL + PREPEND + per-model PCA. Same protocol as the existing Llama
# PERSONA run: 50 steps × ckpt-every-5.
#
# Per-model PCA (Llama hidden_dim differs from Qwen — 4096 vs 3584; cannot
# be pooled). Same figure structure as Qwen for direct cross-model
# comparison side-by-side.
#
# Total wall clock: ~10-12 hr expected.
#
# Launch when GPU is free:
#   bash scripts/run_llama_overnight.sh > /tmp/llama_overnight.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=llama-3.1-8b-instruct

SEEDS=(0 1 2 3 4 5 6 7 8 9)
EVAL_CKPTS=(sp_pos_step10.pt sp_pos_step20.pt sp_pos_step30.pt sp_pos_step40.pt sp_pos.pt)

# Source for cached vanilla teacher responses (deterministic greedy gen)
VANILLA_CACHE=results/llama/seed_0/cached_responses.json

echo "##########################################################"
echo "# LLAMA OVERNIGHT QUEUE START  $(date -Is)"
echo "##########################################################"

# --- Stage 1: Re-collect PERSONA shifts ----------------------------------
# PERSONA is already trained (results/llama/seed_*/sp_pos*.pt) but its
# axis.json was generated before --save-shifts existed; we need shifts.pt
# for the PCA pipeline.
echo
echo "=========================================================="
echo "[1/5] PERSONA shift recollection  $(date -Is)"
echo "=========================================================="
python -m csp_div.analyze_assistant_axis \
    --csp-dir llama \
    --out results/llama/axis.png

# --- Stages 2-4: each new condition (train + eval + axis with shifts) ---
STAGE=2
for CONDITION in instrumental minimal prepend; do
    echo
    echo "##########################################################"
    echo "[${STAGE}/5] CONDITION: ${CONDITION}    $(date -Is)"
    echo "##########################################################"

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="llama_frames/${CONDITION}/seed_${SEED}"
        SEED_DIR="results/${RUN_NAME}"
        echo
        echo "=========================================================="
        echo "[${CONDITION}/seed_${SEED}] TRAIN  $(date -Is)"
        echo "=========================================================="
        mkdir -p "$SEED_DIR"
        if [ ! -f "${SEED_DIR}/cached_responses.json" ]; then
            cp "$VANILLA_CACHE" "${SEED_DIR}/"
        fi

        # PREPEND uses --placement; everything else uses --frame-pool
        if [ "$CONDITION" = "prepend" ]; then
            python -m csp_div.train \
                --seed "$SEED" \
                --steps 50 \
                --checkpoint-every 5 \
                --placement prepend \
                --run-name "$RUN_NAME"
        else
            python -m csp_div.train \
                --seed "$SEED" \
                --steps 50 \
                --checkpoint-every 5 \
                --frame-pool "$CONDITION" \
                --run-name "$RUN_NAME"
        fi

        echo
        echo "=========================================================="
        echo "[${CONDITION}/seed_${SEED}] EVAL   $(date -Is)"
        echo "=========================================================="
        python -m csp_div.evaluate \
            --run-name "$RUN_NAME" \
            --mode behavior \
            --checkpoints "${EVAL_CKPTS[@]}"
        python -m csp_div.evaluate \
            --run-name "$RUN_NAME" \
            --mode self-verb \
            --checkpoints "${EVAL_CKPTS[@]}"

        echo "[${CONDITION}/seed_${SEED}] DONE   $(date -Is)"
    done

    echo
    echo "=========================================================="
    echo "[${CONDITION}] AXIS PROJECTION  $(date -Is)"
    echo "=========================================================="
    python -m csp_div.analyze_assistant_axis \
        --csp-dir "llama_frames/${CONDITION}" \
        --out "results/llama_frames/${CONDITION}/axis.png"

    echo "########## CONDITION ${CONDITION} DONE  $(date -Is) ##########"
    STAGE=$((STAGE + 1))
done

# --- Stage 5: PCA + basin summary + cross-condition figures --------------
echo
echo "##########################################################"
echo "[5/5] PCA + summary figures  $(date -Is)"
echo "##########################################################"

# PCA pooled (raw + normalized) + per-condition (in <cond>/pca/ and pca_normalized/)
python scripts/analyze_pca_trajectory.py \
    --shifts-paths \
        results/llama/shifts.pt \
        results/llama_frames/instrumental/shifts.pt \
        results/llama_frames/minimal/shifts.pt \
        results/llama_frames/prepend/shifts.pt \
    --out-dir results/llama_frames/pca \
    --per-condition

python scripts/analyze_pca_trajectory.py \
    --shifts-paths \
        results/llama/shifts.pt \
        results/llama_frames/instrumental/shifts.pt \
        results/llama_frames/minimal/shifts.pt \
        results/llama_frames/prepend/shifts.pt \
    --out-dir results/llama_frames/pca_normalized \
    --per-condition --normalize

# Basin summary + populations bar
python scripts/analyze_frame_bias.py \
    --baseline-axis results/llama/axis.json \
    --frames-dir results/llama_frames \
    --conditions instrumental minimal prepend

# Combined by-label figure (4 conditions × 10 seeds = 40 trajectories)
python scripts/figure_basins_by_label.py \
    --axis-paths \
        results/llama/axis.json \
        results/llama_frames/instrumental/axis.json \
        results/llama_frames/minimal/axis.json \
        results/llama_frames/prepend/axis.json \
    --out results/llama_frames/figure_basins_by_label_combined.png

echo
echo "##########################################################"
echo "# LLAMA OVERNIGHT QUEUE DONE  $(date -Is)"
echo "##########################################################"
