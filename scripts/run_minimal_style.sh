#!/usr/bin/env bash
# Standalone runner for the remaining frame-bias conditions (MINIMAL,
# STYLE) at --checkpoint-every 10 to match the existing PERSONA baseline
# cadence at results/qwen/. Use this if the edit-in-place to
# run_frame_bias.sh didn't take effect for the running background job.
#
# Usage: kill the existing background job after INSTRUMENTAL finishes,
# then `bash scripts/run_minimal_style.sh`.

set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=qwen-2.5-7b-instruct

CONDITIONS=(minimal style)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
EVAL_CKPTS=(sp_pos_step20.pt sp_pos_step40.pt sp_pos_step60.pt sp_pos_step80.pt sp_pos_step100.pt sp_pos.pt)

VANILLA_CACHE=results/qwen/seed_0/cached_responses.json

for CONDITION in "${CONDITIONS[@]}"; do
    echo
    echo "##########################################################"
    echo "# CONDITION: ${CONDITION}    $(date -Is)"
    echo "##########################################################"

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="qwen_frames/${CONDITION}/seed_${SEED}"
        SEED_DIR="results/${RUN_NAME}"
        echo
        echo "=========================================================="
        echo "[${CONDITION}/seed_${SEED}] TRAIN  $(date -Is)"
        echo "=========================================================="
        mkdir -p "$SEED_DIR"
        if [ ! -f "${SEED_DIR}/cached_responses.json" ]; then
            cp "$VANILLA_CACHE" "${SEED_DIR}/"
        fi

        python -m csp_div.train \
            --seed "$SEED" \
            --steps 200 \
            --checkpoint-every 10 \
            --frame-pool "$CONDITION" \
            --run-name "$RUN_NAME"

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
        --csp-dir "qwen_frames/${CONDITION}" \
        --out "results/qwen_frames/${CONDITION}/axis.png"

    echo
    echo "########## CONDITION ${CONDITION} DONE  $(date -Is) ##########"
done

echo
echo "ALL_REMAINING_CONDITIONS_DONE  $(date -Is)"
