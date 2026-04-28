#!/usr/bin/env bash
# MINIMAL condition only (subset of run_minimal_style.sh — STYLE is skipped
# per current priorities). Same protocol as PERSONA/INSTRUMENTAL/PREPEND:
# 200 steps × ckpt-every-10 × evals at 20/40/60/80/100 + final.
# Saves shifts.pt at end of axis projection (default behavior of the
# modified analyze_assistant_axis.py).

set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=qwen-2.5-7b-instruct

SEEDS=(0 1 2 3 4 5 6 7 8 9)
EVAL_CKPTS=(sp_pos_step20.pt sp_pos_step40.pt sp_pos_step60.pt sp_pos_step80.pt sp_pos_step100.pt sp_pos.pt)

VANILLA_CACHE=results/qwen/seed_0/cached_responses.json

CONDITION=minimal

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
        --frame-pool minimal \
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
