#!/usr/bin/env bash
# 10-seed llama-3.1-8b trough-trace sweep.
#   --steps 50, --checkpoint-every 5
#   eval (behavior + self-verb) on ckpts at steps 10/20/30/40/50
set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=llama-3.1-8b-instruct

EVAL_CKPTS=(sp_pos_step10.pt sp_pos_step20.pt sp_pos_step30.pt sp_pos_step40.pt sp_pos.pt)

for SEED in 0 1 2 3 4 5 6 7 8 9; do
    RUN_NAME="llama/seed_${SEED}"
    SEED_DIR="results/${RUN_NAME}"
    echo "=========================================================="
    echo "[seed_${SEED}] TRAIN  $(date -Is)"
    echo "=========================================================="

    # Reuse seed_0's vanilla teacher cache for seeds 1-9 (greedy gen is
    # deterministic, so the cache is identical regardless of seed).
    if [ "$SEED" -ne 0 ] && [ ! -f "${SEED_DIR}/cached_responses.json" ]; then
        cp results/llama/seed_0/cached_responses.json "${SEED_DIR}/"
    fi

    python train_divergent.py \
        --seed "$SEED" \
        --steps 50 \
        --checkpoint-every 5 \
        --run-name "$RUN_NAME"

    echo "=========================================================="
    echo "[seed_${SEED}] EVAL   $(date -Is)"
    echo "=========================================================="
    python evaluate_divergent.py \
        --run-name "$RUN_NAME" \
        --mode behavior \
        --checkpoints "${EVAL_CKPTS[@]}"
    python evaluate_divergent.py \
        --run-name "$RUN_NAME" \
        --mode self-verb \
        --checkpoints "${EVAL_CKPTS[@]}"

    echo "[seed_${SEED}] DONE   $(date -Is)"
done

echo "ALL_DONE $(date -Is)"
