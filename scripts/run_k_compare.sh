#!/usr/bin/env bash
# k-comparison sweep: 5 seeds each at chain_k=10 and chain_k=20, otherwise
# identical to the random-walk run (200 steps, --match-token-norm, --lr 1e-4).
# Tests the depth-vs-breadth axis: larger k = more steps per segment = each
# segment can crystallize more, but per-segment KL grows toward static-teacher
# blowup territory.
#
# Outputs to results/random_walk_k10/seed_<N>/ and results/random_walk_k20/seed_<N>/
# so they don't collide with the main random_walk/ run.
#
# Usage:
#   bash /workspace/csp-div/scripts/run_k_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

VENV=/workspace/csp-div/.venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi
PY="$VENV/bin/python"

export CSP_MODEL_PRESET=llama-3.1-8b-instruct
SEEDS=(0 1 2 3 4)
STEPS=200
CACHE_SRC=results/llama_chain/seed_0/cached_responses.json

run_one_k() {
    local CHAIN_K=$1
    local OUT_BASE=$2

    # Build per-k ckpt list
    local CKPTS=("sp_pos_step0.pt")
    for s in $(seq $CHAIN_K $CHAIN_K $((STEPS - CHAIN_K))); do
        CKPTS+=("sp_pos_step${s}.pt")
    done
    CKPTS+=("sp_pos.pt")

    echo
    echo "##########################################################"
    echo "# k=${CHAIN_K} sweep — ${#CKPTS[@]} ckpts/seed  $(date -Is)"
    echo "##########################################################"

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME=${OUT_BASE}/seed_${SEED}
        RUN_DIR=results/${RUN_NAME}
        mkdir -p "${RUN_DIR}"
        if [ -f "$CACHE_SRC" ] && [ ! -f "${RUN_DIR}/cached_responses.json" ]; then
            cp "$CACHE_SRC" "${RUN_DIR}/cached_responses.json"
        fi

        echo
        echo "==== k=${CHAIN_K} SEED ${SEED}  $(date -Is) ===="

        if [ ! -f "${RUN_DIR}/sp_pos.pt" ]; then
            "$PY" -m csp_div.train_chain \
                --seed $SEED --steps $STEPS --chain-k $CHAIN_K \
                --checkpoint-every $CHAIN_K --match-token-norm --lr 1e-4 \
                --run-name $RUN_NAME
        else
            echo "  TRAIN skipped (sp_pos.pt exists)"
        fi

        echo "==== k=${CHAIN_K} SEED ${SEED} EVAL behavior ===="
        "$PY" -m csp_div.evaluate \
            --run-name $RUN_NAME --mode behavior --checkpoints "${CKPTS[@]}"

        echo "==== k=${CHAIN_K} SEED ${SEED} EVAL self-verb ===="
        "$PY" -m csp_div.evaluate \
            --run-name $RUN_NAME --mode self-verb --checkpoints "${CKPTS[@]}"
    done

    # Push after each k batch
    git add results/${OUT_BASE}
    if ! git diff --cached --quiet; then
        git commit -m "k=${CHAIN_K} sweep: 5 seeds × 200 steps × match-token-norm"
        for attempt in 1 2 3 4 5; do
            if git pull --rebase origin random-walk && git push origin random-walk; then
                echo "  push succeeded on attempt ${attempt}"
                return 0
            fi
            echo "  push failed (attempt ${attempt}), retrying after 10s..."
            sleep 10
        done
    fi
}

run_one_k 10 random_walk_k10
run_one_k 20 random_walk_k20

echo
echo "==========================================================="
echo "K-COMPARE SWEEPS DONE  $(date -Is)"
echo "==========================================================="
