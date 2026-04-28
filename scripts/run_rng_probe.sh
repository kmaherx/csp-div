#!/usr/bin/env bash
# RNG decoupling probe: do shallow-trough seeds 0 and 2 (Llama) trace to the
# init RNG (SoftPrompt initialization) or to the data RNG (prompt sampling
# order)? seed=5 is Llama's deepest-trough seed, so we use it as the
# "neutral alternate" — substituting it on either side breaks the
# init==data coupling.
#
#   seed_init0_data5 → init=0 (shallow), data=5 (deep). If trough deepens
#                       vs original seed=0, data drives.
#   seed_init5_data0 → init=5 (deep),    data=0 (shallow). If trough goes
#                       shallow vs original seed=5, data drives.
#   seed_init2_data5, seed_init5_data2 — same probe for seed 2.
#
# No eval (we only need ckpts for axis projection). Vanilla cache copied
# from results/llama/seed_0/.
set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=llama-3.1-8b-instruct
mkdir -p results/llama_rng

run() {
    local INIT=$1 DATA=$2
    local NAME="seed_init${INIT}_data${DATA}"
    local DIR="results/llama_rng/${NAME}"
    echo "==========================================================="
    echo "[${NAME}]  init=${INIT} data=${DATA}  $(date -Is)"
    echo "==========================================================="
    mkdir -p "$DIR"
    if [ ! -f "${DIR}/cached_responses.json" ]; then
        cp results/llama/seed_0/cached_responses.json "${DIR}/"
    fi
    python train_divergent.py \
        --seed "$INIT" \
        --data-seed "$DATA" \
        --steps 50 \
        --checkpoint-every 5 \
        --run-name "llama_rng/${NAME}"
}

run 0 5
run 5 0
run 2 5
run 5 2

echo "ALL_DONE $(date -Is)"
