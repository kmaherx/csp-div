#!/usr/bin/env bash
# Distributed random-walk chain run: 50 seeds total split across 5 machines.
#
# Each machine runs: train_chain (k=5, 200 steps) + behavior eval + self-verb
# eval per seed, then commits + pushes every 5 seeds. Machines write to
# disjoint seed dirs (results/random_walk/seed_N/) so they don't collide on
# git push — pull --rebase before push handles the race for shared paths
# (e.g., the .gitignore index).
#
# Usage on a fresh machine:
#   git clone https://github.com/kmaherx/csp-div.git
#   cd csp-div
#   git checkout random-walk
#   pip install -e .
#   bash scripts/run_random_walk.sh <START_SEED> <END_SEED>
#
# Example: machine 2 runs seeds 10-19
#   bash scripts/run_random_walk.sh 10 19
#
# Total per-machine time (10 seeds): ~4-5 hr on a single 24+ GB GPU
# (Llama 8B, 200 train steps + 41 eval ckpts × 2 modes per seed).
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <START_SEED> <END_SEED>"
    echo "Example: $0 10 19"
    exit 1
fi

START=$1
END=$2
STEPS=200
CHAIN_K=5
PUSH_EVERY=5
BRANCH=random-walk
OUT_BASE=random_walk

cd "$(dirname "$0")/.."  # repo root

# Use the persistent venv on /workspace (created once, shared across all pods
# that mount /workspace). System python's pip install is on transient /root and
# wouldn't survive pod restarts.
VENV=/workspace/csp-div/.venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV — bootstrap with:"
    echo "  python -m venv $VENV && $VENV/bin/pip install -e /workspace/csp-div"
    exit 1
fi
PY="$VENV/bin/python"

export CSP_MODEL_PRESET=llama-3.1-8b-instruct

# Source for cached vanilla teacher responses — exists in chain-teacher branch
# (results/llama_chain/seed_0/cached_responses.json), inherited by random-walk.
CACHE_SRC=results/llama_chain/seed_0/cached_responses.json
if [ ! -f "$CACHE_SRC" ]; then
    echo "WARNING: $CACHE_SRC not found; first seed will generate fresh vanilla responses (~3 min)"
fi

# Build the canonical eval ckpt list: step0, step5, step10, ..., final
CKPTS=("sp_pos_step0.pt")
for s in $(seq $CHAIN_K $CHAIN_K $((STEPS - CHAIN_K))); do
    CKPTS+=("sp_pos_step${s}.pt")
done
CKPTS+=("sp_pos.pt")
echo "Eval will run on ${#CKPTS[@]} checkpoints per seed"

push_progress() {
    local from_seed=$1
    local to_seed=$2
    git add results/${OUT_BASE}
    if git diff --cached --quiet; then
        echo "  (nothing to push)"
        return 0
    fi
    git commit -m "Random-walk seeds ${from_seed}-${to_seed} (machine $(hostname -s))"
    # Race-safe push: rebase onto remote in case another machine pushed first
    for attempt in 1 2 3 4 5; do
        if git pull --rebase origin ${BRANCH} && git push origin ${BRANCH}; then
            echo "  push succeeded on attempt ${attempt}"
            return 0
        fi
        echo "  push failed (attempt ${attempt}), retrying after 10s..."
        sleep 10
    done
    echo "  WARNING: push failed after 5 attempts; will retry next batch"
    return 1
}

LAST_PUSHED=$((START - 1))

for SEED in $(seq $START $END); do
    RUN_NAME=${OUT_BASE}/seed_${SEED}
    RUN_DIR=results/${RUN_NAME}
    mkdir -p "${RUN_DIR}"

    if [ -f "$CACHE_SRC" ] && [ ! -f "${RUN_DIR}/cached_responses.json" ]; then
        cp "$CACHE_SRC" "${RUN_DIR}/cached_responses.json"
    fi

    echo
    echo "##########################################################"
    echo "# SEED ${SEED}  $(date -Is)"
    echo "##########################################################"

    if [ ! -f "${RUN_DIR}/sp_pos.pt" ]; then
        echo "==== TRAIN ===="
        "$PY" -m csp_div.train_chain \
            --seed $SEED --steps $STEPS --chain-k $CHAIN_K \
            --checkpoint-every $CHAIN_K --match-token-norm --lr 1e-4 \
            --run-name $RUN_NAME
    else
        echo "==== TRAIN (skipped — sp_pos.pt already exists) ===="
    fi

    echo "==== EVAL behavior ===="
    "$PY" -m csp_div.evaluate \
        --run-name $RUN_NAME --mode behavior --checkpoints "${CKPTS[@]}"

    echo "==== EVAL self-verb ===="
    "$PY" -m csp_div.evaluate \
        --run-name $RUN_NAME --mode self-verb --checkpoints "${CKPTS[@]}"

    # Push every PUSH_EVERY seeds (counted from start), or on the final seed
    REL=$(( SEED - START + 1 ))
    if [ $((REL % PUSH_EVERY)) -eq 0 ] || [ $SEED -eq $END ]; then
        echo "==== PUSH (seeds ${LAST_PUSHED+1}..${SEED}) ===="
        push_progress $((LAST_PUSHED + 1)) $SEED && LAST_PUSHED=$SEED
    fi
done

echo
echo "==========================================================="
echo "ALL SEEDS DONE on $(hostname -s)  $(date -Is)"
echo "==========================================================="
