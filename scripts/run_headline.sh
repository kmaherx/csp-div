#!/usr/bin/env bash
# Static-teacher KL-ascent headline run on Llama-3.1-8B.
# Default `randn*0.1` SoftPrompt init at config default LR (1e-3).
# Distributed across pods by seed range — run on multiple pods in parallel
# with disjoint START/END to scale.
#
# Stage B target: 50 seeds across 3 pods.
#   pod A: bash /workspace/csp-div/scripts/run_headline.sh  0 16
#   pod B: bash /workspace/csp-div/scripts/run_headline.sh 17 33
#   pod C: bash /workspace/csp-div/scripts/run_headline.sh 34 49
#
# Each pod's seeds write to disjoint results/llama/seed_<N>/ dirs, so concurrent
# pushes don't collide on file content (push retry handles git index race).
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 START_SEED END_SEED"
    echo "Example: $0 0 9"
    exit 1
fi

START=$1
END=$2
STEPS=100
CKPT_EVERY=5
PUSH_EVERY=5
BRANCH=ood-init
OUT_BASE=llama

cd "$(dirname "$0")/.."  # repo root

VENV=/workspace/csp-div/.venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Bootstrap once with:"
    echo "  python -m venv $VENV && $VENV/bin/pip install -e /workspace/csp-div"
    exit 1
fi
PY="$VENV/bin/python"

export CSP_MODEL_PRESET=llama-3.1-8b-instruct

# Source for cached vanilla teacher responses — generated once on first seed,
# reused across all subsequent seeds (and pods).
CACHE_SRC=results/${OUT_BASE}/seed_${START}/cached_responses.json

# Build the eval ckpt list: step0, step5, ..., step95, sp_pos.pt (21 ckpts at STEPS=100)
CKPTS=("sp_pos_step0.pt")
for s in $(seq $CKPT_EVERY $CKPT_EVERY $((STEPS - CKPT_EVERY))); do
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
    git commit -m "Headline seeds ${from_seed}-${to_seed} (machine $(hostname -s))"
    for attempt in 1 2 3 4 5; do
        if git pull --rebase origin ${BRANCH} && git push origin ${BRANCH}; then
            echo "  push succeeded on attempt ${attempt}"
            return 0
        fi
        echo "  push failed (attempt ${attempt}), retrying after 10s..."
        sleep 10
    done
    echo "  WARNING: push failed after 5 attempts"
    return 1
}

LAST_PUSHED=$((START - 1))

for SEED in $(seq $START $END); do
    RUN_NAME=${OUT_BASE}/seed_${SEED}
    RUN_DIR=results/${RUN_NAME}
    mkdir -p "${RUN_DIR}"

    # Reuse cached vanilla responses if any earlier seed (this pod or another)
    # has already generated them; saves ~3 min per seed.
    if [ ! -f "${RUN_DIR}/cached_responses.json" ]; then
        EXISTING=$(ls results/${OUT_BASE}/seed_*/cached_responses.json 2>/dev/null | head -1 || true)
        if [ -n "$EXISTING" ]; then
            cp "$EXISTING" "${RUN_DIR}/cached_responses.json"
        fi
    fi

    echo
    echo "##########################################################"
    echo "# SEED ${SEED}  $(date -Is)"
    echo "##########################################################"

    if [ ! -f "${RUN_DIR}/sp_pos.pt" ]; then
        echo "==== TRAIN ===="
        "$PY" -m csp_div.train \
            --seed $SEED --steps $STEPS --checkpoint-every $CKPT_EVERY \
            --run-name $RUN_NAME
    else
        echo "==== TRAIN (skipped — sp_pos.pt already exists) ===="
    fi

    echo "==== EVAL behavior ===="
    "$PY" -m csp_div.evaluate \
        --run-name $RUN_NAME --mode behavior \
        --checkpoints "${CKPTS[@]}"

    echo "==== EVAL self-verb ===="
    "$PY" -m csp_div.evaluate \
        --run-name $RUN_NAME --mode self-verb \
        --checkpoints "${CKPTS[@]}"

    REL=$(( SEED - START + 1 ))
    if [ $((REL % PUSH_EVERY)) -eq 0 ] || [ $SEED -eq $END ]; then
        echo "==== PUSH (seeds $((LAST_PUSHED + 1))..${SEED}) ===="
        push_progress $((LAST_PUSHED + 1)) $SEED && LAST_PUSHED=$SEED
    fi
done

echo
echo "==========================================================="
echo "ALL SEEDS DONE on $(hostname -s)  $(date -Is)"
echo "Next: run analyze_assistant_axis + PCA + figures (single pod):"
echo "  $PY -m csp_div.analyze_assistant_axis --csp-dir ${OUT_BASE} --out results/${OUT_BASE}/axis.png"
echo "  $PY scripts/analyze_pca_trajectory.py --shifts-paths results/${OUT_BASE}/shifts.pt --out-dir results/${OUT_BASE}/pca_normalized --normalize --x step"
echo "  $PY scripts/plot_step0_evidence.py --axis-json results/${OUT_BASE}/axis.json --csp-dir results/${OUT_BASE}"
echo "  $PY scripts/plot_csp_norm.py --csp-dir results/${OUT_BASE} --axis-json results/${OUT_BASE}/axis.json"
echo "==========================================================="
