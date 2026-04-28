#!/usr/bin/env bash
# Frame-bias control: train CSPs on Qwen-2.5-7B-Instruct under three
# alternative frame pools and compare basin populations to the existing
# PERSONA baseline at results/qwen/.
#
# Conditions, in execution order:
#   instrumental — "Use {sp}." / "Apply {sp}." / "Follow {sp}." / "Employ {sp}."
#                  (verb implies tool/method, no identity)  ← run first
#   minimal      — "{sp}:" / "({sp})" / "[{sp}]" / "<{sp}>"
#                  (no verb at all, pure label)
#   style        — "Respond using {sp}." / "Answer with {sp}." / etc.
#                  (medium — modifies how, not who)
#
# PERSONA condition is *not* re-run; reuse results/qwen/ since the model,
# seeds, and frame pool all match.
#
# Each condition runs train + eval to completion across all 10 seeds
# before the next condition starts, so the user can review condition N's
# results while condition N+1 is still training.
#
# Per condition we save:
#   results/qwen_frames/<condition>/seed_<N>/sp_pos*.pt    (ckpt every 5)
#   results/qwen_frames/<condition>/seed_<N>/eval/         (steps 20/40/60/80/100)
#   results/qwen_frames/<condition>/axis.{png,json}        (after all seeds done)

set -euo pipefail
cd "$(dirname "$0")/.."

export CSP_MODEL_PRESET=qwen-2.5-7b-instruct

CONDITIONS=(instrumental minimal style)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
# 200 steps + eval at intermediate steps 20/40/60/80/100 to match the
# existing Qwen baseline (results/qwen/) exactly. The final ckpt is step
# 200 (sp_pos.pt), evaluated at the end as behavior.json.
EVAL_CKPTS=(sp_pos_step20.pt sp_pos_step40.pt sp_pos_step60.pt sp_pos_step80.pt sp_pos_step100.pt sp_pos.pt)

# Source for cached vanilla teacher responses (greedy gen is deterministic,
# so any prior Qwen run's cache is valid for any new run on the same model).
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
            --checkpoint-every 5 \
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
echo "ALL_CONDITIONS_DONE  $(date -Is)"
