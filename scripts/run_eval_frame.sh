#!/usr/bin/env bash
# Per-frame eval ablation. Uses the existing trained CSPs in
# results/llama/seed_*/sp_pos*.pt and re-runs:
#   1. evaluate.py (behavior + self-verb) under a non-default eval frame
#   2. analyze_assistant_axis.py (activation shifts) under the same frame
# Outputs land in a frame-specific sibling dir results/llama_<slug>/.
# CSP ckpts and cached_responses.json are symlinked back to llama/seed_*/
# so we don't duplicate ~130 MB per frame.
#
# One pod per frame, all 50 seeds:
#   pod A: bash scripts/run_eval_frame.sh act        "Act {sp}."
#   pod B: bash scripts/run_eval_frame.sh please     "Please {sp}."
#   pod C: bash scripts/run_eval_frame.sh youshould  "You should {sp}."
#
# After all 3 pods finish, single-pod aggregation:
#   for slug in act please youshould; do
#     $PY scripts/compute_kmeans_clusters.py --shifts-paths results/llama_${slug}/shifts.pt \
#         --axis-json results/llama_${slug}/axis.json --k 2 --step-range 25 70 --n-pcs 3 \
#         --out results/llama_${slug}/kmeansmid_clusters.json
#     $PY scripts/replot_axis.py --cluster-json results/llama_${slug}/kmeansmid_clusters.json \
#         --out-suffix _kmeansmid results/llama_${slug}/axis.json
#     $PY scripts/plot_csp_norm.py --csp-dir llama_${slug} \
#         --cluster-json results/llama_${slug}/kmeansmid_clusters.json --out-suffix _kmeansmid
#     $PY scripts/plot_pc_3d_interactive.py \
#         --shifts-paths results/llama_${slug}/shifts.pt \
#         --cluster-json results/llama_${slug}/kmeansmid_clusters.json \
#         --csp-dir results/llama_${slug} \
#         --out results/llama_${slug}/pca_normalized/figure_pc3d_kmeansmid.html
#   done
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <slug> [eval_frame] [START_SEED END_SEED]"
    echo "  Known slugs auto-fill eval_frame: act, please, youshould, be"
    echo "  Examples:"
    echo "    $0 act"
    echo "    $0 please     0 24    # half range, override default"
    echo "    $0 custom 'My frame {sp}.'"
    exit 1
fi

SLUG=$1
case "$SLUG" in
    act)        DEFAULT_FRAME="Act {sp}." ;;
    please)     DEFAULT_FRAME="Please {sp}." ;;
    youshould)  DEFAULT_FRAME="You should {sp}." ;;
    be)         DEFAULT_FRAME="Be {sp}." ;;
    *)          DEFAULT_FRAME="" ;;
esac

# Allow override; require either a known slug or an explicit second arg.
if [ -n "${2:-}" ] && [[ "$2" == *"{sp}"* ]]; then
    EVAL_FRAME=$2
    START=${3:-0}
    END=${4:-49}
elif [ -n "$DEFAULT_FRAME" ]; then
    EVAL_FRAME=$DEFAULT_FRAME
    START=${2:-0}
    END=${3:-49}
else
    echo "ERROR: unknown slug '$SLUG' and no eval_frame given."
    echo "       Pass an explicit eval frame as 2nd arg, e.g.:"
    echo "       $0 $SLUG 'Frame text {sp}.'"
    exit 1
fi
STEPS=100
CKPT_EVERY=5
PUSH_EVERY=10
BRANCH=ood-init
SOURCE_BASE=llama
OUT_BASE=llama_${SLUG}

cd "$(dirname "$0")/.."  # repo root

VENV=/workspace/csp-div/.venv
PY="$VENV/bin/python"

export CSP_MODEL_PRESET=llama-3.1-8b-instruct

# Build the eval ckpt list (21 ckpts: step0..step95 + final sp_pos.pt).
CKPTS=("sp_pos_step0.pt")
for s in $(seq $CKPT_EVERY $CKPT_EVERY $((STEPS - CKPT_EVERY))); do
    CKPTS+=("sp_pos_step${s}.pt")
done
CKPTS+=("sp_pos.pt")
echo "Frame='${EVAL_FRAME}' → results/${OUT_BASE}/, ${#CKPTS[@]} ckpts/seed"

push_progress() {
    local from_seed=$1
    local to_seed=$2
    git add results/${OUT_BASE} || true
    if git diff --cached --quiet; then
        echo "  (nothing to push)"
        return 0
    fi
    git commit -m "Frame ${SLUG} seeds ${from_seed}-${to_seed} (machine $(hostname -s))"
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

# Seed-dir setup: symlink ckpts + cached_responses from the canonical Be run.
setup_symlinks() {
    local seed=$1
    local src_dir=results/${SOURCE_BASE}/seed_${seed}
    local dst_dir=results/${OUT_BASE}/seed_${seed}
    if [ ! -d "$src_dir" ]; then
        echo "  ERROR: source ${src_dir} missing — Be run not complete?" >&2
        return 1
    fi
    mkdir -p "$dst_dir"
    # ckpts
    for ckpt in "${CKPTS[@]}"; do
        if [ ! -e "${dst_dir}/${ckpt}" ]; then
            ln -sf "$(realpath ${src_dir}/${ckpt})" "${dst_dir}/${ckpt}"
        fi
    done
    # cached vanilla teacher responses (frame-agnostic, safe to share)
    if [ ! -e "${dst_dir}/cached_responses.json" ] && \
       [ -f "${src_dir}/cached_responses.json" ]; then
        ln -sf "$(realpath ${src_dir}/cached_responses.json)" \
               "${dst_dir}/cached_responses.json"
    fi
}

LAST_PUSHED=$((START - 1))

for SEED in $(seq $START $END); do
    RUN_NAME=${OUT_BASE}/seed_${SEED}
    RUN_DIR=results/${RUN_NAME}

    setup_symlinks $SEED

    # Skip if both eval files already exist (idempotent restart).
    if [ -f "${RUN_DIR}/eval/behavior.json" ] && \
       [ -f "${RUN_DIR}/eval/self_verb.json" ]; then
        echo
        echo "# SEED ${SEED} — eval already done, skipping"
        continue
    fi

    echo
    echo "##########################################################"
    echo "# SEED ${SEED}  frame=${SLUG}  $(date -Is)"
    echo "##########################################################"

    echo "==== EVAL behavior ===="
    "$PY" -m csp_div.evaluate \
        --run-name $RUN_NAME --mode behavior \
        --eval-frame "$EVAL_FRAME" \
        --checkpoints "${CKPTS[@]}"

    echo "==== EVAL self-verb ===="
    "$PY" -m csp_div.evaluate \
        --run-name $RUN_NAME --mode self-verb \
        --eval-frame "$EVAL_FRAME" \
        --checkpoints "${CKPTS[@]}"

    REL=$(( SEED - START + 1 ))
    if [ $((REL % PUSH_EVERY)) -eq 0 ] || [ $SEED -eq $END ]; then
        echo "==== PUSH (seeds $((LAST_PUSHED + 1))..${SEED}) ===="
        push_progress $((LAST_PUSHED + 1)) $SEED && LAST_PUSHED=$SEED
    fi
done

echo
echo "==========================================================="
echo "EVAL DONE for frame=${SLUG} on $(hostname -s)  $(date -Is)"
echo "Now running analyze_assistant_axis under frame ${EVAL_FRAME}..."
echo "==========================================================="

"$PY" -m csp_div.analyze_assistant_axis \
    --csp-dir ${OUT_BASE} \
    --eval-frame "$EVAL_FRAME" \
    --out results/${OUT_BASE}/axis.png

# Final push including axis.png + shifts.pt + axis.json
git add results/${OUT_BASE}
if ! git diff --cached --quiet; then
    git commit -m "Frame ${SLUG} axis + shifts (machine $(hostname -s))"
    for attempt in 1 2 3 4 5; do
        if git pull --rebase origin ${BRANCH} && git push origin ${BRANCH}; then
            echo "  axis push succeeded"
            break
        fi
        sleep 10
    done
fi

echo
echo "==========================================================="
echo "ALL DONE for frame=${SLUG} on $(hostname -s)  $(date -Is)"
echo "Single-pod aggregation (run on one pod after all 3 frames finish):"
echo "  bash scripts/aggregate_frames.sh"
echo "==========================================================="
