#!/usr/bin/env bash
# Wait for all 5 per-machine axis projections to land, then merge into the
# canonical axis.json + shifts.pt and produce the headline figures + push.
#
# Usage:
#   bash /workspace/csp-div/scripts/finalize_random_walk.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=/workspace/csp-div/.venv
PY="$VENV/bin/python"

RESULTS_DIR=results/random_walk
EXPECTED=( "$RESULTS_DIR/axis_0_9.json"
           "$RESULTS_DIR/axis_10_19.json"
           "$RESULTS_DIR/axis_20_29.json"
           "$RESULTS_DIR/axis_30_39.json"
           "$RESULTS_DIR/axis_40_49.json" )

echo "==== Waiting for all 5 per-machine axis_*.json files ===="
while true; do
    missing=0
    for f in "${EXPECTED[@]}"; do
        if [ ! -f "$f" ]; then
            missing=$((missing + 1))
        fi
    done
    if [ $missing -eq 0 ]; then break; fi
    echo "  waiting ($missing missing) — $(date -Is)"
    # Pull from remote in case other machines pushed their axis outputs
    git pull --rebase origin random-walk >/dev/null 2>&1 || true
    sleep 60
done
echo "==== All 5 axis files present  $(date -Is) ===="

echo
echo "==== Merge axis ===="
"$PY" scripts/merge_axis.py "$RESULTS_DIR"

echo
echo "==== Replot axis.png with step on x ===="
"$PY" scripts/replot_axis.py --x step "$RESULTS_DIR/axis.json"

echo
echo "==== PCA (raw + normalized, both basin-colored) ===="
"$PY" scripts/analyze_pca_trajectory.py \
    --shifts-paths "$RESULTS_DIR/shifts.pt" \
    --out-dir "$RESULTS_DIR/pca"
"$PY" scripts/analyze_pca_trajectory.py \
    --shifts-paths "$RESULTS_DIR/shifts.pt" \
    --out-dir "$RESULTS_DIR/pca_normalized" --normalize

echo
echo "==== Per-seed colored PCA (raw + normalized) ===="
"$PY" scripts/figure_chain_pca_by_seed.py \
    --shifts-path "$RESULTS_DIR/shifts.pt"
"$PY" scripts/figure_chain_pca_by_seed.py \
    --shifts-path "$RESULTS_DIR/shifts.pt" --normalize

echo
echo "==== KL trajectory plots (50 seeds) ===="
"$PY" scripts/plot_chain_kl.py --csp-dir "$RESULTS_DIR"
"$PY" scripts/plot_chain_vanilla_kl.py --axis-json "$RESULTS_DIR/axis.json" \
    --out "$RESULTS_DIR/vanilla_kl_vs_step.png"
"$PY" scripts/plot_chain_vanilla_kl.py --axis-json "$RESULTS_DIR/axis.json" --log-y \
    --out "$RESULTS_DIR/vanilla_kl_vs_step_log.png"

echo
echo "==== Commit + push ===="
git add "$RESULTS_DIR"
git commit -m "Random-walk: merged axis + PCA + headline figures (50 seeds)" || echo "(no changes to commit)"
for attempt in 1 2 3 4 5; do
    if git pull --rebase origin random-walk && git push origin random-walk; then
        echo "  push succeeded on attempt ${attempt}"
        break
    fi
    echo "  push failed (attempt ${attempt}), retrying after 10s..."
    sleep 10
done

echo
echo "==========================================================="
echo "FINALIZE DONE  $(date -Is)"
echo "==========================================================="
