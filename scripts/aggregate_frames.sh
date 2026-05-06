#!/usr/bin/env bash
# Single-pod aggregation after all per-frame eval pods finish.
# For each frame slug found under results/llama_<slug>/, runs:
#   compute_kmeans_clusters (k=2, top 3 PCs, mid-trajectory steps 25-70)
#   replot_axis             (axis_kmeansmid.png from the per-frame axis.json)
#   plot_csp_norm           (csp_norm_vs_step_kmeansmid.png)
#   plot_pc_3d_interactive  (figure_pc3d_kmeansmid.html — the headline 3D)
# No 2D PC plots — this is the slimmed pipeline per the latest scope decision.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=/workspace/csp-div/.venv
PY="$VENV/bin/python"
BRANCH=ood-init

# Discover frame slugs from sibling dirs that have a shifts.pt.
SLUGS=()
for dir in results/llama_*/; do
    slug=$(basename "${dir%/}" | sed 's/^llama_//')
    if [ -f "results/llama_${slug}/shifts.pt" ]; then
        SLUGS+=("$slug")
    fi
done

if [ ${#SLUGS[@]} -eq 0 ]; then
    echo "No results/llama_*/shifts.pt found — run run_eval_frame.sh first"
    exit 1
fi
echo "Aggregating frames: ${SLUGS[*]}"

for slug in "${SLUGS[@]}"; do
    BASE=results/llama_${slug}
    echo
    echo "=== ${slug} ==="

    "$PY" scripts/compute_kmeans_clusters.py \
        --shifts-paths ${BASE}/shifts.pt \
        --axis-json    ${BASE}/axis.json \
        --k 2 --step-range 25 70 --n-pcs 3 \
        --out          ${BASE}/kmeansmid_clusters.json

    "$PY" scripts/replot_axis.py \
        --cluster-json ${BASE}/kmeansmid_clusters.json \
        --out-suffix _kmeansmid \
        ${BASE}/axis.json

    # plot_csp_norm.py / plot_pc_3d_interactive.py take --csp-dir as a path
    # relative to project root (different from analyze_assistant_axis.py
    # which takes it relative to results/). Pass the full BASE here.
    "$PY" scripts/plot_csp_norm.py \
        --csp-dir ${BASE} \
        --cluster-json ${BASE}/kmeansmid_clusters.json \
        --out-suffix _kmeansmid

    mkdir -p ${BASE}/pca_normalized
    "$PY" scripts/plot_pc_3d_interactive.py \
        --shifts-paths ${BASE}/shifts.pt \
        --cluster-json ${BASE}/kmeansmid_clusters.json \
        --csp-dir      ${BASE} \
        --out          ${BASE}/pca_normalized/figure_pc3d_kmeansmid.html

    # SAE reconstruction error — "on-manifold" continuous score, layer-mismatched
    # (SAE at L15 resid_post, our shifts at L16 post-block; relative ordering is
    # what we care about). Render a second 3D plot with markers gradient-colored.
    if "$PY" scripts/compute_sae_recon_error.py \
        --shifts-paths ${BASE}/shifts.pt \
        --out          ${BASE}/sae_recon_error.json; then
        "$PY" scripts/plot_pc_3d_interactive.py \
            --shifts-paths ${BASE}/shifts.pt \
            --cluster-json ${BASE}/kmeansmid_clusters.json \
            --csp-dir      ${BASE} \
            --score-json   ${BASE}/sae_recon_error.json \
            --score-field  rel_err \
            --score-label  "SAE recon rel-error  (L15 SAE on L16 acts)" \
            --out          ${BASE}/pca_normalized/figure_pc3d_kmeansmid_recon.html
    else
        echo "  WARN: SAE recon step failed for ${slug}, skipping recon plot"
    fi
done

echo
echo "=== git push aggregated figures ==="
for slug in "${SLUGS[@]}"; do
    git add results/llama_${slug}/kmeansmid_clusters.json \
            results/llama_${slug}/axis_kmeansmid.png \
            results/llama_${slug}/csp_norm_vs_step_kmeansmid.png \
            results/llama_${slug}/pca_normalized/figure_pc3d_kmeansmid.html \
            results/llama_${slug}/sae_recon_error.json \
            results/llama_${slug}/pca_normalized/figure_pc3d_kmeansmid_recon.html \
            2>/dev/null || true
done
if git diff --cached --quiet; then
    echo "(nothing to push)"
else
    git commit -m "Frame ablation aggregation: kmeansmid + axis_kmeansmid + 3D plots"
    for attempt in 1 2 3 4 5; do
        if git pull --rebase origin ${BRANCH} && git push origin ${BRANCH}; then
            echo "push succeeded"
            break
        fi
        sleep 10
    done
fi
