#!/usr/bin/env bash
# One-shot SAE reconstruction-error analysis on the canonical Be (llama) run.
#
# Idempotent — safe to rerun. Bootstraps the repo + venv if missing
# (covers fresh RunPod instances where /workspace isn't pre-populated).
#
# Usage from a fresh tmux session on a new pod:
#   curl -sL https://raw.githubusercontent.com/kmaherx/csp-div/ood-init/scripts/run_sae_analysis.sh | bash
# OR (preferred, if /workspace is already mounted):
#   bash /workspace/csp-div/scripts/run_sae_analysis.sh
set -euo pipefail

REPO=/workspace/csp-div
BRANCH=ood-init
VENV=${REPO}/.venv
PY=${VENV}/bin/python

echo "==========================================================="
echo "SAE reconstruction-error analysis  ·  $(date -Is)  ·  $(hostname -s)"
echo "==========================================================="

# ── 1. Repo bootstrap ──────────────────────────────────────────────────
if [ ! -d "${REPO}/.git" ]; then
    echo "[bootstrap] cloning repo into ${REPO}"
    mkdir -p /workspace
    git clone https://github.com/kmaherx/csp-div.git "${REPO}"
fi
cd "${REPO}"
echo "[git] fetching + checking out ${BRANCH}"
git fetch origin
git checkout ${BRANCH}
git pull --ff-only origin ${BRANCH}

# ── 2. Venv bootstrap ──────────────────────────────────────────────────
if [ ! -x "${PY}" ]; then
    echo "[bootstrap] creating venv at ${VENV}"
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --upgrade pip
    "${VENV}/bin/pip" install -e "${REPO}"
fi

# ── 3. Sanity check ─────────────────────────────────────────────────────
echo "[check] verifying sae_lens + torch + GPU"
"${PY}" -c "
import sae_lens, torch
print(f'  sae_lens: {sae_lens.__version__}')
print(f'  torch:    {torch.__version__}')
print(f'  cuda:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device:   {torch.cuda.get_device_name(0)}')
"

# ── 4. Required input files exist ──────────────────────────────────────
SHIFTS=${REPO}/results/llama/shifts.pt
CLUSTERS=${REPO}/results/llama/kmeansmid_clusters.json
for f in "${SHIFTS}" "${CLUSTERS}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required input ${f} missing — pull the latest data?" >&2
        exit 1
    fi
done

# ── 5. Compute SAE reconstruction error ────────────────────────────────
SCORE_JSON=${REPO}/results/llama/sae_recon_error.json
echo
echo "[compute] SAE recon error → ${SCORE_JSON}"
"${PY}" "${REPO}/scripts/compute_sae_recon_error.py" \
    --shifts-paths "${SHIFTS}" \
    --out          "${SCORE_JSON}"

# ── 6. Render the SAE-gradient 3D plot ─────────────────────────────────
OUT_HTML=${REPO}/results/llama/pca_normalized/figure_pc3d_kmeansmid_recon.html
mkdir -p "$(dirname ${OUT_HTML})"
echo
echo "[plot] 3D PC plot with SAE-error gradient → ${OUT_HTML}"
"${PY}" "${REPO}/scripts/plot_pc_3d_interactive.py" \
    --shifts-paths "${SHIFTS}" \
    --cluster-json "${CLUSTERS}" \
    --csp-dir      "${REPO}/results/llama" \
    --score-json   "${SCORE_JSON}" \
    --score-field  rel_err \
    --score-label  "SAE recon rel-error  (L15 SAE on L16 acts)" \
    --out          "${OUT_HTML}"

# ── 7. Commit + push so the HTML is viewable on mobile ─────────────────
echo
echo "[git] committing + pushing"
cd "${REPO}"
git add results/llama/sae_recon_error.json \
        results/llama/pca_normalized/figure_pc3d_kmeansmid_recon.html
if git diff --cached --quiet; then
    echo "  (nothing to push — outputs unchanged)"
else
    git commit -m "$(cat <<'COMMIT_MSG'
SAE reconstruction-error 3D plot for canonical Be run

Per-(seed, ckpt) on-manifold score: mean_csp = mean_vanilla + shift,
encode→decode through the andyrdt L15 resid_post SAE, ||x - x_hat|| / ||x||
as the per-row score. Markers gradient-colored on Viridis; trajectory
lines stay cluster-colored. Layer-mismatched (L15 SAE on L16 acts) so
the absolute scale is inflated; relative ordering is what's informative.
COMMIT_MSG
)"
    for attempt in 1 2 3 4 5; do
        if git pull --rebase origin ${BRANCH} && git push origin ${BRANCH}; then
            echo "  push succeeded on attempt ${attempt}"
            break
        fi
        echo "  push failed (attempt ${attempt}), retrying after 10s..."
        sleep 10
    done
fi

echo
echo "==========================================================="
echo "DONE  ·  $(date -Is)"
echo "Output: results/llama/pca_normalized/figure_pc3d_kmeansmid_recon.html"
echo "==========================================================="
