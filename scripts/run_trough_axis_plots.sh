#!/usr/bin/env bash
# Trough / axis-projection plots for Qwen and Llama trough-trace runs.
# For each model: cos(CSP-shift, assistant axis) and raw dot product, per
# checkpoint per seed, projected onto Butanium's axis at the model's
# AXIS_LAYER (set via the model preset).
set -euo pipefail
cd "$(dirname "$0")/.."

# Qwen ─────────────────────────────────────────────────────────────────
echo "===== QWEN  $(date -Is) ====="
CSP_MODEL_PRESET=qwen-2.5-7b-instruct \
python -m csp_div.analyze_assistant_axis \
    --csp-dir qwen \
    --out results/qwen/axis.png

# Llama ────────────────────────────────────────────────────────────────
echo "===== LLAMA $(date -Is) ====="
CSP_MODEL_PRESET=llama-3.1-8b-instruct \
python -m csp_div.analyze_assistant_axis \
    --csp-dir llama \
    --out results/llama/axis.png

echo "ALL_DONE $(date -Is)"
