"""csp-div — max-divergence contextualized soft prompts."""
import os as _os

# Resolve the project root (two levels up from this package: src/csp_div/ → src/ → root).
# Used as the anchor for `data/` and the default `results/` directory so the
# package works the same regardless of cwd.
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
