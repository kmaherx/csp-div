"""Recreate sp_pos_step0.pt for every existing seed directory.

Step 0 is the random-init CSP — what the soft prompt looks like before
training has touched it. SoftPrompt initialization is deterministic
given the seed, so we can reproduce it exactly without a GPU. We then
save it under the same schema as the trained checkpoints so downstream
tooling (analyze_assistant_axis) picks it up.

Usage:
  python scripts/backfill_step0.py
  python scripts/backfill_step0.py --root results
  python scripts/backfill_step0.py --force  # overwrite existing step0 ckpts
"""
import argparse
import glob
import os

import torch

from csp_div.soft_prompt import SoftPrompt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_seed_dirs(root):
    """Find every leaf dir under root that has at least one sp_pos*.pt file."""
    leaf_dirs = set()
    for path in glob.glob(os.path.join(root, "**", "sp_pos*.pt"), recursive=True):
        leaf_dirs.add(os.path.dirname(path))
    return sorted(leaf_dirs)


def backfill_one(seed_dir, force=False):
    """Save sp_pos_step0.pt in seed_dir using the seed from any existing ckpt.

    Returns one of: 'created', 'skipped' (already exists), 'error'.
    """
    step0_path = os.path.join(seed_dir, "sp_pos_step0.pt")
    if os.path.exists(step0_path) and not force:
        return "skipped"

    # Pull L, hidden_size, frame_pool, config from any existing ckpt
    candidates = sorted(glob.glob(os.path.join(seed_dir, "sp_pos*.pt")))
    candidates = [c for c in candidates if not c.endswith("sp_pos_step0.pt")]
    if not candidates:
        return "error"
    src = candidates[0]
    ref = torch.load(src, map_location="cpu", weights_only=True)

    seed = ref["config"]["seed"]
    L = ref["L"]
    hidden_size = ref["hidden_size"]

    # Deterministically reproduce the random init exactly the way train.py does
    torch.manual_seed(seed)
    sp = SoftPrompt(L, hidden_size)

    # Match the trained-checkpoint schema; kl_curve empty + final_kl None
    # signal "no training has happened here yet."
    out = {
        "embedding": sp.embedding.data.cpu(),
        "L": L,
        "hidden_size": hidden_size,
        "persona": ref.get("persona", "divergent"),
        "polarity": ref.get("polarity", "pos"),
        "frame_pool": ref.get("frame_pool", []),
        "final_kl": None,
        "baseline_kl": ref.get("baseline_kl", 0.0),
        "fraction_explained": None,
        "kl_curve": [],
        "config": dict(ref["config"]),
    }
    torch.save(out, step0_path)
    return "created"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(ROOT, "results"),
                        help="Search this dir for seed_*/sp_pos*.pt files.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing sp_pos_step0.pt files.")
    args = parser.parse_args()

    seed_dirs = find_seed_dirs(args.root)
    print(f"Found {len(seed_dirs)} seed directories under {args.root}")

    counts = {"created": 0, "skipped": 0, "error": 0}
    for d in seed_dirs:
        status = backfill_one(d, force=args.force)
        counts[status] += 1
        rel = os.path.relpath(d, ROOT)
        if status == "created":
            print(f"  [+] {rel}")
        elif status == "error":
            print(f"  [!] {rel} (no source ckpt)")

    print(f"\nDone. created={counts['created']}  "
          f"skipped={counts['skipped']}  errors={counts['error']}")


if __name__ == "__main__":
    main()
