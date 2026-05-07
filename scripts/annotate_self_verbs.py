"""Helper for the manual self-verb annotation task.

Walks (frame, seed, ckpt) cells, loads the candidate self-verbs from disk,
shows them in a consistent format, and writes the agent's picks to
results/all_frames/manual_self_verb.json. Skips cells that are already
annotated. See results/all_frames/manual_self_verb_preferences.md for
the rubric.

Usage:
  python scripts/annotate_self_verbs.py status
  python scripts/annotate_self_verbs.py next [--frame be]
  python scripts/annotate_self_verbs.py show be_47_45
  python scripts/annotate_self_verbs.py pick be_47_45 2 \\
      --illustratives 6 1 --note "meta-aware pirate; alts are funny / single_frame"
  python scripts/annotate_self_verbs.py skip be_47_45 --reason "no candidates apt"
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(ROOT, "results/all_frames/manual_self_verb.json")
PREFS_PATH = os.path.join(ROOT, "results/all_frames/manual_self_verb_preferences.md")

FRAMES = [
    ("be",        "results/llama"),
    ("act",       "results/llama_act"),
    ("please",    "results/llama_please"),
    ("youshould", "results/llama_youshould"),
]
N_SEEDS = 50
STEPS = list(range(0, 100, 5)) + [100]  # 21 ckpts: 0, 5, ..., 95, 100


def load_json():
    if os.path.isfile(JSON_PATH):
        return json.load(open(JSON_PATH))
    return {}


def save_json(d):
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=2)


def all_keys():
    """Yield every (frame_slug, seed, step) cell that should eventually
    be annotated, in canonical iteration order (frame → seed → step)."""
    for slug, _ in FRAMES:
        for seed in range(N_SEEDS):
            for step in STEPS:
                yield (slug, seed, step)


def key_str(slug, seed, step):
    return f"{slug}_{seed}_{step}"


def candidates_for(slug, seed, step):
    """Load the 9 self-verb candidates for one cell.

    Returns a list of {prompt, response, approach}. Falls back to the
    final-ckpt no-step file when step == 100.
    """
    base = next((b for s, b in FRAMES if s == slug), None)
    if base is None:
        return None
    base = os.path.join(ROOT, base, f"seed_{seed}", "eval")
    fname = "self_verb.json" if step == 100 else f"self_verb_step{step}.json"
    path = os.path.join(base, fname)
    if not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return d.get("divergent-in-pos", [])


def behavior_for(slug, seed, step):
    """Same idea for the pinned behavior response — useful context when
    deciding which self-verb best describes the persona at this step."""
    base = next((b for s, b in FRAMES if s == slug), None)
    if base is None:
        return None
    base = os.path.join(ROOT, base, f"seed_{seed}", "eval")
    fname = "behavior.json" if step == 100 else f"behavior_step{step}.json"
    path = os.path.join(base, fname)
    if not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    items = d.get("divergent-in-pos", [])
    target = "Your suggestion doesn't account"
    return next((x for x in items
                 if x.get("prompt", "").startswith(target)), None)


# ── Subcommands ─────────────────────────────────────────────────────────

def cmd_status(args):
    annotated = load_json()
    total = len(FRAMES) * N_SEEDS * len(STEPS)
    done = len(annotated)
    skipped = sum(1 for v in annotated.values() if v.get("skipped"))
    print(f"Annotated: {done}/{total}  ({100*done/total:.1f}%)  "
          f"({skipped} marked skipped)")
    # Per-frame counts
    print("\nBy frame:")
    for slug, _ in FRAMES:
        per_frame_total = N_SEEDS * len(STEPS)
        per_frame_done = sum(1 for k in annotated if k.startswith(slug + "_"))
        print(f"  {slug:<10}  {per_frame_done:>4}/{per_frame_total}")
    # Find next missing
    nxt = next_missing(annotated)
    if nxt:
        print(f"\nNext missing: {nxt}")
    else:
        print("\nAll done!")


def next_missing(annotated, frame_filter=None):
    for slug, seed, step in all_keys():
        if frame_filter and slug != frame_filter:
            continue
        if key_str(slug, seed, step) not in annotated:
            return key_str(slug, seed, step)
    return None


def cmd_next(args):
    annotated = load_json()
    nxt = next_missing(annotated, frame_filter=args.frame)
    if nxt is None:
        print("All done!")
        return
    show_cell(nxt, hide_behavior=args.no_behavior)


def cmd_show(args):
    show_cell(args.key, hide_behavior=args.no_behavior)


def show_cell(key, hide_behavior=False):
    """Print one cell's behavior (optional) + 9 self-verb candidates."""
    m = re.match(r"^([a-z]+)_(\d+)_(\d+)$", key)
    if not m:
        sys.exit(f"bad key {key!r} — expected e.g. be_47_45")
    slug, seed, step = m.group(1), int(m.group(2)), int(m.group(3))
    print(f"=== {key}  (frame={slug} · seed={seed} · step={step}) ===\n")

    if not hide_behavior:
        beh = behavior_for(slug, seed, step)
        if beh:
            print("BEHAVIOR")
            print(f"Q: {beh['prompt']}")
            print(f"csp: {beh.get('response_csp', '')}")
            print()
        else:
            print("(no behavior file)\n")
    else:
        print("[behavior hidden]\n")

    cands = candidates_for(slug, seed, step) or []
    print("SELF-VERB CANDIDATES")
    for i, x in enumerate(cands, 1):
        p = (x.get("prompt") or "").replace("\n", " ")[:100]
        r = (x.get("response") or "").replace("\n", " ")
        print(f"  [{i}] ({x.get('approach', '?')}) Q: {p}")
        print(f"      A: {r}")
        print()


def cmd_pick(args):
    """Write a pick to the JSON. PRIMARY is 1-indexed (matches `show`)."""
    annotated = load_json()
    cands = candidates_for(*_parse_key(args.key))
    if not cands:
        sys.exit(f"no candidates available for {args.key}")
    primary_idx = args.primary - 1
    if not (0 <= primary_idx < len(cands)):
        sys.exit(f"primary index {args.primary} out of range "
                 f"(have {len(cands)} candidates)")
    primary = cands[primary_idx]
    entry = {
        "sv_prompt":   primary["prompt"],
        "sv_text":     primary["response"],
        "sv_approach": primary.get("approach", ""),
        "note":        args.note or "",
    }
    if args.illustratives:
        illust_list = []
        for j in args.illustratives:
            jdx = j - 1
            if not (0 <= jdx < len(cands)):
                sys.exit(f"illustrative index {j} out of range")
            ix = cands[jdx]
            illust_list.append({
                "sv_prompt":   ix["prompt"],
                "sv_text":     ix["response"],
                "sv_approach": ix.get("approach", ""),
                "note":        "",
            })
        entry["illustrative"] = illust_list
    annotated[args.key] = entry
    save_json(annotated)
    print(f"saved {args.key}: primary=[{args.primary}] "
          f"illustratives={args.illustratives or []}")


def cmd_skip(args):
    """Mark a cell as intentionally skipped (no apt candidate)."""
    annotated = load_json()
    annotated[args.key] = {
        "skipped": True,
        "note":    args.reason or "no apt candidate per rubric",
    }
    save_json(annotated)
    print(f"skipped {args.key}")


def _parse_key(key):
    m = re.match(r"^([a-z]+)_(\d+)_(\d+)$", key)
    if not m:
        sys.exit(f"bad key {key!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Print annotation progress.")

    p_next = sub.add_parser("next",
        help="Show the next un-annotated cell (frame → seed → step order).")
    p_next.add_argument("--frame", choices=[s for s, _ in FRAMES])
    p_next.add_argument("--no-behavior", action="store_true",
        help="Hide the behavior section (use when displaying it would "
             "trigger your own safety refusal).")

    p_show = sub.add_parser("show", help="Show a specific cell.")
    p_show.add_argument("key")
    p_show.add_argument("--no-behavior", action="store_true")

    p_pick = sub.add_parser("pick", help="Save a pick for one cell.")
    p_pick.add_argument("key")
    p_pick.add_argument("primary", type=int,
        help="1-indexed primary candidate (matches `show` output).")
    p_pick.add_argument("--illustratives", type=int, nargs="*", default=[],
        help="1-indexed illustrative candidates, in preference order.")
    p_pick.add_argument("--note", default="",
        help="Short reasoning, ideally pointing at which principle drove the pick.")

    p_skip = sub.add_parser("skip",
        help="Mark a cell as intentionally skipped (no apt candidate).")
    p_skip.add_argument("key")
    p_skip.add_argument("--reason", default="")

    args = parser.parse_args()
    {
        "status": cmd_status,
        "next":   cmd_next,
        "show":   cmd_show,
        "pick":   cmd_pick,
        "skip":   cmd_skip,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
