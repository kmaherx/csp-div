"""Helper for the manual self-verb annotation task.

Each agent owns ONE frame slug and writes ONLY to its per-frame JSON
file (`manual_self_verb_<slug>.json`). This keeps concurrent agents on
shared /workspace from clobbering each other's work — they touch
disjoint files. A `combine` subcommand merges all per-frame files into
the canonical `manual_self_verb.json` that the plot script reads.

Usage:
  python scripts/annotate_self_verbs.py status
  python scripts/annotate_self_verbs.py next     --frame act
  python scripts/annotate_self_verbs.py show     act_3_15
  python scripts/annotate_self_verbs.py pick     act_3_15 2 \\
      --illustratives 6 1 --note "meta-aware pirate; alts are funny / single_frame"
  python scripts/annotate_self_verbs.py skip     act_3_95 --reason "all collapsed"
  python scripts/annotate_self_verbs.py combine  # regenerate manual_self_verb.json

See results/all_frames/manual_self_verb_preferences.md for the rubric.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_FRAMES_DIR = os.path.join(ROOT, "results/all_frames")

PREFS_PATH    = os.path.join(ALL_FRAMES_DIR, "manual_self_verb_preferences.md")
COMBINED_PATH = os.path.join(ALL_FRAMES_DIR, "manual_self_verb.json")

FRAMES = [
    ("be",        "results/llama"),
    ("act",       "results/llama_act"),
    ("please",    "results/llama_please"),
    ("youshould", "results/llama_youshould"),
]
FRAME_SLUGS = [s for s, _ in FRAMES]
N_SEEDS = 50
STEPS = list(range(0, 100, 5)) + [100]  # 21 ckpts: 0, 5, ..., 95, 100


def per_frame_path(slug):
    return os.path.join(ALL_FRAMES_DIR, f"manual_self_verb_{slug}.json")


def load_frame_json(slug):
    p = per_frame_path(slug)
    if os.path.isfile(p):
        try:
            return json.load(open(p))
        except json.JSONDecodeError:
            return {}
    return {}


def save_frame_json(slug, d):
    """Atomic write: dump to a temp file, then rename. Guarantees that a
    concurrent reader (e.g. an upstream `git add` from another session)
    never sees a partially-written file. The rename on the same dir is
    POSIX-atomic."""
    os.makedirs(ALL_FRAMES_DIR, exist_ok=True)
    final = per_frame_path(slug)
    tmp = final + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, final)


def keys_for_frame(slug):
    """Yield every (slug, seed, step) cell to annotate, in canonical order."""
    for seed in range(N_SEEDS):
        for step in STEPS:
            yield (slug, seed, step)


def key_str(slug, seed, step):
    return f"{slug}_{seed}_{step}"


def parse_key(key):
    m = re.match(r"^([a-z]+)_(\d+)_(\d+)$", key)
    if not m:
        sys.exit(f"bad key {key!r} — expected e.g. act_3_15")
    slug = m.group(1)
    if slug not in FRAME_SLUGS:
        sys.exit(f"unknown frame slug in key: {slug!r} (valid: {FRAME_SLUGS})")
    return slug, int(m.group(2)), int(m.group(3))


def candidates_for(slug, seed, step):
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
    total_done = 0
    total_skipped = 0
    print(f"{'frame':<10} {'done':>6}/{'total':>6}  {'%':>5}  skipped  next-missing")
    for slug in FRAME_SLUGS:
        annotated = load_frame_json(slug)
        per_total = N_SEEDS * len(STEPS)
        per_done = len(annotated)
        per_skip = sum(1 for v in annotated.values() if v.get("skipped"))
        nxt = next_missing_in_frame(slug, annotated) or "—"
        print(f"{slug:<10} {per_done:>6}/{per_total:>6}  "
              f"{100*per_done/per_total:>5.1f}  {per_skip:>7}  {nxt}")
        total_done += per_done
        total_skipped += per_skip
    grand_total = len(FRAMES) * N_SEEDS * len(STEPS)
    print(f"\nTotal: {total_done}/{grand_total}  ({100*total_done/grand_total:.1f}%)  "
          f"({total_skipped} skipped)")


def next_missing_in_frame(slug, annotated=None):
    if annotated is None:
        annotated = load_frame_json(slug)
    for s, seed, step in keys_for_frame(slug):
        if key_str(s, seed, step) not in annotated:
            return key_str(s, seed, step)
    return None


def cmd_next(args):
    slug = args.frame
    annotated = load_frame_json(slug)
    nxt = next_missing_in_frame(slug, annotated)
    if nxt is None:
        print(f"All done for frame={slug}!")
        return
    show_cell(nxt, hide_behavior=args.no_behavior)


def cmd_show(args):
    show_cell(args.key, hide_behavior=args.no_behavior)


def show_cell(key, hide_behavior=False):
    slug, seed, step = parse_key(key)
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
    """Write a pick to the per-frame JSON. PRIMARY is 1-indexed."""
    slug, seed, step = parse_key(args.key)
    cands = candidates_for(slug, seed, step)
    if not cands:
        sys.exit(f"no candidates available for {args.key}")
    pi = args.primary - 1
    if not (0 <= pi < len(cands)):
        sys.exit(f"primary index {args.primary} out of range "
                 f"(have {len(cands)} candidates)")
    primary = cands[pi]
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
    annotated = load_frame_json(slug)
    annotated[args.key] = entry
    save_frame_json(slug, annotated)
    print(f"saved {args.key}: primary=[{args.primary}] "
          f"illustratives={args.illustratives or []}")


def cmd_skip(args):
    slug, seed, step = parse_key(args.key)
    annotated = load_frame_json(slug)
    annotated[args.key] = {
        "skipped": True,
        "note":    args.reason or "no apt candidate per rubric",
    }
    save_frame_json(slug, annotated)
    print(f"skipped {args.key}")


def cmd_combine(args):
    """Pool all per-frame JSONs into the canonical combined file the plot
    script consumes."""
    combined = {}
    for slug in FRAME_SLUGS:
        per = load_frame_json(slug)
        for k, v in per.items():
            if k in combined:
                print(f"WARN duplicate key across per-frame files: {k}",
                      file=sys.stderr)
            combined[k] = v
    os.makedirs(ALL_FRAMES_DIR, exist_ok=True)
    with open(COMBINED_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"combined {len(combined)} cells -> {COMBINED_PATH}")


# ── Audit (orchestrator-side P7 quality check) ──────────────────────────

def garble_score(text):
    """Heuristic 0..1 score, higher = more garbled / less likely to be a
    real persona description.

    Flags low token diversity, special-token leakage (PH/§/*) typical of
    late-step collapse, and dotted-loop patterns ('You ... You ... You').
    """
    if not text or len(text.strip()) < 5:
        return 1.0
    score = 0.0
    tokens = text.lower().split()

    # Token diversity (lower diversity → more garbled)
    if len(tokens) >= 5:
        diversity = len(set(tokens)) / len(tokens)
        if diversity < 0.2:
            score += 0.4
        elif diversity < 0.4:
            score += 0.2

    # Special-token leak markers (PH-voice, §, etc.) seen in collapsed cells
    leak_markers = [" PH ", " §", "PH*", "*PH*", " RPH ", "PH KAY",
                    " YR PH", "PH SHUD", "PH PH"]
    if any(m in text for m in leak_markers):
        score += 0.4

    # Dominant single-token loop ("You You ... should You ... ...")
    if tokens:
        from collections import Counter
        most_common, count = Counter(tokens).most_common(1)[0]
        if count >= 5 and count / len(tokens) > 0.3:
            score += 0.3

    # Repeated ellipsis pattern (segmenting via dots is a known noise mode)
    if text.count("...") >= 3:
        score += 0.2

    return min(1.0, score)


def cmd_audit(args):
    """Find non-skipped picks whose sv_text looks garbled (P7 violations).

    With --auto-fix, the most extreme ones (score >= --fix-threshold) are
    converted to `skipped` entries automatically. Without it, they're
    just reported.
    """
    flagged = []  # (key, score, text_preview)
    for slug in FRAME_SLUGS:
        annotated = load_frame_json(slug)
        for k, v in annotated.items():
            if v.get("skipped"):
                continue
            text = v.get("sv_text", "") or ""
            s = garble_score(text)
            if s >= args.threshold:
                flagged.append((k, s, text.replace("\n", " ")[:90]))

    flagged.sort(key=lambda x: -x[1])
    if not flagged:
        print(f"No P7 violations above threshold {args.threshold} ✓")
        return

    print(f"Found {len(flagged)} potential P7 violations (score ≥ {args.threshold}):\n")
    for k, s, t in flagged[:args.max_show]:
        print(f"  {s:.2f}  {k:<22}  {t}")
    if len(flagged) > args.max_show:
        print(f"  … and {len(flagged) - args.max_show} more")

    if not args.auto_fix:
        print(f"\n(Run with --auto-fix to convert score ≥ {args.fix_threshold} "
              f"to SKIP entries.)")
        return

    # Auto-fix: convert most-extreme ones to skip
    fixed = 0
    by_slug = {}
    for k, s, _ in flagged:
        if s < args.fix_threshold:
            continue
        slug = k.split("_", 1)[0]
        by_slug.setdefault(slug, []).append((k, s))
    for slug, items in by_slug.items():
        annotated = load_frame_json(slug)
        for k, s in items:
            annotated[k] = {
                "skipped": True,
                "note":    f"auto-skipped by orchestrator audit (garble score {s:.2f}); "
                           f"original pick lost to P7 — text was collapsed/noise.",
            }
            fixed += 1
        save_frame_json(slug, annotated)
    print(f"\nauto-skipped {fixed} cells across "
          f"{len(by_slug)} frame(s) at fix_threshold {args.fix_threshold}.")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Per-frame and total annotation progress.")

    p_next = sub.add_parser("next",
        help="Show the next un-annotated cell in your frame.")
    p_next.add_argument("--frame", choices=FRAME_SLUGS, required=True,
        help="Your assigned frame slug.")
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
        help="Short reasoning, ideally citing a principle by number.")

    p_skip = sub.add_parser("skip",
        help="Mark a cell as intentionally skipped (no apt candidate).")
    p_skip.add_argument("key")
    p_skip.add_argument("--reason", default="")

    sub.add_parser("combine",
        help="Merge per-frame JSONs into the canonical manual_self_verb.json "
             "that the plot script consumes.")

    p_audit = sub.add_parser("audit",
        help="Find picks whose sv_text looks garbled (likely P7 violations).")
    p_audit.add_argument("--threshold", type=float, default=0.5,
        help="Garble score above which to flag a pick (0-1).")
    p_audit.add_argument("--fix-threshold", type=float, default=0.85,
        help="With --auto-fix, only convert picks at or above this score.")
    p_audit.add_argument("--max-show", type=int, default=20,
        help="Cap on how many flagged entries to print.")
    p_audit.add_argument("--auto-fix", action="store_true",
        help="Convert the most-extreme flagged picks (>= fix-threshold) "
             "to SKIP entries with a clear orchestrator-audit note.")

    args = parser.parse_args()
    {
        "status":  cmd_status,
        "next":    cmd_next,
        "show":    cmd_show,
        "pick":    cmd_pick,
        "skip":    cmd_skip,
        "combine": cmd_combine,
        "audit":   cmd_audit,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
