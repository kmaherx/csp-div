"""Build canonical per-(seed, step) self-verb picks by aggregating across
the 4 per-frame agent JSONs.

Decision tree per (seed, step) cell:
  1. If all frames skipped → mark skipped.
  2. If only one frame picked → use it.
  3. If a strict majority of valid picks share the same text → use it.
  4. Otherwise (tie / all-unique) → apply rubric-based scoring:
       - prefer multi_frame > single_frame (P5)
       - penalize persona-voice markers (thou/matey/verily/...) (P1)
       - prefer shorter text (P2)
       - reward explicit persona-naming phrases ("in the voice of", "speak like a")

Source frame is recorded in each canonical entry so we can audit which
agent's pick won, and divergence reasons are noted in `note` for ties.

Output: results/all_frames/manual_self_verb_canonical.json
Keyed by `<seed>_<step>` (no frame prefix), with one canonical entry per cell.
"""
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_FRAMES_DIR = os.path.join(ROOT, "results/all_frames")
FRAMES = ["be", "act", "please", "youshould"]
N_SEEDS = 50
STEPS = list(range(0, 100, 5)) + [100]  # 21 ckpts

# Persona-voice markers — substring count adds to score (lower = better)
VOICE_MARKERS = [
    "thou", "thee", "thy ", "thy.", "verily", "matey", "ye ", "ye.", "ye'",
    "methinks", "hark", "forsooth", "'tis", "wretched", "scurvy", "aye,",
    "hearken", "doth", "dost", "shalt", "swashbucklin", "landlubber",
    "argh", "yarr", "blimey",
]
# Naming phrases — substring count subtracts (lower score = better pick)
NAMING_MARKERS = [
    "in the voice of", "speak like a", "speak in", "as if",
    "in the manner of", "shared theme", "the persona of",
    "like a ", "style of ", "specifically ", "character from",
]


def score_pick(pick):
    """Lower = better. Combines all the rubric heuristics."""
    text = (pick.get("sv_text") or "").lower()
    approach = pick.get("sv_approach", "")
    s = 0.0
    # P5: multi_frame strongly preferred
    if approach != "multi_frame":
        s += 50.0
    # P1: voice bleed (count of persona markers)
    bleed = sum(text.count(m) for m in VOICE_MARKERS)
    s += bleed * 5.0
    # P2: prefer concise (cap to avoid penalizing reasonably long descriptions)
    s += min(len(text), 600) / 100.0
    # P4 / P1 reward: explicit persona-naming
    naming = sum(text.count(m) for m in NAMING_MARKERS)
    s -= naming * 3.0
    return s


def best_pick(picks):
    """Score-min pick; ties broken by appearance order in `picks`."""
    return min(picks, key=lambda p: (score_pick(p["pick"]), p["frame_idx"]))


def main():
    data = {}
    for f in FRAMES:
        path = os.path.join(ALL_FRAMES_DIR, f"manual_self_verb_{f}.json")
        data[f] = json.load(open(path)) if os.path.isfile(path) else {}

    canonical = {}
    decision_log = Counter()

    for seed in range(N_SEEDS):
        for step in STEPS:
            picks = []
            for fi, f in enumerate(FRAMES):
                v = data[f].get(f"{f}_{seed}_{step}")
                if v is None:
                    continue
                if v.get("skipped"):
                    continue
                picks.append({"frame": f, "frame_idx": fi, "pick": v})

            key = f"{seed}_{step}"
            n = len(picks)
            if n == 0:
                # Either all skipped or all missing; canonical = skip.
                canonical[key] = {"skipped": True,
                                  "note": "all 4 frames skipped or unannotated"}
                decision_log["all_skip_or_missing"] += 1
                continue

            if n == 1:
                p = picks[0]["pick"]
                canonical[key] = {
                    "sv_prompt":   p.get("sv_prompt", ""),
                    "sv_text":     p.get("sv_text", ""),
                    "sv_approach": p.get("sv_approach", ""),
                    "source_frame": picks[0]["frame"],
                    "agreement":   "1/1",
                    "note":        "only one frame picked",
                }
                decision_log["only_one_frame_picked"] += 1
                continue

            # Group by exact text
            texts = Counter(p["pick"].get("sv_text", "") for p in picks)
            top_text, top_count = texts.most_common(1)[0]

            if top_count > n / 2:
                # Strict majority (or unanimous)
                rep = next(p for p in picks
                           if p["pick"].get("sv_text", "") == top_text)
                p = rep["pick"]
                if top_count == n:
                    label = f"all_{n}_agree"
                else:
                    label = f"{n}_picks_majority_{top_count}"
                canonical[key] = {
                    "sv_prompt":   p.get("sv_prompt", ""),
                    "sv_text":     p.get("sv_text", ""),
                    "sv_approach": p.get("sv_approach", ""),
                    "source_frame": rep["frame"],
                    "agreement":   f"{top_count}/{n}",
                    "note":        f"majority pick ({top_count}/{n} frames agreed)",
                }
                decision_log[label] += 1
            else:
                # Tie or all unique → rubric-based selection
                winner = best_pick(picks)
                p = winner["pick"]
                # Diagnostic note: which other frames had different picks
                other_summary = []
                for grp_text, c in texts.most_common():
                    if grp_text == p.get("sv_text", ""):
                        continue
                    other_summary.append(f"({c}× alt)")
                canonical[key] = {
                    "sv_prompt":   p.get("sv_prompt", ""),
                    "sv_text":     p.get("sv_text", ""),
                    "sv_approach": p.get("sv_approach", ""),
                    "source_frame": winner["frame"],
                    "agreement":   f"tie among {n}",
                    "note":        f"tie-broken by rubric (score={score_pick(p):.2f}); "
                                   f"other picks: {', '.join(other_summary) or 'none'}",
                }
                decision_log[f"tiebreak_n{n}"] += 1

    # Write canonical JSON
    out_path = os.path.join(ALL_FRAMES_DIR, "manual_self_verb_canonical.json")
    with open(out_path, "w") as f:
        json.dump(canonical, f, indent=2)

    # Print summary
    n_pick = sum(1 for v in canonical.values() if not v.get("skipped"))
    n_skip = sum(1 for v in canonical.values() if v.get("skipped"))
    print(f"\nWrote {len(canonical)} canonical entries → {out_path}")
    print(f"  picks: {n_pick}   skips: {n_skip}\n")
    print("=== Decision breakdown ===")
    for k, v in sorted(decision_log.items(), key=lambda x: -x[1]):
        print(f"  {k:<35} {v:>5}")
    print()
    # Source-frame distribution among non-skipped picks
    src = Counter(v.get("source_frame", "?") for v in canonical.values()
                  if not v.get("skipped"))
    print("=== Source-frame distribution (which agent's pick won) ===")
    for k, v in sorted(src.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v:>5}")


if __name__ == "__main__":
    main()
