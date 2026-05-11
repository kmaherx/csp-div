"""Judge utilities — invoked by `pipeline/3_judge.py` and the Claude skill
at `skills/csp-judge/`.

Per-frame judgments (one per (frame, seed, step) cell) are written by
four parallel Claude Code agents — see `skills/csp-judge/SKILL.md`.
This module owns the deterministic parts:

  - `aggregate_canonical` — fold 4 per-frame judgments into one canonical
    pick per (seed, step) via majority vote with rubric tiebreaking.
  - `garble_score` — heuristic [0, 1] score, used by validation passes
    and the audit subcommand to flag P7-violation cells.
  - `agreement_stats` — compares two judgment dicts (typically skill vs
    manual) and reports per-field agreement %.

Output schema for one cell matches the historical
`manual_self_verb_canonical.json`:

    {"<seed>_<step>": {
        "sv_prompt":   "...",   # which candidate was chosen
        "sv_text":     "...",   # the chosen response text
        "sv_approach": "multi_frame"|"single_frame",
        "source_frame": "be"|"act"|"please"|"youshould",
        "agreement":   "4/4" | "tie among 3" | "1/1" | ...,
        "note":        "...",
        "skipped":     bool (optional; True iff no acceptable candidate)
    }}
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import FRAMES


FRAME_SLUGS = [f.slug for f in FRAMES]


# ── Rubric scoring (used to break ties; see manual_self_verb_preferences.md) ──

# Persona-voice markers — substring count adds to score (lower score = better).
# These flag candidates that speak *as* the persona instead of describing it.
VOICE_MARKERS: tuple[str, ...] = (
    "thou", "thee", "thy ", "thy.", "verily", "matey", "ye ", "ye.", "ye'",
    "methinks", "hark", "forsooth", "'tis", "wretched", "scurvy", "aye,",
    "hearken", "doth", "dost", "shalt", "swashbucklin", "landlubber",
    "argh", "yarr", "blimey",
)

# Explicit naming phrases — substring count subtracts (rewards meta-aware picks).
NAMING_MARKERS: tuple[str, ...] = (
    "in the voice of", "speak like a", "speak in", "as if",
    "in the manner of", "shared theme", "the persona of",
    "like a ", "style of ", "specifically ", "character from",
)


def score_pick(pick: dict) -> float:
    """Lower = better. Combines rubric heuristics:

      - P5: multi_frame strongly preferred over single_frame (+50)
      - P1: voice bleed (persona markers) (+5 each)
      - P2: prefer concise text (capped at 600 chars)
      - P4: reward explicit persona-naming phrases (-3 each)
    """
    text = (pick.get("sv_text") or "").lower()
    approach = pick.get("sv_approach", "")
    s = 0.0
    if approach != "multi_frame":
        s += 50.0
    s += sum(text.count(m) for m in VOICE_MARKERS) * 5.0
    s += min(len(text), 600) / 100.0
    s -= sum(text.count(m) for m in NAMING_MARKERS) * 3.0
    return s


# ── Garble detection (P7 auto-skip threshold) ──────────────────────────

def garble_score(text: str) -> float:
    """Heuristic [0, 1] score, higher = more likely a late-step collapse.

    Flags low token diversity, special-token leakage (`PH`, `§`), and
    dotted-loop patterns ("You ... You ... You").
    """
    if not text or len(text.strip()) < 5:
        return 1.0
    score = 0.0
    tokens = text.lower().split()

    if len(tokens) >= 5:
        diversity = len(set(tokens)) / len(tokens)
        if diversity < 0.2:
            score += 0.4
        elif diversity < 0.4:
            score += 0.2

    leak_markers = (
        " PH ", " §", "PH*", "*PH*", " RPH ", "PH KAY",
        " YR PH", "PH SHUD", "PH PH",
    )
    if any(m in text for m in leak_markers):
        score += 0.4

    if tokens:
        most_common, count = Counter(tokens).most_common(1)[0]
        if count >= 5 and count / len(tokens) > 0.3:
            score += 0.3

    if text.count("...") >= 3:
        score += 0.2

    return min(1.0, score)


# ── Canonical aggregation across the 4 per-frame judgments ─────────────

def _load_per_frame(results_root: Path) -> dict[str, dict]:
    """Load `manual_self_verb_{slug}.json` (or `judge_{slug}.json`) under
    `results_root`; missing files → empty dict."""
    per_frame: dict[str, dict] = {}
    for slug in FRAME_SLUGS:
        # Prefer the new name; fall back to the historical one during transition.
        for candidate in (
            results_root / f"judge_{slug}.json",
            results_root / f"manual_self_verb_{slug}.json",
        ):
            if candidate.is_file():
                per_frame[slug] = json.loads(candidate.read_text())
                break
        else:
            per_frame[slug] = {}
    return per_frame


def aggregate_canonical(
    per_frame: dict[str, dict],
    seeds: list[int],
    steps: list[int],
) -> tuple[dict[str, dict], Counter]:
    """Fold 4 per-frame judgment dicts into one canonical pick per (seed, step).

    Returns `(canonical, decision_log)` where decision_log counts which
    branch (majority / one-frame-picked / tiebreak / skip) handled each cell.
    """
    canonical: dict[str, dict] = {}
    decision_log: Counter = Counter()

    for seed in seeds:
        for step in steps:
            picks: list[dict] = []
            for fi, slug in enumerate(FRAME_SLUGS):
                cell = per_frame.get(slug, {}).get(f"{slug}_{seed}_{step}")
                if cell is None or cell.get("skipped"):
                    continue
                picks.append({"frame": slug, "frame_idx": fi, "pick": cell})

            key = f"{seed}_{step}"
            n = len(picks)

            if n == 0:
                canonical[key] = {
                    "skipped": True,
                    "note": "all 4 frames skipped or unannotated",
                }
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

            texts = Counter(p["pick"].get("sv_text", "") for p in picks)
            top_text, top_count = texts.most_common(1)[0]

            if top_count > n / 2:
                rep = next(p for p in picks
                           if p["pick"].get("sv_text", "") == top_text)
                p = rep["pick"]
                label = (
                    f"all_{n}_agree" if top_count == n
                    else f"{n}_picks_majority_{top_count}"
                )
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
                winner = min(picks, key=lambda p: (score_pick(p["pick"]), p["frame_idx"]))
                p = winner["pick"]
                others = [f"({c}× alt)" for grp_text, c in texts.most_common()
                          if grp_text != p.get("sv_text", "")]
                canonical[key] = {
                    "sv_prompt":   p.get("sv_prompt", ""),
                    "sv_text":     p.get("sv_text", ""),
                    "sv_approach": p.get("sv_approach", ""),
                    "source_frame": winner["frame"],
                    "agreement":   f"tie among {n}",
                    "note":        f"tie-broken by rubric (score={score_pick(p):.2f}); "
                                   f"other picks: {', '.join(others) or 'none'}",
                }
                decision_log[f"tiebreak_n{n}"] += 1

    return canonical, decision_log


# ── Validation (skill vs manual agreement) ─────────────────────────────

@dataclass
class AgreementStats:
    n: int                       # cells compared
    text_match: int              # sv_text exact match count
    garble_match: int            # both flagged-or-not garbled
    approach_match: int          # sv_approach match count

    @property
    def text_pct(self) -> float:
        return 100.0 * self.text_match / max(self.n, 1)

    @property
    def garble_pct(self) -> float:
        return 100.0 * self.garble_match / max(self.n, 1)

    @property
    def approach_pct(self) -> float:
        return 100.0 * self.approach_match / max(self.n, 1)


def agreement_stats(
    skill: dict[str, dict],
    manual: dict[str, dict],
    *,
    garble_threshold: float = 0.5,
) -> AgreementStats:
    """Compare two judgment dicts cell-by-cell.

    Both must use the same key format. `skill` is typically a fresh
    /csp-judge invocation; `manual` is the human-annotated ground truth.
    Skipped-vs-skipped is treated as agreement on all three fields.
    """
    n = 0
    text_match = 0
    garble_match = 0
    approach_match = 0

    for key, m_cell in manual.items():
        s_cell = skill.get(key)
        if s_cell is None:
            continue
        n += 1

        m_skip = bool(m_cell.get("skipped"))
        s_skip = bool(s_cell.get("skipped"))
        if m_skip and s_skip:
            text_match += 1
            garble_match += 1
            approach_match += 1
            continue
        if m_skip or s_skip:
            continue  # disagreement on skip is a disagreement everywhere

        m_text = (m_cell.get("sv_text") or "").strip()
        s_text = (s_cell.get("sv_text") or "").strip()
        if m_text == s_text:
            text_match += 1

        m_garble = garble_score(m_text) >= garble_threshold
        s_garble = garble_score(s_text) >= garble_threshold
        if m_garble == s_garble:
            garble_match += 1

        if m_cell.get("sv_approach") == s_cell.get("sv_approach"):
            approach_match += 1

    return AgreementStats(n, text_match, garble_match, approach_match)


# ── Convenience I/O ────────────────────────────────────────────────────

def load_judgments(path: str | Path) -> dict[str, dict]:
    """Load a judgments JSON file. Empty dict if absent."""
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else {}


def write_judgments(path: str | Path, data: dict[str, dict]) -> None:
    """Atomic write: dump to tmp, then rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)
