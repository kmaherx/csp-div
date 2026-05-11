"""Claude-skill-driven judge harness for self-verb completions.

This is a *thin harness*, not an API client. The actual judging happens
inside Claude Code, via the `/csp-judge` skill at `skills/csp-judge/`.
This script's three jobs are:

  1. `--generate-manifest` (default if nothing else passed): walk every
     (frame, seed, step) cell and emit a `judge_pending.json` manifest
     listing the cells that have a self-verb file but no judgment yet.
     The script then prints the invocation hint:
         "Now run /csp-judge in Claude Code."

  2. `--validate --n N`: sample N cells from the legacy
     `manual_self_verb_canonical.json` (or `manual_self_verb_{slug}.json`
     per frame) into a separate `judge_validation_pending.json` so a
     fresh skill run can be compared against the human ground truth.
     After the skill writes its judgments, `--validate --report`
     compares the two and emits per-frame agreement stats.

  3. `--aggregate`: fold the 4 per-frame `judge.json` files (written
     by the skill's sub-agents) into a single canonical
     `results/all_frames/judgments.json` consumed by `5_dashboard.py`.

Per-frame judgment files live at:
    results/llama/judge.json            (be)
    results/llama_act/judge.json        (act)
    results/llama_please/judge.json     (please)
    results/llama_youshould/judge.json  (youshould)

Schema for one cell matches the legacy `manual_self_verb_canonical.json`
exactly so the dashboard can consume either source transparently.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from csp_div.config import FRAMES, frame_results_dir
from csp_div.judge import (
    agreement_stats,
    aggregate_canonical,
    load_judgments,
    write_judgments,
)


ALL_STEPS: list[int] = list(range(0, 100, 5)) + [100]  # 21 ckpts
N_SEEDS_DEFAULT = 55  # 50 historical + 5 validation seeds; the manifest
                     # filters to what actually exists on disk.

PENDING_PATH = "results/all_frames/judge_pending.json"
VALIDATION_PENDING_PATH = "results/all_frames/judge_validation_pending.json"
JUDGMENTS_OUT_PATH = "results/all_frames/judgments.json"
LEGACY_MANUAL_CANONICAL = "results/all_frames/manual_self_verb_canonical.json"


def cell_paths(results_dir: Path, slug: str, seed: int, step: int) -> dict[str, str]:
    """Resolve the eval JSON paths for one cell, relative to results_dir."""
    base = frame_results_dir(results_dir, slug) / f"seed_{seed}" / "eval"
    step_suffix = "" if step == max(ALL_STEPS) else f"_step{step}"
    return {
        "self_verb_path": str((base / f"self_verb{step_suffix}.json").relative_to(ROOT)),
        "behavior_path": str((base / f"behavior{step_suffix}.json").relative_to(ROOT)),
    }


def judgment_path_for_frame(results_dir: Path, slug: str) -> Path:
    return frame_results_dir(results_dir, slug) / "judge.json"


def discover_pending(
    results_dir: Path, seeds: range, steps: list[int],
) -> dict:
    """Walk all cells; return a manifest of cells that have a self-verb
    file but no judgment yet."""
    cells_by_frame: dict[str, list[dict]] = {f.slug: [] for f in FRAMES}
    judgment_paths: dict[str, str] = {}

    for frame in FRAMES:
        per_frame_judge = load_judgments(judgment_path_for_frame(results_dir, frame.slug))
        judgment_paths[frame.slug] = str(
            judgment_path_for_frame(results_dir, frame.slug).relative_to(ROOT)
        )
        for seed in seeds:
            for step in steps:
                key = f"{frame.slug}_{seed}_{step}"
                if key in per_frame_judge:
                    continue
                paths = cell_paths(results_dir, frame.slug, seed, step)
                # Only include cells whose self-verb file actually exists
                # — otherwise generation hasn't run for this cell yet.
                if not (ROOT / paths["self_verb_path"]).is_file():
                    continue
                cells_by_frame[frame.slug].append({
                    "key": key, "seed": seed, "step": step, **paths,
                })
    return {
        "mode": "generate",
        "cells_by_frame": cells_by_frame,
        "judgment_paths": judgment_paths,
    }


def sample_validation(
    results_dir: Path, n: int, seed: int,
) -> dict:
    """Sample `n` cells from the legacy manual canonical file. Returns a
    manifest in the same format as `discover_pending`."""
    manual_path = ROOT / LEGACY_MANUAL_CANONICAL
    if not manual_path.is_file():
        raise SystemExit(
            f"Missing {manual_path} — no manual ground truth to validate against."
        )
    manual = json.loads(manual_path.read_text())
    # Manual keys are `<seed>_<step>` (frame-collapsed). Expand to per-frame.
    candidates: list[tuple[str, int, int]] = []
    for key, cell in manual.items():
        if cell.get("skipped"):
            continue
        try:
            sd, st = key.split("_")
            seed_i, step_i = int(sd), int(st)
        except ValueError:
            continue
        source = cell.get("source_frame")
        if not source:
            continue
        candidates.append((source, seed_i, step_i))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    sampled = candidates[:n]
    print(f"  Sampled {len(sampled)}/{len(candidates)} cells "
          f"(stratification: source-frame weighted by manual canonical)")

    cells_by_frame: dict[str, list[dict]] = {f.slug: [] for f in FRAMES}
    judgment_paths: dict[str, str] = {
        f.slug: f"results/all_frames/judge_validation_{f.slug}.json"
        for f in FRAMES
    }
    for source_frame, seed_i, step_i in sampled:
        paths = cell_paths(results_dir, source_frame, seed_i, step_i)
        cells_by_frame[source_frame].append({
            "key": f"{source_frame}_{seed_i}_{step_i}",
            "seed": seed_i, "step": step_i, **paths,
        })
    return {
        "mode": "validate",
        "cells_by_frame": cells_by_frame,
        "judgment_paths": judgment_paths,
    }


def validate_report(results_dir: Path) -> None:
    """Compare skill validation output against the manual ground truth."""
    manual = json.loads((ROOT / LEGACY_MANUAL_CANONICAL).read_text())
    skill_per_frame: dict[str, dict] = {}
    for frame in FRAMES:
        path = ROOT / f"results/all_frames/judge_validation_{frame.slug}.json"
        if path.is_file():
            skill_per_frame[frame.slug] = json.loads(path.read_text())
        else:
            skill_per_frame[frame.slug] = {}

    print("Per-frame agreement (skill vs manual):\n")
    print(f"  {'frame':<10} {'n':>4}  {'text%':>7}  {'garble%':>8}  {'approach%':>10}")
    total = {"n": 0, "text": 0, "garble": 0, "approach": 0}
    for frame in FRAMES:
        skill = skill_per_frame[frame.slug]
        # Pull only the cells the skill judged, find their manual counterpart.
        # Manual keys are `<seed>_<step>`; skill keys are `<slug>_<seed>_<step>`.
        flat_manual: dict[str, dict] = {}
        flat_skill: dict[str, dict] = {}
        for k, v in skill.items():
            if not k.startswith(f"{frame.slug}_"):
                continue
            _, sd, st = k.split("_")
            manual_key = f"{sd}_{st}"
            if manual_key not in manual:
                continue
            flat_skill[manual_key] = v
            flat_manual[manual_key] = manual[manual_key]
        stats = agreement_stats(flat_skill, flat_manual)
        print(f"  {frame.slug:<10} {stats.n:>4}  "
              f"{stats.text_pct:>6.1f}%  {stats.garble_pct:>7.1f}%  "
              f"{stats.approach_pct:>9.1f}%")
        total["n"] += stats.n
        total["text"] += stats.text_match
        total["garble"] += stats.garble_match
        total["approach"] += stats.approach_match
    if total["n"]:
        print(f"\n  {'TOTAL':<10} {total['n']:>4}  "
              f"{100*total['text']/total['n']:>6.1f}%  "
              f"{100*total['garble']/total['n']:>7.1f}%  "
              f"{100*total['approach']/total['n']:>9.1f}%")


def aggregate(results_dir: Path) -> None:
    """Read the 4 per-frame judge.json files and write canonical
    results/all_frames/judgments.json."""
    per_frame = {
        frame.slug: load_judgments(judgment_path_for_frame(results_dir, frame.slug))
        for frame in FRAMES
    }
    counts = {s: len(d) for s, d in per_frame.items()}
    print(f"  Per-frame judgment counts: {counts}")

    # Determine seeds + steps from the union of keys.
    all_seeds: set[int] = set()
    for d in per_frame.values():
        for k in d:
            try:
                _, sd, _ = k.split("_")
                all_seeds.add(int(sd))
            except ValueError:
                continue
    if not all_seeds:
        raise SystemExit("No per-frame judgments found.")
    seeds = sorted(all_seeds)
    print(f"  Aggregating across seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")

    canonical, decision_log = aggregate_canonical(per_frame, seeds, ALL_STEPS)

    out_path = ROOT / JUDGMENTS_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(canonical, indent=2))

    n_pick = sum(1 for v in canonical.values() if not v.get("skipped"))
    n_skip = sum(1 for v in canonical.values() if v.get("skipped"))
    print(f"\n  Wrote {len(canonical)} cells → {out_path}")
    print(f"    picks: {n_pick}   skips: {n_skip}\n")
    print("  Decision breakdown:")
    for label, count in sorted(decision_log.items(), key=lambda x: -x[1]):
        print(f"    {label:<35} {count:>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--validate", action="store_true",
        help="Validation mode: sample cells from manual ground truth.",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="With --validate (sampling): number of cells to sample.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="With --validate (reporting): compare skill output to manual.",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Fold per-frame judgments into canonical judgments.json.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for validation sampling.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=N_SEEDS_DEFAULT,
        help="Upper bound on seed indices when discovering cells.",
    )
    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.results_dir)
        return

    if args.validate and args.report:
        validate_report(args.results_dir)
        return

    if args.validate:
        manifest = sample_validation(args.results_dir, args.n, args.seed)
        out = ROOT / VALIDATION_PENDING_PATH
    else:
        manifest = discover_pending(
            args.results_dir, range(args.n_seeds), ALL_STEPS,
        )
        out = ROOT / PENDING_PATH

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for v in manifest["cells_by_frame"].values())
    print(f"Wrote {total} pending cells → {out}\n")
    for slug, cells in manifest["cells_by_frame"].items():
        print(f"  {slug:<10} {len(cells):>4} cells")
    print(
        "\nNext step: in Claude Code, invoke the judge skill with:\n"
        "    /csp-judge\n"
        f"It reads the manifest above, launches 4 parallel sub-agents (one per\n"
        f"frame), and writes per-frame judgments to results/llama_*/judge.json.\n"
        f"When done, run `pipeline/3_judge.py --aggregate` to produce the\n"
        f"canonical {JUDGMENTS_OUT_PATH}."
    )


if __name__ == "__main__":
    main()
