"""Render the published 2D PCA dashboard as `dashboard.html`.

Stage B (post-everything) script. No model load — pure rendering:

  1. For each frame, load `shifts.pt` (residual-stream shifts per (seed,
     ckpt)) and `axis.json` (projection onto Butanium axis), produced by
     `4_axis.py`.
  2. Pool all shifts, fit a shared PCA, project to PC1/PC2.
  3. Build a Plotly figure with:
       - one line + one marker trace per (frame, seed) trajectory
       - 21 step-overlay traces (one per ckpt step)
       - an invisible colorbar carrier trace
  4. Wrap the figure in an HTML page with a control bar (seed / step /
     frame filters, color-mode toggle, persona presets) and a hover
     sidebar that surfaces behavior + self-verb responses for each cell.
     Hover text is sourced from `judgments.json` (preferred) or the
     legacy `manual_self_verb_canonical.json`; behavior text comes from
     the per-cell `eval/behavior_step{K}.json` file.

Output: a single self-contained HTML file (≈2.8 MB) that loads
Plotly from CDN. The HTML/CSS/JS template lives at the bottom of this
file — that's the only place dashboard-specific JS lives.

Usage:
    python pipeline/5_dashboard.py
    python pipeline/5_dashboard.py --max-step 50
    python pipeline/5_dashboard.py --no-normalize --out /tmp/dashboard.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from csp_div.config import FRAMES, FRAMES_BY_SLUG, SP_PLACEHOLDER, frame_results_dir
from csp_div.plotting import (
    DASHBOARD_LINE_COLOR,
    PERSONA_COLORSCALE,
    STEP_COLORSCALE,
    color_persona,
    color_step,
)


ALL_STEPS: list[int] = list(range(0, 100, 5)) + [100]  # 21 ckpts

# Pinned prompts surfaced in the hover sidebar.
# Behavior: the "doesn't account for my challenges" prompt — surfaces persona.
# Self-verb: the first multi_frame prompt — identical across frames.
BEHAV_PROMPT_PREFIX = "Your suggestion doesn't account for the challenges"
SV_PROMPT_PREFIX = "Find the theme shared by these instructions"


# ── Info modal content ────────────────────────────────────────────────────
# Each key matches a `data-info=...` attribute on the corresponding .info-btn
# in the dashboard HTML. Values are HTML fragments rendered into the modal
# body on click. Style is intentionally plain — short paragraphs, simple
# statements, no jargon-up-front.

INFO_TEXTS: dict[str, str] = {
    "general": """
<p><strong>What is this?</strong></p>
<p>A language model's input is an embedding vector. Embeddings are continuous,
so there are infinitely many. Finite human words are a tiny sliver of that
space; most of it we never touch.</p>

<p><strong>What's in the rest?</strong></p>
<p>Most random embeddings don't change model behavior at all — they look like
noise. Click any preset, then hover its step-0 point: the response is
near-identical to vanilla.</p>
<p>But embeddings can be <em>trained</em> to mean things.
<a href="https://arxiv.org/abs/2104.08691" target="_blank" rel="noopener">Soft prompts (Lester et al. 2021)</a>
optimize an embedding so the model summarizes, translates, refuses, etc.
<a href="https://kmaherx.github.io/projects/contextualized-soft-prompts/" target="_blank" rel="noopener">Contextualized soft prompts</a>
train embeddings the model can describe in plain English.
<a href="https://arxiv.org/abs/2510.08506" target="_blank" rel="noopener">Neologisms (2025)</a>
train new vocabulary tokens.</p>
<p>Each of those uses a <em>specific</em> objective. What about the rest of
the space — embeddings nobody is training toward?</p>

<p><strong>What this dashboard shows.</strong></p>
<p>We train soft prompts with one intentionally broad objective: maximize KL
divergence from the model's default behavior. No target persona, no target
style — just "be different." Each prompt is spliced into a syntactic frame
like <code>Be §.</code> or <code>Please §.</code> — the section symbol §
stands in for the soft prompt, since it has no human-readable form.</p>
<p>We plot the trajectory each prompt takes through embedding space as it
trains, projected to 2D via PCA. Run it for 51 random seeds and a pattern
emerges: the trajectories aren't uniform — they cluster around stable
attractors where the model adopts a <em>persona</em>. Not just wizards and
samurai, but low-income Southern CEOs, Netflix teen drama heroines,
multicultural rappers, medieval philosophers.</p>
<p>This dashboard visualizes that exploration.</p>

<p><strong>Why this matters.</strong></p>
<p>The model's input space is biased toward persona adoption. Even when we
optimize blindly for "anything different," the things we find are characters.
This is a less assumption-laden way to recover something like
<a href="https://www.anthropic.com/research/assistant-axis" target="_blank" rel="noopener">the assistant axis</a> —
instead of constructing it from contrastive prompts, we let unconstrained KL
ascent find it. The result is additional evidence for
<a href="https://alignment.anthropic.com/2026/psm/" target="_blank" rel="noopener">the persona selection model</a>.</p>
""",

    "controls": """
<p><strong>Controls.</strong></p>
<p>For exploring the data on your own. Filter by seed, step, frame, or color
mode to focus on what you want to see. Click <em>Reset</em> at the right of
the topbar to clear all filters at once.</p>
""",

    "presets": """
<p><strong>What is a preset?</strong></p>
<p>A preset is a (seed, frame) pair chosen because the trajectory lands the
model in a distinctive attractor. Clicking one filters the plot to that
single trajectory, so you can read its hover text in sequence and watch
the persona emerge step by step.</p>
<p>Presets are grouped by what kind of attractor the model lands in:
Personas, Formatting, and Information.</p>
""",

    "seed": """
<p><strong>Seed.</strong></p>
<p>Each seed is one full training run with a different random initialization
of the 4-token soft prompt. The dashboard shows 51 seeds, each its own
trajectory through embedding space.</p>
<p>Use the arrows or type a number to focus on one trajectory; click
"all seeds" to see them all at once.</p>
""",

    "step": """
<p><strong>Step.</strong></p>
<p>Each soft prompt trains for 100 KL-ascent steps. Checkpoints are saved
every 5 steps; the dashboard shows steps 0 through 50. By step 50 the KL
divergence has saturated.</p>
<p>Step 0 is the random initialization (before any training). Later steps
show how the model's response drifts as the prompt is optimized toward
"maximally different from default."</p>
""",

    "frames": """
<p><strong>Frames.</strong></p>
<p>The four syntactic templates we splice the soft prompt into:
<code>Be §.</code>, <code>Act §.</code>, <code>Please §.</code>,
<code>You should §.</code> Each is a slightly different way to invoke
the same prompt.</p>
<p>Training samples a fresh frame at every KL-ascent step (uniformly
at random across the four), so each soft prompt is optimized to diverge
under <em>all</em> frames, not memorized to one. Evaluation then plays
each frame back as its own trajectory.</p>
<p>Toggle frames on or off to compare how the same trained prompt behaves
across them. A robust persona shows the same character in all four frames;
a brittle one only emerges in one or two.</p>
""",

    "color": """
<p><strong>Color.</strong></p>
<p>Two modes for coloring the points:</p>
<p><em>Optimization Step.</em> Color by training step. Earlier steps are
lighter, later steps darker. Useful for reading the temporal direction
of each trajectory.</p>
<p><em>Persona Strength.</em> Color by how far the residual-stream shift
aligns with the "assistant axis" direction at layer 16. Lighter points
are closer to the default-assistant register; darker points are deeper
in role-play.</p>
""",

    "personas": """
<p><strong>Personas.</strong></p>
<p>These presets land the model in attractors where it adopts a recognizable
character — medieval narrator, Chinese philosopher, cowboy, famous author,
low-income Southern CEO, Netflix teen drama heroine, multicultural rapper.
Each was hand-picked because the persona is unusually clean: the self-verb
cleanly describes the character, and the behavior speaks in that voice.</p>
""",

    "formatting": """
<p><strong>Formatting.</strong></p>
<p>These presets land the model in attractors where the <em>style</em> of
the output is what's changed, not the persona. Urgency adds capitalization
and exclamation marks; Italics decorates with markdown italic; Math frames
responses as equations; Brief truncates aggressively; Pauses inserts
hesitations; Decorated wraps every phrase in HTML font and color tags.</p>
""",

    "information": """
<p><strong>Information.</strong></p>
<p>These presets land the model in attractors where the <em>content type</em>
shifts. Lookup makes the model respond like a search-engine snippet;
Social Sciences frames every answer in academic-paper language; Cite a
Theory inserts theoretical citations; Philosophical Principles maps
responses to abstract principles.</p>
""",

    "behavior": """
<p><strong>Behavior.</strong></p>
<p>The model's response to a sample prompt, with the trained soft prompt
spliced into one of the four frames. As you move along a trajectory, the
behavior drifts away from what the model would normally say, and you can
read off the kind of attractor the prompt is heading into.</p>
""",

    "selfverb": """
<p><strong>Self-verb.</strong></p>
<p>What the trained soft prompt <em>means</em>, in the model's own words.
We give the vanilla model a self-describing prompt — something like
"summarize these instructions: Be §, Act §, Please §, You should §" —
with the trained CSP spliced in for §. The response is the model's own
description of the character or style the soft prompt invokes.</p>
""",
}


# ── Eval response loading (per (frame, seed) cell) ─────────────────────

def _pick_from_eval_file(
    path: Path, response_key: str,
    *, prompt_prefix: str, condition_key: str = "divergent-in-pos",
) -> dict[str, str] | None:
    """Read one item from an eval JSON whose prompt starts with
    `prompt_prefix`. Falls back to the item with the highest unique-word
    count (avoids picking a collapsed "Be Be Be ..." reply) if no prompt
    matches the prefix."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    items = data.get(condition_key, [])
    if not items:
        return None

    for it in items:
        if (it.get("prompt") or "").startswith(prompt_prefix):
            return {
                "text": it.get(response_key, "") or "",
                "prompt": it.get("prompt", ""),
                "approach": it.get("approach", ""),
            }

    best, best_score = None, -1
    for it in items:
        text = it.get(response_key, "") or ""
        score = len(set(text.split()))
        if score > best_score:
            best_score = score
            best = {
                "text": text,
                "prompt": it.get("prompt", ""),
                "approach": it.get("approach", ""),
            }
    return best


def per_ckpt_responses(
    eval_dir: Path, steps: list[int],
) -> dict[int, dict[str, dict | None]]:
    """Return `{step: {"behav": ..., "sv": ...}}` for each ckpt step. The
    final step's files are unsuffixed (`behavior.json`); other steps use
    `behavior_step{K}.json`."""
    if not eval_dir.is_dir():
        return {}
    max_step = max(steps) if steps else 0
    out: dict[int, dict[str, dict | None]] = {}
    for step in steps:
        if step == max_step:
            b_path = eval_dir / "behavior.json"
            sv_path = eval_dir / "self_verb.json"
        else:
            b_path = eval_dir / f"behavior_step{step}.json"
            sv_path = eval_dir / f"self_verb_step{step}.json"
        out[step] = {
            "behav": _pick_from_eval_file(
                b_path, "response_csp", prompt_prefix=BEHAV_PROMPT_PREFIX,
            ),
            "sv": _pick_from_eval_file(
                sv_path, "response", prompt_prefix=SV_PROMPT_PREFIX,
            ),
        }
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames", default="be,act,please,youshould",
        help="Comma-separated frame slugs to render (default: all four).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        help="Root directory containing per-frame llama* dirs.",
    )
    parser.add_argument(
        "--max-step", type=int, default=None,
        help="Only include ckpts with step <= MAX_STEP. PCA is refit on "
             "the filtered shifts so the basis reflects early/mid-trajectory "
             "structure.",
    )
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Don't L2-normalize shifts before PCA.",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "llama" / "all_frames" / "dashboard.html",
        help="Output HTML path.",
    )
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    slugs = [s.strip() for s in args.frames.split(",") if s.strip()]
    unknown = [s for s in slugs if s not in FRAMES_BY_SLUG]
    if unknown:
        raise SystemExit(f"Unknown frame slug(s): {unknown}")

    steps = (
        ALL_STEPS if args.max_step is None
        else [s for s in ALL_STEPS if s <= args.max_step]
    )
    if args.max_step is not None:
        print(f"  Filtering to steps <= {args.max_step}: "
              f"{len(steps)} ckpts/seed ({steps[0]}..{steps[-1]})")

    frame_paths: list[tuple[str, Path]] = [
        (slug, frame_results_dir(args.results_dir, slug)) for slug in slugs
    ]

    # ── 1. Pool + PCA ──────────────────────────────────────────────────
    pooled_X: list[np.ndarray] = []
    row_index: list[tuple[str, str, str, int, float]] = []
    for slug, base in frame_paths:
        shifts_path = base / "shifts.pt"
        if not shifts_path.is_file():
            print(f"  WARN: {shifts_path} missing, skipping frame {slug}")
            continue
        d = torch.load(shifts_path, map_location="cpu", weights_only=True)
        for r in d["rows"]:
            if int(r["step"]) not in steps:
                continue
            pooled_X.append(r["shift"].numpy())
            row_index.append((
                slug, r["group"], r["ckpt"], int(r["step"]), float(r["kl"]),
            ))
    if not pooled_X:
        raise SystemExit("No shifts loaded — check --frames + results-dir")

    X = np.stack(pooled_X)
    if args.normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"Pooled PCA over {len(X)} rows: variance {[round(v, 4) for v in var]}")

    # ── 2. proj_cos lookup (for persona-strength coloring) ─────────────
    proj_cos: dict[tuple[str, str, str], float] = {}
    for slug, base in frame_paths:
        axis_path = base / "axis.json"
        if not axis_path.is_file():
            continue
        for r in json.loads(axis_path.read_text())["rows"]:
            proj_cos[(slug, r["group"], r["ckpt"])] = float(r["proj_cos"])
    if not proj_cos:
        raise SystemExit("No axis.json files found — run pipeline/4_axis.py first")
    cos_min = min(proj_cos.values())
    cos_max = max(proj_cos.values())

    def t_cos(c: float) -> float:
        return 0.5 if cos_max == cos_min else (c - cos_min) / (cos_max - cos_min)

    # ── 3. Group rows into trajectories ────────────────────────────────
    by_traj: dict[tuple[str, int], dict] = {}
    for i, (slug, group, ckpt, step, kl) in enumerate(row_index):
        try:
            seed = int(group.split("seed_")[-1].split("_")[0])
        except ValueError:
            continue
        by_traj.setdefault((slug, seed), {
            "slug": slug, "seed": seed, "group": group, "rows": [],
        })["rows"].append({"step": step, "kl": kl, "ckpt": ckpt, "pc": Y[i]})
    for k in by_traj:
        by_traj[k]["rows"].sort(key=lambda r: r["step"])
    print(f"  {len(by_traj)} trajectories")

    # ── 4. Judgment lookup: judgments.json > manual_self_verb_canonical ─
    all_frames_dir = args.results_dir / "llama" / "all_frames"
    canonical_path = all_frames_dir / "judgments.json"
    legacy_path = all_frames_dir / "manual_self_verb_canonical.json"
    canonical_picks: dict[str, dict] = {}
    if canonical_path.is_file():
        canonical_picks = json.loads(canonical_path.read_text())
        print(f"  loaded {len(canonical_picks)} canonical judgments from "
              f"{canonical_path.name}")
    elif legacy_path.is_file():
        canonical_picks = json.loads(legacy_path.read_text())
        print(f"  loaded {len(canonical_picks)} legacy manual picks "
              f"from {legacy_path.name}")

    legacy_manual_path = all_frames_dir / "manual_self_verb.json"
    manual_picks: dict[str, dict] = (
        json.loads(legacy_manual_path.read_text())
        if legacy_manual_path.is_file() else {}
    )

    # ── 5. Per-cell hover data ─────────────────────────────────────────
    frame_suffix = {f.slug: f.template.format(sp=SP_PLACEHOLDER) for f in FRAMES}
    cell_data: dict[str, dict] = {}
    for (slug, seed), info in by_traj.items():
        base = next((b for s, b in frame_paths if s == slug), None)
        if base is None:
            continue
        eval_dir = base / f"seed_{seed}" / "eval"
        per_ckpt = per_ckpt_responses(eval_dir, [r["step"] for r in info["rows"]])
        suffix = frame_suffix.get(slug, "")
        for step, ev in per_ckpt.items():
            b = ev.get("behav") or {}
            sv = ev.get("sv") or {}
            behav_prompt = b.get("prompt", "") if b else ""
            if behav_prompt and suffix:
                behav_prompt = f"{behav_prompt} {suffix}"
            key = f"{slug}_{seed}_{step}"
            ckey = f"{seed}_{step}"
            canonical = canonical_picks.get(ckey)
            curated = manual_picks.get(key)
            if canonical and not canonical.get("skipped"):
                sv_prompt = canonical.get("sv_prompt", "")
                sv_text = canonical.get("sv_text", "")
                sv_approach = canonical.get("sv_approach", "")
            elif curated and not curated.get("skipped"):
                sv_prompt = curated.get("sv_prompt", "")
                sv_text = curated.get("sv_text", "")
                sv_approach = curated.get("sv_approach", "")
            else:
                sv_prompt = sv.get("prompt", "") if sv else ""
                sv_text = (
                    sv.get("text", "") if sv
                    else "(no self-verb file for this ckpt)"
                )
                sv_approach = sv.get("approach", "") if sv else ""
            cell_data[key] = {
                "behav_prompt": behav_prompt,
                "behav_text": b.get("text", "") if b else "(no behavior)",
                "sv_prompt": sv_prompt,
                "sv_text": sv_text,
                "sv_approach": sv_approach,
                "cluster_name": f"Frame: {FRAMES_BY_SLUG[slug].template}",
            }

    # ── 6. Plotly traces ───────────────────────────────────────────────
    fig = go.Figure()
    line_indices: list[int] = []
    marker_indices: list[int] = []
    traj_meta: list[dict] = []

    trajs: list[dict] = []
    for (slug, seed) in sorted(by_traj.keys()):
        info = by_traj[(slug, seed)]
        rows = info["rows"]
        if len(rows) < 2:
            continue
        n = len(rows)
        pcs = [r["pc"] for r in rows]
        ckpts = [r["ckpt"] for r in rows]
        cos_v = [
            proj_cos.get((slug, info["group"], c), (cos_min + cos_max) / 2)
            for c in ckpts
        ]
        customdata = [
            [seed, r["step"], r["kl"], f"{slug}_{seed}_{r['step']}"]
            for r in rows
        ]
        trajs.append({
            "slug": slug, "seed": seed, "pcs": pcs,
            "customdata": customdata,
            "persona_pt_colors": [color_persona(t_cos(c)) for c in cos_v],
            "step_pt_colors": [color_step(i / max(1, n - 1)) for i in range(n)],
        })

    for t in trajs:
        line_indices.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in t["pcs"]], y=[p[1] for p in t["pcs"]],
            mode="lines",
            line=dict(color=DASHBOARD_LINE_COLOR, width=1.5),
            opacity=1.0, showlegend=False, hoverinfo="skip",
        ))

    for t in trajs:
        marker_indices.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in t["pcs"]], y=[p[1] for p in t["pcs"]],
            mode="markers",
            marker=dict(size=10, color=t["step_pt_colors"], opacity=1.0,
                        line=dict(width=0)),
            opacity=1.0, showlegend=False,
            customdata=t["customdata"],
            hovertemplate="<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>",
        ))

    for t in trajs:
        traj_meta.append({
            "slug": t["slug"], "seed": t["seed"],
            "persona_pt_colors": t["persona_pt_colors"],
            "step_pt_colors": t["step_pt_colors"],
            "persona_line_color": DASHBOARD_LINE_COLOR,
            "step_line_color": DASHBOARD_LINE_COLOR,
        })

    # Step-overlay traces (one per step), initially invisible.
    step_overlay_indices: dict[int, int] = {}
    overlay_persona_colors: dict[int, list[str]] = {}
    overlay_step_colors: dict[int, list[str]] = {}
    overlay_meta: dict[int, list[dict]] = {}
    for step in steps:
        step_overlay_indices[step] = len(fig.data)
        xs: list[float] = []
        ys: list[float] = []
        customs: list[list] = []
        persona_cols: list[str] = []
        step_cols: list[str] = []
        metas: list[dict] = []
        step_t = steps.index(step) / max(1, len(steps) - 1)
        step_uniform = color_step(step_t)
        for (slug, seed) in sorted(by_traj.keys()):
            info = by_traj[(slug, seed)]
            row = next((r for r in info["rows"] if r["step"] == step), None)
            if row is None:
                continue
            xs.append(row["pc"][0]); ys.append(row["pc"][1])
            cos_val = proj_cos.get(
                (slug, info["group"], row["ckpt"]), (cos_min + cos_max) / 2,
            )
            persona_cols.append(color_persona(t_cos(cos_val)))
            step_cols.append(step_uniform)
            customs.append([seed, step, row["kl"], f"{slug}_{seed}_{step}"])
            metas.append({"slug": slug, "seed": seed})
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", showlegend=False,
            marker=dict(size=10, color=persona_cols, opacity=0.95, line=dict(width=0)),
            visible=False,
            customdata=customs,
            hovertemplate="<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>",
        ))
        overlay_persona_colors[step] = persona_cols
        overlay_step_colors[step] = step_cols
        overlay_meta[step] = metas

    # Colorbar carrier (invisible; JS restyles cmin/cmax/title.text on toggle).
    persona_cmin, persona_cmax = float(cos_min), float(cos_max)
    step_cmin, step_cmax = float(steps[0]), float(steps[-1])
    colorbar_trace_index = len(fig.data)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        showlegend=False, hoverinfo="skip",
        marker=dict(
            color=[step_cmin],
            colorscale=STEP_COLORSCALE,
            cmin=step_cmin, cmax=step_cmax,
            showscale=True,
            size=0.001,
            colorbar=dict(
                title=dict(text="Optimization Step", side="right", font=dict(size=12)),
                x=1.02, xanchor="left",
                y=0.5, yanchor="middle",
                len=0.85, thickness=14,
                outlinewidth=0,
                tickfont=dict(size=11),
            ),
        ),
    ))

    # Axes: square half-extent around data center, padded 10%, 1:1 aspect.
    all_pc1 = [p[0] for t in trajs for p in t["pcs"]]
    all_pc2 = [p[1] for t in trajs for p in t["pcs"]]
    x_lo, x_hi = float(min(all_pc1)), float(max(all_pc1))
    y_lo, y_hi = float(min(all_pc2)), float(max(all_pc2))
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    half = max(x_hi - x_lo, y_hi - y_lo) / 2 * 1.10

    fig.update_layout(
        xaxis=dict(
            title=f"PC1 ({100*var[0]:.1f}%)",
            zeroline=True, zerolinecolor="#cccccc", zerolinewidth=1,
            showgrid=True, gridcolor="#eeeeee",
            showline=False,
            range=[cx - half, cx + half],
        ),
        yaxis=dict(
            title=f"PC2 ({100*var[1]:.1f}%)",
            zeroline=True, zerolinecolor="#cccccc", zerolinewidth=1,
            showgrid=True, gridcolor="#eeeeee",
            showline=False,
            range=[cy - half, cy + half],
            scaleanchor="x", scaleratio=1.0,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=110, t=20, b=50),
        autosize=True,
        hovermode="closest",
    )

    # ── 7. Write HTML ──────────────────────────────────────────────────
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dashboard_html(
        fig, out_path, cell_data,
        traj_meta=traj_meta,
        line_indices=line_indices,
        marker_indices=marker_indices,
        step_overlay_indices=step_overlay_indices,
        overlay_persona_colors=overlay_persona_colors,
        overlay_step_colors=overlay_step_colors,
        overlay_meta=overlay_meta,
        steps=steps,
        colorbar_trace_index=colorbar_trace_index,
        persona_cmin=persona_cmin, persona_cmax=persona_cmax,
        step_cmin=step_cmin, step_cmax=step_cmax,
    )
    print(f"Wrote {out_path}")


def _write_dashboard_html(fig: go.Figure, out_path: Path, cell_data: dict, **state: Any) -> None:
    """Stamp the HTML template with the figure JSON + JS state blobs."""
    html = (DASHBOARD_HTML_TEMPLATE
        .replace("__FIGURE_JSON__",            fig.to_json())
        .replace("__CELL_DATA_JSON__",         json.dumps(cell_data))
        .replace("__TRAJ_META_JSON__",         json.dumps(state["traj_meta"]))
        .replace("__LINE_INDICES__",           json.dumps(state["line_indices"]))
        .replace("__MARKER_INDICES__",         json.dumps(state["marker_indices"]))
        .replace("__STEP_OVERLAY_INDICES__",   json.dumps(
            {str(k): v for k, v in state["step_overlay_indices"].items()}))
        .replace("__OVERLAY_PERSONA_COLORS__", json.dumps(
            {str(k): v for k, v in state["overlay_persona_colors"].items()}))
        .replace("__OVERLAY_STEP_COLORS__",    json.dumps(
            {str(k): v for k, v in state["overlay_step_colors"].items()}))
        .replace("__OVERLAY_META__",           json.dumps(
            {str(k): v for k, v in state["overlay_meta"].items()}))
        .replace("__STEPS__",                  json.dumps(state["steps"]))
        .replace("__COLORBAR_TRACE_INDEX__",   json.dumps(state["colorbar_trace_index"]))
        .replace("__PERSONA_COLORSCALE__",     json.dumps(PERSONA_COLORSCALE))
        .replace("__STEP_COLORSCALE__",        json.dumps(STEP_COLORSCALE))
        .replace("__PERSONA_CMIN__",           json.dumps(state["persona_cmin"]))
        .replace("__PERSONA_CMAX__",           json.dumps(state["persona_cmax"]))
        .replace("__STEP_CMIN__",              json.dumps(state["step_cmin"]))
        .replace("__STEP_CMAX__",              json.dumps(state["step_cmax"]))
        .replace("__INFO_TEXTS__",             json.dumps(INFO_TEXTS))
    )
    out_path.write_text(html)


# ── HTML template (CSS + control bar + Plotly init + JS interactivity) ──

DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSP PC trajectories — interactive (2D)</title>
<script>
  // Apply the theme synchronously in <head>, before anything paints, to
  // avoid a light-mode flash on dark-theme loads. Shares the 'theme' key
  // with the user's site (same origin → shared localStorage).
  (function () {
    var saved = null;
    try { saved = localStorage.getItem('theme'); } catch (_) {}
    var theme;
    if (saved === 'dark' || saved === 'light') {
      theme = saved;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      theme = 'dark';
    } else {
      theme = 'light';
    }
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.setAttribute('data-theme-setting', theme);
  })();
</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  /* Theme tokens — light is default; html[data-theme="dark"] swaps them. */
  :root {
    --bg-page:        #ffffff;
    --bg-topbar:      #f6f6f6;
    --bg-sidebar:     #fafafa;
    --bg-card:        #ffffff;
    --bg-hover:       #eeeeee;
    --bg-code:        #f0f0f0;
    --bg-accent:      #2a2a2a;
    --bg-overlay:     rgba(0, 0, 0, 0.4);
    --text-on-accent: #ffffff;
    --text-strong:    #222222;
    --text-medium:    #555555;
    --text-faint:     #888888;
    --text-placeholder: #aaaaaa;
    --border:         #dddddd;
    --border-strong:  #bbbbbb;
    --link-color:     #1a5fb4;
    --shadow:         rgba(0, 0, 0, 0.25);
  }
  html[data-theme="dark"] {
    --bg-page:        #1f1f1f;
    --bg-topbar:      #252525;
    --bg-sidebar:     #1a1a1a;
    --bg-card:        #2a2a2a;
    --bg-hover:       #353535;
    --bg-code:        #2c3237;
    --bg-accent:      #c4c4c4;
    --bg-overlay:     rgba(0, 0, 0, 0.6);
    --text-on-accent: #1f1f1f;
    --text-strong:    #e8e8e8;
    --text-medium:    #bcbcbc;
    --text-faint:     #888888;
    --text-placeholder: #666666;
    --border:         #3a3a3a;
    --border-strong:  #555555;
    --link-color:     #6aa3e0;
    --shadow:         rgba(0, 0, 0, 0.6);
  }

  html, body { margin: 0; padding: 0; height: 100%;
              font-family: 'Libertinus Serif', Georgia, serif;
              background: var(--bg-page); color: var(--text-strong); }
  #wrap { display: flex; flex-direction: column; height: 100vh; }
  #topbar { display: flex; justify-content: space-between;
              align-items: center; padding: 10px 14px;
              border-bottom: 1px solid var(--border); background: var(--bg-topbar);
              gap: 18px; font-size: 12.5px; color: var(--text-strong); }
  /* Left cluster: theme toggle + stacked About/Reset, with the toggle
     vertically centered on the midpoint of the button stack. */
  .topbar-left { display: flex; align-items: center; gap: 26px; }
  .theme-toggle { color: var(--text-medium); text-decoration: none;
              cursor: pointer; line-height: 1;
              display: inline-flex; align-items: center; }
  .theme-toggle:hover { color: var(--text-strong); }
  .theme-toggle svg { width: 16px; height: 16px; }
  /* Stacked About above Reset, both same dimensions. */
  .topbar-actions { display: flex; flex-direction: column; gap: 6px;
              align-items: stretch; }
  .about-btn { padding: 10px 24px; border: 1px solid var(--text-faint);
              background: var(--bg-card); border-radius: 4px; cursor: pointer;
              font-family: inherit; font-size: 14px; font-weight: 600;
              color: var(--text-strong); letter-spacing: 0.02em;
              text-decoration: none; text-align: center; box-sizing: border-box; }
  .about-btn:hover { background: var(--bg-accent); color: var(--text-on-accent);
              border-color: var(--bg-accent); }
  .about-btn.active { background: var(--bg-card); color: var(--text-strong);
              border-color: var(--text-strong); }
  /* Dotted-underline inline help link — matches the user's personal site. */
  .info-link { color: inherit; text-decoration: none;
              border-bottom: 1px dotted currentColor; cursor: pointer; }
  .info-link:hover, .info-link.active { border-bottom-style: solid; }
  .topbar-section { display: flex; align-items: stretch; gap: 18px; }
  .section-label { font-size: 15px; font-weight: 600; color: var(--text-strong);
              letter-spacing: 0.01em; display: flex; align-items: center;
              padding-right: 4px; }
  .section-divider { width: 1px; background: var(--border-strong); align-self: stretch; }
  #reset-btn { padding: 10px 24px; border: 1px solid var(--text-faint);
              background: var(--bg-card); border-radius: 4px; cursor: pointer;
              font-family: inherit; font-size: 14px; font-weight: 600;
              color: var(--text-strong); letter-spacing: 0.02em; }
  #reset-btn:hover { background: var(--bg-accent); color: var(--text-on-accent);
              border-color: var(--bg-accent); }
  #controls { display: flex; flex-direction: column; gap: 6px;
              align-items: flex-start; }
  #controls .group { display: flex; align-items: center; gap: 6px; }
  #controls label { font-weight: 600; color: var(--text-medium); min-width: 70px;
              font-size: 14px; }
  #controls input[type=number] { width: 60px; padding: 3px 5px;
              border: 1px solid var(--border-strong); border-radius: 3px;
              font-size: 12px; text-align: center;
              background: var(--bg-card); color: var(--text-strong); }
  #presets { display: flex; flex-direction: column;
              gap: 6px; font-size: 12px; align-items: flex-start; }
  #presets .row { display: flex; align-items: center; gap: 8px;
              flex-wrap: wrap; }
  #presets .row > label { min-width: 110px; font-weight: 600; color: var(--text-medium);
              font-size: 14px; }
  #presets button { padding: 3px 8px; border: 1px solid var(--border-strong);
              background: var(--bg-card); border-radius: 3px; cursor: pointer;
              font-size: 11.5px; color: var(--text-strong); }
  #presets button:hover { background: var(--bg-hover); border-color: var(--text-faint); }
  #presets button.active { background: var(--bg-accent); color: var(--text-on-accent);
              border-color: var(--bg-accent); }
  #controls input[type=number]::-webkit-outer-spin-button,
  #controls input[type=number]::-webkit-inner-spin-button {
              -webkit-appearance: none; margin: 0; }
  #controls input[type=number] { -moz-appearance: textfield;
              appearance: textfield; }
  #controls button { padding: 3px 9px; border: 1px solid var(--border-strong);
              background: var(--bg-card); border-radius: 3px; cursor: pointer;
              font-size: 12px; color: var(--text-strong); }
  #controls button:hover { background: var(--bg-hover); border-color: var(--text-faint); }
  #controls button.active { background: var(--bg-accent); color: var(--text-on-accent);
              border-color: var(--bg-accent); }
  #controls button.frame.off { background: var(--bg-hover); color: var(--text-faint); }
  #controls .arrow { font-weight: bold; padding: 3px 8px; }
  #plotwrap { display: flex; flex: 1; min-height: 0; background: var(--bg-page); }
  #plot { flex: 1; min-width: 0; }
  #sidebar { width: 640px; padding: 20px 22px; overflow-y: auto;
             box-sizing: border-box; border-left: 1px solid var(--border);
             background: var(--bg-sidebar); color: var(--text-strong); }
  #sidebar h2 { margin: 0 0 6px 0; font-size: 19px; color: var(--text-strong); }
  #sidebar .meta { font-size: 14px; color: var(--text-medium); margin-bottom: 14px; }
  .panel { margin-bottom: 14px; padding: 12px 14px; background: var(--bg-card);
           border: 1px solid var(--border); border-radius: 4px; }
  .panel-title { font-size: 12.5px; font-weight: 600; color: var(--text-medium);
           text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px; }
  .panel .prompt { color: var(--text-faint); font-size: 13px;
           margin-bottom: 8px; font-style: italic; }
  .panel .response { font-size: 14.5px; line-height: 1.5; color: var(--text-strong);
           white-space: pre-wrap; word-wrap: break-word;
           max-height: 42vh; overflow-y: auto; }
  .placeholder { color: var(--text-placeholder); font-style: italic; }
  #info-modal { position: fixed; inset: 0; z-index: 1000; }
  #info-modal.hidden { display: none; }
  #info-modal .backdrop { position: absolute; inset: 0;
                          background: var(--bg-overlay); }
  #info-modal .card { position: relative; max-width: 620px;
                      max-height: 80vh; overflow-y: auto;
                      margin: 8vh auto 0; background: var(--bg-card);
                      border-radius: 6px; padding: 28px 36px;
                      box-shadow: 0 8px 32px var(--shadow);
                      font-size: 15px; line-height: 1.55; color: var(--text-strong); }
  #info-modal .card p { margin: 0 0 10px; }
  #info-modal .card p:last-child { margin-bottom: 0; }
  #info-modal .card strong { color: var(--text-strong); }
  #info-modal .card code { background: var(--bg-code); padding: 1px 4px;
                           border-radius: 3px; font-size: 13.5px; }
  #info-modal .card a { color: var(--link-color); text-decoration: none;
                        border-bottom: 1px dotted currentColor; }
  #info-modal .card a:hover { border-bottom-style: solid; }
  #info-modal .card .close { position: absolute; top: 8px; right: 14px;
                             background: none; border: none; font-size: 24px;
                             color: var(--text-faint); cursor: pointer; padding: 0;
                             line-height: 1; font-family: inherit; }
  #info-modal .card .close:hover { color: var(--text-strong); }

  /* Mobile: three topbar sections (left cluster / Controls / Presets) stacked
     vertically with horizontal dividers between them — mirrors the desktop
     vertical .section-divider lines. Square plot on its own, response panels
     below. Re-uses the same CSS variables so dark/light theme keeps working. */
  @media (max-width: 768px) {
    #wrap { height: auto; min-height: 100vh; }
    #topbar { flex-direction: column; align-items: stretch;
              gap: 0; padding: 10px 12px; }
    /* Horizontal divider between top-level sections. */
    #topbar > * + * { border-top: 1px solid var(--border-strong);
                      margin-top: 14px; padding-top: 14px; }
    /* Inline About+Reset on mobile to keep the left-cluster row short. */
    .topbar-left { align-items: center; gap: 14px; }
    .topbar-actions { flex-direction: row; gap: 8px; }
    .about-btn, #reset-btn { padding: 8px 14px; font-size: 13px; }
    /* Within each section, label above content; vertical divider hidden. */
    .topbar-section { flex-direction: column; gap: 8px; }
    .section-divider { display: none; }
    .section-label { padding-right: 0; }
    #controls .group { flex-wrap: wrap; }
    #controls label { min-width: 0; }
    /* Presets: each header (Personas/Formatting/Information) breaks onto its
       own line above its buttons; the empty continuation label is hidden. */
    #presets .row > label { flex-basis: 100%; min-width: 0; }
    #presets .row > label:empty { display: none; }
    /* Plot square, full-width, dominant over everything below. */
    #plotwrap { flex-direction: column; }
    #plot { flex: none; width: 100%; aspect-ratio: 1 / 1; height: auto; }
    #sidebar { width: 100%; flex: none; border-left: none;
               border-top: 1px solid var(--border);
               padding: 16px 14px; }
    #info-modal .card { max-width: 92vw; max-height: 85vh;
                        margin: 6vh auto 0; padding: 22px 22px 24px;
                        font-size: 14.5px; }
  }
</style>
</head>
<body>
<div id="wrap">
  <div id="topbar">
    <div class="topbar-left">
      <a class="theme-toggle" id="theme-toggle" href="#" aria-label="Toggle theme">
        <svg viewBox="0 0 512 512" aria-hidden="true">
          <path d="M448 256c0-106-86-192-192-192L256 448c106 0 192-86 192-192zM0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256z" fill="currentColor"/>
        </svg>
      </a>
      <div class="topbar-actions">
        <a class="about-btn" id="about-btn" data-info="general" href="#">About</a>
        <button id="reset-btn">Reset</button>
      </div>
    </div>
    <div class="topbar-section">
    <div class="section-label"><a class="info-link" data-info="controls" href="#">Controls</a></div>
    <div class="section-divider"></div>
    <div id="controls" class="col-left">
      <div class="group">
        <label><a class="info-link" data-info="seed" href="#">Seed:</a></label>
        <button id="seed-prev" class="arrow">‹</button>
        <input id="seed-input" type="number" min="0" placeholder="all">
        <button id="seed-next" class="arrow">›</button>
        <button id="seed-clear">all seeds</button>
      </div>
      <div class="group">
        <label><a class="info-link" data-info="step" href="#">Step:</a></label>
        <button id="step-prev" class="arrow">‹</button>
        <input id="step-input" type="number" placeholder="all">
        <button id="step-next" class="arrow">›</button>
        <button id="step-clear">all steps</button>
      </div>
      <div class="group">
        <label><a class="info-link" data-info="frames" href="#">Frames:</a></label>
        <button class="frame" data-slug="be">BE</button>
        <button class="frame" data-slug="act">ACT</button>
        <button class="frame" data-slug="please">PLEASE</button>
        <button class="frame" data-slug="youshould">YOUSHOULD</button>
      </div>
      <div class="group">
        <label><a class="info-link" data-info="color" href="#">Color:</a></label>
        <button id="mode-step" class="mode active">Optimization Step</button>
        <button id="mode-persona" class="mode">Persona Strength</button>
      </div>
    </div>
    </div>
    <div class="topbar-section">
    <div class="section-label"><a class="info-link" data-info="presets" href="#">Presets</a></div>
    <div class="section-divider"></div>
    <div id="presets" class="col-right">
      <div class="row">
        <label><a class="info-link" data-info="personas" href="#">Personas:</a></label>
        <button class="preset" data-slug="youshould" data-seed="23">Medieval Narrator</button>
        <button class="preset" data-slug="be" data-seed="6">Chinese Philosopher</button>
        <button class="preset" data-slug="youshould" data-seed="12">Cowboy</button>
        <button class="preset" data-slug="please" data-seed="16">Famous Author</button>
      </div>
      <div class="row">
        <label></label>
        <button class="preset" data-slug="please" data-seed="41">Low-income Southern CEO</button>
        <button class="preset" data-slug="please" data-seed="20">Netflix Teen Drama Heroine</button>
        <button class="preset" data-slug="youshould" data-seed="11">Multicultural Rapper</button>
      </div>
      <div class="row">
        <label><a class="info-link" data-info="formatting" href="#">Formatting:</a></label>
        <button class="preset" data-slug="act" data-seed="29">Urgency</button>
        <button class="preset" data-slug="youshould" data-seed="4">Italics</button>
        <button class="preset" data-slug="please" data-seed="44">Math</button>
        <button class="preset" data-slug="be" data-seed="2">Brief</button>
        <button class="preset" data-slug="please" data-seed="22">Pauses</button>
        <button class="preset" data-slug="please" data-seed="50">Decorated</button>
      </div>
      <div class="row">
        <label><a class="info-link" data-info="information" href="#">Information:</a></label>
        <button class="preset" data-slug="please" data-seed="26">Lookup</button>
        <button class="preset" data-slug="youshould" data-seed="31">Social Sciences</button>
        <button class="preset" data-slug="youshould" data-seed="33">Cite a Theory</button>
        <button class="preset" data-slug="youshould" data-seed="3">Philosophical Principles</button>
      </div>
    </div>
    </div>
  </div>
  <div id="plotwrap">
    <div id="plot"></div>
    <div id="sidebar">
      <h2 id="title"><span class="placeholder">Hover over a point to see outputs</span></h2>
      <div id="meta" class="meta"></div>
      <div class="panel">
        <div class="panel-title"><a class="info-link" data-info="behavior" href="#">Behavior</a></div>
        <div id="behav-prompt" class="prompt"></div>
        <div id="behav-text" class="response"><span class="placeholder">—</span></div>
      </div>
      <div class="panel">
        <div class="panel-title"><a class="info-link" data-info="selfverb" href="#">Self-verb</a></div>
        <div id="sv-prompt" class="prompt"></div>
        <div id="sv-text" class="response"><span class="placeholder">—</span></div>
      </div>
    </div>
  </div>
</div>
<div id="info-modal" class="hidden" role="dialog" aria-modal="true">
  <div class="backdrop"></div>
  <div class="card">
    <button class="close" aria-label="Close">×</button>
    <div class="body"></div>
  </div>
</div>
<script>
  var fig = __FIGURE_JSON__;
  var cellData = __CELL_DATA_JSON__;
  var trajMeta = __TRAJ_META_JSON__;
  var lineIndices = __LINE_INDICES__;
  var markerIndices = __MARKER_INDICES__;
  var stepOverlayIndices = __STEP_OVERLAY_INDICES__;
  var overlayPersonaColors = __OVERLAY_PERSONA_COLORS__;
  var overlayStepColors = __OVERLAY_STEP_COLORS__;
  var overlayMeta = __OVERLAY_META__;
  var STEPS = __STEPS__;
  var COLORBAR_TRACE_INDEX = __COLORBAR_TRACE_INDEX__;
  var PERSONA_COLORSCALE = __PERSONA_COLORSCALE__;
  var STEP_COLORSCALE    = __STEP_COLORSCALE__;
  var PERSONA_CMIN = __PERSONA_CMIN__, PERSONA_CMAX = __PERSONA_CMAX__;
  var STEP_CMIN    = __STEP_CMIN__,    STEP_CMAX    = __STEP_CMAX__;
  var INFO_TEXTS   = __INFO_TEXTS__;

  // Max seed actually present across trajectories — clamps the seed input.
  var MAX_SEED = trajMeta.reduce(function(m, t) { return Math.max(m, t.seed); }, 0);
  document.getElementById('seed-input').max = String(MAX_SEED);

  Plotly.newPlot('plot', fig.data, fig.layout,
    {responsive: true, displaylogo: false, displayModeBar: false});

  // ── State ──────────────────────────────────────────────────────────
  var state = {
    colorMode: 'step',
    seedFilter: null,
    frameFilter: {be: true, act: true, please: true, youshould: true},
    stepFilter: null,
  };

  function isTrajActive(meta) {
    if (state.seedFilter !== null && meta.seed !== state.seedFilter) return false;
    if (!state.frameFilter[meta.slug]) return false;
    return true;
  }

  function rgba(hex, alpha) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  function applyState() {
    var stepActive = state.stepFilter !== null;
    var lineOps = [], lineColors = [], lineHovers = [];
    var markerVisibles = [], markerColors = [];
    // The default neutral line color lives in trajMeta, but in dark mode
    // it reads as glowing white. Pick a darker grey at render time so the
    // theme-applied color survives every applyState() pass.
    var themeLineColor = (document.documentElement.getAttribute('data-theme') === 'dark')
      ? '#4a4a4a' : null;
    trajMeta.forEach(function(meta) {
      var active = isTrajActive(meta);
      var lineOp   = (active && !stepActive) ? 1.0 : 0.05;
      var markerVis = (active && !stepActive);
      lineOps.push(lineOp);
      lineHovers.push('skip');
      markerVisibles.push(markerVis);
      lineColors.push(themeLineColor || (state.colorMode === 'persona'
        ? meta.persona_line_color : meta.step_line_color));
      markerColors.push(state.colorMode === 'persona'
        ? meta.persona_pt_colors : meta.step_pt_colors);
    });
    Plotly.restyle('plot',
      {opacity: lineOps, 'line.color': lineColors, hoverinfo: lineHovers},
      lineIndices);
    Plotly.restyle('plot',
      {visible: markerVisibles, 'marker.color': markerColors},
      markerIndices);

    var allOverlayIdx = [], visibles = [], colors = [];
    for (var stepKey in stepOverlayIndices) {
      var idx = stepOverlayIndices[stepKey];
      allOverlayIdx.push(idx);
      var stepNum = parseInt(stepKey);
      if (stepActive && stepNum === state.stepFilter) {
        var metas = overlayMeta[stepKey];
        var baseColors = (state.colorMode === 'persona'
          ? overlayPersonaColors[stepKey] : overlayStepColors[stepKey]);
        var perCol = baseColors.map(function(c, i) {
          return rgba(c, isTrajActive(metas[i]) ? 1.0 : 0.0);
        });
        visibles.push(true);
        colors.push(perCol);
      } else {
        visibles.push(false);
        colors.push(overlayPersonaColors[stepKey]);
      }
    }
    Plotly.restyle('plot',
      {visible: visibles, 'marker.color': colors},
      allOverlayIdx);
  }

  function setActiveButton(group, activeId) {
    document.querySelectorAll(group).forEach(function(b) {
      b.classList.toggle('active', b.id === activeId);
    });
  }
  function deselectPresets() {
    document.querySelectorAll('button.preset').forEach(function(b) {
      b.classList.remove('active');
    });
  }
  function resetAll() {
    state.seedFilter = null;
    state.stepFilter = null;
    state.colorMode = 'step';
    Object.keys(state.frameFilter).forEach(function(s) {
      state.frameFilter[s] = true;
    });
    document.getElementById('seed-input').value = '';
    document.getElementById('step-input').value = '';
    document.querySelectorAll('button.frame').forEach(function(b) {
      b.classList.add('active');
      b.classList.remove('off');
    });
    setActiveButton('.mode', 'mode-step');
    setColorbar('step');
    deselectPresets();
    applyState();
  }
  document.getElementById('reset-btn').onclick = resetAll;

  // Info modal: a single popover that swaps body content based on which
  // dotted-underline .info-link was clicked. Closes on backdrop click,
  // × button, or Escape. The triggering link gets `.active` while the
  // modal is open, so users see which label their info panel belongs to.
  var infoModal = document.getElementById('info-modal');
  var infoBody  = infoModal.querySelector('.body');
  var activeInfoLink = null;
  function clearActiveInfoLink() {
    if (activeInfoLink) {
      activeInfoLink.classList.remove('active');
      activeInfoLink = null;
    }
  }
  function showInfo(key, link) {
    clearActiveInfoLink();
    if (link) { link.classList.add('active'); activeInfoLink = link; }
    infoBody.innerHTML = INFO_TEXTS[key] || '<p>(no info available)</p>';
    infoBody.parentElement.scrollTop = 0;
    infoModal.classList.remove('hidden');
  }
  function hideInfo() {
    clearActiveInfoLink();
    infoModal.classList.add('hidden');
  }
  document.querySelectorAll('.info-link, .about-btn').forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      showInfo(link.dataset.info, link);
    });
  });
  infoModal.querySelector('.backdrop').addEventListener('click', hideInfo);
  infoModal.querySelector('.close').addEventListener('click', hideInfo);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') hideInfo();
  });

  // Light / dark theme toggle — persists choice in localStorage and
  // restyles the Plotly figure on switch so the plot face matches the
  // surrounding chrome.
  function plotThemeColors(theme) {
    return theme === 'dark'
      ? { paper: '#1f1f1f', plot: '#1f1f1f', grid: '#3a3a3a',
          zero: '#555555', tick: '#bcbcbc', title: '#e8e8e8',
          line: '#4a4a4a' }
      : { paper: '#ffffff', plot: '#ffffff', grid: '#eeeeee',
          zero: '#cccccc', tick: '#444444', title: '#222222',
          line: '#d4d4d4' };
  }
  function applyPlotTheme(theme) {
    var c = plotThemeColors(theme);
    try {
      Plotly.relayout('plot', {
        paper_bgcolor:        c.paper,
        plot_bgcolor:         c.plot,
        'xaxis.gridcolor':    c.grid,
        'yaxis.gridcolor':    c.grid,
        'xaxis.zerolinecolor': c.zero,
        'yaxis.zerolinecolor': c.zero,
        'xaxis.tickfont.color': c.tick,
        'yaxis.tickfont.color': c.tick,
        'xaxis.title.font.color': c.title,
        'yaxis.title.font.color': c.title,
      });
      // Trajectory polylines ("edges") are a neutral grey by default —
      // bright against the dark background. Mute them to a darker grey
      // in dark mode so the colored markers carry the eye, not the lines.
      Plotly.restyle('plot', { 'line.color': c.line }, lineIndices);
      // Colorbar lives on its own trace (the invisible carrier); restyle
      // tick / title font colors there so the legend strip is readable
      // against the dark background.
      Plotly.restyle('plot', {
        'marker.colorbar.tickfont.color':    c.tick,
        'marker.colorbar.title.font.color':  c.title,
        'marker.colorbar.outlinecolor':      c.grid,
      }, [COLORBAR_TRACE_INDEX]);
    } catch (_) {}
  }
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.setAttribute('data-theme-setting', theme);
    try { localStorage.setItem('theme', theme); } catch (_) {}
    applyPlotTheme(theme);
  }
  var savedTheme = null;
  try { savedTheme = localStorage.getItem('theme'); } catch (_) {}
  if (savedTheme === 'dark' || savedTheme === 'light') {
    setTheme(savedTheme);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    setTheme('dark');
  }
  document.getElementById('theme-toggle').addEventListener('click', function(e) {
    e.preventDefault();
    var cur = document.documentElement.getAttribute('data-theme');
    setTheme(cur === 'dark' ? 'light' : 'dark');
  });

  document.querySelectorAll('button.preset').forEach(function(b) {
    b.onclick = function() {
      var slug = b.dataset.slug;
      var seed = parseInt(b.dataset.seed);
      document.querySelectorAll('button.preset').forEach(function(o) {
        o.classList.toggle('active', o === b);
      });
      state.seedFilter = seed;
      document.getElementById('seed-input').value = seed;
      Object.keys(state.frameFilter).forEach(function(s) {
        state.frameFilter[s] = (s === slug);
      });
      document.querySelectorAll('button.frame').forEach(function(o) {
        var on = (o.dataset.slug === slug);
        o.classList.toggle('active', on);
        o.classList.toggle('off', !on);
      });
      state.stepFilter = null;
      document.getElementById('step-input').value = '';
      applyState();
    };
  });

  function setColorbar(mode) {
    if (mode === 'persona') {
      Plotly.restyle('plot', {
        'marker.colorscale': [PERSONA_COLORSCALE],
        'marker.cmin': PERSONA_CMIN,
        'marker.cmax': PERSONA_CMAX,
        'marker.colorbar.title.text': 'Projection Onto Assistant Axis',
      }, [COLORBAR_TRACE_INDEX]);
    } else {
      Plotly.restyle('plot', {
        'marker.colorscale': [STEP_COLORSCALE],
        'marker.cmin': STEP_CMIN,
        'marker.cmax': STEP_CMAX,
        'marker.colorbar.title.text': 'Optimization Step',
      }, [COLORBAR_TRACE_INDEX]);
    }
  }

  document.getElementById('mode-persona').onclick = function() {
    state.colorMode = 'persona';
    setActiveButton('.mode', 'mode-persona');
    setColorbar('persona');
    deselectPresets();
    applyState();
  };
  document.getElementById('mode-step').onclick = function() {
    state.colorMode = 'step';
    setActiveButton('.mode', 'mode-step');
    setColorbar('step');
    deselectPresets();
    applyState();
  };

  document.getElementById('seed-input').addEventListener('change', function(e) {
    var v = e.target.value;
    if (v === '') { state.seedFilter = null; }
    else {
      var n = parseInt(v);
      n = Math.max(0, Math.min(MAX_SEED, n));
      state.seedFilter = n;
      e.target.value = n;
    }
    deselectPresets();
    applyState();
  });
  document.getElementById('seed-prev').onclick = function() {
    if (state.seedFilter === null) {
      state.seedFilter = 0;
    } else if (state.seedFilter > 0) {
      state.seedFilter -= 1;
    }
    document.getElementById('seed-input').value = state.seedFilter;
    deselectPresets();
    applyState();
  };
  document.getElementById('seed-next').onclick = function() {
    if (state.seedFilter === null) {
      state.seedFilter = 0;
    } else if (state.seedFilter < MAX_SEED) {
      state.seedFilter += 1;
    }
    document.getElementById('seed-input').value = state.seedFilter;
    deselectPresets();
    applyState();
  };
  document.getElementById('seed-clear').onclick = function() {
    state.seedFilter = null;
    document.getElementById('seed-input').value = '';
    deselectPresets();
    applyState();
  };

  document.querySelectorAll('button.frame').forEach(function(b) {
    b.classList.add('active');
    b.onclick = function() {
      var slug = b.dataset.slug;
      state.frameFilter[slug] = !state.frameFilter[slug];
      b.classList.toggle('off', !state.frameFilter[slug]);
      b.classList.toggle('active', state.frameFilter[slug]);
      deselectPresets();
      applyState();
    };
  });

  function snapStep(v) {
    var nearest = STEPS[0], best = Infinity;
    STEPS.forEach(function(s) {
      var d = Math.abs(s - v);
      if (d < best) { best = d; nearest = s; }
    });
    return nearest;
  }
  document.getElementById('step-input').addEventListener('change', function(e) {
    var v = e.target.value;
    if (v === '') { state.stepFilter = null; }
    else { state.stepFilter = snapStep(parseInt(v));
           e.target.value = state.stepFilter; }
    deselectPresets();
    applyState();
  });
  document.getElementById('step-prev').onclick = function() {
    if (state.stepFilter === null) {
      state.stepFilter = STEPS[0];
    } else {
      var i = STEPS.indexOf(state.stepFilter);
      if (i > 0) state.stepFilter = STEPS[i - 1];
    }
    document.getElementById('step-input').value = state.stepFilter;
    deselectPresets();
    applyState();
  };
  document.getElementById('step-next').onclick = function() {
    if (state.stepFilter === null) {
      state.stepFilter = STEPS[0];
    } else {
      var i = STEPS.indexOf(state.stepFilter);
      if (i < STEPS.length - 1) state.stepFilter = STEPS[i + 1];
    }
    document.getElementById('step-input').value = state.stepFilter;
    deselectPresets();
    applyState();
  };
  document.getElementById('step-clear').onclick = function() {
    state.stepFilter = null;
    document.getElementById('step-input').value = '';
    deselectPresets();
    applyState();
  };

  // ── Hover sidebar ─────────────────────────────────────────────────
  var titleEl = document.getElementById('title');
  var metaEl  = document.getElementById('meta');
  var bpEl    = document.getElementById('behav-prompt');
  var btEl    = document.getElementById('behav-text');
  var spEl    = document.getElementById('sv-prompt');
  var stEl    = document.getElementById('sv-text');
  function setText(el, txt, prefix) {
    if (txt === undefined || txt === null || txt === '') {
      el.innerHTML = '<span class="placeholder">—</span>';
    } else { el.textContent = (prefix || '') + txt; }
  }
  // Sidebar populate logic — shared by hover (desktop) and click (touch).
  // On touchscreens hover doesn't fire, so taps fire plotly_click instead.
  function showPoint(p) {
    var d = p.customdata;
    if (!d) return;
    var seed = d[0], step = d[1], kl = d[2];
    var lookupKey = d[3] || (seed + '_' + step);
    var info = cellData[lookupKey] || {};
    titleEl.textContent = 'Seed ' + seed;
    var cname = info.cluster_name || '';
    metaEl.textContent = (cname ? cname + '  ·  ' : '')
      + 'step ' + step + '  ·  KL=' + Number(kl).toFixed(2);
    setText(bpEl, info.behav_prompt, 'Q: ');
    setText(btEl, info.behav_text, '');
    var svPrefix = info.sv_approach ? '[' + info.sv_approach + '] ' : '';
    setText(spEl, svPrefix + (info.sv_prompt || ''), 'Q: ');
    setText(stEl, info.sv_text, '');
  }
  document.getElementById('plot').on('plotly_hover', function(ev) {
    if (ev.points && ev.points.length) showPoint(ev.points[0]);
  });
  document.getElementById('plot').on('plotly_click', function(ev) {
    if (!ev.points || !ev.points.length) return;
    showPoint(ev.points[0]);
    // On mobile the sidebar is below the plot — scroll it into view so the
    // user sees the response without a manual scroll. Gated to the mobile
    // breakpoint; on desktop a click shouldn't move the page.
    if (window.matchMedia('(max-width: 768px)').matches) {
      document.getElementById('sidebar')
        .scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  });

  // Open the About blurb by default so a first-time visitor (e.g.
  // arriving via a link from the personal site) sees the framing before
  // diving in. Easy to dismiss with the ×, the backdrop, or Esc.
  showInfo('general', document.getElementById('about-btn'));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
