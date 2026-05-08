"""All-frames combined 3D PC plot with INTERACTIVE controls for filtering
and color-mode swapping.

Builds on the directional + axiscos sister plots. Same 200 trajectories
pooled into a shared PCA basis, same sidebar hover with curated agent
picks + frame-suffixed Q. Adds a top control bar above the plot:

  * Color mode toggle: persona-alignment (cos to assistant axis, Reds_r)
    OR step-progress (coolwarm).
  * Seed filter: type a seed number → only that seed's 4 trajectories
    (one per frame) stay full-opacity; others fade to 0.05 silhouette.
  * Frame toggles: 4 buttons, click to grey out / restore each frame.
  * Step controls: type a ckpt step + arrow buttons, plus an "isolate"
    toggle that hides everything except the markers at the selected step.

All filters compose. Use seed + isolate to focus a single (seed, step)
across the 4 frames; use frame toggles + color mode to compare
persona-alignment vs step-progress for a slice of the population.
"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib import colormaps
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA

from plot_pc_3d_interactive import (  # type: ignore
    BEHAV_PROMPT_PREFIX, SV_PROMPT_PREFIX,
    per_ckpt_responses,
)
from plot_pc_3d_all_frames import DEFAULT_FRAMES, FRAME_DISPLAY  # type: ignore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAME_SUFFIX = {k: v.format(sp="§") for k, v in FRAME_DISPLAY.items()}
FRAME_SLUGS = ["be", "act", "please", "youshould"]
ALL_STEPS = list(range(0, 100, 5)) + [100]  # 21 ckpts

_REDS_R  = colormaps["Reds_r"]
_RAINBOW = colormaps["rainbow"]   # t=0 → violet (start), t=1 → red (end)

LINE_COLOR = "#bbbbbb"  # neutral grey for both color modes


def color_persona(t):  # t in [0, 1]; t=0 → dark red (persona-aligned)
    return to_hex(_REDS_R(max(0.0, min(1.0, t))))


def color_step(t):  # t in [0, 1]; t=0 → violet (start), t=1 → red (end)
    return to_hex(_RAINBOW(max(0.0, min(1.0, t))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", action="append", nargs=2,
                        metavar=("SLUG", "DIR"), default=None)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--max-step", type=int, default=None,
                        help="Only include ckpts with step <= MAX_STEP. PCA is "
                             "refit on the filtered shifts so the basis reflects "
                             "the early/mid-trajectory structure only.")
    parser.add_argument("--out",
                        default="results/all_frames/figure_pc3d_all_frames_interactive.html")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    frames = args.frame or DEFAULT_FRAMES
    steps = ALL_STEPS if args.max_step is None else [s for s in ALL_STEPS if s <= args.max_step]
    if args.max_step is not None:
        print(f"  filtering to steps <= {args.max_step}: {len(steps)} ckpts/seed "
              f"({steps[0]}..{steps[-1]})")

    # ── 1. Pool + PCA ──────────────────────────────────────────────────
    pooled_X, row_index = [], []
    for slug, base in frames:
        path = os.path.join(base if os.path.isabs(base) else os.path.join(ROOT, base),
                            "shifts.pt")
        d = torch.load(path, map_location="cpu", weights_only=True)
        for r in d["rows"]:
            if int(r["step"]) not in steps:
                continue
            pooled_X.append(r["shift"].numpy())
            row_index.append((slug, r["group"], r["ckpt"], int(r["step"]),
                              float(r["kl"])))
    X = np.stack(pooled_X)
    if args.normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    pca = PCA(n_components=args.n_pcs)
    Y = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"Pooled PCA: variance {[round(v, 4) for v in var]}")

    # ── 2. proj_cos lookup ─────────────────────────────────────────────
    proj_cos = {}
    for slug, base in frames:
        ax = os.path.join(base if os.path.isabs(base) else os.path.join(ROOT, base),
                          "axis.json")
        if not os.path.isfile(ax):
            continue
        for r in json.load(open(ax))["rows"]:
            proj_cos[(slug, r["group"], r["ckpt"])] = float(r["proj_cos"])
    cos_min = min(proj_cos.values())
    cos_max = max(proj_cos.values())

    def t_cos(c):
        if cos_max == cos_min:
            return 0.5
        return float((c - cos_min) / (cos_max - cos_min))

    # ── 3. Trajectories ────────────────────────────────────────────────
    by_traj = {}
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

    # ── 4. Self-verb hover: canonical > per-frame > pinned. ───────────
    canonical_path = os.path.join(ROOT, "results/all_frames/manual_self_verb_canonical.json")
    canonical_picks = json.load(open(canonical_path)) if os.path.isfile(canonical_path) else {}
    manual_path = os.path.join(ROOT, "results/all_frames/manual_self_verb.json")
    manual_picks = json.load(open(manual_path)) if os.path.isfile(manual_path) else {}
    print(f"  loaded {len(canonical_picks)} canonical + "
          f"{len(manual_picks)} per-frame entries")

    # ── 5. Build per-cell hover data ───────────────────────────────────
    cell_data = {}
    for (slug, seed), info in by_traj.items():
        base = next((b for s, b in frames if s == slug), None)
        if base is None:
            continue
        if not os.path.isabs(base):
            base = os.path.join(ROOT, base)
        eval_dir = os.path.join(base, f"seed_{seed}", "eval")
        per_ckpt = per_ckpt_responses(eval_dir, [r["step"] for r in info["rows"]])
        suffix = FRAME_SUFFIX.get(slug, "")
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
                sv_prompt   = canonical.get("sv_prompt", "")
                sv_text     = canonical.get("sv_text", "")
                sv_approach = canonical.get("sv_approach", "")
            elif curated and not curated.get("skipped"):
                sv_prompt   = curated.get("sv_prompt", "")
                sv_text     = curated.get("sv_text", "")
                sv_approach = curated.get("sv_approach", "")
            else:
                sv_prompt   = sv.get("prompt", "") if sv else ""
                sv_text     = (sv.get("text", "") if sv
                               else "(no self-verb file for this ckpt)")
                sv_approach = sv.get("approach", "") if sv else ""
            cell_data[key] = {
                "behav_prompt": behav_prompt,
                "behav_text":   b.get("text", "") if b else "(no behavior)",
                "sv_prompt":    sv_prompt,
                "sv_text":      sv_text,
                "sv_approach":  sv_approach,
                "cluster_name": f"Frame: {FRAME_DISPLAY.get(slug, slug)}",
            }

    # ── 6. Build figure ────────────────────────────────────────────────
    # Trace plan:
    #   For each trajectory (200): 1 main (lines+markers) + 1 start + 1 end = 600
    #   For each step (21): 1 step-overlay holding all 200 trajectories' markers
    #                       at that step (initially invisible)
    # Total: 600 + 21 = 621 traces.
    #
    # JS metadata stored alongside the figure_json so the controls can
    # restyle by index.

    fig = go.Figure()
    # Per trajectory: 2 traces. Lines carry no hover; markers carry hover.
    # Background trajectories hide markers entirely → no hover surface.
    # Construction order: all lines → all markers, so markers win Z-buffer
    # ties at vertex joints (later-added trace wins on depth tie).
    traj_meta = []
    line_indices   = []
    marker_indices = []

    # Pre-compute everything per trajectory in one pass.
    trajs = []
    for (slug, seed) in sorted(by_traj.keys()):
        info = by_traj[(slug, seed)]
        rows = info["rows"]
        if len(rows) < 2:
            continue
        n = len(rows)
        pcs   = [r["pc"] for r in rows]
        ckpts = [r["ckpt"] for r in rows]
        cos_v = [proj_cos.get((slug, info["group"], c), (cos_min + cos_max) / 2)
                 for c in ckpts]
        customdata = [
            [seed, r["step"], r["kl"], f"{slug}_{seed}_{r['step']}"]
            for r in rows
        ]
        trajs.append({
            "slug": slug, "seed": seed, "pcs": pcs,
            "customdata": customdata,
            "persona_pt_colors": [color_persona(t_cos(c)) for c in cos_v],
            "step_pt_colors":    [color_step(i / max(1, n - 1)) for i in range(n)],
            "persona_line_color": color_persona(t_cos(min(cos_v))),
        })

    # Pass 1: line traces (initial mode = step coloring → grey lines)
    for t in trajs:
        line_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in t["pcs"]], y=[p[1] for p in t["pcs"]], z=[p[2] for p in t["pcs"]],
            mode="lines",
            line=dict(color=LINE_COLOR, width=3),
            opacity=1.0, showlegend=False,
            hoverinfo="skip",
        ))
    # Pass 2: per-point marker traces (initial mode = step coloring)
    for t in trajs:
        marker_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in t["pcs"]], y=[p[1] for p in t["pcs"]], z=[p[2] for p in t["pcs"]],
            mode="markers",
            marker=dict(size=8, color=t["step_pt_colors"], opacity=1.0),
            opacity=1.0, showlegend=False,
            customdata=t["customdata"],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
    # Per-trajectory metadata (parallel to line + marker index lists)
    for t in trajs:
        traj_meta.append({
            "slug": t["slug"], "seed": t["seed"],
            "persona_pt_colors":  t["persona_pt_colors"],
            "step_pt_colors":     t["step_pt_colors"],
            "persona_line_color": t["persona_line_color"],
            "step_line_color":    LINE_COLOR,
        })

    # Step-overlay traces (one per step). Each holds 200 markers — one per
    # trajectory's point at that step. Initially invisible. Used when the
    # step filter is set: hides the main trajectories and shows just the
    # markers at the chosen step (with per-marker opacity from seed/frame
    # filter). Persona+step color sets are kept in JS metadata so the color
    # mode toggle restyles them too.
    step_overlay_indices = {}        # step → trace index
    overlay_persona_colors = {}      # step → list of per-marker colors (persona)
    overlay_step_colors    = {}      # step → list of per-marker colors (step)
    overlay_meta = {}                # step → list of {slug, seed} per marker
    for step in steps:
        step_overlay_indices[step] = len(fig.data)
        xs, ys, zs, customs, persona_cols, step_cols, metas = [], [], [], [], [], [], []
        # In step mode, all markers in one overlay share the same color
        # (color of the step itself).
        step_t = steps.index(step) / max(1, len(steps) - 1)
        step_uniform = color_step(step_t)
        for (slug, seed) in sorted(by_traj.keys()):
            info = by_traj[(slug, seed)]
            row = next((r for r in info["rows"] if r["step"] == step), None)
            if row is None:
                continue
            xs.append(row["pc"][0]); ys.append(row["pc"][1]); zs.append(row["pc"][2])
            cos_val = proj_cos.get((slug, info["group"], row["ckpt"]),
                                   (cos_min + cos_max) / 2)
            persona_cols.append(color_persona(t_cos(cos_val)))
            step_cols.append(step_uniform)
            customs.append([seed, step, row["kl"], f"{slug}_{seed}_{step}"])
            metas.append({"slug": slug, "seed": seed})
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers", showlegend=False,
            # No marker.line border — the fill alpha controls visibility,
            # and a separate border was rendering even on alpha=0 markers
            # (showed up as black outlines on filtered-out points).
            marker=dict(size=13, color=persona_cols, opacity=0.95),
            visible=False,
            customdata=customs,
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
        overlay_persona_colors[step] = persona_cols
        overlay_step_colors[step]    = step_cols
        overlay_meta[step]           = metas

    var_str = " · ".join(f"PC{i+1} {100*v:.1f}%" for i, v in enumerate(var))
    fig.update_layout(
        title=dict(
            text=(
                f"PC1/PC2/PC3 trajectories  ({len(by_traj)} trajectories)"
                f"  ·  {var_str}"
            ),
            x=0.02, xanchor="left",
        ),
        scene=dict(
            xaxis=dict(title=f"PC1 ({100*var[0]:.1f}%)", showspikes=False),
            yaxis=dict(title=f"PC2 ({100*var[1]:.1f}%)", showspikes=False),
            zaxis=dict(title=f"PC3 ({100*var[2]:.1f}%)", showspikes=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        autosize=True,
    )

    # ── 7. Write custom HTML with control bar ──────────────────────────
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_interactive_html(
        fig, out_path, cell_data,
        traj_meta=traj_meta,
        line_indices=line_indices,
        marker_indices=marker_indices,
        step_overlay_indices=step_overlay_indices,
        overlay_persona_colors=overlay_persona_colors,
        overlay_step_colors=overlay_step_colors,
        overlay_meta=overlay_meta,
        steps=steps,
    )
    print(f"Saved: {out_path}")


# ── Custom HTML wrapper with control bar ────────────────────────────────

INTERACTIVE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CSP PC trajectories — interactive</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; height: 100%; font-family: 'Libertinus Serif', Georgia, serif; }
  #wrap { display: flex; flex-direction: column; height: 100vh; }
  #topbar { display: flex; padding: 10px 14px; border-bottom: 1px solid #ddd;
              background: #f6f6f6; gap: 18px; font-size: 12.5px; }
  #controls { flex: 1 1 50%; display: flex; flex-direction: column; gap: 6px;
              align-items: flex-start; }
  #controls .group { display: flex; align-items: center; gap: 6px; }
  #controls label { font-weight: 600; color: #444; min-width: 70px; }
  #controls input[type=number] { width: 60px; padding: 3px 5px;
              border: 1px solid #ccc; border-radius: 3px; font-size: 12px;
              text-align: center; }
  #presets { flex: 1 1 50%; display: flex; flex-direction: column;
              gap: 6px; font-size: 12px; align-items: flex-start; }
  #presets .row { display: flex; align-items: center; gap: 8px;
              flex-wrap: wrap; }
  #presets .row > label { min-width: 110px; font-weight: 600; color: #444; }
  #presets button { padding: 3px 8px; border: 1px solid #ccc;
              background: #fff; border-radius: 3px; cursor: pointer;
              font-size: 11.5px; color: #333; }
  #presets button:hover { background: #eef; border-color: #99b; }
  #presets button.active { background: #2c5aa0; color: #fff;
              border-color: #2c5aa0; }
  /* Hide native spinners — the ‹ › buttons on either side do the job. */
  #controls input[type=number]::-webkit-outer-spin-button,
  #controls input[type=number]::-webkit-inner-spin-button {
              -webkit-appearance: none; margin: 0; }
  #controls input[type=number] { -moz-appearance: textfield;
              appearance: textfield; }
  #controls button { padding: 3px 9px; border: 1px solid #ccc;
              background: #fff; border-radius: 3px; cursor: pointer;
              font-size: 12px; }
  #controls button:hover { background: #eee; }
  #controls button.active { background: #2c5aa0; color: #fff; border-color: #2c5aa0; }
  #controls button.frame.off { background: #eee; color: #999; }
  #controls .arrow { font-weight: bold; padding: 3px 8px; }
  #plotwrap { display: flex; flex: 1; min-height: 0; }
  #plot { flex: 1; min-width: 0; }
  #sidebar { width: 460px; padding: 16px 18px; overflow-y: auto;
             box-sizing: border-box; border-left: 1px solid #ddd;
             background: #fafafa; }
  #sidebar h2 { margin: 0 0 4px 0; font-size: 16px; color: #222; }
  #sidebar .meta { font-size: 12px; color: #666; margin-bottom: 12px; }
  .panel { margin-bottom: 12px; padding: 10px 12px; background: #fff;
           border: 1px solid #e0e0e0; border-radius: 4px; }
  .panel-title { font-size: 11px; font-weight: 600; color: #555;
           text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
  .panel .prompt { color: #999; font-size: 11.5px; margin-bottom: 6px; font-style: italic; }
  .panel .response { font-size: 12.5px; line-height: 1.4; color: #222;
           white-space: pre-wrap; word-wrap: break-word;
           max-height: 38vh; overflow-y: auto; }
  .placeholder { color: #aaa; font-style: italic; }
</style>
</head>
<body>
<div id="wrap">
  <div id="topbar">
    <div id="controls">
      <div class="group">
        <label>Seed:</label>
        <button id="seed-prev" class="arrow">‹</button>
        <input id="seed-input" type="number" min="0" max="49" placeholder="all">
        <button id="seed-next" class="arrow">›</button>
        <button id="seed-clear">all seeds</button>
      </div>
      <div class="group">
        <label>Step:</label>
        <button id="step-prev" class="arrow">‹</button>
        <input id="step-input" type="number" placeholder="all">
        <button id="step-next" class="arrow">›</button>
        <button id="step-clear">all steps</button>
      </div>
      <div class="group">
        <label>Frames:</label>
        <button class="frame" data-slug="be">BE</button>
        <button class="frame" data-slug="act">ACT</button>
        <button class="frame" data-slug="please">PLEASE</button>
        <button class="frame" data-slug="youshould">YOUSHOULD</button>
      </div>
      <div class="group">
        <label>Color:</label>
        <button id="mode-step" class="mode active">step</button>
        <button id="mode-persona" class="mode">persona</button>
      </div>
    </div>
    <div id="presets">
      <div class="row">
        <label>Personas:</label>
        <button class="preset" data-slug="please" data-seed="41">Low-income Southerner</button>
        <button class="preset" data-slug="please" data-seed="39">Yoda/Geralt of Rivia</button>
        <button class="preset" data-slug="be" data-seed="6">Chinese Philosopher</button>
      </div>
      <div class="row">
        <label>Formatting:</label>
        <button class="preset" data-slug="act" data-seed="29">Urgency</button>
        <button class="preset" data-slug="be" data-seed="2">All-caps Shouting</button>
        <button class="preset" data-slug="please" data-seed="22">Pauses &amp; Ellipses</button>
      </div>
      <div class="row">
        <label>Analytical:</label>
        <button class="preset" data-slug="act" data-seed="17">Essential Elements</button>
        <button class="preset" data-slug="please" data-seed="44">Math</button>
        <button class="preset" data-slug="youshould" data-seed="0">Code</button>
      </div>
      <div class="row">
        <label>Citations &amp; refs:</label>
        <button class="preset" data-slug="youshould" data-seed="31">Scientific Interpretation</button>
        <button class="preset" data-slug="youshould" data-seed="33">X according to Y</button>
        <button class="preset" data-slug="youshould" data-seed="3">Philosophical Principles</button>
      </div>
    </div>
  </div>
  <div id="plotwrap">
    <div id="plot"></div>
    <div id="sidebar">
      <h2 id="title"><span class="placeholder">Hover a point to see outputs</span></h2>
      <div id="meta" class="meta"></div>
      <div class="panel">
        <div class="panel-title">Behavior</div>
        <div id="behav-prompt" class="prompt"></div>
        <div id="behav-text" class="response"><span class="placeholder">—</span></div>
      </div>
      <div class="panel">
        <div class="panel-title">Self-verb</div>
        <div id="sv-prompt" class="prompt"></div>
        <div id="sv-text" class="response"><span class="placeholder">—</span></div>
      </div>
    </div>
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

  Plotly.newPlot('plot', fig.data, fig.layout, {responsive: true, displaylogo: false});

  // ── State ──────────────────────────────────────────────────────────
  // step is a regular filter (like seed/frame). When set, main trajectories
  // hide and only the selected step's overlay is visible.
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

  // Encode opacity into rgba alpha. Plotly's Scatter3d marker.opacity is
  // a single value (not array), so per-marker filtering has to go through
  // the color string. alpha=0 → invisible, no hover fires.
  function rgba(hex, alpha) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  function applyState() {
    var stepActive = state.stepFilter !== null;

    // Lines: silhouette opacity always; full when active and no step
    // filter. Lines never have hover — hoverinfo='skip' is baked in,
    // and restyled defensively here too in case plotly drops it.
    var lineOps = [], lineColors = [], lineHovers = [];
    // Markers: use visible:true/false, NOT opacity:0. opacity:0 hides
    // the visual but plotly still fires hover events on the invisible
    // hitbox; visible:false removes the trace from the WebGL scene
    // entirely — definitively no hover, no render.
    var markerVisibles = [], markerColors = [];
    trajMeta.forEach(function(meta) {
      var active = isTrajActive(meta);
      var lineOp   = (active && !stepActive) ? 1.0 : 0.05;
      var markerVis = (active && !stepActive);
      lineOps.push(lineOp);
      lineHovers.push('skip');
      markerVisibles.push(markerVis);
      lineColors.push(state.colorMode === 'persona'
        ? meta.persona_line_color : meta.step_line_color);
      markerColors.push(state.colorMode === 'persona'
        ? meta.persona_pt_colors : meta.step_pt_colors);
    });
    Plotly.restyle('plot',
      {opacity: lineOps, 'line.color': lineColors, hoverinfo: lineHovers},
      lineIndices);
    Plotly.restyle('plot',
      {visible: markerVisibles, 'marker.color': markerColors},
      markerIndices);

    // Step overlays. Visible only for the selected step; per-marker
    // visibility encoded into rgba alpha so seed+frame filters compose.
    // Inactive markers get alpha=0 → invisible AND no hover.
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
        colors.push(overlayPersonaColors[stepKey]);   // dummy; not visible
      }
    }
    Plotly.restyle('plot',
      {visible: visibles, 'marker.color': colors},
      allOverlayIdx);
  }

  // ── Control wiring ──────────────────────────────────────────────────
  function setActiveButton(group, activeId) {
    document.querySelectorAll(group).forEach(function(b) {
      b.classList.toggle('active', b.id === activeId);
    });
  }

  document.querySelectorAll('button.preset').forEach(function(b) {
    b.onclick = function() {
      var slug = b.dataset.slug;
      var seed = parseInt(b.dataset.seed);
      // Visual: highlight the active preset button (one at a time).
      document.querySelectorAll('button.preset').forEach(function(o) {
        o.classList.toggle('active', o === b);
      });
      // Isolate (frame, seed); step coloring; no step filter.
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
      state.colorMode = 'step';
      setActiveButton('.mode', 'mode-step');
      state.stepFilter = null;
      document.getElementById('step-input').value = '';
      applyState();
    };
  });

  document.getElementById('mode-persona').onclick = function() {
    state.colorMode = 'persona';
    setActiveButton('.mode', 'mode-persona');
    applyState();
  };
  document.getElementById('mode-step').onclick = function() {
    state.colorMode = 'step';
    setActiveButton('.mode', 'mode-step');
    applyState();
  };

  document.getElementById('seed-input').addEventListener('change', function(e) {
    var v = e.target.value;
    if (v === '') { state.seedFilter = null; }
    else {
      var n = parseInt(v);
      n = Math.max(0, Math.min(49, n));
      state.seedFilter = n;
      e.target.value = n;
    }
    applyState();
  });
  document.getElementById('seed-prev').onclick = function() {
    if (state.seedFilter === null) {
      state.seedFilter = 0;
    } else if (state.seedFilter > 0) {
      state.seedFilter -= 1;
    }
    document.getElementById('seed-input').value = state.seedFilter;
    applyState();
  };
  document.getElementById('seed-next').onclick = function() {
    if (state.seedFilter === null) {
      state.seedFilter = 0;
    } else if (state.seedFilter < 49) {
      state.seedFilter += 1;
    }
    document.getElementById('seed-input').value = state.seedFilter;
    applyState();
  };
  document.getElementById('seed-clear').onclick = function() {
    state.seedFilter = null;
    document.getElementById('seed-input').value = '';
    applyState();
  };

  document.querySelectorAll('button.frame').forEach(function(b) {
    b.classList.add('active');                 // start with all frames on
    b.onclick = function() {
      var slug = b.dataset.slug;
      state.frameFilter[slug] = !state.frameFilter[slug];
      b.classList.toggle('off', !state.frameFilter[slug]);
      b.classList.toggle('active', state.frameFilter[slug]);
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
    applyState();
  });
  document.getElementById('step-prev').onclick = function() {
    if (state.stepFilter === null) {
      // Empty → default to first step (not last)
      state.stepFilter = STEPS[0];
    } else {
      var i = STEPS.indexOf(state.stepFilter);
      if (i > 0) state.stepFilter = STEPS[i - 1];
    }
    document.getElementById('step-input').value = state.stepFilter;
    applyState();
  };
  document.getElementById('step-next').onclick = function() {
    if (state.stepFilter === null) {
      // Empty → default to first step, then increment from there
      state.stepFilter = STEPS[0];
    } else {
      var i = STEPS.indexOf(state.stepFilter);
      if (i < STEPS.length - 1) state.stepFilter = STEPS[i + 1];
    }
    document.getElementById('step-input').value = state.stepFilter;
    applyState();
  };
  document.getElementById('step-clear').onclick = function() {
    state.stepFilter = null;
    document.getElementById('step-input').value = '';
    applyState();
  };

  // ── Hover sidebar (same as the static plots) ───────────────────────
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
  document.getElementById('plot').on('plotly_hover', function(ev) {
    if (!ev.points || !ev.points.length) return;
    var d = ev.points[0].customdata;
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
  });
</script>
</body>
</html>
"""


def write_interactive_html(fig, out_path, cell_data, *,
                           traj_meta, line_indices, marker_indices,
                           step_overlay_indices, overlay_persona_colors,
                           overlay_step_colors, overlay_meta, steps):
    fig_json   = fig.to_json()
    cell_json  = json.dumps(cell_data)
    traj_json  = json.dumps(traj_meta)
    line_json   = json.dumps(line_indices)
    marker_json = json.dumps(marker_indices)
    step_idx_json = json.dumps({str(k): v for k, v in step_overlay_indices.items()})
    persona_cols_json = json.dumps({str(k): v for k, v in overlay_persona_colors.items()})
    step_cols_json    = json.dumps({str(k): v for k, v in overlay_step_colors.items()})
    overlay_meta_json = json.dumps({str(k): v for k, v in overlay_meta.items()})
    steps_json = json.dumps(steps)
    html = (INTERACTIVE_HTML_TEMPLATE
            .replace("__FIGURE_JSON__",            fig_json)
            .replace("__CELL_DATA_JSON__",         cell_json)
            .replace("__TRAJ_META_JSON__",         traj_json)
            .replace("__LINE_INDICES__",           line_json)
            .replace("__MARKER_INDICES__",         marker_json)
            .replace("__STEP_OVERLAY_INDICES__",   step_idx_json)
            .replace("__OVERLAY_PERSONA_COLORS__", persona_cols_json)
            .replace("__OVERLAY_STEP_COLORS__",    step_cols_json)
            .replace("__OVERLAY_META__",           overlay_meta_json)
            .replace("__STEPS__",                  steps_json))
    with open(out_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
