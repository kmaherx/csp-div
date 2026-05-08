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

_REDS_R   = colormaps["Reds_r"]
_COOLWARM = colormaps["coolwarm"]


def color_persona(t):  # t in [0, 1]; t=0 → dark red (persona-aligned)
    return to_hex(_REDS_R(max(0.0, min(1.0, t))))


def color_step(t):  # t in [0, 1]; t=0 → cool blue, t=1 → warm red
    return to_hex(_COOLWARM(max(0.0, min(1.0, t))))


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

    # ── 4. Curated picks for hover ─────────────────────────────────────
    manual_path = os.path.join(ROOT, "results/all_frames/manual_self_verb.json")
    manual_picks = json.load(open(manual_path)) if os.path.isfile(manual_path) else {}
    print(f"  loaded {len(manual_picks)} curated entries")

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
            curated = manual_picks.get(key)
            if curated and not curated.get("skipped"):
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
    traj_meta = []  # parallel to main traces; (slug, seed, persona_marker_colors,
                    #                          step_marker_colors, persona_line_color,
                    #                          step_line_color)
    main_indices  = []
    start_indices = []
    end_indices   = []

    sorted_keys = sorted(by_traj.keys())
    for (slug, seed) in sorted_keys:
        info = by_traj[(slug, seed)]
        rows = info["rows"]
        if len(rows) < 2:
            continue
        n = len(rows)
        pcs    = [r["pc"] for r in rows]
        steps  = [r["step"] for r in rows]
        ckpts  = [r["ckpt"] for r in rows]
        cos_v  = [proj_cos.get((slug, info["group"], c), (cos_min + cos_max) / 2)
                  for c in ckpts]
        customdata = [
            [seed, r["step"], r["kl"], f"{slug}_{seed}_{r['step']}"]
            for r in rows
        ]
        persona_marker_cols = [color_persona(t_cos(c)) for c in cos_v]
        step_marker_cols    = [color_step(i / max(1, n - 1)) for i in range(n)]
        persona_line_col    = color_persona(t_cos(float(np.median(cos_v))))
        step_line_col       = color_step(0.5)  # mid coolwarm = grey-purple-ish

        # Main trajectory trace (initial mode: persona)
        main_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in pcs], y=[p[1] for p in pcs], z=[p[2] for p in pcs],
            mode="lines+markers",
            line=dict(color=persona_line_col, width=4),
            marker=dict(size=4, color=persona_marker_cols, opacity=0.85),
            opacity=1.0,
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))
        traj_meta.append({
            "slug": slug, "seed": seed,
            "persona_marker_color": persona_marker_cols,
            "step_marker_color":    step_marker_cols,
            "persona_line_color":   persona_line_col,
            "step_line_color":      step_line_col,
        })

        # Start marker — open ring (white fill, dark grey border)
        start_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=[pcs[0][0]], y=[pcs[0][1]], z=[pcs[0][2]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color="white",
                        line=dict(color="#333333", width=1.5)),
            customdata=[customdata[0]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]} (start)"
                "<extra></extra>"
            ),
        ))
        # End marker — black filled
        end_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=[pcs[-1][0]], y=[pcs[-1][1]], z=[pcs[-1][2]],
            mode="markers", showlegend=False,
            marker=dict(size=6, color="black"),
            customdata=[customdata[-1]],
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]} (end)"
                "<extra></extra>"
            ),
        ))

    # Step-overlay traces (one per step). Each holds 200 markers — one per
    # trajectory's point at that step. Initially invisible (opacity 0).
    step_overlay_indices = {}  # step → trace index
    for step in steps:
        step_overlay_indices[step] = len(fig.data)
        xs, ys, zs, customs, persona_cols, step_cols = [], [], [], [], [], []
        for meta_i, (slug, seed) in enumerate(sorted_keys):
            info = by_traj[(slug, seed)]
            row = next((r for r in info["rows"] if r["step"] == step), None)
            if row is None:
                continue
            xs.append(row["pc"][0]); ys.append(row["pc"][1]); zs.append(row["pc"][2])
            cos_val = proj_cos.get((slug, info["group"], row["ckpt"]),
                                   (cos_min + cos_max) / 2)
            persona_cols.append(color_persona(t_cos(cos_val)))
            # Step-overlay marker color in step mode = step color (uniform per overlay)
            step_cols.append(color_step(steps.index(step) / max(1, len(steps) - 1)))
            customs.append([seed, step, row["kl"], f"{slug}_{seed}_{step}"])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers", showlegend=False,
            marker=dict(size=10, color=persona_cols, opacity=0.95,
                        line=dict(color="black", width=1)),
            visible=False,  # toggled on by step-isolate
            customdata=customs,
            hovertemplate=(
                "<b>seed %{customdata[0]}</b> · step %{customdata[1]}<extra></extra>"
            ),
        ))

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
            xaxis_title=f"PC1 ({100*var[0]:.1f}%)",
            yaxis_title=f"PC2 ({100*var[1]:.1f}%)",
            zaxis_title=f"PC3 ({100*var[2]:.1f}%)",
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
        main_indices=main_indices,
        start_indices=start_indices,
        end_indices=end_indices,
        step_overlay_indices=step_overlay_indices,
        steps=steps,
        cos_min=cos_min, cos_max=cos_max,
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
  #controls { padding: 10px 14px; border-bottom: 1px solid #ddd;
              background: #f6f6f6; display: flex; flex-wrap: wrap; gap: 18px;
              align-items: center; font-size: 12.5px; }
  #controls .group { display: flex; align-items: center; gap: 6px; }
  #controls label { font-weight: 600; color: #444; }
  #controls input[type=number] { width: 60px; padding: 3px 5px;
              border: 1px solid #ccc; border-radius: 3px; font-size: 12px; }
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
  <div id="controls">
    <div class="group">
      <label>Color:</label>
      <button id="mode-persona" class="mode active">persona</button>
      <button id="mode-step" class="mode">step</button>
    </div>
    <div class="group">
      <label>Seed:</label>
      <input id="seed-input" type="number" min="0" max="49" placeholder="all">
      <button id="seed-clear">all seeds</button>
    </div>
    <div class="group">
      <label>Frames:</label>
      <button class="frame" data-slug="be">BE</button>
      <button class="frame" data-slug="act">ACT</button>
      <button class="frame" data-slug="please">PLEASE</button>
      <button class="frame" data-slug="youshould">YOUSHOULD</button>
    </div>
    <div class="group">
      <label>Step:</label>
      <button id="step-prev" class="arrow">‹</button>
      <input id="step-input" type="number" placeholder="—">
      <button id="step-next" class="arrow">›</button>
      <button id="step-isolate">isolate</button>
      <button id="step-clear">clear</button>
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
        <div class="panel-title">Self-verb (curated)</div>
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
  var mainIndices = __MAIN_INDICES__;
  var startIndices = __START_INDICES__;
  var endIndices = __END_INDICES__;
  var stepOverlayIndices = __STEP_OVERLAY_INDICES__;
  var STEPS = __STEPS__;

  Plotly.newPlot('plot', fig.data, fig.layout, {responsive: true, displaylogo: false});

  // ── State ──────────────────────────────────────────────────────────
  var state = {
    colorMode: 'persona',
    seedFilter: null,
    frameFilter: {be: true, act: true, please: true, youshould: true},
    stepHighlight: null,
    stepIsolate: false,
  };

  // Visibility decision per trajectory: matches all active filters?
  function isTrajActive(meta) {
    if (state.seedFilter !== null && meta.seed !== state.seedFilter) return false;
    if (!state.frameFilter[meta.slug]) return false;
    return true;
  }

  function applyState() {
    var n = trajMeta.length;
    // Compose opacity arrays for main / start / end
    var mainOpacities  = new Array(n);
    var startOpacities = new Array(n);
    var endOpacities   = new Array(n);
    var lineColors     = new Array(n);
    var markerColors   = new Array(n);
    for (var i = 0; i < n; i++) {
      var meta = trajMeta[i];
      var active = isTrajActive(meta);
      var op = active ? 1.0 : 0.05;
      // When step-isolate is on, hide all main/start/end traces entirely
      if (state.stepIsolate) op = 0.0;
      mainOpacities[i]  = op;
      startOpacities[i] = state.stepIsolate ? 0.0 : op;
      endOpacities[i]   = state.stepIsolate ? 0.0 : op;
      // Color mode swap
      if (state.colorMode === 'persona') {
        lineColors[i]   = meta.persona_line_color;
        markerColors[i] = meta.persona_marker_color;
      } else {
        lineColors[i]   = meta.step_line_color;
        markerColors[i] = meta.step_marker_color;
      }
    }
    Plotly.restyle('plot',
      {opacity: mainOpacities, 'line.color': lineColors,
       'marker.color': markerColors},
      mainIndices);
    Plotly.restyle('plot', {opacity: startOpacities}, startIndices);
    Plotly.restyle('plot', {opacity: endOpacities},   endIndices);

    // Step overlay: hide all by default; show the active step-overlay if isolate is on.
    var visibleSteps = {};
    if (state.stepIsolate && state.stepHighlight !== null) {
      visibleSteps[state.stepHighlight] = true;
    }
    var allOverlayIdx = [];
    var overlayVisibles = [];
    var overlayMarkerOpacs = [];
    for (var step in stepOverlayIndices) {
      var idx = stepOverlayIndices[step];
      allOverlayIdx.push(idx);
      var stepNum = parseInt(step);
      if (visibleSteps[stepNum]) {
        // Per-marker opacity within the overlay reflects seed/frame filter
        var trace = fig.data[idx];
        var customs = trace.customdata || [];
        var perMarkerOp = customs.map(function(cd) {
          var key = cd[3];                // "<slug>_<seed>_<step>"
          var slug = key.split('_')[0];
          var seed = parseInt(key.split('_')[1]);
          var meta = {slug: slug, seed: seed};
          return isTrajActive(meta) ? 0.95 : 0.05;
        });
        overlayVisibles.push(true);
        overlayMarkerOpacs.push(perMarkerOp);
      } else {
        overlayVisibles.push(false);
        overlayMarkerOpacs.push(0.0);
      }
    }
    Plotly.restyle('plot',
      {visible: overlayVisibles, 'marker.opacity': overlayMarkerOpacs},
      allOverlayIdx);
  }

  // ── Control wiring ──────────────────────────────────────────────────
  function setActiveButton(group, activeId) {
    document.querySelectorAll(group).forEach(function(b) {
      b.classList.toggle('active', b.id === activeId);
    });
  }

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
    state.seedFilter = (v === '' ? null : parseInt(v));
    applyState();
  });
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
    // Round to nearest valid step in STEPS
    var nearest = STEPS[0], best = Infinity;
    STEPS.forEach(function(s) {
      var d = Math.abs(s - v);
      if (d < best) { best = d; nearest = s; }
    });
    return nearest;
  }
  document.getElementById('step-input').addEventListener('change', function(e) {
    var v = e.target.value;
    if (v === '') { state.stepHighlight = null; }
    else { state.stepHighlight = snapStep(parseInt(v));
           e.target.value = state.stepHighlight; }
    applyState();
  });
  document.getElementById('step-prev').onclick = function() {
    var cur = state.stepHighlight !== null ? state.stepHighlight : STEPS[0];
    var i = STEPS.indexOf(cur);
    if (i > 0) state.stepHighlight = STEPS[i - 1];
    else state.stepHighlight = STEPS[0];
    document.getElementById('step-input').value = state.stepHighlight;
    applyState();
  };
  document.getElementById('step-next').onclick = function() {
    var cur = state.stepHighlight !== null ? state.stepHighlight : STEPS[STEPS.length - 1];
    var i = STEPS.indexOf(cur);
    if (i < STEPS.length - 1) state.stepHighlight = STEPS[i + 1];
    else state.stepHighlight = STEPS[STEPS.length - 1];
    document.getElementById('step-input').value = state.stepHighlight;
    applyState();
  };
  document.getElementById('step-isolate').onclick = function() {
    state.stepIsolate = !state.stepIsolate;
    if (state.stepIsolate && state.stepHighlight === null) {
      state.stepHighlight = 25;          // sensible default
      document.getElementById('step-input').value = 25;
    }
    document.getElementById('step-isolate').classList.toggle('active', state.stepIsolate);
    applyState();
  };
  document.getElementById('step-clear').onclick = function() {
    state.stepHighlight = null;
    state.stepIsolate = false;
    document.getElementById('step-input').value = '';
    document.getElementById('step-isolate').classList.remove('active');
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
                           traj_meta, main_indices, start_indices, end_indices,
                           step_overlay_indices, steps, cos_min, cos_max):
    fig_json   = fig.to_json()
    cell_json  = json.dumps(cell_data)
    traj_json  = json.dumps(traj_meta)
    main_json  = json.dumps(main_indices)
    start_json = json.dumps(start_indices)
    end_json   = json.dumps(end_indices)
    step_idx_json = json.dumps({str(k): v for k, v in step_overlay_indices.items()})
    steps_json = json.dumps(steps)
    html = (INTERACTIVE_HTML_TEMPLATE
            .replace("__FIGURE_JSON__",        fig_json)
            .replace("__CELL_DATA_JSON__",     cell_json)
            .replace("__TRAJ_META_JSON__",     traj_json)
            .replace("__MAIN_INDICES__",       main_json)
            .replace("__START_INDICES__",      start_json)
            .replace("__END_INDICES__",        end_json)
            .replace("__STEP_OVERLAY_INDICES__", step_idx_json)
            .replace("__STEPS__",              steps_json))
    with open(out_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
