"""Central configuration for the CSP-divergence pipeline.

Dataclasses for training, generation, and axis settings; constants for the
frame pool and placeholder token. Pipeline scripts override the defaults
via CLI; library code reads the defaults directly.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Model preset ────────────────────────────────────────────────────────

MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
AXIS_REPO: str = "Butanium/llama-3.1-8b-instruct-assistant-axis"
AXIS_LAYER: int = 16


# ── Frames ──────────────────────────────────────────────────────────────
# The CSP is spliced into a `§` placeholder inside one of these frames.
# Each frame has a stable slug used in result paths (e.g. results/llama_be/).

SP_PLACEHOLDER: str = "§"


@dataclass(frozen=True)
class Frame:
    slug: str       # filesystem-safe identifier, e.g. "youshould"
    template: str   # e.g. "You should {sp}."


FRAMES: tuple[Frame, ...] = (
    Frame("be",        "Be {sp}."),
    Frame("act",       "Act {sp}."),
    Frame("please",    "Please {sp}."),
    Frame("youshould", "You should {sp}."),
)

FRAMES_BY_SLUG: dict[str, Frame] = {f.slug: f for f in FRAMES}

POSITIVE_FRAMES: list[str] = [f.template for f in FRAMES]
TRAIN_FRAME_POOL: list[str] = POSITIVE_FRAMES   # sampled per step during KL ascent


# ── Configs ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    L: int = 4                          # soft prompt length (tokens)
    steps: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    prompts_per_step: int = 50
    max_new_tokens: int = 128           # teacher vanilla-response cap
    checkpoint_every: int = 5           # save intermediate ckpt every N steps
    seed: int = 42


@dataclass
class GenerateConfig:
    n_eval_prompts: int = 30            # behavior eval prompt count
    n_behavior_samples: int = 5         # subset shown side-by-side w/ vanilla
    max_new_tokens_verb: int = 64       # per self-verb response
    max_new_tokens_behavior: int = 128  # per behavior response
    max_new_tokens_act: int = 64        # tokens generated when capturing shifts


@dataclass
class AxisConfig:
    layer: int = AXIS_LAYER
    max_new_tokens: int = 64            # tokens averaged when capturing shifts
    n_eval_prompts: int = 30
