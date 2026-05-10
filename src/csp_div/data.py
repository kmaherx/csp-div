"""Question dataset loading.

The 240-question file lives at `data/questions.jsonl`, sourced from
safety-research/assistant-axis under MIT (see `data/README.md`).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import PROJECT_ROOT


DEFAULT_QUESTIONS_PATH = Path(PROJECT_ROOT) / "data" / "questions.jsonl"


def load_questions(path: str | Path | None = None) -> list[str]:
    """Load the question prompts as a list of strings.

    Reads any JSONL field named `question`, `text`, or `prompt`. Skips
    lines lacking all three.
    """
    p = Path(path) if path is not None else DEFAULT_QUESTIONS_PATH
    questions: list[str] = []
    with open(p) as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question") or obj.get("text") or obj.get("prompt")
            if q:
                questions.append(q)
    return questions
