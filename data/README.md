# `questions.jsonl`

240 conversational prompts used as the evaluation set throughout the
pipeline (vanilla teacher response cache, behavior generation, shift
capture). One JSON object per line, schema:

```jsonl
{"question": "...", "id": 0}
```

## Provenance

This file is `extraction_questions.jsonl` from
[safety-research/assistant-axis](https://github.com/safety-research/assistant-axis/blob/master/data/extraction_questions.jsonl),
renamed to `questions.jsonl` here. Content is byte-identical (240
questions, IDs 0–239).

Used under the MIT license. If you use this dataset (directly or
through this repo), cite:

```bibtex
@misc{lu2026assistant,
  title  = {The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models},
  author = {Lu, Christina and Gallagher, Jack and Michala, Jonathan and Fish, Kyle and Lindsey, Jack},
  year   = {2026},
  eprint = {2601.10387},
  archivePrefix = {arXiv},
}
```
