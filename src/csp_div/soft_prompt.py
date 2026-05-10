"""SoftPrompt: trainable parameter block spliced into a frame at the
`§` placeholder. One CSP per training run.

Init regime: `randn * 0.1`, giving per-token L2 norm ~6.4 on Llama-3.1-8B —
about 10x typical token-embedding rows. This is intentional: the model
treats the init as noise at step 0, and KL ascent walks the embedding
inward toward an attractor. See README for the rationale.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    def __init__(self, length: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(length, hidden_size) * 0.1)

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu",
    ) -> tuple["SoftPrompt", dict]:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        sp = cls(ckpt["L"], ckpt["hidden_size"]).to(device)
        sp.embedding.data = ckpt["embedding"].to(device)
        return sp, ckpt
