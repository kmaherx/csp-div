import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    def __init__(self, length: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(length, hidden_size) * 0.1)

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        sp = cls(ckpt["L"], ckpt["hidden_size"]).to(device)
        sp.embedding.data = ckpt["embedding"].to(device)
        return sp, ckpt
