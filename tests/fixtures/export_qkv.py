"""A tiny torch.export factory used by the ingest test and the CLI demo."""

import torch
import torch.export as export
import torch.nn.functional as F


class QKV(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(3072, 1024, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w)


def build():
    model = QKV().half().eval()
    x = torch.randn(128, 1024, dtype=torch.float16)
    return export.export(model, (x,))
