"""A tiny torch.export factory for the reference attention block."""

import torch
import torch.export as export

from nomad.reference.block import BlockParams, BlockShape, block


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shape = BlockShape(batch=1, seq=8, d_model=32, n_heads=4)
        self.p = BlockParams(
            g=torch.ones(32, dtype=torch.float16),
            w_qkv=torch.randn(96, 32, dtype=torch.float16),
            w_o=torch.randn(32, 32, dtype=torch.float16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return block(x, self.p, self.shape)


def build():
    model = Block().half().eval()
    x = torch.randn(1, 8, 32, dtype=torch.float16)
    return export.export(model, (x,))
