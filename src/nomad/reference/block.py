"""PyTorch reference for the fused attention block: RMSNorm -> QKV -> SDPA -> out-proj (fp16).

`block` is the math. Its realizations:
  reference_fp32  correctness oracle: fp32 throughout, rounded to the input dtype exactly once
  eager           fp16 eager with an explicit softmax              (latency baseline)
  sdpa            fp16 eager with F.scaled_dot_product_attention   (latency baseline)
  compiled        torch.compile over the eager block               (latency baseline)
Only the oracle judges correctness; the baselines are what NOMAD's kernels are timed against.
Keep torch.backends.cuda.matmul.allow_tf32 at its default (False) or the oracle loses ~3 digits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, partial
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockShape:
    batch: int
    seq: int
    d_model: int
    n_heads: int
    causal: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def qkv_gemm(self) -> tuple[int, int, int]:
        """(M, K, N) of the fused QKV projection, in tests/gpu/test_qkv_tile.py's convention."""
        return self.batch * self.seq, self.d_model, 3 * self.d_model

    @property
    def id(self) -> str:
        causal = "c" if self.causal else ""
        return f"b{self.batch}s{self.seq}d{self.d_model}h{self.n_heads}{causal}"


# Block shape table; the first four reproduce the (M, K, N) rows of tests/gpu/test_qkv_tile.py.
SHAPES: tuple[BlockShape, ...] = (
    BlockShape(batch=1, seq=128, d_model=1024, n_heads=16),
    BlockShape(batch=4, seq=256, d_model=1024, n_heads=16),
    BlockShape(batch=4, seq=1024, d_model=1024, n_heads=16),
    BlockShape(batch=1, seq=2048, d_model=2048, n_heads=16),
    BlockShape(batch=2, seq=77, d_model=1024, n_heads=16),  # ragged: seq not a multiple of 64
    BlockShape(batch=2, seq=256, d_model=1024, n_heads=16, causal=True),
)


@dataclass(frozen=True)
class BlockParams:
    g: torch.Tensor  # [D] RMSNorm gain
    w_qkv: torch.Tensor  # [3D, D] nn.Linear layout, rows ordered q | k | v
    w_o: torch.Tensor  # [D, D]
    eps: float = 1e-6

    def to(self, dtype: torch.dtype) -> BlockParams:
        return BlockParams(self.g.to(dtype), self.w_qkv.to(dtype), self.w_o.to(dtype), self.eps)


def make_inputs(
    shape: BlockShape, *, dtype: torch.dtype = torch.float16, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, BlockParams]:
    """Seeded inputs, drawn in fp32 and cast so every dtype sees the same values."""
    gen = torch.Generator(device=device).manual_seed(seed)
    d = shape.d_model
    x = torch.randn(shape.batch, shape.seq, d, generator=gen, device=device)
    params = BlockParams(
        g=1.0 + 0.1 * torch.randn(d, generator=gen, device=device),
        w_qkv=torch.randn(3 * d, d, generator=gen, device=device) / math.sqrt(d),
        w_o=torch.randn(d, d, generator=gen, device=device) / math.sqrt(d),
    )
    return x.to(dtype), params.to(dtype)


def block(
    x: torch.Tensor, p: BlockParams, shape: BlockShape, *, attention: str = "explicit"
) -> torch.Tensor:
    """x [B, S, D] -> out_proj(attention(split_heads(rmsnorm(x) @ w_qkv^T))) [B, S, D]."""
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + p.eps) * p.g
    q, k, v = (
        t.unflatten(-1, (shape.n_heads, shape.head_dim)).transpose(1, 2)  # [B, H, S, Dh]
        for t in F.linear(h, p.w_qkv).chunk(3, dim=-1)
    )
    if attention == "sdpa":
        o = F.scaled_dot_product_attention(q, k, v, is_causal=shape.causal)
    else:
        s = (q @ k.transpose(-2, -1)) * shape.head_dim**-0.5
        if shape.causal:
            future = torch.ones(shape.seq, shape.seq, dtype=torch.bool, device=x.device).triu(1)
            s = s.masked_fill(future, float("-inf"))
        o = s.softmax(-1) @ v
    return F.linear(o.transpose(1, 2).flatten(2), p.w_o)


def reference_fp32(x: torch.Tensor, p: BlockParams, shape: BlockShape) -> torch.Tensor:
    """Correctness oracle: the same math in fp32, rounded to x.dtype exactly once."""
    return block(x.float(), p.to(torch.float32), shape).to(x.dtype)


@cache
def _compiled_block():
    return torch.compile(block, dynamic=False)


def compiled(x: torch.Tensor, p: BlockParams, shape: BlockShape) -> torch.Tensor:
    return _compiled_block()(x, p, shape)


BASELINES = {"eager": block, "sdpa": partial(block, attention="sdpa"), "compile": compiled}


def graph_qkv_proj() -> dict[str, Any]:
    return {
        "graph": "qkv_proj",
        "params": ["M", "K", "N"],
        "assume": ["K % 8 == 0"],
        "tensors": {
            "x": {"shape": ["M", "K"], "dtype": "f16"},
            "w": {"shape": ["N", "K"], "dtype": "f16", "weight": True},
            "y": {"shape": ["M", "N"], "dtype": "f16", "out": True},
        },
        "ops": [{"id": "qkv_proj", "kind": "linear", "in": ["x", "w"], "out": ["y"]}],
    }


def graph_attention_block() -> dict[str, Any]:
    return {
        "graph": "attention_block",
        "params": ["B", "S", "D", "H"],
        "assume": ["D % H == 0"],
        "tensors": {
            "x": {"shape": ["B * S", "D"], "dtype": "f16"},
            "g": {"shape": ["D"], "dtype": "f16", "weight": True},
            "w_qkv": {"shape": ["3 * D", "D"], "dtype": "f16", "weight": True},
            "w_o": {"shape": ["D", "D"], "dtype": "f16", "weight": True},
            "h": {"shape": ["B * S", "D"], "dtype": "f16"},
            "qkv": {"shape": ["B * S", "3 * D"], "dtype": "f16"},
            "q": {"shape": ["B", "H", "S", "D / H"], "dtype": "f16"},
            "k": {"shape": ["B", "H", "S", "D / H"], "dtype": "f16"},
            "v": {"shape": ["B", "H", "S", "D / H"], "dtype": "f16"},
            "o": {"shape": ["B", "H", "S", "D / H"], "dtype": "f16"},
            "o2": {"shape": ["B * S", "D"], "dtype": "f16"},
            "y": {"shape": ["B * S", "D"], "dtype": "f16", "out": True},
        },
        "ops": [
            {"id": "norm", "kind": "rmsnorm", "in": ["x", "g"], "out": ["h"], "eps": 1.0e-6},
            {"id": "qkv", "kind": "linear", "in": ["h", "w_qkv"], "out": ["qkv"]},
            {"id": "split", "kind": "layout", "in": ["qkv"], "out": ["q", "k", "v"]},
            {"id": "attn", "kind": "sdpa", "in": ["q", "k", "v"], "out": ["o"], "causal": False},
            {"id": "merge", "kind": "layout", "in": ["o"], "out": ["o2"]},
            {"id": "out", "kind": "linear", "in": ["o2", "w_o"], "out": ["y"]},
        ],
    }


TASK_SHAPES = {
    "qkv_v0": [
        {"M": 128, "K": 1024, "N": 3072},
        {"M": 1000, "K": 1000, "N": 3000},
        {"M": 1, "K": 64, "N": 64},
    ],
    "block_v0": [
        {"B": 2, "S": 77, "D": 1024, "H": 16},
        {"B": 1, "S": 8, "D": 256, "H": 8},
    ],
}
