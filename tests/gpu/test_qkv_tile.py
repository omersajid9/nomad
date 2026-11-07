"""I1 target kernel check: Y = X @ W^T in fp16 against PyTorch, on the RTX 3070.

Run on the GPU machine (needs PyTorch with CUDA, ninja, and an nvcc matching PyTorch's CUDA
version):
    python test_qkv_tile.py
or, under pytest:
    uv run pytest -m gpu tests/gpu
The first run compiles the extension, which takes about a minute. Nothing is compiled at import
time -- CI has no CUDA toolchain and only deselection follows collection.
"""

import os
from functools import partial

import pytest
import torch
from torch.utils.cpp_extension import load

from nomad.difftest.harness import time_ms
from nomad.reference.block import SHAPES as BLOCK_SHAPES

HERE = os.path.dirname(os.path.abspath(__file__))
pytestmark = pytest.mark.gpu

# (M, K, N): M = batch * seq tokens, K = hidden size D, N = 3 * D for a fused QKV projection.
# The first rows come from the reference block's shape table.
SHAPES = [
    *dict.fromkeys(s.qkv_gemm for s in BLOCK_SHAPES),
    # Ragged edges: M and N not multiples of the 64x64 tile, K a multiple of 8 but not of 32.
    (1000, 1000, 3000),
    (77, 264, 200),
    (1, 64, 64),
]


def build_ext():
    """Compile the CUDA extension on demand -- never at import time (CI has no toolchain)."""
    return load(
        name="qkv_tile_ext",
        sources=[os.path.join(HERE, "qkv_tile_kernel.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def ext():
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    return build_ext()


def make_inputs(M, K, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=g) / K**0.5
    return x, w


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_matches_fp32_reference(ext, M, K, N):
    x, w = make_inputs(M, K, N)
    y = ext.qkv_proj(x, w)
    ref = torch.nn.functional.linear(x.float(), w.float()).half()  # fp32 accumulation, rounded once
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


def main():
    ext = build_ext()
    print(torch.cuda.get_device_name(), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
    print(
        f"{'M':>6} {'K':>6} {'N':>6} {'max|err|':>10} {'ours ms':>9} "
        f"{'torch ms':>9} {'ours TF/s':>10} {'torch TF/s':>11}"
    )
    for M, K, N in SHAPES:
        x, w = make_inputs(M, K, N)
        y = ext.qkv_proj(x, w)
        # fp32 accumulation, rounded once
        ref = torch.nn.functional.linear(x.float(), w.float()).half()
        err = (y.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)
        ours = time_ms(partial(ext.qkv_proj, x, w))
        theirs = time_ms(partial(torch.nn.functional.linear, x, w))
        flop = 2 * M * N * K
        print(
            f"{M:>6} {K:>6} {N:>6} {err:>10.2e} {ours:>9.3f} {theirs:>9.3f} "
            f"{flop / (ours * 1e9):>10.1f} {flop / (theirs * 1e9):>11.1f}"
        )
    print("all shapes match the fp32 reference")


if __name__ == "__main__":
    main()
