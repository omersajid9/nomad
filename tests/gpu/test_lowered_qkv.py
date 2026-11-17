"""I1 acceptance on the RTX 3070: the kernel lowered from examples/qkv_proj.pkir.yaml.

It must match the fp32 reference like the hand-written oracle does and, because it realizes the
same tiles, stages and MMA order, match the oracle bit for bit. If allclose passes but the bitwise
check fails, diff `lower(doc).source` against tests/gpu/qkv_tile_kernel.cu.

    uv run pytest -m gpu tests/gpu/test_lowered_qkv.py
    uv run python tests/gpu/test_lowered_qkv.py   # timing: lowered vs oracle vs torch
"""

from functools import partial
from pathlib import Path

import pytest
import torch
import yaml
from test_qkv_tile import SHAPES, build_ext, make_inputs

from nomad.difftest.harness import time_ms
from nomad.lowering.cuda.tiled_gemm import lower

# test_qkv_tile is the sibling module: pytest (default import mode) and `python tests/gpu/...`
# both put tests/gpu on sys.path.
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qkv_proj.pkir.yaml"
pytestmark = pytest.mark.gpu


def build_lowered():
    return lower(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))).build()


@pytest.fixture(scope="module")
def lowered():
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    return build_lowered()


@pytest.fixture(scope="module")
def oracle():
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    return build_ext()


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_lowered_matches_the_reference_and_the_oracle(lowered, oracle, M, K, N):
    x, w = make_inputs(M, K, N)
    y = lowered.qkv_proj(x, w)
    ref = torch.nn.functional.linear(x.float(), w.float()).half()
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(y, oracle.qkv_proj(x, w)), "lowered kernel diverged from the oracle"


def test_assume_is_checked_on_the_host(lowered):
    with pytest.raises(RuntimeError, match="assume violated: K % 8 == 0"):
        lowered.qkv_proj(*make_inputs(64, 60, 64))


def main():
    lowered, oracle = build_lowered(), build_ext()
    print(torch.cuda.get_device_name(), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
    print(f"{'M':>6} {'K':>6} {'N':>6} {'lowered ms':>11} {'oracle ms':>10} {'torch ms':>9}")
    for M, K, N in SHAPES:
        x, w = make_inputs(M, K, N)
        fns = (lowered.qkv_proj, oracle.qkv_proj, torch.nn.functional.linear)
        ours, theirs, torch_ms = (time_ms(partial(f, x, w)) for f in fns)
        print(f"{M:>6} {K:>6} {N:>6} {ours:>11.3f} {theirs:>10.3f} {torch_ms:>9.3f}")


if __name__ == "__main__":
    main()