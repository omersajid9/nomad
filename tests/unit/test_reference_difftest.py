"""CPU checks for the reference block and the differential harness (what CI runs)."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from nomad.difftest.harness import run
from nomad.reference.block import SHAPES, BlockShape, block

TINY = (
    BlockShape(batch=2, seq=8, d_model=32, n_heads=4),
    BlockShape(batch=1, seq=5, d_model=32, n_heads=2, causal=True),
)
run_cpu = partial(run, shapes=TINY, device="cpu", dtype=torch.float32)


def test_block_shapes_line_up_with_the_qkv_gpu_table() -> None:
    assert [s.qkv_gemm for s in SHAPES[:4]] == [
        (128, 1024, 3072),
        (1024, 1024, 3072),
        (4096, 1024, 3072),
        (2048, 2048, 6144),
    ]


@pytest.mark.parametrize("attention", ["explicit", "sdpa"])
def test_both_attention_paths_match_the_oracle(attention: str) -> None:
    assert all(r.ok for r in run_cpu(partial(block, attention=attention)))


def test_a_dropped_causal_mask_fails_only_the_causal_shape() -> None:
    results = run_cpu(lambda x, p, s: block(x, p, replace(s, causal=False)))
    assert [r.ok for r in results] == [True, False]


def test_a_ragged_edge_bug_is_located_at_the_last_token() -> None:
    def drops_last_token(x, p, shape):  # the skipped-tile / OOB-tile flavor of mutant
        y = block(x, p, shape)
        y[:, -1] = 0
        return y

    for shape, r in zip(TINY, run_cpu(drops_last_token), strict=True):
        assert not r.ok and r.worst_index[1] == shape.seq - 1


def test_a_crash_is_reported_not_raised() -> None:
    def crashes(x, p, shape):
        raise RuntimeError("illegal memory access")

    assert all(not r.ok and "illegal memory access" in r.error for r in run_cpu(crashes))
