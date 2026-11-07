"""Differential tests: a candidate realization against the fp32 oracle over the block's shape table.

A candidate is any callable with the reference signature (x, params, shape) -> y. Each shape is
compared with torch.testing's closeness rule at atol = rtol = 2e-2 (fixed by docs/pkir-spec.md §8)
and yields one ShapeResult. A failure names the worst element, so the agent gets a located
diagnostic instead of a bare "mismatch"; a crash is a result, not a harness exception.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from nomad.reference.block import SHAPES, BlockParams, BlockShape, make_inputs, reference_fp32

ATOL = RTOL = 2e-2

Candidate = Callable[[torch.Tensor, BlockParams, BlockShape], torch.Tensor]


@dataclass(frozen=True)
class ShapeResult:
    shape: str
    ok: bool
    error: str | None = None  # crash, or wrong shape/dtype: no numeric comparison was made
    max_abs_err: float = 0.0
    mismatched: int = 0  # elements with |y - ref| > atol + rtol * |ref|, NaN included
    numel: int = 0
    worst_index: tuple[int, ...] | None = None  # argmax |y - ref|


def compare(
    y: torch.Tensor, ref: torch.Tensor, shape_id: str, *, atol: float = ATOL, rtol: float = RTOL
) -> ShapeResult:
    if y.shape != ref.shape or y.dtype != ref.dtype:
        got, want = f"{tuple(y.shape)} {y.dtype}", f"{tuple(ref.shape)} {ref.dtype}"
        return ShapeResult(shape_id, ok=False, error=f"got {got}, expected {want}")
    if ref.numel() == 0:
        return ShapeResult(shape_id, ok=True)
    y32, r32 = y.float(), ref.float()
    bad = ~torch.isclose(y32, r32, atol=atol, rtol=rtol)
    err = torch.nan_to_num((y32 - r32).abs(), nan=float("inf"))
    worst = torch.unravel_index(err.argmax(), err.shape)
    return ShapeResult(
        shape_id,
        ok=not bool(bad.any()),
        max_abs_err=float(err.max()),
        mismatched=int(bad.sum()),
        numel=err.numel(),
        worst_index=tuple(int(i) for i in worst),
    )


def run(
    candidate: Candidate,
    shapes: Iterable[BlockShape] = SHAPES,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> list[ShapeResult]:
    """Run a candidate over the shape table. A CUDA fault poisons the context for later shapes,
    so the mutant study should call this once per candidate in a fresh subprocess."""
    results = []
    for shape in shapes:
        x, params = make_inputs(shape, dtype=dtype, device=device, seed=seed)
        ref = reference_fp32(x, params, shape)
        try:
            y = candidate(x, params, shape)
            if y.is_cuda:
                torch.cuda.synchronize()  # surface asynchronous kernel faults here
        except Exception as exc:
            results.append(ShapeResult(shape.id, ok=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        results.append(compare(y, ref, shape.id, atol=atol, rtol=rtol))
    return results


def time_ms(fn: Callable[[], object], *, iters: int = 100, warmup: int = 10) -> float:
    """Mean milliseconds per call on the current CUDA stream, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
