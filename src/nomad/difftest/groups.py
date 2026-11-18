"""G4: differential-test one lowered group against its PyTorch reference.

v0 covers the `single` linear region because that is the only PKIR v0.1 template. The shape
bindings come from `task.yaml` (`shape_sets`), not from the lowering, so a wider tile or a different
schedule is tested over the same block shapes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nomad.difftest.harness import ShapeResult, compare
from nomad.graph.plan import Group, PlanDoc
from nomad.graph.schema import GraphDoc
from nomad.pkir.expr import try_eval

ATOL = RTOL = 2e-2


class GroupCaseError(ValueError):
    """The group is not covered by the v0 G4 adapter."""


def shape_bindings(task: dict[str, Any]) -> list[dict[str, int]]:
    shapes = task.get("shape_sets", {}).get(task.get("shapes"), [])
    return [{str(k): int(v) for k, v in binding.items()} for binding in shapes]


def _shape(tensor_shape: list[Any], env: dict[str, int]) -> tuple[int, ...]:
    out: list[int] = []
    for part in tensor_shape:
        value = try_eval(str(part), env)
        if value is None:
            raise GroupCaseError(f"cannot evaluate shape component {part!r} with {env}")
        out.append(int(value))
    return tuple(out)


def linear_case(
    graph: GraphDoc, plan: PlanDoc, group: Group, binding: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Return `(x, w, reference, shape_id)` for one single-linear group."""
    if len(group.ops) != 1 or graph.op(group.ops[0]).kind != "linear":
        raise GroupCaseError(
            f"group {group.id!r} is not a single linear; G4 v0 has no adapter for it"
        )
    op = graph.op(group.ops[0])
    if len(op.in_) != 2 or len(op.out_) != 1:
        raise GroupCaseError(f"linear {op.id!r} must have two inputs and one output")
    x_name, w_name = op.in_
    y_name = op.out_[0]
    x_shape = _shape(graph.tensors[x_name].shape, binding)
    w_shape = _shape(graph.tensors[w_name].shape, binding)
    y_shape = _shape(graph.tensors[y_name].shape, binding)
    if len(x_shape) != 2 or len(w_shape) != 2 or len(y_shape) != 2:
        raise GroupCaseError("G4 v0 supports 2-D linear regions only")
    m, k = x_shape
    n, k_w = w_shape
    if k_w != k or y_shape != (m, n):
        raise GroupCaseError(f"linear shapes disagree: x{x_shape}, w{w_shape}, y{y_shape}")
    shape_id = "x".join(str(v) for v in (m, k, n))
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=generator)
    w = torch.randn(n, k, dtype=torch.float16, device="cuda", generator=generator)
    ref = F.linear(x.float(), w.float()).half()
    return x, w, ref, shape_id


def run_group(
    module: Any,
    kernel: str,
    graph: GraphDoc,
    plan: PlanDoc,
    group: Group,
    task: dict[str, Any],
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> list[ShapeResult]:
    """Run a lowered extension function over every bound shape and compare to PyTorch."""
    fn = getattr(module, kernel)
    results: list[ShapeResult] = []
    for binding in shape_bindings(task):
        x, w, ref, shape_id = linear_case(graph, plan, group, binding)
        try:
            y = fn(x, w)
            torch.cuda.synchronize()
        except Exception as exc:
            results.append(ShapeResult(shape_id, ok=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        results.append(compare(y, ref, shape_id, atol=atol, rtol=rtol))
    return results


def time_group(
    module: Any,
    kernel: str,
    graph: GraphDoc,
    plan: PlanDoc,
    group: Group,
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    """Time one lowered function against PyTorch over the shape bindings."""
    from functools import partial

    from nomad.difftest.harness import time_ms

    fn = getattr(module, kernel)
    rows: list[dict[str, Any]] = []
    for binding in shape_bindings(task):
        x, w, _, shape_id = linear_case(graph, plan, group, binding)
        ours = time_ms(partial(fn, x, w))
        theirs = time_ms(partial(F.linear, x, w))
        rows.append(
            {
                "shape": shape_id,
                "kernel_ms": ours,
                "torch_ms": theirs,
                "speedup": theirs / ours if ours > 0 else None,
            }
        )
    return rows
