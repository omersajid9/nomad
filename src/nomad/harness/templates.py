"""Scaffold PKIR documents for plan groups.

A ``single`` linear group gets the v0 GEMM template from ``examples/qkv_proj.pkir.yaml`` with the
group's boundary tensor names. Other groups get a schema-valid skeleton; G1 then reports
``S-UNSUPPORTED`` until PKIR v0.2 can express them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nomad.graph.plan import Group, PlanDoc
from nomad.graph.schema import GraphDoc
from nomad.graph.static import group_inputs, group_outputs
from nomad.pkir.models import PkirDoc


def _names(graph: GraphDoc, plan: PlanDoc, group: Group) -> tuple[str, str, str]:
    inputs = group_inputs(graph, plan, group)
    outputs = group_outputs(graph, plan, group)
    activation = inputs[0] if inputs else "x"
    weight = inputs[1] if len(inputs) > 1 else "w"
    out = outputs[0] if outputs else "y"
    return activation, weight, out


def _gemm_doc(activation: str, weight: str, out: str, kernel: str, device: str) -> dict[str, Any]:
    return {
        "pkir": 0,
        "kernel": kernel,
        "params": ["M", "N", "K"],
        "assume": ["K % 8 == 0"],
        "tensors": {
            activation: {"shape": ["M", "K"], "dtype": "f16"},
            weight: {"shape": ["N", "K"], "dtype": "f16"},
            out: {"shape": ["M", "N"], "dtype": "f16", "out": True},
        },
        "tile": {"m": 64, "n": 64, "k": 32},
        "grid": {"bm": "ceil_div(M, tile.m)", "bn": "ceil_div(N, tile.n)"},
        "roles": {"load": {"kind": "move"}, "mma": {"kind": "matrix"}},
        "buffers": {
            "A": {
                "tier": "shared",
                "shape": ["tile.m", "tile.k"],
                "dtype": "f16",
                "stages": 2,
                "stride": ["tile.k + 8", 1],
            },
            "B": {
                "tier": "shared",
                "shape": ["tile.n", "tile.k"],
                "dtype": "f16",
                "stages": 2,
                "stride": ["tile.k + 8", 1],
            },
            "acc": {"tier": "register", "shape": ["tile.m", "tile.n"], "dtype": "f32"},
            "C": {
                "tier": "shared",
                "shape": ["tile.m", "tile.n"],
                "dtype": "f32",
                "stride": ["tile.n + 4", 1],
            },
        },
        "placement": {
            device: {
                "warps": 4,
                "specialize": False,
                "warp_tile": {"mma": ["tile.m / 2", "tile.n / 2"]},
            }
        },
        "body": [
            {"fill": {"role": "mma", "dst": "acc", "value": 0.0}},
            {
                "for": {
                    "var": "kt",
                    "trips": "ceil_div(K, tile.k)",
                    "body": [
                        {
                            "copy": {
                                "role": "load",
                                "src": (
                                    f"{activation}[bm*tile.m : bm*tile.m + tile.m, "
                                    "kt*tile.k : kt*tile.k + tile.k]"
                                ),
                                "dst": "A[kt]",
                                "pad": "zero",
                                "async": True,
                                "wait": ["A.free"],
                                "signal": ["A.full"],
                            }
                        },
                        {
                            "copy": {
                                "role": "load",
                                "src": (
                                    f"{weight}[bn*tile.n : bn*tile.n + tile.n, "
                                    "kt*tile.k : kt*tile.k + tile.k]"
                                ),
                                "dst": "B[kt]",
                                "pad": "zero",
                                "async": True,
                                "wait": ["B.free"],
                                "signal": ["B.full"],
                            }
                        },
                        {
                            "matmul": {
                                "role": "mma",
                                "acc": "acc",
                                "a": "A[kt]",
                                "b": "B[kt]",
                                "transpose_b": True,
                                "wait": ["A.full", "B.full"],
                                "signal": ["A.free", "B.free"],
                            }
                        },
                    ],
                }
            },
            {"copy": {"role": "mma", "src": "acc", "dst": "C", "signal": ["C.full"]}},
            {
                "copy": {
                    "role": "load",
                    "src": "C",
                    "dst": f"{out}[bm*tile.m : bm*tile.m + tile.m, bn*tile.n : bn*tile.n + tile.n]",
                    "cast": "f16",
                    "mask": True,
                    "wait": ["C.full"],
                }
            },
        ],
    }


def _skeleton_doc(kernel: str, device: str) -> dict[str, Any]:
    return {
        "pkir": 0,
        "kernel": kernel,
        "params": ["M", "N", "K"],
        "assume": [],
        "tensors": {},
        "tile": {"m": 64, "n": 64, "k": 32},
        "grid": {"bm": "ceil_div(M, tile.m)", "bn": "ceil_div(N, tile.n)"},
        "roles": {},
        "buffers": {},
        "placement": {
            device: {
                "warps": 4,
                "specialize": False,
                "warp_tile": {"mma": [32, 32]},
            }
        },
        "body": [],
    }


def kernel_doc_for_group(
    graph: GraphDoc, plan: PlanDoc, group: Group, device: str, *, template: bool = True
) -> dict[str, Any]:
    kinds = [graph.op(op_id).kind for op_id in group.ops]
    if template and group.pattern == "single" and kinds == ["linear"]:
        activation, weight, out = _names(graph, plan, group)
        return _gemm_doc(activation, weight, out, group.id, device)
    return _skeleton_doc(group.id, device)


def scaffold_group(
    run_dir: str | Path,
    graph: GraphDoc,
    plan: PlanDoc,
    group: Group,
    device: str,
    *,
    template: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    doc = kernel_doc_for_group(graph, plan, group, device, template=template)
    PkirDoc.model_validate(doc)
    path = run_dir / group.pkir
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def scaffold_all(
    run_dir: str | Path, graph: GraphDoc, plan: PlanDoc, device: str, *, template: bool = True
) -> list[Path]:
    return [
        scaffold_group(run_dir, graph, plan, group, device, template=template)
        for group in plan.groups
    ]
