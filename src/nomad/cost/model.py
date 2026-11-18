"""Analytic cost estimate for ``nomad cost`` (pkir-spec §6; G5 is not implemented)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nomad.devicespec.spec import load_device_from_path
from nomad.graph.catalog import load_catalog
from nomad.graph.plan import PlanDoc
from nomad.graph.schema import GraphDoc
from nomad.graph.static import is_supported_group
from nomad.harness.workspace import read_task
from nomad.pkir.expr import try_eval
from nomad.pkir.models import Buffer, parse_pkir
from nomad.yaml_lines import load_yaml

_ELEM_BYTES = {"f16": 2, "f32": 4}


def _shape(shape: list[Any], env: dict[str, int]) -> list[int] | None:
    out: list[int] = []
    for part in shape:
        if isinstance(part, int):
            out.append(part)
            continue
        value = try_eval(str(part), env)
        if value is None:
            return None
        out.append(int(value))
    return out


def _footprint(buffer: Buffer, tile: dict[str, int]) -> int | None:
    if not buffer.shape:
        return None
    rows = try_eval(str(buffer.shape[0]), tile)
    if rows is None:
        return None
    if buffer.stride:
        width = try_eval(str(buffer.stride[0]), tile)
    elif len(buffer.shape) > 1:
        width = try_eval(str(buffer.shape[1]), tile)
    else:
        width = 1
    if width is None:
        return None
    return int(rows) * int(width) * _ELEM_BYTES[buffer.dtype] * (buffer.stages or 1)


def estimate(run_dir: str | Path, group_id: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    task = read_task(run_dir)
    graph_data, _ = load_yaml(run_dir / "graph.yaml")
    graph = GraphDoc.model_validate(graph_data)
    plan_data, _ = load_yaml(run_dir / "plan.yaml")
    plan = PlanDoc.model_validate(plan_data)
    catalog = load_catalog(run_dir / "catalog.yaml")
    device = load_device_from_path(run_dir / "devicespec.yaml")

    shape_name = task.get("shapes")
    shape_sets = task.get("shape_sets") or {}
    bindings = (shape_sets.get(shape_name) or [{}])[0]
    env = {k: int(v) for k, v in bindings.items()}

    shared_capacity = device.memory_tiers["shared"].capacity_bytes
    warp_capacity = device.runtime.warps_per_sm_max or 48
    groups: list[dict[str, Any]] = []
    for group in plan.groups:
        if group_id is not None and group.id != group_id:
            continue
        if not is_supported_group(graph, catalog, group):
            groups.append({"group": group.id, "status": "unsupported"})
            continue
        doc, diags = parse_pkir(run_dir / group.pkir, display=group.pkir)
        if doc is None:
            groups.append({"group": group.id, "status": "unparsed", "diagnostics": diags})
            continue
        op = graph.op(group.ops[0])
        act_shape = _shape(graph.tensors[op.in_[0]].shape, env)
        weight_shape = _shape(graph.tensors[op.in_[1]].shape, env)
        if (
            act_shape is None
            or weight_shape is None
            or len(act_shape) < 2
            or len(weight_shape) != 2
        ):
            groups.append({"group": group.id, "status": "symbolic"})
            continue
        m, k = act_shape[0], act_shape[-1]
        n, wk = weight_shape
        if wk != k:
            groups.append({"group": group.id, "status": "shape-mismatch"})
            continue
        tile = {"tile.m": doc.tile.m, "tile.n": doc.tile.n, "tile.k": doc.tile.k}
        k_tiles = (k + doc.tile.k - 1) // doc.tile.k
        flops = 2 * m * n * k
        bytes_per_block = (doc.tile.m + doc.tile.n) * doc.tile.k * 2 * k_tiles
        bytes_per_block += doc.tile.m * doc.tile.n * 2
        shared = 0
        register = 0
        for buffer in doc.buffers.values():
            size = _footprint(buffer, tile)
            if size is None:
                continue
            if buffer.tier == "shared":
                shared += size
            else:
                register += size
        warps = doc.placement[device.name].warps
        occupancy = min(
            shared_capacity // shared if shared else 10**9,
            (device.runtime.warps_per_sm_max or warp_capacity) // warps,
        )
        groups.append(
            {
                "group": group.id,
                "status": "ok",
                "kernel": doc.kernel,
                "shape": {"M": m, "N": n, "K": k},
                "tile": {"m": doc.tile.m, "n": doc.tile.n, "k": doc.tile.k},
                "flops": flops,
                "bytes_per_block": bytes_per_block,
                "shared_bytes": shared,
                "register_bytes": register,
                "occupancy_hint": occupancy,
                "estimated": True,
            }
        )

    if group_id is not None and not groups:
        raise KeyError(f"unknown group {group_id!r}")
    return {
        "device": device.name,
        "model": device.model,
        "shapes": shape_name,
        "bindings": env,
        "groups": groups,
        "estimated": True,
    }
