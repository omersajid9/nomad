"""torch.export -> canonical graph.yaml (harness-only).

The agent never sees torch. This adapter covers the v0 admitted region: one bias-free linear,
either in the high-level export or after ``run_decompositions`` (``permute/view/mm``), plus the
reference attention block (RMSNorm -> linear -> head split -> SDPA -> merge -> linear). The block
match is structural: it names the canonical graph only when the exported op sequence is the one
``reference.block`` emits, and otherwise asks for a supplied ``graph.yaml`` rather than guessing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from nomad.reference.block import graph_attention_block, graph_qkv_proj


def _load_factory(path: Path):
    spec = importlib.util.spec_from_file_location(f"nomad_export_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import export factory {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build"):
        raise ValueError(f"{path} must define build() -> ExportedProgram")
    return module.build()


def load_export(path: str | Path):
    path = Path(path)
    if path.suffix == ".py":
        return _load_factory(path)
    if path.suffix == ".pt2":
        import torch.export as export

        return export.load(str(path))
    raise ValueError(f"expected a .py factory or a .pt2 archive, got {path.suffix!r}")


def _shape(node) -> tuple[int, ...]:
    value = node.meta.get("val")
    if value is None or not hasattr(value, "shape"):
        raise ValueError(f"node {node.name!r} has no FakeTensor shape metadata")
    return tuple(int(dim) if isinstance(dim, int) else str(dim) for dim in value.shape)


def _is_linear(target: Any) -> bool:
    return str(target).startswith("aten.linear")


def _is_mm(target: Any) -> bool:
    return str(target).startswith("aten.mm")


def _is_permute(target: Any) -> bool:
    return str(target).startswith("aten.permute")


def _is_view(target: Any) -> bool:
    return str(target).startswith("aten.view")


def _flatten_shape(shape: tuple) -> tuple:
    if len(shape) <= 2:
        return shape
    m = 1
    for dim in shape[:-1]:
        if not isinstance(dim, int):
            m = "M"
            break
        m *= dim
    return (m, shape[-1])


def _producer(node):
    return node.args[0] if getattr(node, "args", None) else None


def _dtype(node) -> str:
    value = node.meta.get("val")
    name = str(getattr(value, "dtype", ""))
    if name.endswith("float16"):
        return "f16"
    if name.endswith("float32"):
        return "f32"
    raise ValueError(f"unsupported export dtype {name!r}; export the module in fp16")


def _is_attention_block(nodes) -> bool:
    """Whether the call-function sequence is the reference block's explicit or SDPA form."""
    targets = [str(node.target) for node in nodes if node.op == "call_function"]
    linears = sum(target.startswith("aten.linear") for target in targets)
    if linears != 2:
        return False
    if any("scaled_dot_product_attention" in target for target in targets):
        return True
    return (
        sum(target.startswith("aten.matmul") for target in targets) == 2
        and sum(target.startswith("aten.softmax") for target in targets) == 1
        and sum(target.startswith("aten.mean") for target in targets) == 1
        and sum(target.startswith("aten.rsqrt") for target in targets) == 1
        and sum(target.startswith("aten.chunk") for target in targets) == 1
        and sum(target.startswith("aten.unflatten") for target in targets) == 3
    )


def graph_from_export(ep, name: str = "qkv_proj") -> dict[str, Any]:
    """Convert a supported export to the canonical graph document."""

    nodes = list(ep.graph.nodes)
    linears = [node for node in nodes if node.op == "call_function" and _is_linear(node.target)]
    mms = [node for node in nodes if node.op == "call_function" and _is_mm(node.target)]
    if len(linears) == 1 and not mms:
        node = linears[0]
        args = list(node.args)
        if len(args) >= 3:
            raise ValueError("linear with bias is not supported by the v0 ingest")
        x_node, w_node = args[0], args[1]
        weight = _shape(w_node)
        out_shape = _flatten_shape(_shape(node))
        x_shape = _flatten_shape(_shape(x_node))
        dtype = _dtype(w_node)
        if _dtype(x_node) != "f16" or dtype != "f16":
            raise ValueError("v0 ingest requires an fp16 export")
        if len(weight) != 2 or len(out_shape) != 2 or len(x_shape) != 2:
            raise ValueError("only 2-D linear regions are supported by the v0 ingest")
        n, k = weight
        m = out_shape[0]
        graph = graph_qkv_proj()
        graph["graph"] = name
        graph["tensors"]["x"]["shape"] = [m, k]
        graph["tensors"]["x"]["dtype"] = dtype
        graph["tensors"]["w"]["shape"] = [n, k]
        graph["tensors"]["w"]["dtype"] = dtype
        graph["tensors"]["y"]["shape"] = [m, n]
        graph["tensors"]["y"]["dtype"] = dtype
        graph["ops"][0]["id"] = name
        return graph

    if len(mms) == 1 and not linears:
        node = mms[0]
        left, right = node.args[0], node.args[1]
        # One side is a view of the activation, the other a permute of the weight parameter.
        if _is_permute(right.target):
            w_param = _producer(right)
            x_side = left
        elif _is_permute(left.target):
            w_param = _producer(left)
            x_side = right
        else:
            raise ValueError("decomposed mm without a permuted weight is not supported")
        while _is_view(x_side.target):
            x_side = _producer(x_side)
        weight = _shape(w_param)
        x_shape = _flatten_shape(_shape(x_side))
        out_shape = _flatten_shape(_shape(node))
        dtype = _dtype(w_param)
        if _dtype(x_side) != "f16" or dtype != "f16":
            raise ValueError("v0 ingest requires an fp16 export")
        if len(weight) != 2 or len(x_shape) != 2 or len(out_shape) != 2:
            raise ValueError("only 2-D linear regions are supported by the v0 ingest")
        n, k = weight
        m = out_shape[0]
        graph = graph_qkv_proj()
        graph["graph"] = name
        graph["tensors"]["x"]["shape"] = [m, k]
        graph["tensors"]["x"]["dtype"] = dtype
        graph["tensors"]["w"]["shape"] = [n, k]
        graph["tensors"]["w"]["dtype"] = dtype
        graph["tensors"]["y"]["shape"] = [m, n]
        graph["tensors"]["y"]["dtype"] = dtype
        graph["ops"][0]["id"] = name
        return graph

    if _is_attention_block(nodes):
        graph = graph_attention_block()
        graph["graph"] = name
        return graph

    raise ValueError(
        "v0 torch.export ingest supports one bias-free linear or the reference attention block; "
        "other graphs must be supplied as graph.yaml"
    )
