"""G1 PKIR checks: capabilities, slices, placement and capacity (agent-contract §6).

Structural checks only. Slot discipline, equivalence and timing belong to G2/G4/G5.
Every failure carries a file, line, JSON Pointer and group.
"""

from __future__ import annotations

from typing import Any

from nomad.devicespec.spec import DeviceSpec
from nomad.diagnostics import Diagnostic
from nomad.graph.plan import Group, PlanDoc
from nomad.graph.schema import GraphDoc
from nomad.graph.static import group_inputs, group_outputs
from nomad.pkir.expr import ExprError, names, parse_slice, try_eval
from nomad.pkir.models import Buffer, CopyBody, ForBody, PkirDoc, Step
from nomad.yaml_lines import LineMap, loc_line, loc_pointer

_ELEM_BYTES = {"f16": 2, "f32": 4}
_EXPR_CALLS = {"ceil", "ceil_div", "min", "max"}


class _Check:
    def __init__(
        self,
        doc: PkirDoc,
        device: DeviceSpec,
        display: str,
        lines: LineMap,
        group: str,
    ) -> None:
        self.doc = doc
        self.device = device
        self.display = display
        self.lines = lines
        self.group = group
        self.params = set(doc.params)
        self.tile = {"tile.m": doc.tile.m, "tile.n": doc.tile.n, "tile.k": doc.tile.k}
        self.diags: list[Diagnostic] = []

    def add(
        self,
        code: str,
        message: str,
        path: tuple[Any, ...],
        hint: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        self.diags.append(
            Diagnostic(
                code=code,
                message=message,
                file=self.display,
                line=loc_line(self.lines, path),
                pointer=loc_pointer(path),
                group=self.group,
                hint=hint,
                where=where,
            )
        )

    def expr(
        self, expr: str, allowed: set[str], path: tuple[Any, ...], code: str = "S-CAPABILITY"
    ) -> None:
        try:
            used = names(expr)
        except ExprError as exc:
            self.add(code, str(exc), path)
            return
        unknown = sorted(used - allowed - _EXPR_CALLS)
        if unknown:
            self.add(code, f"expression {expr!r} uses undeclared name(s) {unknown}", path)


def _iter_steps(body: list[Step], path: tuple[Any, ...], loop_vars: set[str]):
    for index, step in enumerate(body):
        here = path + (index,)
        yield step, here, loop_vars
        if hasattr(step, "for_"):
            for_body: ForBody = step.for_
            yield from _iter_steps(
                for_body.body, here + ("for", "body"), loop_vars | {for_body.var}
            )


def _buffer_bytes(buffer: Buffer, tile: dict[str, int]) -> int | None:
    rows = try_eval(str(buffer.shape[0]), tile)
    width = (
        try_eval(str(buffer.stride[0]), tile)
        if buffer.stride
        else try_eval(str(buffer.shape[1]), tile)
        if len(buffer.shape) > 1
        else 1
    )
    if rows is None or width is None:
        return None
    return int(rows) * int(width) * _ELEM_BYTES[buffer.dtype] * (buffer.stages or 1)


def _check_placement(chk: _Check) -> None:
    keys = set(chk.doc.placement)
    if keys != {chk.device.name}:
        chk.add(
            "S-PLACEMENT",
            f"placement must be keyed by the target device {chk.device.name!r}; "
            f"found {sorted(keys)}",
            ("placement",),
        )
        return
    placement = chk.doc.placement[chk.device.name]
    macro, runtime = chk.doc, chk.device.runtime
    if placement.warps > runtime.warps_per_block_max:
        chk.add(
            "S-PLACEMENT",
            f"warps {placement.warps} exceeds runtime.warps_per_block_max "
            f"{runtime.warps_per_block_max}",
            ("placement", chk.device.name, "warps"),
        )
    threads = placement.warps * runtime.threads_per_warp
    if threads > runtime.threads_per_block_max:
        chk.add(
            "S-PLACEMENT",
            f"{placement.warps} warps implies {threads} threads, over "
            f"runtime.threads_per_block_max {runtime.threads_per_block_max}",
            ("placement", chk.device.name, "warps"),
        )
    mma = placement.warp_tile.mma
    if len(mma) != 2:
        chk.add(
            "S-PLACEMENT",
            "warp_tile.mma must have exactly two extents",
            ("placement", chk.device.name, "warp_tile", "mma"),
        )
    for axis, expr in enumerate(mma):
        value = try_eval(str(expr), chk.tile)
        extent = (macro.tile.m, macro.tile.n)[axis] if axis < 2 else macro.tile.k
        path = ("placement", chk.device.name, "warp_tile", "mma", axis)
        if value is None:
            chk.add(
                "S-PLACEMENT", f"warp_tile.mma[{axis}] {expr!r} is not numeric in tile sizes", path
            )
        elif value <= 0 or extent % int(value) != 0:
            chk.add(
                "S-PLACEMENT",
                f"warp_tile.mma[{axis}]={int(value)} does not divide the tile extent {extent}",
                path,
            )


def _check_capacity(chk: _Check) -> None:
    shared = register = 0
    for buffer in chk.doc.buffers.values():
        size = _buffer_bytes(buffer, chk.tile)
        if size is None:
            continue
        if buffer.tier == "shared":
            shared += size
        else:
            register += size
    shared_limit = chk.device.memory_tiers["shared"].max_per_block_bytes
    if shared_limit is not None and shared > shared_limit:
        chk.add(
            "S-CAPACITY",
            f"shared footprint {shared} B exceeds "
            f"memory_tiers.shared.max_per_block_bytes {shared_limit} B",
            ("buffers",),
            where={"shared_bytes": shared, "limit": shared_limit},
        )
    register_limit = chk.device.memory_tiers["register"].capacity_bytes
    if register > register_limit:
        chk.add(
            "S-CAPACITY",
            f"register footprint {register} B exceeds "
            f"memory_tiers.register.capacity_bytes {register_limit} B",
            ("buffers",),
            where={"register_bytes": register, "limit": register_limit},
        )


def _check_slice_ref(
    chk: _Check,
    ref: str,
    path: tuple[Any, ...],
    *,
    allow_overrun: bool,
) -> None:
    if "[" not in ref:
        if ref.strip() not in chk.doc.tensors and ref.strip() not in chk.doc.buffers:
            chk.add("S-SLICE", f"undeclared tensor or buffer {ref.strip()!r}", path)
        return
    try:
        part = parse_slice(ref)
    except ExprError as exc:
        chk.add("S-SLICE", str(exc), path)
        return
    if part.name not in chk.doc.tensors and part.name not in chk.doc.buffers:
        chk.add("S-SLICE", f"undeclared tensor or buffer {part.name!r}", path)
        return
    if part.staged_index:
        buffer = chk.doc.buffers.get(part.name)
        if buffer is None or buffer.stages is None:
            chk.add(
                "S-SLICE",
                f"{part.name!r} is indexed as a staged buffer but is not declared with stages",
                path,
            )
        return
    if part.name in chk.doc.buffers and chk.doc.buffers[part.name].stages is not None:
        chk.add(
            "S-SLICE",
            f"staged buffer {part.name!r} must be addressed as {part.name}[trip]",
            path,
        )
    shape = (
        chk.doc.tensors[part.name].shape
        if part.name in chk.doc.tensors
        else chk.doc.buffers[part.name].shape
    )
    extents = [try_eval(str(extent), {}) for extent in shape]
    for axis, (lo_expr, hi_expr) in enumerate(part.ranges):
        if hi_expr is None:
            chk.add(
                "S-SLICE",
                f"unbounded slice {part.name}[...] on axis {axis}; give an explicit hi bound",
                path,
            )
            continue
        lo, hi = try_eval(lo_expr, {}), try_eval(hi_expr, {})
        if lo is not None and hi is not None and hi < lo:
            chk.add("S-SLICE", f"empty range {lo_expr}:{hi_expr}", path)
        extent = extents[axis] if axis < len(extents) else None
        if extent is not None and hi is not None and hi > extent and not allow_overrun:
            chk.add(
                "S-SLICE",
                f"range {lo_expr}:{hi_expr} exceeds extent {extent} on axis {axis}",
                path,
            )


def _ref_dtype(chk: _Check, ref: str) -> str | None:
    name = ref.strip()
    if "[" in name:
        try:
            name = parse_slice(name).name
        except ExprError:
            return None
    if name in chk.doc.buffers:
        return chk.doc.buffers[name].dtype
    if name in chk.doc.tensors:
        return chk.doc.tensors[name].dtype
    return None


def _check_slot(chk: _Check, slot: str, path: tuple[Any, ...]) -> None:
    if "." not in slot:
        chk.add("S-CAPABILITY", f"slot {slot!r} must be <buffer>.<full|free>", path)
        return
    buffer_name, state = slot.rsplit(".", 1)
    if buffer_name not in chk.doc.buffers:
        chk.add("S-CAPABILITY", f"slot names unknown buffer {buffer_name!r}", path)
    elif state not in {"full", "free"}:
        chk.add("S-CAPABILITY", f"slot state {state!r} must be full or free", path)


def _check_steps(chk: _Check) -> None:
    doc = chk.doc
    base = chk.params | {"tile.m", "tile.n", "tile.k", "bm", "bn"}
    for step, path, loop_vars in _iter_steps(doc.body, ("body",), set()):
        allowed = base | loop_vars
        if hasattr(step, "fill"):
            fill = step.fill
            if fill.role not in doc.roles:
                chk.add("S-CAPABILITY", f"unknown role {fill.role!r}", path + ("fill", "role"))
            if fill.dst not in doc.buffers:
                chk.add(
                    "S-CAPABILITY",
                    f"fill destination {fill.dst!r} is not a buffer",
                    path + ("fill", "dst"),
                )
        if hasattr(step, "copy_"):
            copy: CopyBody = step.copy_
            if copy.role not in doc.roles:
                chk.add("S-CAPABILITY", f"unknown role {copy.role!r}", path + ("copy", "role"))
            _check_slice_ref(
                chk, copy.src, path + ("copy", "src"), allow_overrun=copy.pad == "zero"
            )
            _check_slice_ref(chk, copy.dst, path + ("copy", "dst"), allow_overrun=bool(copy.mask))
            if copy.async_ and not chk.device.copy_engines.async_:
                chk.add(
                    "S-CAPABILITY",
                    "async copy is not available on this device",
                    path + ("copy", "async"),
                )
            if copy.cast == "f16" and ["f32", "f16"] not in chk.device.capabilities.convert.pairs:
                chk.add(
                    "S-CAPABILITY",
                    "f32 -> f16 cast is not available on this device",
                    path + ("copy", "cast"),
                )
            if copy.pad == "zero" and not chk.device.copy_engines.zero_fill_on_guard_false:
                chk.add(
                    "S-CAPABILITY",
                    "zero fill on an out-of-extent copy is not supported",
                    path + ("copy", "pad"),
                )
            src_dtype, dst_dtype = _ref_dtype(chk, copy.src), _ref_dtype(chk, copy.dst)
            if src_dtype and dst_dtype and src_dtype != dst_dtype and copy.cast != dst_dtype:
                chk.add(
                    "S-CAPABILITY",
                    f"copy converts {src_dtype} to {dst_dtype} without cast: {dst_dtype}",
                    path + ("copy", "cast"),
                )
            for index, slot in enumerate((copy.wait or []) + (copy.signal or [])):
                _check_slot(chk, slot, path + ("copy", index))
        if hasattr(step, "matmul"):
            mm = step.matmul
            if mm.role not in doc.roles:
                chk.add("S-CAPABILITY", f"unknown role {mm.role!r}", path + ("matmul", "role"))
            acc = doc.buffers.get(mm.acc)
            if acc is None or acc.tier != "register":
                chk.add(
                    "S-CAPABILITY",
                    f"matmul accumulator {mm.acc!r} must be a register buffer",
                    path + ("matmul", "acc"),
                )
            elif acc.dtype != "f32":
                chk.add(
                    "S-CAPABILITY",
                    f"matmul accumulator {mm.acc!r} must be f32",
                    path + ("matmul", "acc"),
                )
            for field, name in (("a", mm.a), ("b", mm.b)):
                ref = name
                if "[" in name:
                    try:
                        ref = parse_slice(name).name
                    except ExprError as exc:
                        chk.add("S-SLICE", str(exc), path + ("matmul", field))
                        continue
                dtype = (
                    doc.buffers[ref].dtype
                    if ref in doc.buffers
                    else doc.tensors[ref].dtype
                    if ref in doc.tensors
                    else None
                )
                if dtype is None:
                    chk.add(
                        "S-CAPABILITY",
                        f"matmul input {name!r} is not declared",
                        path + ("matmul", field),
                    )
                elif dtype != "f16":
                    chk.add(
                        "S-CAPABILITY",
                        f"matmul input {name!r} must be f16",
                        path + ("matmul", field),
                    )
            for index, slot in enumerate((mm.wait or []) + (mm.signal or [])):
                _check_slot(chk, slot, path + ("matmul", index))
        if hasattr(step, "for_"):
            for_body: ForBody = step.for_
            chk.expr(for_body.trips, allowed | {for_body.var}, path + ("for", "trips"))
            if for_body.var in base:
                chk.add(
                    "S-CAPABILITY",
                    f"loop variable {for_body.var!r} shadows an existing name",
                    path + ("for", "var"),
                )


def _check_boundary(chk: _Check, group: Group, graph: GraphDoc, plan: PlanDoc) -> None:
    expected_in = group_inputs(graph, plan, group)
    expected_out = group_outputs(graph, plan, group)
    expected = set(expected_in) | set(expected_out)
    actual = set(chk.doc.tensors)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing:
        chk.add(
            "S-CAPABILITY",
            f"PKIR tensors are missing group boundary tensor(s) {missing}",
            ("tensors",),
        )
    if extra:
        chk.add(
            "S-CAPABILITY",
            f"PKIR tensors include non-boundary tensor(s) {extra}; "
            "internal data belongs in buffers",
            ("tensors",),
        )
    out_names = {name for name, tensor in chk.doc.tensors.items() if tensor.out}
    if out_names != set(expected_out):
        chk.add(
            "S-CAPABILITY",
            f"out: true tensors {sorted(out_names)} do not match group outputs "
            f"{sorted(expected_out)}",
            ("tensors",),
        )


def check_pkir(
    doc: PkirDoc,
    device: DeviceSpec,
    group: Group,
    graph: GraphDoc,
    plan: PlanDoc,
    display: str,
    lines: LineMap,
) -> list[Diagnostic]:
    chk = _Check(doc, device, display, lines, group.id)
    if doc.kernel != group.id:
        chk.add(
            "S-CAPABILITY",
            f"kernel {doc.kernel!r} does not match group id {group.id!r}",
            ("kernel",),
        )
    _check_placement(chk)
    _check_capacity(chk)
    _check_steps(chk)
    _check_boundary(chk, group, graph, plan)

    shape_names = chk.params | {"tile.m", "tile.n", "tile.k"}
    for name, tensor in doc.tensors.items():
        for axis, part in enumerate(tensor.shape):
            if isinstance(part, str):
                chk.expr(part, shape_names, ("tensors", name, "shape", axis))
    for name, buffer in doc.buffers.items():
        for axis, part in enumerate(buffer.shape):
            if isinstance(part, str):
                chk.expr(part, shape_names, ("buffers", name, "shape", axis))
        for axis, part in enumerate(buffer.stride or []):
            if isinstance(part, str):
                chk.expr(part, shape_names, ("buffers", name, "stride", axis))
    for index, assumption in enumerate(doc.assume):
        chk.expr(assumption, chk.params, ("assume", index))
    for key in ("bm", "bn"):
        chk.expr(getattr(doc.grid, key), chk.params | set(chk.tile) | {"bm", "bn"}, ("grid", key))
    return chk.diags
