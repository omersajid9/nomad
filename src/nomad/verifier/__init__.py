"""G2 verifier v0: slot discipline for the tiled-GEMM template.

This is deliberately a structural verifier, not the symbolic-equivalence core. It checks the
publication/consumption edges the lowering's template relies on: a staged operand copy must wait
for its `free` slot and publish `full`; the matmul must consume `full` and release `free`; the
epilogue must publish `C.full` before the store consumes it. Missing pad/mask semantics and
numeric equivalence are G4's job, because only a reference comparison can decide them.

The checks are phrased in PKIR names (`A.full`, `C.full`), never CUDA instructions, so the same
diagnostics serve every backend.
"""

from __future__ import annotations

from typing import Any

from nomad.diagnostics import Diagnostic
from nomad.graph.plan import Group
from nomad.pkir.models import PkirDoc, Step
from nomad.yaml_lines import LineMap, loc_line, loc_pointer


def _iter_steps(body: list[Step], base: tuple[Any, ...]):
    for index, step in enumerate(body):
        here = base + (index,)
        yield step, here
        if hasattr(step, "for_"):
            yield from _iter_steps(step.for_.body, here + ("for", "body"))


def _buffer(name: str) -> str:
    return name.strip().split("[", 1)[0].strip()


def _slots(value: Any) -> set[str]:
    return set(value or [])


def _diag(
    code: str, message: str, display: str, lines: LineMap, path: tuple[Any, ...], group: str | Group
):
    return Diagnostic(
        code=code,
        message=message,
        file=display,
        line=loc_line(lines, path),
        pointer=loc_pointer(path),
        group=group.id if isinstance(group, Group) else group,
    )


def _check_slots(
    slots: set[str],
    buffers: dict[str, Any],
    display: str,
    lines: LineMap,
    path: tuple[Any, ...],
    group: str,
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    for slot in sorted(slots):
        name, _, edge = slot.partition(".")
        if name not in buffers:
            diags.append(
                _diag(
                    "V-OOB",
                    f"slot {slot!r} names unknown buffer {name!r}",
                    display,
                    lines,
                    path,
                    group,
                )
            )
        elif edge not in {"full", "free"}:
            diags.append(
                _diag(
                    "V-RACE",
                    f"slot {slot!r} must end in .full or .free",
                    display,
                    lines,
                    path,
                    group,
                )
            )
    return diags


def verify_region(
    doc: PkirDoc,
    group: Group,
    display: str,
    lines: LineMap,
) -> list[Diagnostic]:
    """Return G2 diagnostics for one region, empty when its declared edges are complete."""
    diags: list[Diagnostic] = []
    buffers = doc.buffers
    steps = list(_iter_steps(doc.body, ("body",)))

    for step, path in steps:
        if hasattr(step, "copy_"):
            copy = step.copy_
            src_buf, dst_buf = _buffer(copy.src), _buffer(copy.dst)
            waits, signals = _slots(copy.wait), _slots(copy.signal)
            for edge, slots in (("wait", waits), ("signal", signals)):
                diags.extend(
                    _check_slots(slots, buffers, display, lines, path + ("copy", edge), group)
                )
            src = buffers.get(src_buf)
            dst = buffers.get(dst_buf)
            staged = bool(
                (src and src.stages and src.stages > 1) or (dst and dst.stages and dst.stages > 1)
            )
            if copy.async_ and dst is not None:
                full = f"{dst_buf}.full"
                if full not in signals:
                    diags.append(
                        _diag(
                            "V-RACE",
                            f"async copy into {dst_buf!r} must signal {full!r}",
                            display,
                            lines,
                            path + ("copy", "signal"),
                            group,
                        )
                    )
            if staged and dst is not None:
                free = f"{dst_buf}.free"
                if free not in waits:
                    diags.append(
                        _diag(
                            "V-RACE",
                            f"staged copy into {dst_buf!r} must wait for {free!r}",
                            display,
                            lines,
                            path + ("copy", "wait"),
                            group,
                        )
                    )
            # Epilogue: shared/register -> global must consume the epilogue publication.
            if src is not None and dst is None and src.tier == "shared":
                full = f"{src_buf}.full"
                if full not in waits:
                    diags.append(
                        _diag(
                            "V-RACE",
                            f"store from {src_buf!r} must wait for {full!r}",
                            display,
                            lines,
                            path + ("copy", "wait"),
                            group,
                        )
                    )
            if src is not None and dst is not None and src.tier == "register":
                full = f"{dst_buf}.full"
                if full not in signals:
                    diags.append(
                        _diag(
                            "V-RACE",
                            f"register->shared copy into {dst_buf!r} must signal {full!r}",
                            display,
                            lines,
                            path + ("copy", "signal"),
                            group,
                        )
                    )
        elif hasattr(step, "matmul"):
            mm = step.matmul
            waits, signals = _slots(mm.wait), _slots(mm.signal)
            for edge, slots in (("wait", waits), ("signal", signals)):
                diags.extend(
                    _check_slots(slots, buffers, display, lines, path + ("matmul", edge), group)
                )
            for operand in (mm.a, mm.b):
                name = _buffer(operand)
                if name not in buffers:
                    diags.append(
                        _diag(
                            "V-OOB",
                            f"matmul operand {operand!r} names unknown buffer {name!r}",
                            display,
                            lines,
                            path + ("matmul", "a" if operand == mm.a else "b"),
                            group,
                        )
                    )
                    continue
                if f"{name}.full" not in waits:
                    diags.append(
                        _diag(
                            "V-RACE",
                            f"matmul must wait for {name}.full",
                            display,
                            lines,
                            path + ("matmul", "wait"),
                            group,
                        )
                    )
                if f"{name}.free" not in signals:
                    diags.append(
                        _diag(
                            "V-RACE",
                            f"matmul must release {name}.free",
                            display,
                            lines,
                            path + ("matmul", "signal"),
                            group,
                        )
                    )
    return diags
