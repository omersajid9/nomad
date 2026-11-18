"""PKIR v0.1 document models (docs/pkir-spec.md §3) and YAML loading with located errors."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nomad.diagnostics import Diagnostic
from nomad.yaml_lines import YamlLoadError, load_yaml, loc_line, loc_pointer, normalize_loc

Expr = str | int


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Tensor(_Model):
    shape: list[Expr]
    dtype: Literal["f16", "f32"]
    out: bool = False


class Tile(_Model):
    m: int = Field(gt=0)
    n: int = Field(gt=0)
    k: int = Field(gt=0)


class Grid(_Model):
    bm: str
    bn: str


class Role(_Model):
    kind: Literal["move", "matrix"]


class WarpTile(_Model):
    mma: list[Expr]


class Placement(_Model):
    warps: int = Field(gt=0)
    specialize: bool = False
    warp_tile: WarpTile


class Buffer(_Model):
    tier: Literal["shared", "register"]
    shape: list[Expr]
    dtype: Literal["f16", "f32"]
    stages: int | None = Field(default=None, gt=0)
    stride: list[Expr] | None = None


class FillBody(_Model):
    role: str
    dst: str
    value: float


class CopyBody(_Model):
    role: str
    src: str
    dst: str
    pad: Literal["zero"] | None = None
    async_: bool | None = Field(default=None, alias="async")
    cast: Literal["f16"] | None = None
    mask: bool | None = None
    wait: list[str] | None = None
    signal: list[str] | None = None


class MatmulBody(_Model):
    role: str
    acc: str
    a: str
    b: str
    transpose_b: bool = False
    wait: list[str] | None = None
    signal: list[str] | None = None


class FillStep(_Model):
    fill: FillBody


class CopyStep(_Model):
    copy_: CopyBody = Field(alias="copy")


class MatmulStep(_Model):
    matmul: MatmulBody


class ForBody(_Model):
    var: str
    trips: str
    body: list[Step]


class ForStep(_Model):
    for_: ForBody = Field(alias="for")


Step = FillStep | ForStep | CopyStep | MatmulStep

ForBody.model_rebuild()


class PkirDoc(_Model):
    pkir: Literal[0]
    kernel: str
    params: list[str]
    assume: list[str] = Field(default_factory=list)
    tensors: dict[str, Tensor]
    tile: Tile
    grid: Grid
    roles: dict[str, Role]
    buffers: dict[str, Buffer]
    placement: dict[str, Placement]
    body: list[Step]


PkirDoc.model_rebuild()


_STEP_CLASSES = {"FillStep": "fill", "ForStep": "for", "CopyStep": "copy", "MatmulStep": "matmul"}


def _lookup(data, path: tuple) -> object:
    value = data
    for part in path:
        try:
            value = value[part]
        except (KeyError, IndexError, TypeError):
            return None
    return value


def _filter_union_errors(errors: list[dict], data) -> list[dict]:
    """Pydantic's smart union reports every branch; keep only the step's own branch.

    Step unions are nested (a ``for`` body contains steps), so walk the loc once, dropping
    branch class names and rejecting an error whose branch does not match the actual YAML key.
    """

    kept: list[dict] = []
    for error in errors:
        filtered: list = []
        path: list = []
        drop = False
        for part in error.get("loc", ()):
            if part in _STEP_CLASSES:
                step = _lookup(data, tuple(path))
                actual = next(
                    (
                        key
                        for key in _STEP_CLASSES.values()
                        if isinstance(step, dict) and key in step
                    ),
                    None,
                )
                if actual is not None and _STEP_CLASSES[part] != actual:
                    drop = True
                    break
                continue
            if part == "root":
                continue
            filtered.append(part)
            path.append(part)
        if drop:
            continue
        kept.append({**error, "loc": tuple(filtered)})

    seen: set[tuple] = set()
    unique: list[dict] = []
    for error in kept:
        key = (error.get("type"), tuple(error.get("loc", ())), error.get("msg"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(error)
    return unique


def _diagnostic_from_error(
    exc: ValidationError,
    display: str,
    lines,
    data,
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    for error in _filter_union_errors(exc.errors(), data):
        loc = tuple(error.get("loc", ()))
        code = "P-UNKNOWN-KEY" if error.get("type") == "extra_forbidden" else "P-SCHEMA"
        if code == "P-UNKNOWN-KEY" and loc:
            message = f"unknown key {loc[-1]!r}"
        else:
            message = str(error.get("msg", "invalid value"))
        line = loc_line(lines, normalize_loc(loc))
        diags.append(
            Diagnostic(
                code=code,
                message=message,
                file=display,
                line=line,
                pointer=loc_pointer(loc),
            )
        )
    return diags


def parse_pkir(
    path: str | Path, display: str | None = None
) -> tuple[PkirDoc | None, list[Diagnostic]]:
    """Parse one PKIR document. Returns ``(doc, [])`` or ``(None, diagnostics)``."""

    path = Path(path)
    display = display or str(path)
    try:
        data, lines = load_yaml(path)
    except YamlLoadError as exc:
        return None, [
            Diagnostic(
                code="P-SCHEMA", message=exc.message, file=display, line=exc.line, pointer=""
            )
        ]
    try:
        return PkirDoc.model_validate(data), []
    except ValidationError as exc:
        return None, _diagnostic_from_error(exc, display, lines, data)
