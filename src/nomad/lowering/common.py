"""Shared by every lowering: the error type and PKIR expression handling.

The expression helpers accept the PKIR v0.1 grammar subset the lowerings need (integer literals,
declared names, `tile.*`, `+ - * / %`, `ceil_div`, one comparison) and reject everything else with
the field's JSON Pointer. They stand in for `nomad.pkir.expr` until that module lands.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping


class LoweringError(Exception):
    """A region a lowering cannot realize, located by JSON Pointer into the PKIR document.

    `code` follows docs/agent-contract.md §7: `L-RESOURCE` is the agent's to fix (tiles, stages,
    placement); `L-TEMPLATE` means no template covers the region, i.e. a harness error.
    """

    def __init__(self, code: str, pointer: str, message: str) -> None:
        super().__init__(f"{code} at {pointer or '/'}: {message}")
        self.code, self.pointer, self.message = code, pointer, message


_ARITH = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%"}
_EVAL = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,
    ast.Mod: operator.mod,
}
_CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


def _parse(expr: str | int, where: str) -> ast.expr:
    try:
        return ast.parse(str(expr), mode="eval").body
    except SyntaxError as exc:
        raise LoweringError("L-TEMPLATE", where, f"cannot parse {expr!r}") from exc


def _name(node: ast.expr) -> str | None:
    """`K` -> "K", `tile.k` -> "tile.k", anything else -> None."""
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=ast.Name(id="tile"), attr=attr):
            return f"tile.{attr}"
    return None


def const_int(expr: str | int, tile: Mapping[str, int], where: str) -> int:
    """Evaluate an expression over integer literals and `tile.*` (PKIR `/` is floor division)."""

    def ev(node: ast.expr) -> int:
        match node:
            case ast.Constant(value=int() as value) if not isinstance(value, bool):
                return value
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _EVAL:
                a, b = ev(left), ev(right)
                if b == 0 and isinstance(op, ast.Div | ast.Mod):
                    raise LoweringError("L-TEMPLATE", where, f"division by zero in {expr!r}")
                return _EVAL[type(op)](a, b)
            case ast.Call(func=ast.Name(id="ceil_div"), args=[a, b], keywords=[]):
                num, den = ev(a), ev(b)
                if den == 0:
                    raise LoweringError("L-TEMPLATE", where, f"division by zero in {expr!r}")
                return -(-num // den)
        name = _name(node)
        if name and name.startswith("tile.") and name[5:] in tile:
            return int(tile[name[5:]])
        raise LoweringError("L-TEMPLATE", where, f"{expr!r} is not a compile-time constant")

    return ev(_parse(expr, where))


def to_cpp(expr: str | int, names: Mapping[str, str], where: str) -> str:
    """Print an expression as C++ over the given name map (e.g. {"K": "K", "tile.k": "BK"}).

    All operands are non-negative (pkir-spec §3), so C++ truncating `/` equals PKIR floor division.
    """

    def p(node: ast.expr) -> str:
        match node:
            case ast.Constant(value=int() as value) if not isinstance(value, bool):
                return str(value)
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _ARITH:
                return f"({p(left)} {_ARITH[type(op)]} {p(right)})"
            case ast.Call(func=ast.Name(id="ceil_div"), args=[a, b], keywords=[]):
                return f"(({p(a)} + {p(b)} - 1) / {p(b)})"
            case ast.Compare(left=left, ops=[op], comparators=[right]) if type(op) in _CMP:
                return f"{p(left)} {_CMP[type(op)]} {p(right)}"
        name = _name(node)
        if name in names:
            return names[name]
        raise LoweringError("L-TEMPLATE", where, f"unsupported expression {expr!r}")

    return p(_parse(expr, where))
