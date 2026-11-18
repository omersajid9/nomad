"""Affine expression and slice parsing for PKIR.

The grammar is ``+ - * / %``, ``ceil``, ``ceil_div``, ``min``, ``max``, comparisons and
``&&``/``||``/``!`` over params, ``tile.*``, grid ids and loop vars. We parse by rewriting to
Python syntax and walking a whitelisted AST; evaluation is only used by the cost model and by
numeric slice bounds, never to discharge a verifier obligation.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_ALLOWED_CALLS = {"ceil", "ceil_div", "min", "max"}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class ExprError(ValueError):
    pass


def _to_python(expr: str) -> str:
    text = expr.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    return text.strip()


def parse_expr(expr: str, *, where: str = "expression") -> ast.Expression:
    try:
        tree = ast.parse(_to_python(expr), mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"{where}: cannot parse {expr!r}: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"{where}: {type(node).__name__} is not in the expression grammar")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                raise ExprError(
                    f"{where}: call to {getattr(node.func, 'id', '?')!r} is not in "
                    + ", ".join(sorted(_ALLOWED_CALLS))
                )
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "tile":
                raise ExprError(f"{where}: only tile.* attributes are allowed")
    return tree


class _NameVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.found.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            self.found.add(f"{node.value.id}.{node.attr}")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


def names(expr: str, *, where: str = "expression") -> set[str]:
    visitor = _NameVisitor()
    visitor.visit(parse_expr(expr, where=where))
    return visitor.found


def _eval(node: ast.AST, env: dict[str, int | float]) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ExprError("non-numeric constant")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExprError(f"unbound name {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.Attribute):
        key = f"{node.value.id}.{node.attr}"  # type: ignore[attr-defined]
        if key not in env:
            raise ExprError(f"unbound name {key!r}")
        return env[key]
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.Not):
            return not value
        raise ExprError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left // right  # PKIR / is floor division on non-negative operands
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ExprError("unsupported binary operator")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, env)
            ok = {
                ast.Eq: left == right,
                ast.NotEq: left != right,
                ast.Lt: left < right,
                ast.LtE: left <= right,
                ast.Gt: left > right,
                ast.GtE: left >= right,
            }.get(type(op))
            if ok is None:
                raise ExprError("unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        values = [_eval(value, env) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Call):
        func = node.func.id  # type: ignore[attr-defined]
        args = [_eval(arg, env) for arg in node.args]
        if func == "ceil":
            import math

            return math.ceil(args[0])
        if func == "ceil_div":
            import math

            return math.ceil(args[0] / args[1])
        if func == "min":
            return min(args)
        if func == "max":
            return max(args)
    raise ExprError("unsupported expression")


def eval_expr(expr: str, env: dict[str, int | float], *, where: str = "expression") -> int | float:
    return _eval(parse_expr(expr, where=where), env)


def try_eval(expr: str, env: dict[str, int | float]) -> int | float | None:
    try:
        return eval_expr(expr, env)
    except (ExprError, ZeroDivisionError):
        return None


def _split_top_level(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(body):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    parts.append(body[start:].strip())
    return parts


@dataclass(frozen=True)
class SlicePart:
    name: str
    ranges: tuple[tuple[str, str | None], ...]
    staged_index: bool = False


def parse_slice(text: str, *, where: str = "slice") -> SlicePart:
    """Parse ``name[lo:hi, ...]`` or a staged ``name[kt]`` reference."""

    text = text.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]", text)
    if not match:
        raise ExprError(f"{where}: expected name[...] slice, got {text!r}")
    name, body = match.group(1), match.group(2)
    raw_parts = _split_top_level(body)
    if len(raw_parts) == 1 and ":" not in raw_parts[0]:
        index = raw_parts[0]
        parse_expr(index, where=f"{where} index")
        return SlicePart(name=name, ranges=(), staged_index=True)
    ranges: list[tuple[str, str | None]] = []
    for part in raw_parts:
        if part.count(":") != 1:
            raise ExprError(f"{where}: range {part!r} must be lo:hi")
        lo, hi = (piece.strip() for piece in part.split(":"))
        parse_expr(lo, where=f"{where} lower bound")
        if hi:
            parse_expr(hi, where=f"{where} upper bound")
        ranges.append((lo, hi or None))
    return SlicePart(name=name, ranges=tuple(ranges))
