"""YAML loading with path-to-line mapping, for the agent contract's ``file``/``line`` diagnostics.

The contract requires every diagnostic to name a file and line. Pydantic reports a ``loc`` path,
not a line, so we compose the YAML node tree once and map each path to the line of its key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml.tokens import AliasToken, AnchorToken, TagToken


class YamlLoadError(Exception):
    def __init__(self, message: str, line: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


@dataclass
class LineMap:
    """Maps a YAML path (mapping keys and sequence indices) to a 1-based line number."""

    lines: dict[tuple[Any, ...], int] = field(default_factory=dict)

    def get(self, path: tuple[Any, ...], default: int = 1) -> int:
        path = tuple(path)
        while path:
            if path in self.lines:
                return self.lines[path]
            path = path[:-1]
        return default

    def pointer(self, path: tuple[Any, ...]) -> str:
        return "".join(f"/{part}" for part in path)


def _walk(node: yaml.Node, path: tuple[Any, ...], lines: LineMap) -> None:
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            key = key_node.value
            lines.lines[path + (key,)] = key_node.start_mark.line + 1
            _walk(value_node, path + (key,), lines)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            lines.lines[path + (index,)] = item.start_mark.line + 1
            _walk(item, path + (index,), lines)


def _find_anchor_or_tag(node: yaml.Node) -> str | None:
    anchor = getattr(node, "anchor", None)
    if anchor:
        return f"anchors and aliases are not allowed (found &{anchor})"
    if node.tag and not node.tag.startswith("tag:yaml.org,2002:"):
        return f"custom YAML tags are not allowed (found {node.tag})"
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            found = _find_anchor_or_tag(key_node) or _find_anchor_or_tag(value_node)
            if found:
                return found
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            found = _find_anchor_or_tag(item)
            if found:
                return found
    return None


def load_yaml(path: str | Path) -> tuple[Any, LineMap]:
    """Load one YAML document, rejecting multi-document streams, anchors and custom tags."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        for token in yaml.scan(text, Loader=yaml.SafeLoader):
            if isinstance(token, (AnchorToken, AliasToken)):
                raise YamlLoadError(
                    f"anchors and aliases are not allowed (found {token.value!r})",
                    token.start_mark.line + 1,
                )
            if isinstance(token, TagToken) and not token.value.startswith("tag:yaml.org,2002:"):
                raise YamlLoadError(
                    f"custom YAML tags are not allowed (found {token.value!r})",
                    token.start_mark.line + 1,
                )
        docs = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = (mark.line + 1) if mark is not None else 1
        raise YamlLoadError(f"invalid YAML: {getattr(exc, 'problem', str(exc))}", line) from exc

    if len(docs) != 1:
        raise YamlLoadError(f"expected exactly one YAML document, found {len(docs)}", 1)

    node = docs[0]
    problem = _find_anchor_or_tag(node)
    if problem:
        raise YamlLoadError(problem, node.start_mark.line + 1)

    lines = LineMap()
    _walk(node, (), lines)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - compose_all already parsed it
        mark = getattr(exc, "problem_mark", None)
        line = (mark.line + 1) if mark is not None else 1
        raise YamlLoadError(f"invalid YAML: {getattr(exc, 'problem', str(exc))}", line) from exc

    return data, lines


def normalize_loc(loc: tuple[Any, ...]) -> tuple[Any, ...]:
    """Map pydantic field names back to YAML keys where they differ (``async_`` -> ``async``)."""

    out: list[Any] = []
    for part in loc:
        if isinstance(part, str) and part.endswith("_") and part != "_":
            out.append(part[:-1])
        else:
            out.append(part)
    return tuple(out)


def loc_line(lines: LineMap, loc: tuple[Any, ...]) -> int:
    candidates = (tuple(loc), normalize_loc(tuple(loc)))
    for candidate in candidates:
        if candidate in lines.lines:
            return lines.lines[candidate]
    return lines.get(tuple(loc), 1)


def loc_pointer(loc: tuple[Any, ...]) -> str:
    return "".join(f"/{part}" for part in loc)
