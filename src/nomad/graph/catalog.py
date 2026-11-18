"""Fusion patterns, rewrite rules and layout set (agent-contract §4)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nomad.yaml_lines import load_yaml

DEFAULT_CATALOG = Path(__file__).resolve().parent / "catalog" / "v0.yaml"


class Pattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    ops: list[str]
    supported: bool = False
    note: str | None = None


class Rewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    patterns: list[Pattern]
    rewrites: list[Rewrite]
    layouts: dict[str, list[str]]

    def pattern(self, pattern_id: str) -> Pattern:
        for pattern in self.patterns:
            if pattern.id == pattern_id:
                return pattern
        raise KeyError(pattern_id)

    def rewrite(self, rule_id: str) -> Rewrite:
        for rule in self.rewrites:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    def matches(self, pattern_id: str, kinds: list[str]) -> bool:
        pattern = self.pattern(pattern_id)
        if pattern.ops == ["*"]:
            return len(kinds) == 1
        return pattern.ops == kinds


def load_catalog(path: str | Path | None = None) -> Catalog:
    data, _ = load_yaml(Path(path) if path is not None else DEFAULT_CATALOG)
    return Catalog.model_validate(data)
