"""Plan document (agent-contract §4)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nomad.yaml_lines import LineMap, load_yaml


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pattern: str
    ops: list[str]
    pkir: str


class Rewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: str


class PlanDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: Literal[0]
    graph: str
    groups: list[Group]
    layouts: dict[str, str] = Field(default_factory=dict)
    rewrites: list[Rewrite] = Field(default_factory=list)

    def group(self, group_id: str) -> Group:
        for group in self.groups:
            if group.id == group_id:
                return group
        raise KeyError(group_id)


def load_plan(path: str | Path) -> tuple[PlanDoc, LineMap]:
    data, lines = load_yaml(path)
    return PlanDoc.model_validate(data), lines


def plan_to_dict(plan: PlanDoc) -> dict:
    return plan.model_dump()
