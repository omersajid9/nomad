"""Block graph document (agent-contract §3)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nomad.yaml_lines import LineMap, load_yaml


class TensorDecl(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    shape: list[int | str]
    dtype: str
    weight: bool = False
    out: bool = False


class OpDecl(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    kind: str
    in_: list[str] = Field(alias="in")
    out_: list[str] = Field(alias="out")


class GraphDoc(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    graph: str
    params: list[str]
    assume: list[str] = Field(default_factory=list)
    tensors: dict[str, TensorDecl]
    ops: list[OpDecl]

    def op(self, op_id: str) -> OpDecl:
        for op in self.ops:
            if op.id == op_id:
                return op
        raise KeyError(op_id)

    @property
    def outputs(self) -> list[str]:
        return [name for name, tensor in self.tensors.items() if tensor.out]


def load_graph(path: str | Path) -> tuple[GraphDoc, LineMap]:
    data, lines = load_yaml(path)
    return GraphDoc.model_validate(data), lines
