"""The contract loop: init -> edit -> check -> repair -> check -> submit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nomad.agent.driver import AgentError, OpenRouterCompleter
from nomad.graph.ingest import graph_from_export
from nomad.harness.check import run_check
from nomad.harness.workspace import init_run, submit


def test_repair_loop(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    code, report = run_check(run, upto="G1")
    assert code == 0 and report.verdict == "admitted"

    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    original = kernel.read_text()
    kernel.write_text(original.replace("tile:\n  m: 64", "tile:\n  m: 64\n  mm: 1"))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "P-UNKNOWN-KEY" for d in report.bounded())

    kernel.write_text(original)
    code, report = run_check(run, upto="G1")
    assert code == 0
    _, result = submit(run)
    assert result["status"] == "admitted"
    assert result["checks_used"] == 3
    assert json.loads((run / "result.json").read_text())["status"] == "admitted"


def test_export_ingest_linear(tmp_path):
    torch = __import__("torch")
    export = __import__("torch.export", fromlist=["export"])

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(3072, 1024, dtype=torch.float16))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.w)

    ep = export.export(M(), (torch.randn(128, 1024, dtype=torch.float16),))
    graph = graph_from_export(ep, "qkv_proj")
    assert graph["tensors"]["x"]["shape"] == [128, 1024]
    assert graph["tensors"]["w"]["shape"] == [3072, 1024]
    assert graph["tensors"]["y"]["shape"] == [128, 3072]
    assert graph["ops"][0]["kind"] == "linear"

    decomposed = ep.run_decompositions()
    graph2 = graph_from_export(decomposed, "qkv_proj")
    assert graph2["tensors"]["w"]["shape"] == [3072, 1024]


def test_init_from_graph_file(tmp_path):
    graph = {
        "graph": "my_linear",
        "params": ["M", "K", "N"],
        "assume": ["K % 8 == 0"],
        "tensors": {
            "a": {"shape": ["M", "K"], "dtype": "f16"},
            "b": {"shape": ["N", "K"], "dtype": "f16", "weight": True},
            "c": {"shape": ["M", "N"], "dtype": "f16", "out": True},
        },
        "ops": [{"id": "proj", "kind": "linear", "in": ["a", "b"], "out": ["c"]}],
    }
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(graph, sort_keys=False))
    run = init_run("ignored", tmp_path / "run", start="unfused", graph_path=path)
    plan = yaml.safe_load((run / "plan.yaml").read_text())
    assert plan["graph"] == "my_linear"
    assert plan["groups"][0]["id"] == "proj"
    code, report = run_check(run, upto="G1")
    assert code == 0, report.summary()


def test_init_from_export_factory(tmp_path):
    run = init_run(
        "qkv_proj",
        tmp_path / "run",
        start="unfused",
        export_path=Path(__file__).resolve().parents[1] / "fixtures" / "export_qkv.py",
    )
    graph = yaml.safe_load((run / "graph.yaml").read_text())
    assert graph["graph"] == "qkv_proj"
    assert graph["tensors"]["x"]["shape"] == [128, 1024]
    code, report = run_check(run, upto="G1")
    assert code == 0, report.summary()


def test_init_from_attention_export(tmp_path):
    run = init_run(
        "attention_block",
        tmp_path / "block",
        start="unfused",
        export_path=Path(__file__).resolve().parents[1] / "fixtures" / "export_block.py",
    )
    graph = yaml.safe_load((run / "graph.yaml").read_text())
    assert graph["graph"] == "attention_block"
    assert [op["id"] for op in graph["ops"]] == [
        "norm",
        "qkv",
        "split",
        "attn",
        "merge",
        "out",
    ]
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(diag.code == "S-UNSUPPORTED" for diag in report.diagnostics)


def test_openrouter_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    completer = OpenRouterCompleter()
    assert completer.url == "https://openrouter.ai/api/v1/chat/completions"
    assert completer.model == "deepseek/deepseek-v4-flash"
    assert completer.extra_headers["X-Title"] == "NOMAD"


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.setenv("NOMAD_NO_DOTENV", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AgentError, match="OPENROUTER_API_KEY"):
        OpenRouterCompleter()
