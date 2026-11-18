"""G2/G3/G4 wiring, lowering artifacts, and the agent driver on a CPU checkout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from nomad.agent.driver import AgentError, apply_updates, parse_updates, render_prompt, run_agent
from nomad.harness.check import run_check
from nomad.harness.cli import main
from nomad.harness.workspace import init_run, submit


def qkv_run(tmp_path: Path) -> Path:
    return init_run("qkv_proj", tmp_path / "run", start="unfused")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def test_g2_rejects_dropped_slot_edges(tmp_path):
    run = qkv_run(tmp_path)
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    data = load_yaml(kernel)
    for step in data["body"][1]["for"]["body"][:2]:
        del step["copy"]["wait"]
    kernel.write_text(yaml.safe_dump(data, sort_keys=False))
    code, report = run_check(run, upto="G2")
    assert code == 1
    assert report.gate == "verify"
    assert any(diag.code == "V-RACE" for diag in report.diagnostics)


def test_g3_writes_cuda_and_admits_source(tmp_path):
    run = qkv_run(tmp_path)
    code, report = run_check(run, upto="G3")
    assert code == 0, report.summary()
    assert report.gate == "lower"
    source = run / "lowered" / "qkv_proj.cu"
    assert source.is_file()
    assert "gemm_kernel" in source.read_text()
    meta = json.loads((run / "lowered" / "qkv_proj.json").read_text())
    assert meta["kernel"] == "qkv_proj"
    assert meta["shared_bytes"] == 37_888
    assert report.metrics["lowered"]["qkv_proj"]["source"] == "lowered/qkv_proj.cu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA host runs the real G4")
def test_g4_is_explicitly_unavailable_without_cuda(tmp_path):
    run = qkv_run(tmp_path)
    code, report = run_check(run, upto="G4")
    assert code == 1
    assert report.gate == "difftest"
    assert report.diagnostics[0].code == "D-UNAVAILABLE"


def test_lower_command_emits_source(tmp_path, capsys):
    run = qkv_run(tmp_path)
    assert main(["lower", str(run)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lowered"]["qkv_proj"]["source"] == "lowered/qkv_proj.cu"
    assert (run / "lowered" / "qkv_proj.cu").is_file()


def test_agent_driver_applies_a_file_and_admits(tmp_path):
    run = qkv_run(tmp_path)
    plan = (run / "plan.yaml").read_text()
    seen = []

    def complete(prompt: str) -> str:
        seen.append(prompt)
        return f"### FILE: plan.yaml\n```yaml\n{plan}```\n"

    result = run_agent(run, complete, upto="G1", max_steps=2)
    assert result.status == "admitted"
    assert result.steps == 1
    assert result.files == ("plan.yaml",)
    assert "plan.yaml" in seen[0]
    assert "diagnostics" in seen[0]


def test_agent_rejects_paths_outside_the_workspace(tmp_path):
    try:
        parse_updates("### FILE: ../plan.yaml\n```yaml\nplan: 0\n```\n")
    except AgentError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsafe path was accepted")


def test_agent_apply_is_confined(tmp_path):
    run = qkv_run(tmp_path)
    written = apply_updates(run, parse_updates("### FILE: plan.yaml\n```yaml\nplan: 0\n```\n"))
    assert written == ("plan.yaml",)
    assert (run / "plan.yaml").is_file()


def test_prompt_contains_current_files(tmp_path):
    run = qkv_run(tmp_path)
    prompt = render_prompt(run, gate="G1")
    assert "## graph.yaml" in prompt
    assert "qkv_proj.pkir.yaml" in prompt
    assert "Target gate: G1" in prompt


def test_submit_includes_lowered_snapshot(tmp_path):
    run = qkv_run(tmp_path)
    code, report = run_check(run, upto="G3")
    assert code == 0, report.summary()
    _, result = submit(run)
    assert result["status"] == "admitted"
    snapshot = run / result["snapshot"]
    assert (snapshot / "lowered" / "qkv_proj.cu").is_file()
