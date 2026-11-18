"""CLI exit codes and the free commands (agent-contract §5)."""

from __future__ import annotations

import json

from nomad.harness.cli import main
from nomad.harness.workspace import init_run, read_state, write_state


def test_check_exit_codes(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    assert main(["check", str(run), "--upto", "G1"]) == 0

    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    original = kernel.read_text()
    kernel.write_text(original.replace("tile:\n  m: 64", "tile:\n  m: 64\n  bad: 1"))
    assert main(["check", str(run), "--upto", "G1"]) == 1
    kernel.write_text(original)
    assert main(["check", str(run), "--upto", "G1"]) == 0

    (run / "task.yaml").write_text((run / "task.yaml").read_text() + "\n# edited\n")
    assert main(["check", str(run), "--upto", "G1"]) == 2


def test_budget_exit_code(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    state = read_state(run)
    state["checks_used"] = 40
    write_state(run, state)
    assert main(["check", str(run), "--upto", "G1"]) == 3


def test_cost_scaffold_explain_submit(tmp_path, capsys):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    assert main(["cost", str(run), "--group", "qkv_proj"]) == 0
    cost = json.loads(capsys.readouterr().out)
    assert cost["groups"][0]["shared_bytes"] == 37888

    assert main(["scaffold", str(run), "qkv_proj"]) == 0
    assert main(["explain", "S-UNSUPPORTED"]) == 0
    assert main(["explain", "NO-SUCH-CODE"]) == 2
    assert main(["check", str(run), "--upto", "G1"]) == 0
    assert main(["submit", str(run)]) == 0
    assert json.loads((run / "result.json").read_text())["status"] == "admitted"


def test_explore_lists_feasible_orders(tmp_path, capsys):
    run = init_run("attention_block", tmp_path / "block", start="unfused")
    assert main(["explore", str(run)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["applicable"] == ["fold_norm_gain", "fold_attn_scale"]
    assert ["fold_norm_gain", "norm_as_row_scale"] in result["sequences"]
    assert ["norm_as_row_scale", "fold_norm_gain"] not in result["sequences"]
    assert result["count"] == 8
