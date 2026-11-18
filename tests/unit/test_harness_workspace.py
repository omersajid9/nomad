"""Workspace lifecycle, G0/G1, diagnostics, budget, submit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nomad.harness.check import BudgetExhausted, HarnessError, run_check
from nomad.harness.workspace import init_run, read_state, submit, write_state

ROOT = Path(__file__).resolve().parents[2]


def qkv_run(tmp_path: Path) -> Path:
    return init_run("qkv_proj", tmp_path / "run", start="unfused")


def block_run(tmp_path: Path) -> Path:
    return init_run("attention_block", tmp_path / "block", start="unfused")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def dump_yaml(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_init_writes_workspace_and_scaffold(tmp_path):
    run = qkv_run(tmp_path)
    for name in [
        "task.yaml",
        "devicespec.yaml",
        "graph.yaml",
        "catalog.yaml",
        "plan.yaml",
        "run.json",
    ]:
        assert (run / name).is_file(), name
    plan = load_yaml(run / "plan.yaml")
    assert plan["groups"] == [
        {
            "id": "qkv_proj",
            "pattern": "single",
            "ops": ["qkv_proj"],
            "pkir": "kernels/qkv_proj.pkir.yaml",
        }
    ]
    kernel = load_yaml(run / "kernels" / "qkv_proj.pkir.yaml")
    assert kernel["kernel"] == "qkv_proj"
    assert kernel["placement"].keys() == {"sm86"}
    assert read_state(run)["hashes"]["task.yaml"]


def test_qkv_is_admitted_and_snapshotted(tmp_path):
    run = qkv_run(tmp_path)
    code, report = run_check(run, upto="G1")
    assert code == 0
    assert report.verdict == "admitted"
    report_json = json.loads((run / "diagnostics" / "latest.json").read_text())
    assert report_json["verdict"] == "admitted"
    assert report_json["diagnostics"] == []
    assert report_json["gate"] == "static"
    assert (run / "snapshots" / "001" / "plan.yaml").is_file()
    assert (run / "log.jsonl").read_text().strip()


def test_unknown_key_is_located(tmp_path):
    run = qkv_run(tmp_path)
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    text = kernel.read_text().replace("tile:\n  m: 64", "tile:\n  m: 64\n  mm: 1")
    kernel.write_text(text)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert report.gate == "parse"
    unknown = [d for d in report.bounded() if d.code == "P-UNKNOWN-KEY"]
    assert unknown and unknown[0].line > 0
    assert unknown[0].pointer.endswith("/tile/mm")


def test_missing_kernel_is_parse_failure(tmp_path):
    run = qkv_run(tmp_path)
    (run / "kernels" / "qkv_proj.pkir.yaml").unlink()
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert report.gate == "parse"
    assert any(d.code == "P-SCHEMA" for d in report.bounded())


def test_plan_partition_error(tmp_path):
    run = qkv_run(tmp_path)
    plan = load_yaml(run / "plan.yaml")
    plan["groups"][0]["ops"] = []
    dump_yaml(run / "plan.yaml", plan)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert [d.code for d in report.bounded()] == ["S-PARTITION"]


def test_plan_pattern_error(tmp_path):
    run = qkv_run(tmp_path)
    plan = load_yaml(run / "plan.yaml")
    plan["groups"][0]["pattern"] = "norm_gemm"
    dump_yaml(run / "plan.yaml", plan)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert [d.code for d in report.bounded()] == ["S-PATTERN"]


def test_layout_must_be_in_catalog(tmp_path):
    run = qkv_run(tmp_path)
    plan = load_yaml(run / "plan.yaml")
    plan["layouts"] = {"y": "nonsense"}
    dump_yaml(run / "plan.yaml", plan)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert [d.code for d in report.bounded()] == ["S-LAYOUT"]


def test_rewrite_side_condition(tmp_path):
    run = qkv_run(tmp_path)
    plan = load_yaml(run / "plan.yaml")
    plan["rewrites"] = [{"rule": "fold_norm_gain"}]
    dump_yaml(run / "plan.yaml", plan)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert [d.code for d in report.bounded()] == ["S-REWRITE"]


def test_placement_key_and_capacity(tmp_path):
    run = qkv_run(tmp_path)
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = load_yaml(kernel)
    doc["placement"] = {"a40": doc["placement"]["sm86"]}
    dump_yaml(kernel, doc)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-PLACEMENT" for d in report.bounded())

    doc["placement"] = {"sm86": doc["placement"]["a40"]}
    doc["buffers"]["A"]["stages"] = 10
    doc["buffers"]["B"]["stages"] = 10
    dump_yaml(kernel, doc)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-CAPACITY" for d in report.bounded())


def test_kernel_name_must_match_group(tmp_path):
    run = qkv_run(tmp_path)
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = load_yaml(kernel)
    doc["kernel"] = "not_qkv"
    dump_yaml(kernel, doc)
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-CAPABILITY" for d in report.bounded())


def test_readonly_change_is_harness_error(tmp_path):
    run = qkv_run(tmp_path)
    (run / "task.yaml").write_text((run / "task.yaml").read_text() + "\n# edited\n")
    with pytest.raises(HarnessError):
        run_check(run, upto="G1")


def test_budget_exhaustion(tmp_path):
    run = qkv_run(tmp_path)
    state = read_state(run)
    state["checks_used"] = 40
    write_state(run, state)
    with pytest.raises(BudgetExhausted):
        run_check(run, upto="G1")


def test_submit_chooses_admitted_snapshot(tmp_path):
    run = qkv_run(tmp_path)
    run_check(run, upto="G1")
    path, result = submit(run)
    assert path.is_file()
    assert result["status"] == "admitted"
    assert result["snapshot"] == "snapshots/001"


def test_attention_block_unfused_reports_capability_gap(tmp_path):
    run = block_run(tmp_path)
    code, report = run_check(run, upto="G1")
    assert code == 1
    codes = [d.code for d in report.bounded()]
    assert codes and set(codes) == {"S-UNSUPPORTED"}
    groups = {d.group for d in report.bounded()}
    assert {"norm", "split", "attn", "merge"} <= groups


def test_attention_target_plan_still_unsupported(tmp_path):
    run = block_run(tmp_path)
    plan = load_yaml(run / "plan.yaml")
    plan["rewrites"] = [{"rule": "norm_as_row_scale"}, {"rule": "fold_norm_gain"}]
    dump_yaml(run / "plan.yaml", plan)
    code, report = run_check(run, upto="G1")
    assert code == 1
    rewrite = [d for d in report.bounded() if d.code == "S-REWRITE"]
    assert rewrite and "fold_norm_gain" in rewrite[0].message
