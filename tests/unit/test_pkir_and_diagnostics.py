"""PKIR models/expr/static and the bounded diagnostic document."""

from __future__ import annotations

from pathlib import Path

import yaml

from nomad.diagnostics import Diagnostic, Report
from nomad.harness.check import run_check
from nomad.harness.workspace import init_run
from nomad.pkir.expr import ExprError, eval_expr, names, parse_slice
from nomad.pkir.models import parse_pkir

ROOT = Path(__file__).resolve().parents[2]


def test_worked_example_parses():
    doc, diags = parse_pkir(ROOT / "examples" / "qkv_proj.pkir.yaml")
    assert diags == []
    assert doc is not None
    assert doc.kernel == "qkv_proj"
    assert doc.tile.m == 64
    assert len(doc.body) == 4


def test_unknown_key_has_line(tmp_path):
    path = tmp_path / "bad.pkir.yaml"
    data = yaml.safe_load((ROOT / "examples" / "qkv_proj.pkir.yaml").read_text())
    data["tile"]["mm"] = 1
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    doc, diags = parse_pkir(path, display="bad.pkir.yaml")
    assert doc is None
    unknown = [d for d in diags if d.code == "P-UNKNOWN-KEY"]
    assert unknown and unknown[0].file == "bad.pkir.yaml"
    assert unknown[0].line > 0
    assert unknown[0].pointer.endswith("/tile/mm")


def test_expr_grammar_and_names():
    assert names("ceil_div(M, tile.m)") == {"M", "tile.m"}
    assert eval_expr("ceil_div(K, tile.k)", {"K": 1024, "tile.k": 32}) == 32
    assert eval_expr("min(4, max(2, 3))", {}) == 3
    try:
        names("foo(M)")
    except ExprError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown call must be rejected")


def test_slice_parser():
    part = parse_slice("X[bm*tile.m : bm*tile.m + tile.m, 0:32]")
    assert part.name == "X"
    assert len(part.ranges) == 2
    staged = parse_slice("A[kt]")
    assert staged.name == "A" and staged.staged_index


def test_report_is_bounded_and_short():
    diagnostics = [
        Diagnostic("S-CAPACITY", "x" * 500, "kernels/a.pkir.yaml", line=i + 1) for i in range(8)
    ]
    report = Report(submission=1, verdict="rejected", gate="static", diagnostics=diagnostics)
    payload = report.to_dict()
    assert len(payload["diagnostics"]) == 5
    assert all(len(d["message"]) <= 300 for d in payload["diagnostics"])
    assert payload["budget"] == {"checks_left": 0, "tokens_used": None}


def test_static_capacity_and_placement(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = yaml.safe_load(kernel.read_text())
    doc["buffers"]["A"]["stages"] = 10
    doc["buffers"]["B"]["stages"] = 10
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-CAPACITY" for d in report.bounded())

    doc["buffers"]["A"]["stages"] = 2
    doc["buffers"]["B"]["stages"] = 2
    doc["placement"] = {"a40": doc["placement"]["sm86"]}
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-PLACEMENT" for d in report.bounded())


def test_undeclared_slice_name(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = yaml.safe_load(kernel.read_text())
    doc["body"][3]["copy"]["src"] = "missing[0:1, 0:1]"
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-SLICE" for d in report.bounded())


def test_unbounded_slice_is_a_defect(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = yaml.safe_load(kernel.read_text())
    doc["body"][3]["copy"]["src"] = "x[0:, 0:32]"
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-SLICE" for d in report.bounded())


def test_yaml_anchor_is_rejected(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    kernel.write_text("pkir: &v 0\n" + kernel.read_text().split("\n", 1)[1])
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "P-SCHEMA" for d in report.bounded())


def test_shape_expression_is_checked(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = yaml.safe_load(kernel.read_text())
    doc["tensors"]["x"]["shape"][0] = "M +"
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-CAPABILITY" for d in report.bounded())


def test_warp_tile_arity_is_checked(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = yaml.safe_load(kernel.read_text())
    doc["placement"]["sm86"]["warp_tile"]["mma"] = ["tile.m / 2"]
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-PLACEMENT" for d in report.bounded())


def test_copy_without_required_cast(tmp_path):
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    kernel = run / "kernels" / "qkv_proj.pkir.yaml"
    doc = yaml.safe_load(kernel.read_text())
    doc["body"][-1]["copy"].pop("cast")
    kernel.write_text(yaml.safe_dump(doc, sort_keys=False))
    code, report = run_check(run, upto="G1")
    assert code == 1
    assert any(d.code == "S-CAPABILITY" for d in report.bounded())
