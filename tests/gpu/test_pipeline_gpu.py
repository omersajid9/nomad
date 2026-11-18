"""Full QKV pipeline on the CUDA box: G0 through G5, measured, no skips."""

from __future__ import annotations

import pytest
import torch

from nomad.harness.check import run_check
from nomad.harness.workspace import init_run

pytestmark = pytest.mark.gpu


def test_qkv_pipeline_through_timing(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    run = init_run("qkv_proj", tmp_path / "run", start="unfused")
    code, report = run_check(run, upto="G5")
    assert code == 0, report.summary()
    assert report.verdict == "admitted"
    assert report.gate == "timing"
    assert report.metrics is not None
    assert report.metrics["lowered"]["qkv_proj"]["source"].endswith(".cu")
    assert report.metrics["difftest"]["qkv_proj"]
    assert report.metrics["timing"]["qkv_proj"]
    assert report.metrics["geomean_speedup"] is not None
    assert (run / "lowered" / "qkv_proj.cu").is_file()
    assert (run / "snapshots" / "001" / "lowered" / "qkv_proj.cu").is_file()
