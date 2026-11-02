"""Example PKIR programs parse, and stay equal to the spec's normative listing."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SPEC = ROOT / "docs" / "pkir-spec.md"


def spec_listing() -> dict:
    """The single normative PKIR listing carried by the spec."""
    blocks = re.findall(r"```yaml\n(.*?)```", SPEC.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 1, "the spec carries exactly one yaml listing"
    return yaml.safe_load(blocks[0])


def test_qkv_example_matches_the_spec_listing() -> None:
    """A lowering is built against the example file, so the two copies may not drift apart."""
    example = yaml.safe_load((EXAMPLES / "qkv_proj.pkir.yaml").read_text(encoding="utf-8"))
    assert example == spec_listing()
