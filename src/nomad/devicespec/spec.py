"""DeviceSpec models and loader: one YAML per device parameterizes the whole stack."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nomad.yaml_lines import load_yaml


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RuntimeSpec(_Model):
    threads_per_warp: int = Field(gt=0)
    warps_per_block_max: int = Field(gt=0)
    threads_per_block_max: int = Field(gt=0)
    warps_per_sm_max: int | None = Field(default=None, gt=0)
    sm_count: int | None = Field(default=None, gt=0)


class MemoryTier(_Model):
    capacity_bytes: int = Field(gt=0)
    alignment_bytes: int = Field(gt=0)
    max_per_block_bytes: int | None = Field(default=None, gt=0)


class CopyEngines(_Model):
    chunk_bytes: int = Field(gt=0)
    element_bytes: int = Field(gt=0)
    async_: bool = Field(default=False, alias="async")
    commit_groups: bool = False
    zero_fill_on_guard_false: bool = False


class MmaOperand(_Model):
    dtype: str
    layout: str


class TensorCoreMma(_Model):
    shapes: list[list[int]]
    a: MmaOperand
    b: MmaOperand
    acc: str
    ldm_multiple: int = Field(gt=0)


class Convert(_Model):
    pairs: list[list[str]]
    rounding: str


class Capabilities(_Model):
    tensor_core_mma: TensorCoreMma
    convert: Convert


class SyncSpec(_Model):
    barrier_scope: str
    async_wait: str


class CostSpec(_Model):
    peak_flops: dict[str, float]
    bandwidth_bytes_s: float | None = None


class DeviceSpec(_Model):
    name: str
    model: str | None = None
    arch: str
    runtime: RuntimeSpec
    memory_tiers: dict[str, MemoryTier]
    copy_engines: CopyEngines
    capabilities: Capabilities
    sync: SyncSpec
    cost: CostSpec

    def tier(self, name: str) -> MemoryTier:
        try:
            return self.memory_tiers[name]
        except KeyError as exc:
            raise KeyError(f"device {self.name!r} has no memory tier {name!r}") from exc


DEFAULT_DEVICES_DIR = Path(__file__).resolve().parents[1] / "devices"


def load_device(name: str, specs_dir: str | Path | None = None) -> DeviceSpec:
    """Load ``<specs_dir>/<name>.yaml``; defaults to the canonical ``src/nomad/devices``."""

    root = Path(specs_dir) if specs_dir is not None else DEFAULT_DEVICES_DIR
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"device spec not found: {path}")
    data, _ = load_yaml(path)
    spec = DeviceSpec.model_validate(data)
    if spec.name != name:
        raise ValueError(f"device file {path} declares name {spec.name!r}, expected {name!r}")
    return spec


def load_device_from_path(path: str | Path) -> DeviceSpec:
    data, _ = load_yaml(path)
    return DeviceSpec.model_validate(data)
