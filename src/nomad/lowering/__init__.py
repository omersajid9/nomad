"""PKIR template lowerings; device-specific code lives only under this package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nomad.devicespec.spec import DeviceSpec
from nomad.lowering.common import LoweringError
from nomad.lowering.cuda.tiled_gemm import (
    CudaTarget,
    LoweredKernel,
)
from nomad.lowering.cuda.tiled_gemm import (
    lower as lower_cuda,
)
from nomad.pkir.models import PkirDoc

__all__ = [
    "CudaTarget",
    "LoweredKernel",
    "LoweringError",
    "lower",
    "lower_cuda",
    "target_for",
]


def target_for(device: DeviceSpec) -> CudaTarget:
    """Return the CUDA target a DeviceSpec addresses, or raise `L-TARGET`."""
    return CudaTarget.from_device(device)


def lower(doc: PkirDoc | Mapping[str, Any], device: DeviceSpec) -> LoweredKernel:
    """Lower one PKIR region for `device`.

    Only the sm86 tiled-GEMM template exists in v0.1. Other devices raise `L-TARGET`; regions the
    template does not cover raise `L-TEMPLATE` or `L-RESOURCE` from the CUDA matcher.
    """
    data = doc.model_dump(by_alias=True, exclude_none=True) if isinstance(doc, PkirDoc) else doc
    return lower_cuda(data, target_for(device))
