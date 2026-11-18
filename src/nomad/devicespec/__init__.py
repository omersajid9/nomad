"""DeviceSpec models and loader: one YAML per device parameterizes the whole stack."""

from nomad.devicespec.spec import (
    DeviceSpec,
    load_device,
    load_device_from_path,
)

__all__ = ["DeviceSpec", "load_device", "load_device_from_path"]
