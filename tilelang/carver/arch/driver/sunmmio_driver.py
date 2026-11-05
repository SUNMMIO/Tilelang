from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SunmmioDeviceProperties:
    RsramPerCore: int = 1536000  # 1.5MB
    WSRAMperCore: int = 2097152  # 2MB
    ASRAMPerCore: int = 1048576  # 1MB


def get_sunmmio_device_properties(device_id: int = 0) -> SunmmioDeviceProperties:
    # TODO: Get device prperties from torch
    return SunmmioDeviceProperties()
