from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SunmmioDeviceProperties:
    mesh_config = (
        4,
        4,
    )
    RsramPerCore: int = 1536000  # 1.5MB
    WSRAMperCore: int = 2097152  # 2MB
    ASRAMPerCore: int = 1048576  # 1MB


def get_sunmmio_device_properties(device_id: int = 0) -> SunmmioDeviceProperties:
    # TODO: Get device prperties from torch
    return SunmmioDeviceProperties()


def get_device_mesh_config(device_id: int = 0) -> tuple[int, int]:
    """
    Get the mesh configurations on the Sunmmio device, e.g., i.e.,
    #cores_per_row & #cores_per_col

    Args:
        device_id (int, optional): The Sunmmio device ID. Defaults to 0.

    Returns:
        (int, int): The number of cores per row and per column on the device.

    Raises:
        RuntimeError: If unable to get the device properties.
    """
    prop = get_sunmmio_device_properties(device_id)
    return prop.mesh_config
