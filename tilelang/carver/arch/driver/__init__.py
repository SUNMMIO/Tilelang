from .cuda_driver import (
    get_cuda_device_properties,  # noqa: F401
    get_device_name,  # noqa: F401
    get_shared_memory_per_block,  # noqa: F401
    get_device_attribute,  # noqa: F401
    get_max_dynamic_shared_size_bytes,  # noqa: F401
    get_persisting_l2_cache_max_size,  # noqa: F401
    get_num_sms,  # noqa: F401
    get_registers_per_block,  # noqa: F401
)
from .sunmmio_driver import (
    get_sunmmio_device_properties,  # noqa: F401
    get_device_mesh_config as get_sunmmio_device_mesh_config,  # noqa: F401
)
