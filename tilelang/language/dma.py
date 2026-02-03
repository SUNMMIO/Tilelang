# tilelang/lang/dma.py

from __future__ import annotations
from tvm import tir
from tilelang.utils.language import (get_buffer_region_from_load, legalize_pairwise_extents,
                                     to_buffer_region)


def _get_extent(data):
    """Detect extent from Buffer / BufferRegion / BufferLoad."""
    if isinstance(data, tir.Buffer):
        return list(data.shape)

    elif isinstance(data, tir.BufferRegion):
        return [r.extent for r in data.region]

    elif isinstance(data, tir.BufferLoad):
        region = get_buffer_region_from_load(data)
        if region is None:
            return None
        return [r.extent for r in region.region]

    return None


def dma_load(src, dst, eviction_policy: int = 0):
    """Global -> Shared TMA Load."""
    src_extent = _get_extent(src)
    dst_extent = _get_extent(dst)

    assert src_extent or dst_extent, "Can't deduce extents for dma_load()"

    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    # Pairwise extent legalize (same as T.copy)
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    src_region = to_buffer_region(src, "r", src_extent)
    dst_region = to_buffer_region(dst, "w", dst_extent)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dma_load"),
        src_region,
        dst_region,
        eviction_policy,
    )


def dma_store(src, dst, eviction_policy: int = 0):
    """Shared -> Global TMA Store."""
    src_extent = _get_extent(src)
    dst_extent = _get_extent(dst)

    assert src_extent or dst_extent, "Can't deduce extents for dma_store()"

    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    src_region = to_buffer_region(src, "r", src_extent)
    dst_region = to_buffer_region(dst, "w", dst_extent)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dma_store"),
        src_region,
        dst_region,
        eviction_policy,
    )
