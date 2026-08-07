"""Matrix transpose operations exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.utils import get_extent
from tilelang.utils.language import to_buffer_region
from tvm import tir


def transpose(src: BufferLikeType, dst: BufferLikeType) -> tir.PrimExpr:
    """Transpose a complete 2D matrix from ``src[M, N]`` to ``dst[N, M]``.

    On Sunmmio this operation is lowered to the A4E asynchronous ODMA
    transpose path. Source and destination must be complete rank-2 RSRAM
    buffers with bfloat16 or float32 elements. Synchronization is inserted by
    the compiler before a dependent access.
    """
    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    if src_extent is None or dst_extent is None:
        raise TypeError("T.transpose requires source and destination regions with known extents")

    src_extent = list(src_extent)
    dst_extent = list(dst_extent)
    if len(src_extent) != 2 or len(dst_extent) != 2:
        raise ValueError(f"T.transpose requires rank-2 source and destination regions, got ranks {len(src_extent)} and {len(dst_extent)}")

    src_region = to_buffer_region(src, access_type="r", extents=src_extent)
    dst_region = to_buffer_region(dst, access_type="w", extents=dst_extent)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.transpose"), src_region, dst_region)
