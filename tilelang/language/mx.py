"""MX physical pack/unpack operations."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.mesh_tensor import _unwrap_mesh_tensor
from tilelang.language.utils import get_extent
from tilelang.utils.language import to_buffer_region
from tvm import tir


def _full_region_arg(buffer: BufferLikeType, access_type: str):
    buffer = _unwrap_mesh_tensor(buffer)
    extents = get_extent(buffer)
    if extents is None:
        if isinstance(buffer, tir.Buffer):
            extents = list(buffer.shape)
        else:
            raise TypeError(f"Cannot deduce full region extent from {type(buffer)}")
    return to_buffer_region(buffer, access_type=access_type, extents=list(extents))


def mx_pack(data: BufferLikeType, scale: BufferLikeType, mx: BufferLikeType) -> tir.PrimExpr:
    """Copy prepared data/scale buffers into an MX container buffer."""
    data_region = _full_region_arg(data, "r")
    scale_region = _full_region_arg(scale, "r")
    mx_region = _full_region_arg(mx, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.mx_pack"),
        data_region,
        scale_region,
        mx_region,
    )


def mx_unpack(mx: BufferLikeType, data: BufferLikeType, scale: BufferLikeType) -> tir.PrimExpr:
    """Copy data/scale fields out of an MX container buffer."""
    mx_region = _full_region_arg(mx, "r")
    data_region = _full_region_arg(data, "w")
    scale_region = _full_region_arg(scale, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.mx_unpack"),
        mx_region,
        data_region,
        scale_region,
    )
