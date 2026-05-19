"""Sunmmio named layout constructors (Python FFI wrappers)."""

from __future__ import annotations

import tvm_ffi
import tvm.tir as tir
from tvm.tir import Buffer, BufferLoad, BufferRegion

from .swizzle import _get_buffer_info

_make_row_major = tvm_ffi.get_global_func("tl.sunmmio.make_row_major")
_make_zz = tvm_ffi.get_global_func("tl.sunmmio.make_zz")
_make_zn = tvm_ffi.get_global_func("tl.sunmmio.make_zn")
_make_zzz = tvm_ffi.get_global_func("tl.sunmmio.make_zzz")
_make_nzz = tvm_ffi.get_global_func("tl.sunmmio.make_nzz")


def make_row_major(shape):
    """Create a row-major CuteLayout."""
    return _make_row_major(shape)


def _to_expr(v):
    if isinstance(v, int):
        return tir.IntImm("int32", v)
    return v


def _normalize_shape(shape_or_buffer):
    if isinstance(shape_or_buffer, (Buffer, BufferLoad, BufferRegion)):
        _, shape, _ = _get_buffer_info(shape_or_buffer)
        return list(shape)
    if isinstance(shape_or_buffer, (tuple, list)):
        return list(shape_or_buffer)
    if isinstance(shape_or_buffer, tvm_ffi.container.Array):
        return list(shape_or_buffer)
    raise ValueError(f"Invalid shape or buffer: {shape_or_buffer}")


def _normalize_axes(axes, rank):
    if axes is None:
        if rank < 2:
            raise ValueError(f"make_zz_layout requires rank >= 2 when axes is omitted, got rank {rank}")
        return [rank - 2, rank - 1]
    return list(axes)


def make_zz_layout(shape_or_buffer, axes=None, block_shape=(32, 32)):
    """Create a ZZ (blockwise row-major) CuteLayout."""
    shape = _normalize_shape(shape_or_buffer)
    axes = _normalize_axes(axes, len(shape))
    block_shape = [_to_expr(v) for v in block_shape]
    return _make_zz(shape, axes, block_shape)


def make_zn_layout(shape, axes, block_shape):
    """Create a ZN (blockwise column-major) CuteLayout."""
    return _make_zn(shape, axes, block_shape)


def make_zzz_layout(shape, axes, block_shape, cluster_shape):
    """Create a ZZZ (clustered row-major) CuteLayout."""
    return _make_zzz(shape, axes, block_shape, cluster_shape)


def make_nzz_layout(shape, axes, block_shape, cluster_shape):
    """Create a NZZ (clustered column-major) CuteLayout."""
    return _make_nzz(shape, axes, block_shape, cluster_shape)
