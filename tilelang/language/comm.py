"""Communication intrinsics wrappers for TileLang.

This module provides small helper functions that prepare arguments and
emit TIR intrinsics for inter-core communication on a target mesh.
"""

from __future__ import annotations

from typing import Literal

import tvm_ffi
from tvm import ir, tir
import tilelang.language as T
from tilelang._typing import BufferLikeType
from tilelang.utils.language import (
    to_buffer_region,
)

from tilelang.carver.arch.driver import get_sunmmio_device_mesh_config

# Mirror of kAttrSrcOffsetByte in src/target/sunmmio_utils.h. Resolved at
# import time via FFI so the single source of truth stays in C++.
ATTR_SRC_OFFSET_BYTE = str(tvm_ffi.get_global_func("tl.target.GetAttrSrcOffsetByte")())

DIRECTION_MAP = {"horizontal": 0, "h": 0, "vertical": 1, "v": 1, "all": 2, "a": 2}
REDUCE_TYPE_LIST = (
    "sum",
    "abssum",
    "max",
    "min",
    "absmax",
    "bitand",
    "bitor",
    "bitxor",
)

CoreCoord = int | tir.PrimExpr
CoreSpec = CoreCoord | tuple[CoreCoord, CoreCoord]


def get_target_mesh_shape() -> dict[str, int]:
    """Get the target mesh shape as a dictionary with 'nrow' and 'ncol' keys."""
    nrow, ncol = get_sunmmio_device_mesh_config()
    return {"nrow": nrow, "ncol": ncol}


def _check_core_coord(coord: CoreCoord, limit: int, name: str):
    if isinstance(coord, bool):
        raise TypeError(f"{name} must be an integer or TIR PrimExpr, got bool.")
    coord_int = _const_int(coord)
    if coord_int is not None:
        assert 0 <= coord_int < limit, f"{name} {coord_int} out of bounds for limit {limit}."
    elif not isinstance(coord, tir.PrimExpr):
        raise TypeError(f"{name} must be an integer or TIR PrimExpr, got {type(coord)}.")


def core_to_id(core_id: CoreSpec, name: str = "core") -> CoreCoord:
    """Normalize a linear core id or 2D mesh coordinate into a linear core id.

    Parameters
    ----------
    core_id : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Either a linear core id, or a tuple specifying the (row, col)
        coordinates of the core on the mesh.
    name : str
        User-facing argument name used in diagnostics.

    Returns
    -------
    int | tir.PrimExpr
        The normalized linear core id.

    Notes
    -----
    Dynamic TIR expressions are allowed. Compile-time bounds checks are only
    performed when the id or coordinate is statically known.
    """
    mesh_shape = get_target_mesh_shape()
    if isinstance(core_id, tuple):
        assert len(core_id) == 2, f"{name} must be a linear core id or a tuple of (row, col)."
        row, col = core_id
        _check_core_coord(row, mesh_shape["nrow"], f"{name} row")
        _check_core_coord(col, mesh_shape["ncol"], f"{name} col")
        return row * mesh_shape["ncol"] + col

    _check_core_coord(core_id, mesh_shape["nrow"] * mesh_shape["ncol"], name)
    return core_id


def core_tuple_to_id(core_id: tuple[CoreCoord, CoreCoord]) -> CoreCoord:
    """Convert 2D (row, col) coordinates on the mesh into a linear core id."""
    assert isinstance(core_id, tuple) and len(core_id) == 2, "core_id must be a tuple of (row, col)."
    return core_to_id(core_id)


def _const_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tir.IntImm):
        return int(value.value)
    return None


def _extent_equal(lhs, rhs) -> bool:
    lhs_int = _const_int(lhs)
    rhs_int = _const_int(rhs)
    if lhs_int is not None and rhs_int is not None:
        return lhs_int == rhs_int
    try:
        return bool(ir.structural_equal(lhs, rhs))
    except (TypeError, ValueError):
        return False


def _extent_is_one(extent) -> bool:
    extent_int = _const_int(extent)
    if extent_int is not None:
        return extent_int == 1
    return _extent_equal(extent, tir.IntImm("int32", 1))


def _shape_equal(lhs, rhs) -> bool:
    return len(lhs) == len(rhs) and all(_extent_equal(lhs_extent, rhs_extent) for lhs_extent, rhs_extent in zip(lhs, rhs))


def _shape_compatible(lhs, rhs) -> bool:
    return len(lhs) == len(rhs) and all(
        _extent_equal(lhs_extent, rhs_extent) or _extent_is_one(lhs_extent) or _extent_is_one(rhs_extent)
        for lhs_extent, rhs_extent in zip(lhs, rhs)
    )


def _const_product(extents):
    result = 1
    for extent in extents:
        extent_int = _const_int(extent)
        if extent_int is None:
            return None
        result *= extent_int
    return result


def _check_size(size: int, extents, op_name: str):
    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    elements = _const_product(extents)
    if size >= 0 and elements is not None:
        assert size <= elements, f"size {size} exceeds {op_name} buffer size {elements}."


def _get_buffer_info(buf: BufferLikeType):
    region = to_buffer_region(buf)
    if not isinstance(region, tir.BufferRegion):
        raise TypeError(f"Expected a buffer-like object, got {type(buf)}.")
    return region.buffer, region.buffer.dtype, list(r.extent for r in region.region)


def broadcast(
    src: BufferLikeType,
    dst: BufferLikeType,
    src_core: CoreSpec,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"] = "all",
    size: int = -1,
):
    """Broadcast data from a source buffer on a specific source core to a destination buffer
    on all cores in the specified direction by emitting the TIR intrinsic tl.tileop.comm_broadcast.
    Parameters
    ----------
    src : BufferLikeType
        Source buffer containing data to broadcast.
    dst : BufferLikeType
        Destination buffer to receive the broadcasted data.
    src_core : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Linear source core id, or (row, col) coordinates of the source core on
        the target mesh. Dynamic TIR expressions such as the block id returned
        by ``T.Kernel`` are allowed.
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of broadcast: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    size : int
        Number of elements to broadcast. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_broadcast`.
    Examples
    --------
    >>> broadcast(A, B, (1, 2), direction="horizontal")
    >>> broadcast(A, B, cid, direction="horizontal")
    """
    _, src_dtype, src_shape = _get_buffer_info(src)
    _, dst_dtype, dst_shape = _get_buffer_info(dst)

    assert src_dtype == dst_dtype, f"Source and destination buffer dtypes must match for broadcast. Got {src_dtype} vs {dst_dtype}."
    if not _shape_compatible(src_shape, dst_shape):
        raise ValueError("Source and destination buffer must have the same number of dimensions for broadcast.")

    _check_size(size, src_shape, "source")

    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"

    src_region = to_buffer_region(src, access_type="r")
    dst_region = to_buffer_region(dst, access_type="w")
    src_core_id = core_to_id(src_core, "src_core")

    args = (
        src_region,
        dst_region,
        size,
        src_core_id,
        DIRECTION_MAP[direction.lower()],
    )
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_broadcast"), *args)


def put(
    src: BufferLikeType,
    dst: BufferLikeType,
    src_core: CoreSpec,
    dst_core: CoreSpec,
    size: int = -1,
):
    """Put data from a source buffer on a specific source core to a destination buffer on a specific destination core
    by emitting the TIR intrinsic tl.tileop.comm_put.
    Parameters
    ----------
    src : BufferLikeType
        Source buffer containing data to put.
    dst : BufferLikeType
        Destination buffer to receive the data.
    src_core : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Linear source core id, or (row, col) coordinates of the source core on
        the target mesh. Dynamic TIR expressions such as the block id returned
        by ``T.Kernel`` are allowed.
    dst_core : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Linear destination core id, or (row, col) coordinates of the destination
        core on the target mesh. Dynamic TIR expressions are allowed.
    size : int
        Number of elements to put. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_put`.
    Examples
    --------
    >>> put(A, B, (1, 2), (2, 3))
    >>> put(A, B, cid, (cid + 1) % 16)
    """
    _, src_dtype, src_shape = _get_buffer_info(src)
    _, dst_dtype, dst_shape = _get_buffer_info(dst)

    assert src_dtype == dst_dtype, f"Source and destination buffer dtypes must match for put. Got {src_dtype} vs {dst_dtype}."
    if not _shape_compatible(src_shape, dst_shape):
        raise ValueError("Source and destination buffer must have the same number of dimensions for put.")

    _check_size(size, src_shape, "source")

    src_region = to_buffer_region(src, access_type="r")
    dst_region = to_buffer_region(dst, access_type="w")
    src_core_id = core_to_id(src_core, "src_core")
    dst_core_id = core_to_id(dst_core, "dst_core")
    args = (src_region, dst_region, size, src_core_id, dst_core_id)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_put"), *args)


def all_gather(
    send_buffer: BufferLikeType,
    recv_buffer: BufferLikeType,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"] = "all",
    size: int = -1,
    axis: int | None = None,
    src_offset_byte: int = 0,
):
    """Perform an all-gather operation from a send buffer to a receive buffer
    by emitting the TIR intrinsic tl.tileop.comm_allgather.
    Parameters
    ----------
    send_buffer : BufferLikeType
        Buffer containing data to send.
    recv_buffer : BufferLikeType
        Buffer to receive gathered data.
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of all-gather: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    size : int
        Number of elements to send from each core. If -1, the entire send buffer is used.
    axis : int, optional
        Axis along which gathered data is concatenated. When ``axis`` is ``None``
        (default), a new leading axis is introduced and ``recv_buffer`` must have
        shape ``[K, *send_buffer.shape]`` where ``K`` is the number of contributing
        cores. When ``axis`` is an integer, gathered data is concatenated along
        that existing axis: ``recv_buffer.shape[axis] == K * send_buffer.shape[axis]``
        and all other dimensions match ``send_buffer``. Only ``axis=0`` and
        ``axis=-1`` (the last dim) are currently supported.
    src_offset_byte : int
        Byte offset added to the source pointer at codegen. Default 0. Set by the
        Sunmmio bf16 GEMM legalization pass to re-stage south-bound A data into a
        destination buffer's north bank. User code should leave this at 0.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_allgather`.
    Examples
    --------
    >>> all_gather(A_local, C_local, direction="horizontal")
    >>> # send [d0, d1], 4-col mesh, axis=0 -> recv [4*d0, d1]
    >>> all_gather(A_local, R_local, direction="horizontal", axis=0)
    >>> # send [d0, d1], 4-col mesh, axis=-1 -> recv [d0, 4*d1]
    >>> all_gather(A_local, R_local, direction="horizontal", axis=-1)
    """
    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"

    _, send_dtype, send_shape = _get_buffer_info(send_buffer)
    _, recv_dtype, recv_shape = _get_buffer_info(recv_buffer)
    assert send_dtype == recv_dtype, f"Source and destination buffer dtypes must match for all_gather. Got {send_dtype} vs {recv_dtype}."
    mesh_shape = get_target_mesh_shape()

    recv_num = 1
    if direction.lower() in ["horizontal", "h"]:
        recv_num = mesh_shape["ncol"]
    elif direction.lower() in ["vertical", "v"]:
        recv_num = mesh_shape["nrow"]
    elif direction.lower() in ["all", "a"]:
        recv_num = mesh_shape["nrow"] * mesh_shape["ncol"]

    # Sentinel -1 in the wire format means "no axis specified" (legacy
    # new-leading-axis semantics). User-facing axis is normalized to a
    # non-negative index before being forwarded.
    if axis is None:
        axis_arg = -1
        expected_recv_shape = [recv_num] + list(send_shape)
    else:
        ndim = len(send_shape)
        assert isinstance(axis, int) and -ndim <= axis < ndim, f"axis {axis} out of range for send buffer with {ndim} dimensions."
        normalized_axis = axis if axis >= 0 else axis + ndim
        assert normalized_axis == 0 or normalized_axis == ndim - 1, (
            f"Only axis=0 or axis=-1 (last dim) are currently supported, got axis={axis} "
            f"(normalized to {normalized_axis}) for {ndim}-D send buffer."
        )
        axis_arg = normalized_axis
        expected_recv_shape = list(send_shape)
        expected_recv_shape[normalized_axis] = recv_num * send_shape[normalized_axis]

    assert _shape_equal(recv_shape, expected_recv_shape), (
        f"Receive buffer shape must be {expected_recv_shape} to hold gathered data from {recv_num} cores, but got {recv_shape}."
    )

    _check_size(size, send_shape, "send")

    assert isinstance(src_offset_byte, int) and src_offset_byte >= 0, "src_offset_byte must be a non-negative integer."

    send_buffer_region = to_buffer_region(send_buffer, access_type="r")
    recv_buffer_region = to_buffer_region(recv_buffer, access_type="w")
    cid = T.get_block_binding(0)

    args = (
        send_buffer_region,
        recv_buffer_region,
        DIRECTION_MAP[direction.lower()],
        size,
        axis_arg,
        cid,
    )
    ann = {ATTR_SRC_OFFSET_BYTE: src_offset_byte} if src_offset_byte != 0 else None
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_allgather"), *args, annotations=ann)


def all_reduce(
    buffer: BufferLikeType,
    out: BufferLikeType,
    reduce_type: str,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"],
    dim: int = -1,
    clear: bool = True,
):
    """Perform an all-reduce operation on a buffer and store the result in an output buffer
    by emitting the TIR intrinsic tl.tileop.comm_allreduce.
    Parameters
    ----------
    buffer : BufferLikeType
        Input buffer containing data to reduce.
    out : BufferLikeType
        Output buffer to store the reduced result.
    reduce_type : str
        Type of reduction operation (e.g., "sum", "max", etc.).
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of all-reduce: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    dim : int
        Dimension along which to perform the reduction. Default is -1 (last dimension).
    clear : bool
        Whether to clear the output buffer before reduction. Default is True.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_allreduce`.
    Examples
    --------
    >>> all_reduce(A_local, E_local, "sum", "all", dim=-1, clear=False)
    """
    _, _, buffer_shape = _get_buffer_info(buffer)
    out_buffer, out_dtype, out_shape = _get_buffer_info(out)

    assert isinstance(dim, int) and dim >= -1 and dim < len(buffer_shape), (
        f"dim {dim} out of bounds for buffer with {len(buffer_shape)} dimensions."
    )
    if dim == -1:
        dim = len(buffer_shape) - 1

    expected_shapes = [
        buffer_shape[:dim] + buffer_shape[dim + 1 :],
        buffer_shape[:dim] + [1] + buffer_shape[dim + 1 :],
    ]
    if not any(_shape_equal(out_shape, expected_shape) for expected_shape in expected_shapes):
        expected_shapes_str = " or ".join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buffer_shape}, dim is {dim}, "
            f"output shape is {out_shape}, expected shapes are {expected_shapes_str}"
        )

    reduce_type = reduce_type.lower()
    assert reduce_type in REDUCE_TYPE_LIST, f"Reduction op must be one of {REDUCE_TYPE_LIST}, but got {reduce_type}."

    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"
    assert clear in [True, False], "clear must be a boolean value."

    mesh_shape = get_target_mesh_shape()

    # Create temporary buffers for row and column allgather results.  Keep the
    # temporaries in the output scope because the lowered Sunmmio path feeds
    # them back into ReduceOp and broadcast_.
    out_scope = out_buffer.scope()

    def alloc_tmp(shape):
        if out_scope.startswith("shared"):
            return T.alloc_shared(shape, out_dtype, scope=out_scope)
        return T.alloc_fragment(shape, out_dtype, scope=out_scope)

    row_allgather = alloc_tmp([mesh_shape["ncol"]] + list(out_shape))
    col_allgather = alloc_tmp([mesh_shape["nrow"]] + list(out_shape))

    buffer_region = to_buffer_region(buffer, access_type="r")
    out_region = to_buffer_region(out, access_type="w")
    row_allgather_region = to_buffer_region(row_allgather, access_type="rw")
    col_allgather_region = to_buffer_region(col_allgather, access_type="rw")
    cid = T.get_block_binding(0)

    args = (
        buffer_region,
        out_region,
        row_allgather_region,
        col_allgather_region,
        reduce_type,
        DIRECTION_MAP[direction.lower()],
        dim,
        clear,
        cid,
    )

    # If not clearing, allocate an output copy buffer to hold intermediate results
    if not clear:
        out_copy = alloc_tmp(list(out_shape))
        out_copy_region = to_buffer_region(out_copy, access_type="rw")
        args = (
            buffer_region,
            out_region,
            row_allgather_region,
            col_allgather_region,
            reduce_type,
            DIRECTION_MAP[direction.lower()],
            dim,
            clear,
            out_copy_region,
            cid,
        )

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_allreduce"), *args)
