"""Communication intrinsics wrappers for TileLang.

This module provides small helper functions that prepare arguments and
emit TIR intrinsics for inter-core communication on a target mesh.
"""

from __future__ import annotations

from typing import Literal
from collections.abc import Iterable

from tvm import tir
import tilelang.language as T
from tilelang.utils.language import to_buffer_region

DIRECTION_MAP = {"horizontal": 0, "h": 0, "vertical": 1, "v": 1, "all": 2, "a": 2}


def get_target_mesh_shape(target: str = "auto") -> dict[str, int]:
    """Get the shape of the target mesh as a dictionary with 'x' and 'y' keys.
    Args:
        target: The target mesh type. Supported values are
            'sunmmio-a4e', 'sunmmio-a4e-lite', and 'auto'. If 'auto' is specified,
            the function defaults to 'sunmmio-a4e'.
    Returns:
        A dictionary with integer keys 'x' and 'y' representing
        the 2D mesh size in each dimension.
    Raises:
        ValueError: If an unknown target is specified.
    """
    if target == "auto":
        target = "sunmmio-a4e"

    if target == "sunmmio-a4e":
        return {"x": 4, "y": 4}
    elif target == "sunmmio-a4e-lite":
        return {"x": 2, "y": 4}
    else:
        raise ValueError(f"Unknown target: {target}")


def core_tuple_to_id(core_id: tuple[int, int]) -> int:
    """Convert 2D (row, col) coordinates on the mesh into a linear core id.

    Parameters
    ----------
    core_id : tuple[int, int]
        A tuple specifying the (row, col) coordinates of the core on the mesh.

    Returns
    -------
    int
        The linear core id corresponding to the provided coordinates.

    Notes
    -----
    The conversion uses the current target mesh shape obtained via
    get_target_mesh_shape("auto").
    """
    mesh_shape = get_target_mesh_shape("auto")
    row, col = core_id
    assert (0 <= row < mesh_shape["x"]), f"Row {row} out of bounds for mesh shape {mesh_shape}."
    assert (0 <= col < mesh_shape["y"]), f"Col {col} out of bounds for mesh shape {mesh_shape}."
    core_id_value = row * mesh_shape["y"] + col
    return core_id_value


def core_id_to_tuple(core_id: tir.Call) -> tuple[int, int]:
    """Convert a linear core id into 2D (row, col) coordinates on the mesh.

    Parameters
    ----------
    core_id : tir.Call
        A linear core identifier (or a TIR expression that yields one).

    Returns
    -------
    tuple[int, int]
        The (row, col) coordinates corresponding to the linear core id.

    Notes
    -----
    The conversion uses the current target mesh shape obtained via
    get_target_mesh_shape("auto").
    """
    mesh_shape = get_target_mesh_shape("auto")
    core_id_value = core_id
    row = core_id_value // mesh_shape["y"]
    col = core_id_value % mesh_shape["y"]
    return (row, col)


def CoreId(core_id: int | tuple[int, int]):
    """Convert a core identifier to a linear core ID for the target mesh.

    Parameters
    ----------
    core_id : int or tuple[int, int]
        Either a linear core id (int) or a 2-tuple (row, col) specifying the
        core coordinates on the target mesh.

    Returns
    -------
    int
        The linear core id mapped into [0, mesh_x * mesh_y).

    Raises
    ------
    AssertionError, ValueError
        If the provided coordinates are out of bounds or the type is invalid.
    """
    mesh_shape = get_target_mesh_shape("auto")
    if isinstance(core_id, tuple):
        row, col = core_id
        assert (0 <= row < mesh_shape["x"]), f"Row {row} out of bounds for mesh shape {mesh_shape}"
        assert (0 <= col < mesh_shape["y"]), f"Col {col} out of bounds for mesh shape {mesh_shape}"
        # Convert 2D coordinates into a linear core id.
        core_id_value = row * mesh_shape["x"] + col
    elif isinstance(core_id, int):
        core_id_value = core_id
        assert (0 <= core_id_value < mesh_shape["x"] * mesh_shape["y"]
               ), f"Core ID {core_id_value} out of bounds for mesh shape {mesh_shape}"
    else:
        raise ValueError("core_id must be either a tuple[int, int] or an int.")
    return tir.call_intrin("handle", tir.op.Op.get("tl.CoreId"), core_id_value)


def current_core():
    """Get the current core's identifier.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_current_core`.

    Examples
    --------
    >>> current_core()
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_current_core"))


def broadcast(
    src: T.Buffer,
    dst: T.Buffer,
    src_core: tuple[int, int],
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"] = "all",
    size: int = -1,
):
    """Broadcast data from a source buffer on a specific source core to a destination buffer
    on all cores in the specified direction by emitting the TIR intrinsic tl.tileop.comm_broadcast.
    Parameters
    ----------
    src : T.Buffer
        Source buffer containing data to broadcast.
    dst : T.Buffer
        Destination buffer to receive the broadcasted data.
    src_core : tuple[int, int]
        (row, col) coordinates of the source core on the target mesh.
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
    """
    assert (
        src.dtype == dst.dtype
    ), f"Source and destination buffer dtypes must match for broadcast. Got {src.dtype} vs {dst.dtype}."
    if len(src.shape) != len(dst.shape):
        raise ValueError(
            "Source and destination buffer must have the same number of dimensions for broadcast.")
    for i in range(len(src.shape)):
        assert (
            src.shape[i] == dst.shape[i] or src.shape[i] == 1 or dst.shape[i] == 1
        ), f"Source buffer shape  and destination buffer shape must match for broadcast. Got {src.shape} vs {dst.shape}."

    mesh_shape = get_target_mesh_shape("auto")
    assert (isinstance(src_core, tuple) and
            len(src_core) == 2), "src_core must be a tuple of (row, col)."
    assert (0 <= src_core[0] < mesh_shape["x"]
           ), f"src_core row {src_core[0]} out of bounds for mesh shape {mesh_shape}."
    assert (0 <= src_core[1] < mesh_shape["y"]
           ), f"src_core col {src_core[1]} out of bounds for mesh shape {mesh_shape}."

    src_elements = 1
    for dim in src.shape:
        src_elements *= dim
    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    assert size <= src_elements, f"size {size} exceeds source buffer size {src_elements}."

    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"

    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)
    src_core_id = core_tuple_to_id(src_core)
    dst_offset = 0  # Always 0 for now

    args = (
        src_region,
        dst_region,
        size,
        dst_offset,
        src_core_id,
        DIRECTION_MAP[direction.lower()],
    )
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_broadcast"), *args)


def put(
    src: T.Buffer,
    dst: T.Buffer,
    src_core: tuple[int, int],
    dst_core: tuple[int, int],
    size: int = -1,
):
    """Put data from a source buffer on a specific source core to a destination buffer on a specific destination core
    by emitting the TIR intrinsic tl.tileop.comm_put.
    Parameters
    ----------
    src : T.Buffer
        Source buffer containing data to put.
    dst : T.Buffer
        Destination buffer to receive the data.
    src_core : tuple[int, int]
        (row, col) coordinates of the source core on the target mesh.
    dst_core : tuple[int, int]
        (row, col) coordinates of the destination core on the target mesh.
    size : int
        Number of elements to put. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_put`.
    Examples
    --------
    >>> put(A, B, (1, 2), (2, 3))
    """
    assert (
        src.dtype == dst.dtype
    ), f"Source and destination buffer dtypes must match for broadcast. Got {src.dtype} vs {dst.dtype}."
    if len(src.shape) != len(dst.shape):
        raise ValueError(
            "Source and destination buffer must have the same number of dimensions for broadcast.")
    for i in range(len(src.shape)):
        assert (
            src.shape[i] == dst.shape[i] or src.shape[i] == 1 or dst.shape[i] == 1
        ), f"Source buffer shape  and destination buffer shape must match for broadcast. Got {src.shape} vs {dst.shape}."

    mesh_shape = get_target_mesh_shape("auto")
    assert (isinstance(src_core, tuple) and
            len(src_core) == 2), "src_core must be a tuple of (row, col)."
    assert (0 <= src_core[0] < mesh_shape["x"]
           ), f"src_core row {src_core[0]} out of bounds for mesh shape {mesh_shape}."
    assert (0 <= src_core[1] < mesh_shape["y"]
           ), f"src_core col {src_core[1]} out of bounds for mesh shape {mesh_shape}."
    assert (isinstance(dst_core, tuple) and
            len(dst_core) == 2), "dst_core must be a tuple of (row, col)."
    assert (0 <= dst_core[0] < mesh_shape["x"]
           ), f"dst_core row {dst_core[0]} out of bounds for mesh shape {mesh_shape}."
    assert (0 <= dst_core[1] < mesh_shape["y"]
           ), f"dst_core col {dst_core[1]} out of bounds for mesh shape {mesh_shape}."
    src_elements = 1
    for dim in src.shape:
        src_elements *= dim
    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    assert (size <= src_elements), f"size {size} exceeds source buffer size {src_elements}."

    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)
    src_core_id = core_tuple_to_id(src_core)
    dst_core_id = core_tuple_to_id(dst_core)
    args = (src_region, dst_region, size, src_core_id, dst_core_id)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_put"), *args)


def all_gather(
    send_buffer: T.Buffer,
    recv_buffer: T.Buffer,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"] = "all",
    size: int = -1,
):
    """Perform an all-gather operation from a send buffer to a receive buffer
    by emitting the TIR intrinsic tl.tileop.comm_allgather.
    Parameters
    ----------
    send_buffer : T.Buffer
        Buffer containing data to send.
    recv_buffer : T.Buffer
        Buffer to receive gathered data.
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of all-gather: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    size : int
        Number of elements to send from each core. If -1, the entire send buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_allgather`.
    Examples
    --------
    >>> all_gather(A_local, C_local, direction="horizontal")
    """
    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"

    assert (
        send_buffer.dtype == recv_buffer.dtype
    ), f"Source and destination buffer dtypes must match for broadcast. Got {send_buffer.dtype} vs {recv_buffer.dtype}."
    mesh_shape = get_target_mesh_shape("auto")

    recv_num = 1
    if direction.lower() in ["horizontal", "h"]:
        recv_num = mesh_shape["y"]
    elif direction.lower() in ["vertical", "v"]:
        recv_num = mesh_shape["x"]
    elif direction.lower() in ["all", "a"]:
        recv_num = mesh_shape["x"] * mesh_shape["y"]

    expected_recv_shape = [recv_num] + list(send_buffer.shape)
    assert (
        list(recv_buffer.shape) == expected_recv_shape
    ), f"Receive buffer shape must be {expected_recv_shape} to hold gathered data from {recv_num} cores, but got {recv_buffer.shape}."

    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    send_elements = 1
    for dim in send_buffer.shape:
        send_elements *= dim
    assert (size <= send_elements), f"size {size} exceeds send buffer size {send_elements}."

    send_buffer_region = to_buffer_region(send_buffer)
    recv_buffer_region = to_buffer_region(recv_buffer)

    direction_map = {"horizontal": 0, "h": 0, "vertical": 1, "v": 1, "all": 2, "a": 2}
    args = (
        send_buffer_region,
        recv_buffer_region,
        direction_map[direction.lower()],
        size,
    )
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_allgather"), *args)


def barrier(group: Iterable[tuple[int, int]] | None = None):
    """Insert a synchronization barrier among a group of cores.

    Parameters
    ----------
    group : iterable of tuple[int, int] | None
        Optional set of core coordinates to synchronize. If omitted, the
        runtime's default participant set is used.
        Optional set of core coordinates to synchronize. If omitted, the
        runtime's default participant set is used.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_barrier`.

    Examples
    --------
    >>> barrier()
    >>> barrier(group=[(0,0),(0,1)])
    """
    if group is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.comm_barrier"))
    else:
        group = [core_tuple_to_id(core_id) for core_id in group]
        return tir.call_intrin("handle", tir.op.Op.get("tl.comm_barrier"), *group)


def fence():
    """Emit a memory/communication fence intrinsic.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_fence`.

    Examples
    --------
    >>> fence()
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_fence"))
