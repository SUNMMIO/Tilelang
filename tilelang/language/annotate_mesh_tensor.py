"""Utilities for annotating and manipulating mesh-distributed tensors.

This module provides helper functions used by tilelang passes to record
mesh layout metadata for buffers and to compute per-tile shapes and copies.
"""

from copy import deepcopy

from tvm import tir
import tilelang.language as T


def mesh_tensor_functions(mesh_shape: dict[str, int]):
    """Return helpers for mesh-tensor operations.

    Args:
        mesh_shape: mapping from mesh axis name to device count.

    Returns:
        dict with keys:
          - "annotate_mesh_tensor_info": function to attach per-buffer mesh info
          - "mesh_tensor_copy": function to copy blocks between buffers
          - "tile_shape": function to compute per-tile shape for a buffer
    """

    # Internal storage for parsed mesh tensor metadata keyed by buffer.data.
    _mesh_tensor_info = {}
    # Validate and store the mesh shape.
    if isinstance(mesh_shape, dict) and "x" in mesh_shape and "y" in mesh_shape:
        _mesh_shape = deepcopy(mesh_shape)
    else:
        raise ValueError(f"mesh_shape must be a dict with 'x' and 'y' keys.")

    def annotate_mesh_tensor_info(mesh_tensor_info: dict):
        """Validate and store mesh tensor metadata.

        The expected input maps buffer objects to info dicts containing at least
        'block_shape', 'program_id', and 'sharding'. The info is deep-copied and
        stored under the buffer's `.data` key.
        """

        nonlocal _mesh_tensor_info
        _mesh_tensor_info = {}
        for buffer, info in mesh_tensor_info.items():
            # Basic validation of the info dict.
            if (
                not isinstance(info, dict)
                or "block_shape" not in info
                or "program_id" not in info
                or "sharding" not in info
            ):
                raise ValueError(f"Invalid mesh tensor info: {info}")
            else:
                # Store a copy to avoid external mutation.
                _mesh_tensor_info[buffer.data] = deepcopy(info)

        # Return a function attribute dict compatible with tilelang passes.
        return T.func_attr({"mesh_tensor_info": _mesh_tensor_info})

    def get_tile_shape(buffer: tir.Buffer):
        """Compute the shape of `buffer` on a single mesh tile.

        The global tensor shape is split across mesh axes listed in the
        'sharding' mapping. For each mapped axis we compute the per-device size
        by ceil-dividing the global size by the mesh dimension.
        """

        tensor_shape = buffer.shape
        nonlocal _mesh_tensor_info
        info = _mesh_tensor_info.get(buffer.data, None)
        if info is None:
            raise ValueError(f"MeshTensor information for buffer {buffer} not found.")

        # Indices of the tensor dims that map to mesh x/y axes.
        sharding_x = info["sharding"]["x"]
        sharding_y = info["sharding"]["y"]

        tile_shape = list(tensor_shape)
        # Replace the global dims with per-tile sizes (ceil-divide).
        tile_shape[sharding_x] = T.ceildiv(tile_shape[sharding_x], _mesh_shape["x"])
        tile_shape[sharding_y] = T.ceildiv(tile_shape[sharding_y], _mesh_shape["y"])
        return tuple(tile_shape)

    def mesh_tensor_copy(
        src: tir.Buffer,
        dst: tir.Buffer,
        *,
        src_coord: tuple[int] | None = None,
        dst_coord: tuple[int] | None = None,
    ):
        """Copy data from `src` to `dst` with optional block coordinates.

        If `src_coord`/`dst_coord` are provided they are interpreted as integer
        block coordinates and converted to element offsets using the recorded
        `block_shape` for the corresponding buffer.
        """

        nonlocal _mesh_tensor_info
        if src_coord is not None:
            try:
                info = _mesh_tensor_info[src.data]
                block_shape = info["block_shape"]
                # Convert block coordinates to element indices for slicing.
                src = src[tuple(i * b for i, b in zip(src_coord, block_shape))]
            except KeyError as e:
                raise ValueError(
                    f"MeshTensor information for buffer {src} not found."
                ) from e
        if dst_coord is not None:
            try:
                info = _mesh_tensor_info[dst.data]
                block_shape = info["block_shape"]
                dst = dst[tuple(i * b for i, b in zip(dst_coord, block_shape))]
            except KeyError as e:
                raise ValueError(
                    f"MeshTensor information for buffer {dst} not found."
                ) from e
        # Use tilelang's copy primitive to perform the copy.
        return T.copy(src, dst)

    return {
        "annotate_mesh_tensor_info": annotate_mesh_tensor_info,
        "mesh_tensor_copy": mesh_tensor_copy,
        "get_tile_shape": get_tile_shape,
    }
