from __future__ import annotations
import tvm_ffi
from tvm.ir import Node
from tvm.tir import Buffer, BufferLoad, BufferRegion
from tilelang import _ffi_api
from tilelang.layout.swizzle import _get_buffer_info


@tvm_ffi.register_object("tl.TileView")
class TileView(Node):
    """
    TileView represents how a buffer is partitioned into tiles for processing.

    A TileView captures:
    - buffer_shape: The original shape of the buffer
    - tile_shape: The shape of each tile (e.g., (16, 32) for a 2D tile)
    - index_map: Which dimensions of the buffer are tiled
    - tiled_buffer_shape: The resulting shape after tiling

    For Sunmmio target, the Tile unit processes fixed-size 2D tiles with
    constraints: width must be 32, height can be 8/16/32.
    """

    def __init__(self, buffer_shape, tile_shape, index_map):
        """
        Initialize a TileView object.

        Parameters
        ----------
        buffer_shape : list of int or PrimExpr
            The shape of the buffer being tiled.
        tile_shape : list of int or PrimExpr
            The shape of each tile.
        index_map : list of int
            Which dimensions of the buffer are tiled.
            Supports negative indices (e.g., -1 for last dim, -2 for second to last).

        Raises
        ------
        RuntimeError
            If buffer dimensions are not divisible by tile dimensions.
        """
        self.__init_handle_by_constructor__(_ffi_api.TileView, buffer_shape, tile_shape, index_map)

    @property
    def tile_shape(self):
        """
        Get the tile shape.

        Returns
        -------
        List[PrimExpr]
            The shape of each tile.
        """
        return _ffi_api.TileView_tile_shape(self)

    @property
    def index_map(self):
        """
        Get the index map (which dimensions are tiled).

        Returns
        -------
        List[PrimExpr]
            The indices of dimensions being tiled.
        """
        return _ffi_api.TileView_index_map(self)

    @property
    def buffer_shape(self):
        """
        Get the original buffer shape.

        Returns
        -------
        List[PrimExpr]
            The shape of the original buffer.
        """
        return _ffi_api.TileView_buffer_shape(self)

    @property
    def tiled_buffer_shape(self):
        """
        Get the tiled buffer shape.

        The tiled shape is computed by replacing each tiled dimension with
        num_tiles (buffer_dim / tile_dim), then appending the tile dimensions.

        For example, buffer_shape=(64, 128), tile_shape=(16, 32), index_map=(-2, -1)
        results in tiled_buffer_shape=(4, 4, 16, 32).

        Returns
        -------
        List[PrimExpr]
            The shape of the tiled buffer.
        """
        return _ffi_api.TileView_tiled_buffer_shape(self)

    @property
    def vector_lanes(self):
        """
        Compute the vector lanes (total elements per tile).

        Returns
        -------
        PrimExpr
            Product of tile_shape dimensions.
        """
        return _ffi_api.TileView_vector_lanes(self)

    def is_equal(self, other: TileView) -> bool:
        """
        Check if this TileView is equal to another.

        Parameters
        ----------
        other : TileView
            The TileView to compare with.

        Returns
        -------
        bool
            True if the TileViews are equal.
        """
        return _ffi_api.TileView_is_equal(self, other)

    def __repr__(self):
        return (f"TileView(buffer_shape={list(self.buffer_shape)}, "
                f"tile_shape={list(self.tile_shape)}, "
                f"index_map={list(self.index_map)}) -> "
                f"tiled_shape={list(self.tiled_buffer_shape)}")


def make_tileview(buffer: Buffer | BufferLoad | BufferRegion, tile_shape, index_map):
    """
    Create a TileView from a buffer, tile shape, and index map.

    Parameters
    ----------
    buffer : Buffer | BufferLoad | BufferRegion
        The buffer being tiled. The shape will be extracted automatically.
    tile_shape : list of int or PrimExpr
        The shape of each tile.
    index_map : list of int
        Which dimensions of the buffer are tiled.
        Supports negative indices (e.g., -1 for last dim).

    Returns
    -------
    TileView
        A TileView object.

    Raises
    ------
    RuntimeError
        If buffer dimensions are not divisible by tile dimensions.

    Examples
    --------
    >>> # Tile a buffer into (16, 32) tiles on the last two dimensions
    >>> tv = make_tileview(my_buffer, [16, 32], [-2, -1])
    >>> print(tv.tiled_buffer_shape)  # e.g., [4, 4, 16, 32] for a (64, 128) buffer
    """
    _, buffer_shape, _ = _get_buffer_info(buffer)
    return _ffi_api.make_tileview(buffer_shape, tile_shape, index_map)
