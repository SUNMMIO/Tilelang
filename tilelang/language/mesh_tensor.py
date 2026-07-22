"""MeshTensor: Distributed tensor abstraction for multi-chip mesh execution."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, TYPE_CHECKING

from tvm import ir, tir
from tvm.tir import PrimExpr, IntImm
from tvm.script.ir_builder.tir import buffer as tir_buffer

import tvm_ffi

from tilelang._typing import DType, ShapeType
from tilelang.dtypes import dtype as tilelang_dtype
from tilelang.language import dtypes as _dtypes
from tilelang.language.placement import (
    PlacementSpec,
    _normalize_placement,
    _placement_metadata,
    _validate_placement,
)
from tilelang.language.proxy import TensorProxy
from tilelang.language.mesh_symbols import mesh_ncols, mesh_nrows

__all__ = [
    "PlacementSpec",
    "MeshTensor",
    "TensorWithMeta",
    "get_local_extent",
]

# FFI functions for layout operations
_make_row_major = tvm_ffi.get_global_func("tl.sunmmio.make_row_major")
_make_mx_row_major = tvm_ffi.get_global_func("tl.sunmmio.make_mx_row_major")
_derive_layout_like = tvm_ffi.get_global_func("tl.DeriveLayoutLike")
_derive_mx_layout_like = tvm_ffi.get_global_func("tl.sunmmio.derive_mx_layout_like")


class TensorWithMeta:
    """A tensor buffer paired with metadata (e.g., global shape/strides)."""

    def __init__(self, buffer: tir.Buffer, meta_data: dict):
        self.buffer = buffer
        self.meta_data = meta_data
        self._attach_meta(buffer, meta_data)

    @staticmethod
    def _attach_meta(buffer: tir.Buffer, meta_data: dict) -> None:
        with suppress(AttributeError):
            buffer._tilelang_mesh_tensor_meta = meta_data

    @property
    def global_shape(self):
        """Return the user-visible global tensor shape."""
        return self.meta_data["global_shape"]

    @property
    def local_shape(self):
        """Return the uniform physical local buffer shape."""
        return self.meta_data["local_shape"]

    def get_local_extent(self, cid):
        """Return the valid local extent on core ``cid``."""
        return get_local_extent(self, cid)


class MeshTensorValue:
    """Frontend value for a MeshTensor parameter inside a TileLang function."""

    def __init__(self, buffer: tir.Buffer, meta_data: dict):
        self.buffer = buffer
        self.meta_data = meta_data
        TensorWithMeta._attach_meta(buffer, meta_data)

    @property
    def global_shape(self):
        """Return the user-visible global tensor shape."""
        return self.meta_data["global_shape"]

    @property
    def local_shape(self):
        """Return the uniform physical local buffer shape."""
        return self.meta_data["local_shape"]

    def get_local_extent(self, cid):
        """Return the valid local extent on core ``cid``."""
        return get_local_extent(self, cid)

    def __getitem__(self, keys):
        return self.buffer[keys]

    def __setitem__(self, keys, value):
        self.buffer[keys] = value

    def __getattr__(self, name):
        if name == "shape":
            raise AttributeError("MeshTensor.shape is ambiguous. Use `.global_shape` or `.local_shape` instead.")
        return getattr(self.buffer, name)

    def __repr__(self):
        return f"MeshTensorValue(buffer={self.buffer!r}, global_shape={self.global_shape}, local_shape={self.local_shape})"


def _unwrap_mesh_tensor(value):
    """Return the backing TIR buffer for MeshTensor wrapper values."""
    if isinstance(value, (TensorWithMeta, MeshTensorValue)):
        return value.buffer
    return value


def _ceildiv(a, b):
    """Ceiling division that works for both Python int and TVM PrimExpr."""
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    return tir.ceildiv(a, b)


def _to_primexpr(v):
    """Convert a value to PrimExpr if it isn't one already."""
    if isinstance(v, int):
        return IntImm("int32", v)
    return v


def _to_python_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, IntImm):
        return int(v.value)
    return None


def distribute_valid_count(D, k, n):
    """Return the number of valid elements on core index ``k``.

    The first ``D % n`` cores receive one extra element. Supports Python ints
    and TIR PrimExpr values.
    """
    d_int = _to_python_int(D)
    k_int = _to_python_int(k)
    n_int = _to_python_int(n)
    if d_int is not None and k_int is not None and n_int is not None:
        base, rem = divmod(d_int, n_int)
        return base + (1 if k_int < rem else 0)

    base = D // n
    rem = D % n
    rem_int = _to_python_int(rem)
    if rem_int == 0:
        return base
    if rem_int is not None and k_int is not None:
        return base + (1 if k_int < rem_int else 0)
    return base + tir.Select(_to_primexpr(k) < _to_primexpr(rem), IntImm("int32", 1), IntImm("int32", 0))


def lookup_mesh_tensor_meta(mesh_tensor):
    """Return MeshTensor metadata from a wrapper, dict, or annotated Buffer."""
    if isinstance(mesh_tensor, (TensorWithMeta, MeshTensorValue)):
        return mesh_tensor.meta_data
    if isinstance(mesh_tensor, dict):
        return mesh_tensor
    meta = getattr(mesh_tensor, "_tilelang_mesh_tensor_meta", None)
    if meta is not None:
        return meta
    raise TypeError(f"Expected a MeshTensor value with metadata, got {type(mesh_tensor)}")


def get_local_extent(mesh_tensor, cid):
    """Return the valid local extent for ``mesh_tensor`` on linear core id ``cid``.

    When both mesh axes shard the same tensor dimension, cores are ordered by
    the row-major linear index ``row * ncols + col``.
    """
    meta = lookup_mesh_tensor_meta(mesh_tensor)
    global_shape = meta["global_shape"]
    nrows, ncols = meta["mesh_shape"]
    row = cid // ncols
    col = cid % ncols

    row_kind, row_dim, col_kind, col_dim = (_to_python_int(value) for value in meta["placement"])
    local_extent = list(global_shape)
    for dim, extent in enumerate(global_shape):
        row_shards = row_kind == 1 and row_dim == dim
        col_shards = col_kind == 1 and col_dim == dim
        if row_shards and col_shards:
            local_extent[dim] = distribute_valid_count(extent, row * ncols + col, nrows * ncols)
        elif row_shards:
            local_extent[dim] = distribute_valid_count(extent, row, nrows)
        elif col_shards:
            local_extent[dim] = distribute_valid_count(extent, col, ncols)

    return tuple(local_extent)


def _is_mesh_config(value):
    if not isinstance(value, tuple):
        return False
    if len(value) != 2:
        return False
    return all(isinstance(v, (int, PrimExpr)) for v in value)


def _is_dtype_like(value):
    return isinstance(value, (str, type, tilelang_dtype, ir.Type))


def _is_mx_dtype(dtype):
    return str(_dtypes.normalize_dtype(dtype)) in {"custom[mxfp8]8", "custom[mxfp4]4"}


class MeshTensorProxy:
    """Proxy for creating distributed mesh tensors.

    Computes per-core shapes from a row/column placement, then delegates to the
    standard TIR buffer creation.
    """

    @staticmethod
    def _get_sharded_shape(
        shape: tuple[Any, ...],
        placement: PlacementSpec,
        nrows: int,
        ncols: int,
    ) -> tuple[Any, ...]:
        placement = _validate_placement(placement, len(shape))
        sharded_shape = list(shape)

        for dim, extent in enumerate(sharded_shape):
            row_shards = placement.row_dim == dim
            col_shards = placement.col_dim == dim
            if not row_shards and not col_shards:
                continue
            shard_factor = 1
            if row_shards:
                shard_factor *= nrows
            if col_shards:
                shard_factor *= ncols
            sharded_shape[dim] = _ceildiv(extent, shard_factor)

        return tuple(sharded_shape)

    def __call__(
        self,
        shape: ShapeType,
        placement: PlacementSpec,
        device_mesh_config: tuple[int | PrimExpr, int | PrimExpr] | DType | None = None,
        dtype: DType = "float32",
        layout=None,
    ) -> TensorWithMeta:
        placement = _normalize_placement(placement)
        if isinstance(shape, (int, PrimExpr)):
            shape = (shape,)
        if device_mesh_config is not None and not _is_mesh_config(device_mesh_config):
            if not _is_dtype_like(device_mesh_config):
                raise TypeError("device_mesh_config must be a tuple of (nrows, ncols). To omit it, pass dtype as the third argument.")
            dtype = device_mesh_config
            device_mesh_config = None
        if device_mesh_config is None:
            device_mesh_config = (mesh_nrows(), mesh_ncols())
        dtype = _dtypes.normalize_dtype(dtype)
        nrows, ncols = device_mesh_config
        sharded_shape = self._get_sharded_shape(shape, placement, nrows, ncols)
        sharded_strides = TensorProxy._construct_strides(sharded_shape)
        shape_exprs = [_to_primexpr(s) for s in shape]
        sharded_shape_exprs = [_to_primexpr(s) for s in sharded_shape]

        meta_data = dict(
            global_shape=shape,
            global_strides=TensorProxy._construct_strides(shape),
            local_shape=sharded_shape,
            local_strides=sharded_strides,
            mesh_shape=(nrows, ncols),
            placement=_placement_metadata(placement),
        )

        # Build global layout (CuteLayout object).
        if layout is not None:
            global_layout = layout
        else:
            # Default: row-major CuteLayout
            if _is_mx_dtype(dtype):
                global_layout = _make_mx_row_major(shape_exprs, dtype)
            else:
                global_layout = _make_row_major(shape_exprs)

        # Derive sharded layout via DeriveLayoutLike.
        if _is_mx_dtype(dtype):
            sharded_layout = _derive_mx_layout_like(global_layout, sharded_shape_exprs, dtype)
            if not sharded_layout:
                raise ValueError(
                    "MeshTensor with SUVM MX dtype only supports MX row-major, "
                    "MXZZ, or MXZNN external layouts. Omit layout or use "
                    "make_mx_row_major_layout(...), make_mxzz_layout(...), "
                    "or make_mxznn_layout(...)."
                )
        else:
            sharded_layout = _derive_layout_like(global_layout, sharded_shape_exprs, None)

        meta_data["global_layout"] = global_layout
        meta_data["sharded_layout"] = sharded_layout

        buf = tir_buffer(
            sharded_shape,
            dtype=_dtypes.normalize_dtype(dtype),
            strides=sharded_strides,
            scope="global",
        )
        return TensorWithMeta(buf, meta_data)


if TYPE_CHECKING:

    class MeshTensor:
        global_shape: tuple[Any, ...]
        local_shape: tuple[Any, ...]

        def __new__(
            cls,
            shape: ShapeType,
            placement: PlacementSpec,
            device_mesh_config: tuple[int | PrimExpr, int | PrimExpr] | DType | None = None,
            dtype: DType = "float32",
            layout=None,
        ) -> TensorWithMeta: ...

        def get_local_extent(self, cid) -> tuple[Any, ...]: ...

else:
    MeshTensor = MeshTensorProxy()
