"""MeshTensor: Distributed tensor abstraction for multi-chip mesh execution."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, TYPE_CHECKING

from tvm import ir, tir
from tvm.tir import PrimExpr, IntImm
from tvm.script.ir_builder.tir import buffer as tir_buffer

import tvm_ffi
from tvm_ffi.container import Map

from tilelang._typing import DType, ShapeType
from tilelang.dtypes import dtype as tilelang_dtype
from tilelang.language import dtypes as _dtypes
from tilelang.language.placement import (
    MeshReplicationType,
    MeshShardingPolicy,
    PlacementSpec,
    _PlacementKind,
    _placement_metadata,
    _validate_placement,
)
from tilelang.language.dist import (
    RankPlacementSpec,
    _current_world_size,
    _rank_placement_metadata,
    _validate_rank_placement,
    placement as rank_placement_namespace,
)
from tilelang.language.proxy import TensorProxy
from tilelang.language.mesh_symbols import mesh_ncols, mesh_nrows

__all__ = [
    "MeshReplicationType",
    "MeshShardingPolicy",
    "PlacementSpec",
    "MeshTensor",
    "TensorWithMeta",
    "get_rank_extent",
    "get_local_extent",
]

# FFI functions for layout operations
_make_aligned_row_major = tvm_ffi.get_global_func("tl.sunmmio.make_aligned_row_major")
_make_zz = tvm_ffi.get_global_func("tl.sunmmio.make_zz")
_make_mxzz = tvm_ffi.get_global_func("tl.sunmmio.make_mxzz")
_derive_layout_like = tvm_ffi.get_global_func("tl.DeriveLayoutLike")
_derive_mx_layout_like = tvm_ffi.get_global_func("tl.sunmmio.derive_mx_layout_like")

_DEFAULT_ZZ_BLOCK_SHAPE = (32, 32)
_DEFAULT_1D_ALIGNMENT_BYTES = 1024


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

    @property
    def rank_shape(self):
        """Return the uniform physical shape held by one Rank."""
        return self.meta_data.get("rank_shape", self.global_shape)

    def get_rank_extent(self, rank_id):
        """Return the valid extent held by ``rank_id``."""
        return get_rank_extent(self, rank_id)

    def get_local_extent(self, cid=None, *, rank_id=None):
        """Return the valid local extent on ``cid`` or the current kernel core."""
        return get_local_extent(self, cid, rank_id=rank_id)


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

    @property
    def rank_shape(self):
        """Return the uniform physical shape held by one Rank."""
        return self.meta_data.get("rank_shape", self.global_shape)

    def get_rank_extent(self, rank_id):
        """Return the valid extent held by ``rank_id``."""
        return get_rank_extent(self, rank_id)

    def get_local_extent(self, cid=None, *, rank_id=None):
        """Return the valid local extent on ``cid`` or the current kernel core."""
        return get_local_extent(self, cid, rank_id=rank_id)

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
    return tir.ceildiv(_to_primexpr(D) - _to_primexpr(k), _to_primexpr(n))


def lookup_mesh_tensor_meta(mesh_tensor):
    """Return MeshTensor metadata from a wrapper, dict, or annotated Buffer."""
    if isinstance(mesh_tensor, (TensorWithMeta, MeshTensorValue)):
        return mesh_tensor.meta_data
    if isinstance(mesh_tensor, (dict, Map)):
        return mesh_tensor
    meta = getattr(mesh_tensor, "_tilelang_mesh_tensor_meta", None)
    if meta is not None:
        return meta
    raise TypeError(f"Expected a MeshTensor value with metadata, got {type(mesh_tensor)}")


def get_rank_extent(mesh_tensor, rank_id):
    """Return the valid logical extent assigned to ``rank_id``."""

    meta = lookup_mesh_tensor_meta(mesh_tensor)
    global_shape = meta["global_shape"]
    placement_kind, shard_dim = (_to_python_int(value) for value in meta.get("rank_placement", (0, -1)))
    if placement_kind == 0:
        return tuple(global_shape)

    world_size = meta.get("world_size")
    if world_size is None:
        raise ValueError("Rank-sharded MeshTensor metadata is missing world_size")
    rank_extent = list(global_shape)
    rank_extent[shard_dim] = distribute_valid_count(global_shape[shard_dim], rank_id, world_size)
    return tuple(rank_extent)


def get_local_extent(mesh_tensor, cid=None, *, rank_id=None):
    """Return the valid local extent for ``mesh_tensor`` on linear core id ``cid``.

    When ``cid`` is omitted inside a kernel, use its current block binding.

    Full sharding preserves the physical mesh-axis order: row sharding is
    applied first, then column sharding is applied to the row-local extent.
    ``mesh_as_line`` instead uses the row-major linear core id.
    """
    if cid is None:
        from tilelang.language.kernel import get_block_binding

        cid = get_block_binding(0)

    meta = lookup_mesh_tensor_meta(mesh_tensor)
    placement_kind = _to_python_int(meta.get("rank_placement", (0, -1))[0])
    if placement_kind == 1:
        if rank_id is None:
            raise ValueError("rank_id is required for get_local_extent on a Rank-sharded MeshTensor")
        rank_extent = get_rank_extent(meta, rank_id)
    else:
        rank_extent = tuple(meta["global_shape"])
    nrows, ncols = meta["mesh_shape"]
    row = cid // ncols
    col = cid % ncols

    placement_desc = meta.get("placement")
    if placement_desc is None:
        # Support metadata created before PlacementSpec became canonical.
        local_extent = list(rank_extent)
        cross_mesh_dim = meta.get("cross_mesh_dim", -1)
        if cross_mesh_dim != -1:
            local_extent[cross_mesh_dim] = distribute_valid_count(rank_extent[cross_mesh_dim], cid, nrows * ncols)
            return tuple(local_extent)

        shard_y = meta.get("shard_y", -1)
        if shard_y != -1:
            local_extent[shard_y] = distribute_valid_count(rank_extent[shard_y], row, nrows)

        shard_x = meta.get("shard_x", -1)
        if shard_x != -1:
            local_extent[shard_x] = distribute_valid_count(local_extent[shard_x], col, ncols)
        return tuple(local_extent)

    row_kind, row_dim, col_kind, col_dim = (_to_python_int(value) for value in placement_desc)
    local_extent = list(rank_extent)
    core_placement_kind = _to_python_int(meta.get("placement_kind", -1))
    if core_placement_kind == _PlacementKind.MESH_AS_LINE:
        local_extent[row_dim] = distribute_valid_count(rank_extent[row_dim], cid, nrows * ncols)
        return tuple(local_extent)

    for dim, extent in enumerate(rank_extent):
        row_shards = row_kind == 1 and row_dim == dim
        col_shards = col_kind == 1 and col_dim == dim
        if row_shards:
            local_extent[dim] = distribute_valid_count(extent, row, nrows)
        if col_shards:
            local_extent[dim] = distribute_valid_count(local_extent[dim], col, ncols)

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


def _make_default_mesh_tensor_layout(shape, dtype):
    rank = len(shape)
    if rank == 0:
        raise ValueError("MeshTensor requires rank >= 1")

    if rank == 1:
        if _is_mx_dtype(dtype):
            raise ValueError("Rank-1 MeshTensor does not support SUVM MX dtypes")
        return _make_aligned_row_major(shape, dtype, _DEFAULT_1D_ALIGNMENT_BYTES)

    axes = [rank - 2, rank - 1]
    if _is_mx_dtype(dtype):
        return _make_mxzz(shape, axes, dtype)

    block_shape = [_to_primexpr(v) for v in _DEFAULT_ZZ_BLOCK_SHAPE]
    return _make_zz(shape, axes, block_shape)


class MeshTensorProxy:
    """Proxy for creating distributed mesh tensors.

    Computes per-core shapes from a row/column placement, then delegates to the
    standard TIR buffer creation.
    """

    @staticmethod
    def _get_sharded_shape(
        shape: tuple[Any, ...],
        placement: PlacementSpec | MeshShardingPolicy,
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

    @staticmethod
    def _get_rank_shape(
        shape: tuple[Any, ...],
        placement: RankPlacementSpec,
        world_size: int,
    ) -> tuple[Any, ...]:
        rank_shape = list(shape)
        if placement.kind == 1:
            rank_shape[placement.dim] = _ceildiv(rank_shape[placement.dim], world_size)
        return tuple(rank_shape)

    def __call__(
        self,
        shape: ShapeType,
        placement: PlacementSpec | MeshShardingPolicy | None = None,
        device_mesh_config: tuple[int | PrimExpr, int | PrimExpr] | DType | None = None,
        dtype: DType = "float32",
        layout=None,
        *,
        sharding_policy: PlacementSpec | MeshShardingPolicy | None = None,
        rank_placement: RankPlacementSpec | None = None,
    ) -> TensorWithMeta:
        if sharding_policy is not None:
            if placement is not None:
                raise TypeError("Specify only one of placement or the legacy sharding_policy argument")
            placement = sharding_policy
        if placement is None:
            raise TypeError("MeshTensor requires a placement")
        if isinstance(shape, (int, PrimExpr)):
            shape = (shape,)
        placement = _validate_placement(placement, len(shape))
        configured_world_size = _current_world_size(allow_none=True)
        explicit_rank_placement = rank_placement is not None
        if rank_placement is None:
            rank_placement = rank_placement_namespace.replicated()
        else:
            rank_placement = _validate_rank_placement(rank_placement, len(shape))
        world_size = configured_world_size if configured_world_size is not None else 1
        if device_mesh_config is not None and not _is_mesh_config(device_mesh_config):
            if not _is_dtype_like(device_mesh_config):
                raise TypeError("device_mesh_config must be a tuple of (nrows, ncols). To omit it, pass dtype as the third argument.")
            dtype = device_mesh_config
            device_mesh_config = None
        if device_mesh_config is None:
            device_mesh_config = (mesh_nrows(), mesh_ncols())
        dtype = _dtypes.normalize_dtype(dtype)
        nrows, ncols = device_mesh_config
        rank_shape = self._get_rank_shape(shape, rank_placement, world_size)
        sharded_shape = self._get_sharded_shape(rank_shape, placement, nrows, ncols)
        rank_strides = TensorProxy._construct_strides(rank_shape)
        sharded_strides = TensorProxy._construct_strides(sharded_shape)
        shape_exprs = [_to_primexpr(s) for s in shape]
        rank_shape_exprs = [_to_primexpr(s) for s in rank_shape]
        sharded_shape_exprs = [_to_primexpr(s) for s in sharded_shape]

        meta_data = dict(
            global_shape=shape,
            global_strides=TensorProxy._construct_strides(shape),
            rank_shape=rank_shape,
            rank_strides=rank_strides,
            local_shape=sharded_shape,
            local_strides=sharded_strides,
            mesh_shape=(nrows, ncols),
            rank_placement=_rank_placement_metadata(rank_placement),
            placement=_placement_metadata(placement),
            placement_kind=placement.kind,
        )
        if configured_world_size is not None or explicit_rank_placement:
            meta_data["world_size"] = world_size

        # Build global and per-shard layouts (CuteLayout objects).
        if layout is not None:
            global_layout = layout
            if _is_mx_dtype(dtype):
                rank_layout = _derive_mx_layout_like(global_layout, rank_shape_exprs, dtype)
                if not rank_layout:
                    raise ValueError("MeshTensor Rank sharding cannot derive a supported SUVM MX layout.")
                sharded_layout = _derive_mx_layout_like(rank_layout, sharded_shape_exprs, dtype)
                if not sharded_layout:
                    raise ValueError(
                        "MeshTensor with SUVM MX dtype only supports MX row-major, "
                        "MXZZ, or MXZNZ external layouts. Omit layout or use "
                        "make_mx_row_major_layout(...), make_mxzz_layout(...), "
                        "or make_mxznz_layout(...)."
                    )
            else:
                rank_layout = _derive_layout_like(global_layout, rank_shape_exprs, None)
                sharded_layout = _derive_layout_like(rank_layout, sharded_shape_exprs, None)
        else:
            global_layout = _make_default_mesh_tensor_layout(shape_exprs, dtype)
            rank_layout = _make_default_mesh_tensor_layout(rank_shape_exprs, dtype)
            sharded_layout = _make_default_mesh_tensor_layout(sharded_shape_exprs, dtype)

        meta_data["global_layout"] = global_layout
        meta_data["rank_layout"] = rank_layout
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
        rank_shape: tuple[Any, ...]
        local_shape: tuple[Any, ...]

        def __new__(
            cls,
            shape: ShapeType,
            placement: PlacementSpec | MeshShardingPolicy | None = None,
            device_mesh_config: tuple[int | PrimExpr, int | PrimExpr] | DType | None = None,
            dtype: DType = "float32",
            layout=None,
            *,
            sharding_policy: PlacementSpec | MeshShardingPolicy | None = None,
            rank_placement: RankPlacementSpec | None = None,
        ) -> TensorWithMeta: ...

        def get_rank_extent(self, rank_id) -> tuple[Any, ...]: ...

        def get_local_extent(self, cid=None, *, rank_id=None) -> tuple[Any, ...]: ...

else:
    MeshTensor = MeshTensorProxy()
