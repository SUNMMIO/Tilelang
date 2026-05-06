"""MeshTensor: Distributed tensor abstraction for multi-chip mesh execution."""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

from tvm import tir
from tvm.tir import PrimExpr, IntImm
from tvm.script.ir_builder.tir import buffer as tir_buffer

import tvm_ffi

from tilelang._typing import DType, ShapeType
from tilelang.language.proxy import TensorProxy

__all__ = [
    "MeshReplicationType",
    "MeshShardingPolicy",
    "MeshTensor",
    "TensorWithMeta",
]

# FFI functions for layout operations
_make_row_major = tvm_ffi.get_global_func("tl.sunmmio.make_row_major")
_derive_layout_like = tvm_ffi.get_global_func("tl.DeriveLayoutLike")


class MeshReplicationType(Enum):
    NONE = 0  # no replication (each core has unique data)
    ROW = 1  # replicate across X (same row)
    COLUMN = 2  # replicate across Y (same column)
    ALL = 3  # replicate on all cores


class MeshShardingPolicy:
    """Sharding Policy for MeshTensor."""

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        replicate: int = MeshReplicationType.NONE,
        cross_mesh_dim: int | None = None,
    ):
        if cross_mesh_dim is not None and (x is not None or y is not None):
            raise ValueError("cross_mesh_dim is mutually exclusive with x/y splits")
        if sum(v is not None for v in [x, y, cross_mesh_dim]) > 2:
            raise ValueError("Invalid layout: too many splits")

        self.x = x
        self.y = y
        self.replicate = replicate
        self.cross_mesh_dim = cross_mesh_dim

    def __repr__(self):
        if self.cross_mesh_dim is not None:
            return f"MeshLayout(split_dim={self.cross_mesh_dim} across XxY)"
        parts = []
        if self.x is not None:
            parts.append(f"x→dim{self.x}")
        if self.y is not None:
            parts.append(f"y→dim{self.y}")
        if self.replicate != MeshReplicationType.NONE:
            parts.append(f"replicate={self.replicate.name}")
        return "MeshLayout(" + ", ".join(parts) + ")" if parts else "MeshLayout(replicated)"


class TensorWithMeta:
    """A tensor buffer paired with metadata (e.g., global shape/strides)."""

    def __init__(self, buffer: tir.Buffer, meta_data: dict):
        self.buffer = buffer
        self.meta_data = meta_data


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


class MeshTensorProxy:
    """Proxy for creating distributed mesh tensors.

    Adapts MeshShardingPolicy to compute per-core sharded shapes,
    then delegates to the standard TIR buffer creation.
    """

    @staticmethod
    def _get_sharded_shape(
        shape: tuple[Any, ...],
        policy: MeshShardingPolicy,
        nrows: int,
        ncols: int,
    ) -> tuple[Any, ...]:
        sharded_shape = list(shape)

        if policy.replicate == MeshReplicationType.ALL:
            return tuple(sharded_shape)

        if policy.cross_mesh_dim is not None:
            if not 0 <= policy.cross_mesh_dim < len(sharded_shape):
                raise ValueError(f"Invalid cross_mesh_dim: {policy.cross_mesh_dim}, tensor rank is {len(sharded_shape)}")
            sharded_shape[policy.cross_mesh_dim] = _ceildiv(sharded_shape[policy.cross_mesh_dim], nrows * ncols)
            return tuple(sharded_shape)

        if policy.replicate == MeshReplicationType.ROW:
            if policy.x is not None:
                raise ValueError("Cannot shard on x-axis when replicating on rows")
            if policy.y is not None:
                if not 0 <= policy.y < len(sharded_shape):
                    raise ValueError(f"Invalid y-split dimension: {policy.y}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.y] = _ceildiv(sharded_shape[policy.y], nrows)
        elif policy.replicate == MeshReplicationType.COLUMN:
            if policy.y is not None:
                raise ValueError("Cannot shard on y-axis when replicating on columns")
            if policy.x is not None:
                if not 0 <= policy.x < len(sharded_shape):
                    raise ValueError(f"Invalid x-split dimension: {policy.x}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.x] = _ceildiv(sharded_shape[policy.x], ncols)
        elif policy.replicate == MeshReplicationType.NONE:
            if policy.x is not None:
                if not 0 <= policy.x < len(sharded_shape):
                    raise ValueError(f"Invalid x-split dimension: {policy.x}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.x] = _ceildiv(sharded_shape[policy.x], ncols)
            if policy.y is not None:
                if not 0 <= policy.y < len(sharded_shape):
                    raise ValueError(f"Invalid y-split dimension: {policy.y}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.y] = _ceildiv(sharded_shape[policy.y], nrows)

        return tuple(sharded_shape)

    def __call__(
        self,
        shape: ShapeType,
        sharding_policy: MeshShardingPolicy,
        device_mesh_config: tuple[int, int],
        dtype: DType = "float32",
        layout=None,
    ) -> TensorWithMeta:
        if isinstance(shape, (int, PrimExpr)):
            shape = (shape,)
        nrows, ncols = device_mesh_config
        sharded_shape = self._get_sharded_shape(shape, sharding_policy, nrows, ncols)
        sharded_strides = TensorProxy._construct_strides(sharded_shape)

        meta_data = dict(
            global_shape=shape,
            global_strides=TensorProxy._construct_strides(shape),
        )

        # Build global layout (CuteLayout object).
        if layout is not None:
            global_layout = layout
        else:
            # Default: row-major CuteLayout
            global_layout = _make_row_major([_to_primexpr(s) for s in shape])

        # Derive sharded layout via DeriveLayoutLike.
        sharded_shape_exprs = [_to_primexpr(s) for s in sharded_shape]
        sharded_layout = _derive_layout_like(global_layout, sharded_shape_exprs, None)

        meta_data["global_layout"] = global_layout
        meta_data["sharded_layout"] = sharded_layout

        buf = tir_buffer(
            sharded_shape,
            dtype=dtype,
            strides=sharded_strides,
            scope="global",
        )
        return TensorWithMeta(buf, meta_data)


if TYPE_CHECKING:

    class MeshTensor:
        def __new__(
            cls,
            shape: ShapeType,
            sharding_policy: MeshShardingPolicy,
            device_mesh_config: tuple[int, int],
            dtype: DType = "float32",
            layout=None,
        ) -> TensorWithMeta: ...

else:
    MeshTensor = MeshTensorProxy()
