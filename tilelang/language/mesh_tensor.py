"""MeshTensor: Distributed tensor abstraction for multi-chip mesh execution."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, TYPE_CHECKING

from tvm import tir
from tvm.tir import PrimExpr
from tvm.script.ir_builder.tir import buffer as tir_buffer

from tilelang._typing import DType, ShapeType
from tilelang.language.proxy import TensorProxy

__all__ = [
    "MeshReplicationType",
    "MeshShardingPolicy",
    "MeshTensor",
    "TensorWithMeta",
]


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
        replicate: MeshReplicationType = MeshReplicationType.NONE,
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
            sharded_shape[policy.cross_mesh_dim] = int(math.ceil(sharded_shape[policy.cross_mesh_dim] / (nrows * ncols)))
            return tuple(sharded_shape)

        if policy.replicate == MeshReplicationType.ROW:
            if policy.x is not None:
                raise ValueError("Cannot shard on x-axis when replicating on rows")
            if policy.y is not None:
                if not 0 <= policy.y < len(sharded_shape):
                    raise ValueError(f"Invalid y-split dimension: {policy.y}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.y] = int(math.ceil(sharded_shape[policy.y] / nrows))
        elif policy.replicate == MeshReplicationType.COLUMN:
            if policy.y is not None:
                raise ValueError("Cannot shard on y-axis when replicating on columns")
            if policy.x is not None:
                if not 0 <= policy.x < len(sharded_shape):
                    raise ValueError(f"Invalid x-split dimension: {policy.x}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.x] = int(math.ceil(sharded_shape[policy.x] / ncols))
        elif policy.replicate == MeshReplicationType.NONE:
            if policy.x is not None:
                if not 0 <= policy.x < len(sharded_shape):
                    raise ValueError(f"Invalid x-split dimension: {policy.x}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.x] = int(math.ceil(sharded_shape[policy.x] / ncols))
            if policy.y is not None:
                if not 0 <= policy.y < len(sharded_shape):
                    raise ValueError(f"Invalid y-split dimension: {policy.y}, tensor rank is {len(sharded_shape)}")
                sharded_shape[policy.y] = int(math.ceil(sharded_shape[policy.y] / nrows))

        return tuple(sharded_shape)

    @staticmethod
    def _get_sharded_hierarchical_layout(
        hdims: tuple[int, ...],
        hgroups: tuple[tuple[int, int], ...],
        policy: MeshShardingPolicy,
        nrows: int,
        ncols: int,
    ) -> tuple[int, ...]:
        sharded_hdims = list(hdims)
        sharded_dims = []

        if policy.cross_mesh_dim is not None:
            sharded_dims.append((policy.cross_mesh_dim, nrows * ncols))
        else:
            if policy.y is not None:
                sharded_dims.append((policy.y, nrows))
            if policy.x is not None:
                sharded_dims.append((policy.x, ncols))

        for logical_dim_to_shard, shard_factor in sharded_dims:
            if shard_factor == 1:
                continue
            group_start, group_end = hgroups[logical_dim_to_shard]
            if group_start < group_end:
                hdim_to_shard_idx = group_start
                hdim_to_shard = hdims[hdim_to_shard_idx]
                if hdim_to_shard % shard_factor != 0:
                    raise ValueError(
                        f"The most significant hierarchical dimension ({hdim_to_shard}) of logical dimension "
                        f"{logical_dim_to_shard} is not divisible by the shard factor ({shard_factor})."
                    )
                sharded_hdims[hdim_to_shard_idx] = hdim_to_shard // shard_factor
        return tuple(sharded_hdims)

    @staticmethod
    def _derive_sharded_hstrides(
        sharded_hdims: tuple[int, ...],
        global_hstrides: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not sharded_hdims:
            return ()

        perm = sorted(range(len(global_hstrides)), key=lambda i: global_hstrides[i])

        sharded_hstrides = [0] * len(sharded_hdims)
        current_stride = 1
        for idx in perm:
            sharded_hstrides[idx] = current_stride
            current_stride *= sharded_hdims[idx]

        return tuple(sharded_hstrides)

    def __call__(
        self,
        shape: ShapeType,
        sharding_policy: MeshShardingPolicy,
        device_mesh_config: tuple[int, int],
        dtype: DType = "float32",
        hierarchical_dims: tuple[int, ...] | None = None,
        hierarchical_strides: tuple[int, ...] | None = None,
        hierarchical_groups: tuple[tuple[int, int], ...] | None = None,
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

        if hierarchical_dims is not None:
            sharded_hdims = self._get_sharded_hierarchical_layout(hierarchical_dims, hierarchical_groups, sharding_policy, nrows, ncols)
            sharded_hstrides = self._derive_sharded_hstrides(sharded_hdims, hierarchical_strides)
            meta_data["global_hdims"] = hierarchical_dims
            meta_data["global_hstrides"] = hierarchical_strides
            meta_data["global_hgroups"] = hierarchical_groups
            meta_data["sharded_hdims"] = sharded_hdims
            meta_data["sharded_hstrides"] = sharded_hstrides
            meta_data["sharded_hgroups"] = hierarchical_groups
        else:
            meta_data["global_hdims"] = shape
            meta_data["global_hstrides"] = TensorProxy._construct_strides(shape)
            meta_data["global_hgroups"] = tuple((i, i + 1) for i in range(len(shape)))
            meta_data["sharded_hdims"] = sharded_shape
            meta_data["sharded_hstrides"] = sharded_strides
            meta_data["sharded_hgroups"] = tuple((i, i + 1) for i in range(len(shape)))

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
            hierarchical_dims: tuple[int, ...] | None = None,
            hierarchical_strides: tuple[int, ...] | None = None,
            hierarchical_groups: tuple[tuple[int, int], ...] | None = None,
        ) -> TensorWithMeta: ...

else:
    MeshTensor = MeshTensorProxy()
