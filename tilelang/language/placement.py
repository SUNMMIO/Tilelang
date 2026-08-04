"""Placement specifications for distributed SunMMIO tensors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

__all__ = [
    "PlacementSpec",
    "MeshShardingPolicy",
    "MeshReplicationType",
    "replicated",
    "row_shard",
    "col_shard",
    "full_shard",
    "mesh_as_line",
]


class _PlacementKind(IntEnum):
    REPLICATED = 0
    ROW_SHARD = 1
    COL_SHARD = 2
    FULL_SHARD = 3


def _validate_dim(dim: int, name: str) -> None:
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError(f"{name} must be a non-negative int, got {dim!r}")


@dataclass(frozen=True)
class PlacementSpec:
    """Immutable description of how a tensor is placed on a 2D mesh."""

    _kind: _PlacementKind
    row_dim: int = -1
    col_dim: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self._kind, _PlacementKind):
            raise TypeError("PlacementSpec values must be constructed with T.placement")

        if self._kind == _PlacementKind.REPLICATED:
            if self.row_dim != -1 or self.col_dim != -1:
                raise ValueError("Replicated placement cannot have shard dimensions")
        elif self._kind == _PlacementKind.ROW_SHARD:
            _validate_dim(self.row_dim, "dim")
            if self.col_dim != -1:
                raise ValueError("RowShard placement cannot have a column shard dimension")
        elif self._kind == _PlacementKind.COL_SHARD:
            if self.row_dim != -1:
                raise ValueError("ColShard placement cannot have a row shard dimension")
            _validate_dim(self.col_dim, "dim")
        elif self._kind == _PlacementKind.FULL_SHARD:
            _validate_dim(self.row_dim, "row_dim")
            _validate_dim(self.col_dim, "col_dim")

    @property
    def kind(self) -> int:
        """Return the placement kind identifier used by torch-sunmmio."""
        return int(self._kind)

    @staticmethod
    def replicated() -> PlacementSpec:
        return PlacementSpec(_PlacementKind.REPLICATED)

    @staticmethod
    def row_shard(dim: int) -> PlacementSpec:
        return PlacementSpec(_PlacementKind.ROW_SHARD, row_dim=dim)

    @staticmethod
    def col_shard(dim: int) -> PlacementSpec:
        return PlacementSpec(_PlacementKind.COL_SHARD, col_dim=dim)

    @staticmethod
    def full_shard(row_dim: int, col_dim: int) -> PlacementSpec:
        return PlacementSpec(_PlacementKind.FULL_SHARD, row_dim=row_dim, col_dim=col_dim)

    @staticmethod
    def mesh_as_line(dim: int) -> PlacementSpec:
        return PlacementSpec.full_shard(dim, dim)

    def __repr__(self) -> str:
        if self._kind == _PlacementKind.REPLICATED:
            return "Replicated()"
        if self._kind == _PlacementKind.ROW_SHARD:
            return f"RowShard({self.row_dim})"
        if self._kind == _PlacementKind.COL_SHARD:
            return f"ColShard({self.col_dim})"
        if self._kind == _PlacementKind.FULL_SHARD:
            return f"FullShard({self.row_dim}, {self.col_dim})"
        raise AssertionError(f"Unknown placement kind: {self._kind!r}")


class MeshReplicationType(Enum):
    """Legacy replication modes retained for source compatibility."""

    NONE = 0
    ROW = 1
    COLUMN = 2
    ALL = 3


class MeshShardingPolicy:
    """Legacy mesh placement syntax backed by :class:`PlacementSpec`.

    New code should construct placements with ``T.placement``. This wrapper
    keeps existing ``x/y/replicate/cross_mesh_dim`` call sites working.
    """

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        replicate: int | MeshReplicationType = MeshReplicationType.NONE,
        cross_mesh_dim: int | None = None,
    ) -> None:
        if isinstance(replicate, int):
            replicate = MeshReplicationType(replicate)
        if not isinstance(replicate, MeshReplicationType):
            raise TypeError(f"replicate must be a MeshReplicationType or int, got {type(replicate).__name__}")
        if cross_mesh_dim is not None and (x is not None or y is not None):
            raise ValueError("cross_mesh_dim is mutually exclusive with x/y splits")

        self.x = x
        self.y = y
        self.replicate = replicate
        self.cross_mesh_dim = cross_mesh_dim

    def validate(self, tensor_rank: int) -> None:
        """Validate legacy shard dimensions against a tensor rank."""
        if self.replicate == MeshReplicationType.ALL:
            return
        if self.cross_mesh_dim is not None:
            if not self._is_valid_dim(self.cross_mesh_dim, tensor_rank):
                raise ValueError(f"Invalid cross_mesh_dim: {self.cross_mesh_dim}, tensor rank is {tensor_rank}")
            return
        if self.replicate == MeshReplicationType.ROW and self.x is not None:
            raise ValueError("Cannot shard on x-axis when replicating on rows")
        if self.replicate == MeshReplicationType.COLUMN and self.y is not None:
            raise ValueError("Cannot shard on y-axis when replicating on columns")
        if self.y is not None and not self._is_valid_dim(self.y, tensor_rank):
            raise ValueError(f"Invalid y-split dimension: {self.y}, tensor rank is {tensor_rank}")
        if self.x is not None and not self._is_valid_dim(self.x, tensor_rank):
            raise ValueError(f"Invalid x-split dimension: {self.x}, tensor rank is {tensor_rank}")

    def to_placement_spec(self) -> PlacementSpec:
        """Convert the legacy policy to the canonical placement representation."""
        if self.replicate == MeshReplicationType.ALL:
            return PlacementSpec.replicated()
        if self.cross_mesh_dim is not None:
            return PlacementSpec.mesh_as_line(self.cross_mesh_dim)
        if self.replicate == MeshReplicationType.ROW:
            return PlacementSpec.row_shard(self.y) if self.y is not None else PlacementSpec.replicated()
        if self.replicate == MeshReplicationType.COLUMN:
            return PlacementSpec.col_shard(self.x) if self.x is not None else PlacementSpec.replicated()
        if self.y is not None and self.x is not None:
            return PlacementSpec.full_shard(self.y, self.x)
        if self.y is not None:
            return PlacementSpec.row_shard(self.y)
        if self.x is not None:
            return PlacementSpec.col_shard(self.x)
        return PlacementSpec.replicated()

    @staticmethod
    def _is_valid_dim(dim: object, tensor_rank: int) -> bool:
        return not isinstance(dim, bool) and isinstance(dim, int) and 0 <= dim < tensor_rank

    def __repr__(self) -> str:
        return f"MeshShardingPolicy(x={self.x!r}, y={self.y!r}, replicate={self.replicate!r}, cross_mesh_dim={self.cross_mesh_dim!r})"


def replicated() -> PlacementSpec:
    """Replicate a tensor along both mesh axes."""
    return PlacementSpec.replicated()


def row_shard(dim: int) -> PlacementSpec:
    """Shard a tensor dimension along mesh rows and replicate along columns."""
    return PlacementSpec.row_shard(dim)


def col_shard(dim: int) -> PlacementSpec:
    """Replicate along mesh rows and shard a tensor dimension along columns."""
    return PlacementSpec.col_shard(dim)


def full_shard(row_dim: int, col_dim: int) -> PlacementSpec:
    """Shard a tensor along both mesh axes."""
    return PlacementSpec.full_shard(row_dim, col_dim)


def mesh_as_line(dim: int) -> PlacementSpec:
    """Shard one tensor dimension across the row-major linearized mesh."""
    return PlacementSpec.mesh_as_line(dim)


def _normalize_placement(placement: PlacementSpec | MeshShardingPolicy) -> PlacementSpec:
    if isinstance(placement, MeshShardingPolicy):
        return placement.to_placement_spec()
    if isinstance(placement, PlacementSpec):
        return placement
    raise TypeError(
        f"placement must be a PlacementSpec constructed with T.placement or a MeshShardingPolicy, got {type(placement).__name__}"
    )


def _validate_placement(placement: PlacementSpec | MeshShardingPolicy, tensor_rank: int) -> PlacementSpec:
    """Validate shard dimensions against a tensor rank."""
    if isinstance(placement, MeshShardingPolicy):
        placement.validate(tensor_rank)
    placement = _normalize_placement(placement)
    for axis_name, dim in (("row", placement.row_dim), ("col", placement.col_dim)):
        if dim >= tensor_rank:
            raise ValueError(f"Invalid {axis_name} shard dimension: {dim}, tensor rank is {tensor_rank}")
    return placement


def _placement_metadata(placement: PlacementSpec | MeshShardingPolicy) -> tuple[int, int, int, int]:
    placement = _normalize_placement(placement)
    return (
        int(placement.row_dim >= 0),
        placement.row_dim,
        int(placement.col_dim >= 0),
        placement.col_dim,
    )
