"""Mesh placement primitives for distributed Sunmmio tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

__all__ = [
    "Placement",
    "Shard",
    "Replicate",
    "MeshShardingPolicy",
    "PlacementSpec",
    "S",
    "R",
    "full_shard",
    "row_shard",
    "col_shard",
    "replicated",
    "full_replicated",
]


class Placement:
    """Base class for a placement on one mesh axis."""


@dataclass(frozen=True)
class Shard(Placement):
    """Shard a tensor dimension along one mesh axis."""

    dim: int

    def __post_init__(self) -> None:
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 0:
            raise ValueError(f"Shard.dim must be a non-negative int, got {self.dim!r}")

    def __repr__(self) -> str:
        return f"S({self.dim})"


@dataclass(frozen=True)
class Replicate(Placement):
    """Replicate a tensor along one mesh axis."""

    def __repr__(self) -> str:
        return "R()"


AxisPlacement = Union[Shard, Replicate]


class MeshReplicationType(Enum):
    """Legacy replication modes retained for source compatibility."""

    NONE = 0
    ROW = 1
    COLUMN = 2
    ALL = 3


@dataclass(frozen=True, init=False)
class MeshShardingPolicy:
    """Placement of a tensor on the ordered ``[row, col]`` mesh axes.

    New code should pass a two-element ``[row, col]`` placement sequence directly
    to ``MeshTensor``. This wrapper retains support for the legacy
    ``x/y/replicate/cross_mesh_dim`` arguments while existing programs migrate.
    """

    placements: tuple[AxisPlacement, AxisPlacement]
    _legacy_args: tuple[int | None, int | None, MeshReplicationType, int | None] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __init__(
        self,
        placements: Sequence[AxisPlacement] | int | None = None,
        y: int | None = None,
        replicate: int | MeshReplicationType = MeshReplicationType.NONE,
        cross_mesh_dim: int | None = None,
        *,
        x: int | None = None,
    ) -> None:
        replicate = self._normalize_replicate(replicate)
        if isinstance(placements, Sequence) and not isinstance(placements, (str, bytes)):
            if x is not None or y is not None or cross_mesh_dim is not None or replicate != MeshReplicationType.NONE:
                raise ValueError("placements cannot be combined with legacy sharding arguments")
            axes = self._normalize_axes(placements)
            object.__setattr__(self, "placements", axes)
            object.__setattr__(self, "_legacy_args", None)
            return

        if placements is not None:
            if x is not None:
                raise TypeError("x was specified both positionally and by keyword")
            if isinstance(placements, bool) or not isinstance(placements, int):
                raise TypeError("the first legacy argument x must be an int or None")
            x = placements
        if cross_mesh_dim is not None and (x is not None or y is not None):
            raise ValueError("cross_mesh_dim is mutually exclusive with x/y splits")

        axes = self._from_legacy_args(x, y, replicate, cross_mesh_dim)
        object.__setattr__(self, "placements", axes)
        object.__setattr__(self, "_legacy_args", (x, y, replicate, cross_mesh_dim))

    @staticmethod
    def _normalize_replicate(replicate: int | MeshReplicationType) -> MeshReplicationType:
        if isinstance(replicate, int):
            replicate = MeshReplicationType(replicate)
        if not isinstance(replicate, MeshReplicationType):
            raise TypeError(f"replicate must be a MeshReplicationType or int, got {type(replicate).__name__}")
        return replicate

    @staticmethod
    def _normalize_axes(placements: Sequence[AxisPlacement]) -> tuple[AxisPlacement, AxisPlacement]:
        if isinstance(placements, (str, bytes)) or not isinstance(placements, Sequence):
            raise TypeError("placements must be a sequence of [row_placement, col_placement]")
        if len(placements) != 2:
            raise ValueError(f"placements must contain exactly two mesh axes, got {len(placements)}")
        row, col = placements
        for axis, value in (("row", row), ("col", col)):
            if not isinstance(value, (Shard, Replicate)):
                raise TypeError(f"{axis} placement must be Shard or Replicate, got {type(value).__name__}")
        return row, col

    @staticmethod
    def _from_legacy_args(
        x: int | None,
        y: int | None,
        replicate: MeshReplicationType,
        cross_mesh_dim: int | None,
    ) -> tuple[AxisPlacement, AxisPlacement]:
        if replicate == MeshReplicationType.ALL:
            return Replicate(), Replicate()
        if cross_mesh_dim is not None:
            return Shard(cross_mesh_dim), Shard(cross_mesh_dim)
        if replicate == MeshReplicationType.ROW:
            return (Shard(y) if y is not None else Replicate()), Replicate()
        if replicate == MeshReplicationType.COLUMN:
            return Replicate(), (Shard(x) if x is not None else Replicate())
        return (
            Shard(y) if y is not None else Replicate(),
            Shard(x) if x is not None else Replicate(),
        )

    @property
    def row(self) -> AxisPlacement:
        return self.placements[0]

    @property
    def col(self) -> AxisPlacement:
        return self.placements[1]

    def validate(self, tensor_rank: int) -> None:
        """Validate shard dimensions against a tensor rank."""
        if self._legacy_args is not None:
            x, y, replicate, cross_mesh_dim = self._legacy_args
            if replicate == MeshReplicationType.ALL:
                return
            if cross_mesh_dim is not None:
                if not 0 <= cross_mesh_dim < tensor_rank:
                    raise ValueError(f"Invalid cross_mesh_dim: {cross_mesh_dim}, tensor rank is {tensor_rank}")
                return
            if replicate == MeshReplicationType.ROW and x is not None:
                raise ValueError("Cannot shard on x-axis when replicating on rows")
            if replicate == MeshReplicationType.COLUMN and y is not None:
                raise ValueError("Cannot shard on y-axis when replicating on columns")
            if y is not None and not 0 <= y < tensor_rank:
                raise ValueError(f"Invalid y-split dimension: {y}, tensor rank is {tensor_rank}")
            if x is not None and not 0 <= x < tensor_rank:
                raise ValueError(f"Invalid x-split dimension: {x}, tensor rank is {tensor_rank}")
            return

        for axis_name, axis in (("row", self.row), ("col", self.col)):
            if isinstance(axis, Shard) and axis.dim >= tensor_rank:
                raise ValueError(f"Invalid {axis_name} shard dimension: {axis.dim}, tensor rank is {tensor_rank}")

    @property
    def x(self) -> int | None:
        if self._legacy_args is not None:
            return self._legacy_args[0]
        return self.col.dim if isinstance(self.col, Shard) else None

    @property
    def y(self) -> int | None:
        if self._legacy_args is not None:
            return self._legacy_args[1]
        return self.row.dim if isinstance(self.row, Shard) else None

    @property
    def replicate(self) -> MeshReplicationType:
        if self._legacy_args is not None:
            return self._legacy_args[2]
        if isinstance(self.row, Replicate) and isinstance(self.col, Replicate):
            return MeshReplicationType.ALL
        if isinstance(self.col, Replicate):
            return MeshReplicationType.ROW
        if isinstance(self.row, Replicate):
            return MeshReplicationType.COLUMN
        return MeshReplicationType.NONE

    @property
    def cross_mesh_dim(self) -> int | None:
        if self._legacy_args is not None:
            return self._legacy_args[3]
        if isinstance(self.row, Shard) and isinstance(self.col, Shard) and self.row.dim == self.col.dim:
            return self.row.dim
        return None

    def __iter__(self):
        return iter(self.placements)

    def __len__(self) -> int:
        return len(self.placements)

    def __getitem__(self, index: int) -> AxisPlacement:
        return self.placements[index]

    def __repr__(self) -> str:
        return f"MeshShardingPolicy([{self.row!r}, {self.col!r}])"


PlacementSpec = Union[MeshShardingPolicy, Sequence[AxisPlacement]]


def _normalize_placement(placement: PlacementSpec) -> MeshShardingPolicy:
    """Convert a public placement specification to the internal policy form."""
    if isinstance(placement, MeshShardingPolicy):
        return placement
    if isinstance(placement, Sequence) and not isinstance(placement, (str, bytes)):
        return MeshShardingPolicy(placement)
    raise TypeError(
        f"placement must be a two-element sequence of Shard/Replicate values or a MeshShardingPolicy, got {type(placement).__name__}"
    )


def S(dim: int) -> Shard:
    """Construct a shard placement."""
    return Shard(dim)


def R() -> Replicate:
    """Construct a replicated placement."""
    return Replicate()


def full_shard(row_dim: int, col_dim: int) -> MeshShardingPolicy:
    """Shard along both mesh axes."""
    return MeshShardingPolicy((Shard(row_dim), Shard(col_dim)))


def row_shard(dim: int) -> MeshShardingPolicy:
    """Shard along the mesh row axis and replicate along columns."""
    return MeshShardingPolicy((Shard(dim), Replicate()))


def col_shard(dim: int) -> MeshShardingPolicy:
    """Replicate along rows and shard along the mesh column axis."""
    return MeshShardingPolicy((Replicate(), Shard(dim)))


def replicated() -> MeshShardingPolicy:
    """Replicate a tensor along both mesh axes."""
    return MeshShardingPolicy((Replicate(), Replicate()))


def full_replicated() -> MeshShardingPolicy:
    """Alias for :func:`replicated`."""
    return replicated()


def _placement_metadata(policy: MeshShardingPolicy) -> tuple[int, int, int, int]:
    def encode(axis: AxisPlacement) -> tuple[int, int]:
        if isinstance(axis, Shard):
            return 1, axis.dim
        return 0, -1

    row_kind, row_dim = encode(policy.row)
    col_kind, col_dim = encode(policy.col)
    return row_kind, row_dim, col_kind, col_dim
