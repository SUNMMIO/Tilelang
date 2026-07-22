"""Mesh placement primitives for distributed Sunmmio tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

__all__ = [
    "Placement",
    "Shard",
    "Replicate",
    "PlacementSpec",
    "S",
    "R",
    "full_shard",
    "row_shard",
    "col_shard",
    "replicated",
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
PlacementSpec = Sequence[AxisPlacement]


def _normalize_placement(placement: PlacementSpec) -> tuple[AxisPlacement, AxisPlacement]:
    """Validate and freeze a public ``[row, col]`` placement sequence."""
    if isinstance(placement, (str, bytes)) or not isinstance(placement, Sequence):
        raise TypeError(f"placement must be a two-element sequence, got {type(placement).__name__}")
    if len(placement) != 2:
        raise ValueError(f"placement must contain exactly two mesh axes, got {len(placement)}")

    row, col = placement
    for axis_name, axis in (("row", row), ("col", col)):
        if not isinstance(axis, (Shard, Replicate)):
            raise TypeError(f"{axis_name} placement must be S(dim) or R(), got {type(axis).__name__}")
    return row, col


def _validate_placement(
    placement: PlacementSpec,
    tensor_rank: int,
) -> tuple[AxisPlacement, AxisPlacement]:
    """Validate shard dimensions against a tensor rank."""
    axes = _normalize_placement(placement)
    for axis_name, axis in (("row", axes[0]), ("col", axes[1])):
        if isinstance(axis, Shard) and axis.dim >= tensor_rank:
            raise ValueError(f"Invalid {axis_name} shard dimension: {axis.dim}, tensor rank is {tensor_rank}")
    return axes


def S(dim: int) -> Shard:
    """Construct a shard placement."""
    return Shard(dim)


def R() -> Replicate:
    """Construct a replicated placement."""
    return Replicate()


def full_shard(row_dim: int, col_dim: int) -> list[AxisPlacement]:
    """Shard along both mesh axes."""
    return [Shard(row_dim), Shard(col_dim)]


def row_shard(dim: int) -> list[AxisPlacement]:
    """Shard along the mesh row axis and replicate along columns."""
    return [Shard(dim), Replicate()]


def col_shard(dim: int) -> list[AxisPlacement]:
    """Replicate along rows and shard along the mesh column axis."""
    return [Replicate(), Shard(dim)]


def replicated() -> list[AxisPlacement]:
    """Replicate a tensor along both mesh axes."""
    return [Replicate(), Replicate()]


def _placement_metadata(placement: PlacementSpec) -> tuple[int, int, int, int]:
    def encode(axis: AxisPlacement) -> tuple[int, int]:
        if isinstance(axis, Shard):
            return 1, axis.dim
        return 0, -1

    row, col = _normalize_placement(placement)
    row_kind, row_dim = encode(row)
    col_kind, col_dim = encode(col)
    return row_kind, row_dim, col_kind, col_dim
