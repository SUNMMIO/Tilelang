"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations

from dataclasses import dataclass

import math

from tvm.tir import Buffer, BufferLoad, BufferRegion
from tilelang import _ffi_api


@dataclass
class HierarchicalLayout:
    hdims: list[int]
    hstrides: list[int]
    groups: list[tuple[int]]

    def __post_init__(self):
        assert len(self.hdims) == len(self.hstrides), "hdims and hstrides must have the same length"

    @property
    def ndim(self) -> int:
        return len(self.groups)

    @property
    def logical_shape(self) -> tuple[int]:
        shape = []
        for group in self.groups:
            group_dims = self.hdims[group[0]:group[1]]
            shape.append(math.prod(group_dims))
        return tuple(shape)

    def _decompose_index(self, logical_dim: int, idx: int) -> list[int]:
        """
        Decompose a logical index into hierarchical indices
        for the given logical dimension.
        """
        start, end = self.groups[logical_dim]
        factors = self.hdims[start:end]

        indices = []
        rem = idx
        for f in reversed(factors):
            indices.append(rem % f)
            rem //= f

        return list(reversed(indices))

    def get_hierarchical_indices(self, logical_indices: list[int]) -> list[int]:
        """
        Convert logical indices to hierarchical indices.
        """
        hierarchical_indices = []
        for dim, idx in enumerate(logical_indices):
            h_indices = self._decompose_index(dim, idx)
            hierarchical_indices.extend(h_indices)
        return hierarchical_indices

    def get_logical_indices(self, hierarchical_indices: list[int]) -> list[int]:
        """
        Convert hierarchical indices to logical indices.
        """
        logical_indices = []
        h_idx_offset = 0
        for group in self.groups:
            group_start, group_end = group
            group_len = group_end - group_start

            h_indices_group = hierarchical_indices[h_idx_offset:h_idx_offset + group_len]
            factors = self.hdims[group_start:group_end]

            logical_idx = 0
            if h_indices_group:
                strides = [math.prod(factors[i + 1:]) for i in range(len(factors))]
                for i, h_idx in enumerate(h_indices_group):
                    logical_idx += h_idx * strides[i]

            logical_indices.append(logical_idx)
            h_idx_offset += group_len

        return logical_indices

    def __str__(self) -> str:

        def format_grouped_list(lst):
            parts = []
            for start, end in self.groups:
                parts.append("(" + ", ".join(map(str, lst[start:end])) + ")")
            return ", ".join(parts)

        hdims_str = format_grouped_list(self.hdims)
        hstrides_str = format_grouped_list(self.hstrides)
        return f"HierarchicalLayout(hdims=({hdims_str}), hstrides=({hstrides_str}))"


def make_hierarchical_layout(arg):
    """
    Args:
        args: Buffer/BufferLoad/BufferRegion or HierarchicalLayout
    Examples:
        make_hierarchical_layout(buffer)
        make_hierarchical_layout(hlayout)
    """
    if isinstance(arg, (Buffer, BufferLoad, BufferRegion)):
        # Obtain hshape and hstride from attr/block_attr/func_attr
        pass
    elif isinstance(arg, HierarchicalLayout):
        hlayout = arg
    else:
        raise ValueError("Invalid number of arguments to make_hierarchical_layout")
    return _ffi_api.make_hierarchical_layout(hlayout)
