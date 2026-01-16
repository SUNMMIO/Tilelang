"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations

import math
from dataclasses import dataclass

import tvm.runtime
from tvm.tir import Buffer, BufferLoad, BufferRegion
from tilelang import _ffi_api


@dataclass
class HierarchicalLayout:
    """A layout that describes a multi-level memory layout for a tensor.

    This class defines a mapping from a logical tensor coordinate to a physical
    memory offset. This mapping is defined by a hierarchy of dimensions and strides.

    - `hdims` (hierarchical dimensions): The dimensions of the hierarchy.
    - `hstrides` (hierarchical strides): The strides corresponding to the `hdims`.
    - `groups`: Groups the `hdims` and `hstrides` to form the logical dimensions of the tensor.

    The physical memory offset is the dot product of the hierarchical indices and the
    hierarchical strides.

    **Visualization Example**

    This visualization demonstrates how a 4x4 tensor, initially in a standard
    row-major layout, is rearranged into a 1D physical memory space by the
    `HierarchicalLayout`.

    **1. Logical 2D View (Row-Major)**

    This is our starting point: a 4x4 tensor where the values (0-15) are laid
    out in a simple row-by-row order.
    ```
    ┌────┬────┬────┬────┐
    │  0 │  1 │  2 │  3 │
    ├────┼────┼────┼────┤
    │  4 │  5 │  6 │  7 │
    ├────┼────┼────┼────┤
    │  8 │  9 │ 10 │ 11 │
    ├────┼────┼────┼────┤
    │ 12 │ 13 │ 14 │ 15 │
    └────┴────┴────┴────┘
    ```

    **2. Applying the Hierarchical Layout**

    We apply the `HierarchicalLayout` with the following parameters:
    - `hdims`: `[2, 2, 2, 2]`
    - `hstrides`: `[8, 2, 4, 1]`
    - `groups`: `[(0, 2), (2, 4)]`

    **3. Physical 1D Memory View**

    This is the final arrangement of the tensor's values in the 1D physical
    memory. Notice how the elements from the logical view have been reordered
    to form contiguous blocks.

    Memory Index:
    ```
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ]
    ```

    Value from Logical Tensor:
    ```
    [ 0  1  4  5  2  3  6  7  8  9 12 13 10 11 14 15 ]
    ```
    This shows that the tensor is laid out in 2x2 blocks. For instance, the
    top-left 2x2 block of the logical tensor (`0, 1, 4, 5`) is stored as the
    first contiguous chunk in memory. This blocked arrangement can be
    beneficial for data locality and performance on certain hardware
    architectures.
    """
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
        """Decompose a logical index into hierarchical indices.

        This function performs a mixed-radix decomposition of the logical index
        `idx` to find the corresponding hierarchical indices.

        Formula:
            This method finds the unique sequence of hierarchical indices
            `H_d = (h_{d,0}, ..., h_{d,k-1})` that satisfies the equation:

            `I_d = sum_{i=0}^{k-1} [ h_{d,i} * (product_{j=i+1}^{k-1} f_{d,j}) ]`

            where:
            - `I_d` is the logical index (`idx`).
            - `h_{d,i}` is the i-th hierarchical index.
            - `f_{d,j}` is the j-th hierarchical dimension (`hdims`) in the group.
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
        """Convert hierarchical indices to logical indices.

        This function reconstructs the logical index from its hierarchical
        indices by directly implementing the mixed-radix representation formula.

        Formula:
            `I_d = sum_{i=0}^{k-1} [ h_{d,i} * (product_{j=i+1}^{k-1} f_{d,j}) ]`

            where:
            - `I_d` is the resulting logical index for a given dimension.
            - `h_{d,i}` is the i-th hierarchical index.
            - `f_{d,j}` is the j-th hierarchical dimension (`hdims`) in the group.
            - The product term is the stride of the i-th hierarchical index.
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
    Creates a C++ Layout object from a Python HierarchicalLayout instance or extracts
    it from a Buffer/BufferLoad/BufferRegion object.

    Parameters
    ----------
    arg : Union[Buffer, BufferLoad, BufferRegion, HierarchicalLayout]
        The input object from which to create or extract the hierarchical layout.

    Returns
    -------
    tvm.tl.Layout
        A C++ Layout object representing the hierarchical layout.

    Raises
    ------
    ValueError
        If an invalid argument type is provided.
    """
    hlayout_instance: HierarchicalLayout
    if isinstance(arg, (Buffer, BufferLoad, BufferRegion)):
        # TODO: Implement extraction of hdims, hstrides, and groups from buffer attributes.
        # This will involve reading attributes from the Buffer object.
        # Once extracted, a HierarchicalLayout instance should be constructed:
        # hlayout_instance = HierarchicalLayout(hdims=..., hstrides=..., groups=...)
        raise NotImplementedError("Extracting HierarchicalLayout from Buffer/BufferLoad/BufferRegion is not yet implemented.")
    elif isinstance(arg, HierarchicalLayout):
        hlayout_instance = arg
    else:
        raise ValueError("Invalid argument type for make_hierarchical_layout. "
                         "Expected Buffer, BufferLoad, BufferRegion, or HierarchicalLayout.")

    # Convert Python lists/tuples from the hlayout_instance to TVM runtime Arrays for the C++ FFI call
    hdims_arr = tvm.runtime.convert(hlayout_instance.hdims)
    hstrides_arr = tvm.runtime.convert(hlayout_instance.hstrides)
    groups_arr = tvm.runtime.convert([list(g) for g in hlayout_instance.groups]) # Convert inner tuples to lists
    logical_shape_arr = tvm.runtime.convert(list(hlayout_instance.logical_shape))

    return _ffi_api.make_hierarchical_layout(hdims_arr, hstrides_arr, groups_arr, logical_shape_arr)
