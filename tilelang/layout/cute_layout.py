"""CuteLayout Python FFI wrapper."""

import tvm_ffi

_make_cute = tvm_ffi.get_global_func("tl.make_cute_layout")
_logical_shape = tvm_ffi.get_global_func("tl.CuteLayout_logical_shape")
_mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")
_mode_stride = tvm_ffi.get_global_func("tl.CuteLayout_mode_stride")
_dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
_covered_shape = tvm_ffi.get_global_func("tl.CuteLayout_covered_shape")
_storage_size = tvm_ffi.get_global_func("tl.CuteLayout_storage_size")
_same_layout = tvm_ffi.get_global_func("tl.CuteLayout_same_layout")
_is_same_layout = tvm_ffi.get_global_func("tl.IsSameLayout")
_derive_layout_like = tvm_ffi.get_global_func("tl.DeriveLayoutLike")
_is_layout_match = tvm_ffi.get_global_func("tl.IsLayoutMatch")


class CuteLayout:
    """Python wrapper around the C++ CuteLayoutNode.

    Provides structural layout metadata (mode_shape, mode_stride,
    dim_levels) while the underlying C++ object also caches the
    affine forward_index for backward compatibility.
    """

    def __init__(self, logical_shape, mode_shape, mode_stride, dim_levels):
        self._layout = _make_cute(logical_shape, mode_shape, mode_stride, dim_levels)

    @property
    def _inner(self):
        return self._layout

    @property
    def logical_shape(self):
        return _logical_shape(self._layout)

    @property
    def mode_shape(self):
        return _mode_shape(self._layout)

    @property
    def mode_stride(self):
        return _mode_stride(self._layout)

    @property
    def dim_levels(self):
        return _dim_levels(self._layout)

    @property
    def covered_shape(self):
        return _covered_shape(self._layout)

    @property
    def storage_size(self):
        return _storage_size(self._layout)

    def same_layout(self, other):
        """Structural equality comparison."""
        other_inner = other._layout if isinstance(other, CuteLayout) else other
        return _same_layout(self._layout, other_inner)

    def map_forward_index(self, indices):
        return self._layout.map_forward_index(indices)


def is_same_layout(lhs, rhs):
    """Exact structural layout comparison."""
    lhs_inner = lhs._layout if isinstance(lhs, CuteLayout) else lhs
    rhs_inner = rhs._layout if isinstance(rhs, CuteLayout) else rhs
    return _is_same_layout(lhs_inner, rhs_inner)


def derive_layout_like(src, dst_shape, axis_map=None):
    """Build a layout for dst_shape using src as the structural template."""
    src_inner = src._layout if isinstance(src, CuteLayout) else src
    return _derive_layout_like(src_inner, dst_shape, axis_map)


def is_layout_match(lhs, rhs):
    """Same layout kind, possibly for different logical shapes."""
    lhs_inner = lhs._layout if isinstance(lhs, CuteLayout) else lhs
    rhs_inner = rhs._layout if isinstance(rhs, CuteLayout) else rhs
    return _is_layout_match(lhs_inner, rhs_inner)
