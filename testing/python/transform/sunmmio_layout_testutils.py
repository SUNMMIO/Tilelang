"""Shared utilities for Sunmmio layout-inference tests.

Every layout-inference test runs the *same* canonical pass pipeline and reads
the resulting per-buffer layout map the same way. Centralizing both here keeps
the pipeline from drifting between test files and makes assertions declarative.

Usage::

    from sunmmio_layout_testutils import run_sunmmio_layout_inference, assert_layout

    layouts = run_sunmmio_layout_inference(mod, target)   # name -> Layout
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))
"""

import tvm_ffi
from tilelang import tvm as tvm
from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass
import tilelang as tl

# Re-exported so tests have a single import site for layout constructors.
from tilelang.layout import (  # noqa: F401
    make_row_major,
    make_zz_layout,
    make_zn_layout,
    make_zzz_layout,
    make_nzz_layout,
)
from tilelang.layout.cute_layout import is_same_layout, is_layout_match  # noqa: F401

# Raw FFI accessors for structural layout introspection (work on the C++
# CuteLayout objects stored in layout_map, not just the Python wrapper).
_dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
_mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")
_mode_stride = tvm_ffi.get_global_func("tl.CuteLayout_mode_stride")

# ---------------------------------------------------------------------------
# Layout capture
# ---------------------------------------------------------------------------

# Populated by LayoutVisual as a side effect; run_sunmmio_layout_inference
# returns a fresh copy so tests never touch this global directly.
collected_result = {}


@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):
    def visit_block_(self, op: tir.Block) -> None:
        anns = op.annotations
        # ApplyToIR annotates every block with the final maps; capture both the
        # SRAM layout_map and the DRAM global_layout_map, merged by buffer name.
        if "layout_map" not in anns and "global_layout_map" not in anns:
            return
        collected_result.clear()
        for map_key in ("layout_map", "global_layout_map"):
            if map_key in anns:
                for key, layout in anns[map_key].items():
                    collected_result[key.name] = layout


def LayoutVisual():
    """Pass that records the final layout_map block annotation into
    ``collected_result`` (keyed by buffer name)."""

    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)


# ---------------------------------------------------------------------------
# Canonical pipeline
# ---------------------------------------------------------------------------


def run_sunmmio_layout_inference(mod, target):
    """Run the canonical Sunmmio layout-inference pipeline and return the
    inferred per-buffer layouts as ``{buffer_name: Layout}``.

    This is the single source of truth for the pass order used by every
    layout-inference test. Negative tests that expect a failure inside
    ``SunmmioLayoutInference`` should wrap this call in ``pytest.raises`` (they
    never reach the LayoutVisual capture step).
    """
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.AddWrapperForSingleBufStore()(mod)
        mod = tl.transform.LegalizeNegativeIndex()(mod)
        mod = tl.transform.InjectAssumes()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioDataPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        LayoutVisual()(mod)
    return dict(collected_result)


# ---------------------------------------------------------------------------
# Layout classification (declarative assertions)
# ---------------------------------------------------------------------------


def _per_dim_modes(layout):
    """Split the flat (mode_shape, mode_stride) arrays into per-dim modes.

    Returns a list (one entry per logical dim) of dicts with ``shape`` and
    ``stride`` lists ordered innermost-mode-first (index 0 is innermost), to
    match the C++ CuteLayout convention.
    """
    inner = getattr(layout, "_layout", layout)
    dim_levels = [int(x) for x in _dim_levels(inner)]
    shapes = [_maybe_int(x) for x in _mode_shape(inner)]
    strides = [_maybe_int(x) for x in _mode_stride(inner)]
    dims = []
    i = 0
    for n in dim_levels:
        dims.append({"shape": shapes[i : i + n], "stride": strides[i : i + n]})
        i += n
    return dims


def _maybe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x  # symbolic stride/shape; leave as-is


def layout_kind(layout):
    """Classify a CuteLayout as ``"RowMajor"``, ``"ZZ"`` (any ZZ-like:
    ZZ/ZZZ/NZZ, i.e. row-major innermost), or ``"ZN"`` (col-major innermost).

    Mirrors the C++ ``sunmmio::IsZZLike`` test: among the last two blocked
    dims, ZZ-like means the higher-indexed dim's innermost stride is smaller.
    """
    dims = _per_dim_modes(layout)
    blocked = [d for d, m in enumerate(dims) if len(m["shape"]) > 1]
    if len(blocked) < 2:
        return "RowMajor"
    ax0, ax1 = blocked[-2], blocked[-1]
    s0 = dims[ax0]["stride"][0]
    s1 = dims[ax1]["stride"][0]
    if isinstance(s0, int) and isinstance(s1, int):
        return "ZZ" if s1 < s0 else "ZN"
    # Fallback: stride 1 on the inner dim is the common ZZ-like pattern.
    return "ZZ" if s1 == 1 else "ZN"


def block_shape(layout):
    """Return ``(height, width)`` block extent of a blocked layout, or None."""
    dims = _per_dim_modes(layout)
    blocked = [d for d, m in enumerate(dims) if len(m["shape"]) > 1]
    if len(blocked) < 2:
        return None
    ax0, ax1 = blocked[-2], blocked[-1]
    return (dims[ax0]["shape"][0], dims[ax1]["shape"][0])


def assert_layout(layouts, name, kind, block=(32, 32)):
    """Assert ``layouts[name]`` has the expected kind (and block shape, unless
    RowMajor or ``block`` is None)."""
    assert name in layouts, f"{name!r} not in layout_map; have {sorted(layouts)}"
    lay = layouts[name]
    got = layout_kind(lay)
    assert got == kind, f"{name}: expected {kind}, got {got}  ({lay})"
    if kind != "RowMajor" and block is not None:
        got_block = block_shape(lay)
        assert tuple(got_block) == tuple(block), f"{name}: expected block {tuple(block)}, got {got_block}  ({lay})"
