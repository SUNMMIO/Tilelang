"""Tests for Sunmmio layout Python wrappers."""

import random

import pytest
import tvm_ffi

import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import (
    is_same_layout,
    make_zz_layout,
    make_row_major,
    make_aligned_row_major,
)


ANA = tvm.arith.Analyzer()


def _imm(v):
    return tvm.tir.IntImm("int32", v)


def _imms(*vals):
    return [_imm(v) for v in vals]


def _eval(layout, *indices):
    idx = layout.map_forward_index(_imms(*indices))
    return int(ANA.simplify(idx[0]))


_storage_size = tvm_ffi.get_global_func("tl.CuteLayout_storage_size")
_covered_shape = tvm_ffi.get_global_func("tl.CuteLayout_covered_shape")


def _covered(layout):
    return [int(x) for x in _covered_shape(layout)]


def _zz_expected_offset(row, col, shape, block_size):
    _, columns = shape
    block_rows, block_cols = block_size
    column_tiles = (columns + block_cols - 1) // block_cols
    row_tile, row_inner = divmod(row, block_rows)
    col_tile, col_inner = divmod(col, block_cols)
    return row_tile * (block_rows * block_cols * column_tiles) + row_inner * block_cols + col_tile * (block_rows * block_cols) + col_inner


@pytest.mark.parametrize(
    "shape, block_size",
    [
        ((128, 128), (32, 32)),
        ((128, 256), (32, 32)),
        ((128, 256), (16, 32)),
        ((32, 32), (32, 32)),
        ((32, 64), (32, 32)),
        ((64, 64), (1, 32)),
        ((60, 90), (10, 15)),
        ((48, 48), (6, 8)),
        ((63, 127), (32, 32)),
    ],
)
def test_make_zz_layout_matches_manual_offset(shape, block_size):
    layout = make_zz_layout(shape, block_shape=block_size)

    random.seed(42)
    for _ in range(100):
        row = random.randint(0, shape[0] - 1)
        col = random.randint(0, shape[1] - 1)
        expected = _zz_expected_offset(row, col, shape, block_size)
        assert _eval(layout, row, col) == expected


def test_make_zz_layout_allows_padded_shape():
    shape = (63, 127)
    block_size = (32, 32)
    layout = make_zz_layout(shape, block_shape=block_size)
    offset = _eval(layout, 62, 126)

    assert offset == _zz_expected_offset(62, 126, shape, block_size)
    assert offset >= shape[0] * shape[1]


def test_make_zz_layout_accepts_buffer():
    shape = (128, 128)
    block_size = (32, 32)
    buffer = tvm.tir.decl_buffer(shape, "float16")
    layout = make_zz_layout(buffer, block_shape=block_size)
    direct = make_zz_layout(_imms(*shape), [0, 1], _imms(*block_size))

    assert is_same_layout(layout, direct)
    assert _eval(layout, 33, 5) == _zz_expected_offset(33, 5, shape, block_size)


def test_make_zz_layout_defaults_to_last_two_axes():
    shape = (2, 8, 4, 128)
    default_layout = make_zz_layout(shape, block_shape=(4, 32))
    explicit_layout = make_zz_layout(shape, axes=[2, 3], block_shape=(4, 32))
    assert is_same_layout(default_layout, explicit_layout)


def test_make_zz_layout_rejects_rank_one_default_axes():
    with pytest.raises(ValueError, match="requires rank >= 2"):
        make_zz_layout((128,))


def test_make_dynamic_zz_layout():
    from tvm.tir.stmt_functor import substitute

    M = T.dynamic("m")
    K = T.dynamic("k")
    block_size = (32, 32)

    dyn = make_zz_layout((M, K), block_shape=block_size)
    print(dyn)

    # Two independent constructions with the same dynamic extents are equal.
    assert is_same_layout(dyn, make_zz_layout((M, K), block_shape=block_size))

    # The dynamic layout's forward index must match the concrete-shape layout
    # once concrete extents are substituted in — including non-divisible shapes.
    for m_val, k_val in [(128, 128), (96, 256), (64, 96), (63, 127)]:
        concrete = make_zz_layout((m_val, k_val), block_shape=block_size)
        vmap = {M: _imm(m_val), K: _imm(k_val)}
        for row, col in [(0, 0), (33, 5), (31, 63), (m_val - 1, k_val - 1)]:
            dyn_idx = dyn.map_forward_index(_imms(row, col))[0]
            dyn_off = int(ANA.simplify(substitute(dyn_idx, vmap)))
            assert dyn_off == _eval(concrete, row, col)


@pytest.mark.parametrize(
    "shape_spec, axes, dyn_vals",
    [
        # 3D, two dynamic tiled extents (default axes = last two).
        ((4, "s", "d"), None, {"s": 128, "d": 96}),
        # 3D, two dynamic tiled extents, non-divisible.
        ((4, "s", "d"), None, {"s": 63, "d": 127}),
        # 4D attention, two dynamic tiled extents on axes (1, 3).
        ((2, "s", 4, "d"), [1, 3], {"s": 96, "d": 160}),
        # 4D, three dynamic extents: dynamic (non-tiled) batch + two tiled.
        # The non-tiled batch sits physically outside the tiled counts, so the
        # layout has several symbolic strides.
        (("b", "s", 4, "d"), [1, 3], {"b": 3, "s": 64, "d": 128}),
    ],
)
def test_make_dynamic_zz_layout_higher_rank(shape_spec, axes, dyn_vals):
    """Higher-rank ZZ with 2-3 dynamic logical extents: the dynamic layout's
    forward index must match the concrete-shape layout once extents are bound."""
    from tvm.tir.stmt_functor import substitute

    vars_ = {name: T.dynamic(name) for name in dyn_vals}
    dyn_shape = tuple(vars_[s] if isinstance(s, str) else s for s in shape_spec)
    concrete_shape = tuple(dyn_vals[s] if isinstance(s, str) else s for s in shape_spec)
    block_size = (32, 32)

    dyn = make_zz_layout(dyn_shape, axes=axes, block_shape=block_size)
    concrete = make_zz_layout(concrete_shape, axes=axes, block_shape=block_size)
    sub_map = {vars_[name]: _imm(val) for name, val in dyn_vals.items()}

    rank = len(shape_spec)
    coords = [tuple(0 for _ in range(rank)), tuple(d - 1 for d in concrete_shape)]
    random.seed(0)
    for _ in range(30):
        coords.append(tuple(random.randint(0, d - 1) for d in concrete_shape))

    for c in coords:
        dyn_idx = dyn.map_forward_index(_imms(*c))[0]
        dyn_off = int(ANA.simplify(substitute(dyn_idx, sub_map)))
        assert dyn_off == _eval(concrete, *c)


# ---------------------------------------------------------------------------
# make_aligned_row_major: alignment-padded row-major for RSRAM
# ---------------------------------------------------------------------------


def test_make_aligned_row_major_pads_leading_dim_bf16():
    # 64B / 2B(bf16) = 32 elems; pitch = round_up(40, 32) = 64.
    layout = make_aligned_row_major((2, 40), "float16", 64)
    assert _eval(layout, 0, 0) == 0
    assert _eval(layout, 0, 39) == 39
    assert _eval(layout, 1, 0) == 64  # padded leading-dim pitch, not 40
    assert _eval(layout, 1, 39) == 103


def test_make_aligned_row_major_pads_leading_dim_fp32():
    # 64B / 4B(fp32) = 16 elems; pitch = round_up(40, 16) = 48.
    layout = make_aligned_row_major((2, 40), "float32", 64)
    assert _eval(layout, 1, 0) == 48


def test_make_aligned_row_major_propagates_to_outer_dims():
    # (4,2,40) bf16: innermost pitch 64, so stride[1]=64 and stride[0]=2*64=128.
    layout = make_aligned_row_major((4, 2, 40), "float16", 64)
    assert _eval(layout, 0, 1, 0) == 64
    assert _eval(layout, 1, 0, 0) == 128


def test_make_aligned_row_major_distinct_from_plain():
    assert not is_same_layout(make_aligned_row_major((2, 40), "float16", 64), make_row_major((2, 40)))


def test_make_aligned_row_major_noop_when_already_aligned():
    # 64 is already a multiple of 32 -> no padding -> identical to plain.
    assert is_same_layout(make_aligned_row_major((2, 64), "float16", 64), make_row_major((2, 64)))


def test_make_aligned_row_major_rank1_pads_extent():
    # [40] has no leading-dim stride; padding lands in the covered extent, so it
    # is distinct from the plain [40] row-major, but a no-op once 32-aligned.
    assert not is_same_layout(make_aligned_row_major((40,), "float16", 64), make_row_major((40,)))
    assert is_same_layout(make_aligned_row_major((64,), "float16", 64), make_row_major((64,)))


def test_make_aligned_row_major_storage_and_covered():
    layout = make_aligned_row_major((2, 40), "float16", 64)
    assert _covered(layout) == [2, 64]
    assert int(_storage_size(layout)) == 128  # (2-1)*64 + (64-1)*1 + 1
    rank1 = make_aligned_row_major((40,), "float16", 64)
    assert _covered(rank1) == [64]
    assert int(_storage_size(rank1)) == 64


def test_make_aligned_row_major_boundary_exact_multiple():
    # 32 is already a multiple of 32 -> no padding; 33 rounds up to 64.
    assert is_same_layout(make_aligned_row_major((2, 32), "float16", 64), make_row_major((2, 32)))
    assert _eval(make_aligned_row_major((2, 33), "float16", 64), 1, 0) == 64
    assert _covered(make_aligned_row_major((2, 33), "float16", 64)) == [2, 64]


def test_make_aligned_row_major_inner_extent_one():
    # A single-column inner dim still pads up to the alignment granularity.
    layout = make_aligned_row_major((2, 1), "float16", 64)
    assert _covered(layout) == [2, 32]
    assert _eval(layout, 1, 0) == 32


def test_make_aligned_row_major_int8_uses_64_elem_granularity():
    # int8: 64B / 1B = 64 elems; round_up(100, 64) = 128.
    layout = make_aligned_row_major((2, 100), "int8", 64)
    assert _eval(layout, 1, 0) == 128
    assert _covered(layout) == [2, 128]


def test_make_aligned_row_major_fp4_uses_128_elem_granularity():
    # fp4 (4-bit): 64B = 128 elems; round_up(100, 128) = 128, exact for 256.
    layout = make_aligned_row_major((2, 100), "float4_e2m1fn", 64)
    assert _eval(layout, 1, 0) == 128
    assert _covered(layout) == [2, 128]
    assert is_same_layout(make_aligned_row_major((2, 256), "float4_e2m1fn", 64), make_row_major((2, 256)))


def test_make_aligned_row_major_rank4_propagates():
    # (2,3,4,40) bf16: inner 40->64; strides 4*64=256, 3*256=768.
    layout = make_aligned_row_major((2, 3, 4, 40), "float16", 64)
    assert _eval(layout, 0, 0, 1, 0) == 64
    assert _eval(layout, 0, 1, 0, 0) == 256
    assert _eval(layout, 1, 0, 0, 0) == 768
