"""Tests for Sunmmio layout Python wrappers."""

import random

import pytest

import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import is_same_layout, make_zz_layout


ANA = tvm.arith.Analyzer()


def _imm(v):
    return tvm.tir.IntImm("int32", v)


def _imms(*vals):
    return [_imm(v) for v in vals]


def _eval(layout, *indices):
    idx = layout.map_forward_index(_imms(*indices))
    return int(ANA.simplify(idx[0]))


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
