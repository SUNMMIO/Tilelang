"""Tests for CuTe layout algebra operations (complement, coalesce,
zippedProduct, logicalDivide) exposed via FFI."""

import pytest

from tilelang import tvm as tvm

tir = tvm.tir
_ffi = tvm.ffi


# ---------------------------------------------------------------------------
# FFI helpers
# ---------------------------------------------------------------------------


def _get(name):
    return _ffi.get_global_func(name)


complement = _get("tl.cute.complement")
coalesce = _get("tl.cute.coalesce")
zipped_product = _get("tl.cute.zipped_product")
logical_divide = _get("tl.cute.logical_divide")

ANA = tvm.arith.Analyzer()


def _imm(v):
    return tir.IntImm("int32", v)


def _imms(*vals):
    return [_imm(v) for v in vals]


def _to_ints(arr):
    """Convert array of PrimExpr to list of Python ints."""
    return [int(ANA.simplify(x)) for x in arr]


# ---------------------------------------------------------------------------
# complement tests
# ---------------------------------------------------------------------------


class TestComplement:
    def test_basic_contiguous(self):
        """complement((4, 1), 16) → shape=(4,), stride=(4,)"""
        shapes, strides = complement(_imms(4), _imms(1), 16)
        assert _to_ints(shapes) == [4]
        assert _to_ints(strides) == [4]

    def test_single_element(self):
        """complement((1, 1), 8) → shape=(8,), stride=(1,)"""
        shapes, strides = complement(_imms(1), _imms(1), 8)
        assert _to_ints(shapes) == [8]
        assert _to_ints(strides) == [1]

    def test_full_coverage(self):
        """complement((8, 1), 8) → shape=(1,) (trivial complement)."""
        shapes, strides = complement(_imms(8), _imms(1), 8)
        assert _to_ints(shapes) == [1]
        # Stride of a shape-1 mode is irrelevant (no offset contribution).

    def test_strided(self):
        """complement((4, 2), 16) → stride 2, shape 4 → free = (2,1),(2,8)
        After coalesce: (2, 1) and (2, 8)."""
        shapes, strides = complement(_imms(4), _imms(2), 16)
        s = _to_ints(shapes)
        d = _to_ints(strides)
        # Complement covers indices 0,2,4,6 → free indices are 1,3,5,7,8..15
        # Result: (2, 1), (2, 8) after coalesce
        assert s == [2, 2]
        assert d == [1, 8]


# ---------------------------------------------------------------------------
# coalesce tests
# ---------------------------------------------------------------------------


class TestCoalesce:
    def test_merge_contiguous(self):
        """(8,4,2) strides (1,8,32) → coalesce → (64, 1)"""
        shapes, strides = coalesce(_imms(8, 4, 2), _imms(1, 8, 32))
        assert _to_ints(shapes) == [64]
        assert _to_ints(strides) == [1]

    def test_already_coalesced(self):
        """(16, 1) → no change → (16, 1)"""
        shapes, strides = coalesce(_imms(16), _imms(1))
        assert _to_ints(shapes) == [16]
        assert _to_ints(strides) == [1]

    def test_non_mergeable(self):
        """(4, 4) strides (1, 8) → gap between modes → stays (4, 4)"""
        shapes, strides = coalesce(_imms(4, 4), _imms(1, 8))
        assert _to_ints(shapes) == [4, 4]
        assert _to_ints(strides) == [1, 8]


# ---------------------------------------------------------------------------
# zippedProduct tests
# ---------------------------------------------------------------------------


class TestZippedProduct:
    def test_2d(self):
        """zippedProduct of rank-2 block and tiler.

        block = (4, 4) strides (4, 1)  (row-major 4×4)
        tiler = (2, 2) strides (2, 1)  (row-major 2×2)
        Result should cover 64 elements (4*2 × 4*2 = 8×8).
        """
        shapes, strides = zipped_product(_imms(4, 4), _imms(4, 1), _imms(2, 2), _imms(2, 1))
        s = _to_ints(shapes)
        total = 1
        for x in s:
            total *= x
        assert total == 64


# ---------------------------------------------------------------------------
# logicalDivide tests
# ---------------------------------------------------------------------------


class TestLogicalDivide:
    def test_basic_divide(self):
        """logicalDivide((8, 1), (4, 1)) → shapes (4, 2)"""
        shapes, strides = logical_divide(_imms(8), _imms(1), _imms(4), _imms(1))
        s = _to_ints(shapes)
        # Within-block = 4, across-block = ceil(8/4) = 2
        assert s == [4, 2]

    def test_non_divisible(self):
        """logicalDivide((10, 1), (4, 1)) → shapes (4, 3)
        ceildiv(10, 4) = 3"""
        shapes, strides = logical_divide(_imms(10), _imms(1), _imms(4), _imms(1))
        s = _to_ints(shapes)
        assert s == [4, 3]

    def test_tiler_equals_size(self):
        """logicalDivide((8, 1), (8, 1)) → shapes (8, 1)"""
        shapes, strides = logical_divide(_imms(8), _imms(1), _imms(8), _imms(1))
        s = _to_ints(shapes)
        assert s == [8, 1]

    def test_tiler_one(self):
        """logicalDivide((8, 1), (1, 1)) → shapes (1, 8)"""
        shapes, strides = logical_divide(_imms(8), _imms(1), _imms(1), _imms(1))
        s = _to_ints(shapes)
        assert s == [1, 8]


# ---------------------------------------------------------------------------
# Integration: logicalDivide → MakeZZ equivalence
# ---------------------------------------------------------------------------


class TestAlgebraConstructorEquivalence:
    """Verify that logicalDivide produces the same mode shapes as the
    MakeZZ/MakeZN constructors."""

    def _get_constructor_mode_shapes(self, constructor, shape, axes, block_shape):
        make_fn = _get(f"tl.sunmmio.make_{constructor}")
        get_ms = _get("tl.CuteLayout_mode_shape")
        layout = make_fn(_imms(*shape), [_imm(a) for a in axes], _imms(*block_shape))
        return _to_ints(get_ms(layout))

    def test_zz_shapes_from_divide(self):
        """For ZZ(128×128, block 32×32), logicalDivide on each axis
        should give BM=32,QM=4 and BN=32,QN=4."""
        M, N = 128, 128
        BM, BN = 32, 32

        # Divide M dimension
        ms, _ = logical_divide(_imms(M), _imms(1), _imms(BM), _imms(1))
        m_modes = _to_ints(ms)  # [BM, QM]

        # Divide N dimension
        ns, _ = logical_divide(_imms(N), _imms(1), _imms(BN), _imms(1))
        n_modes = _to_ints(ns)  # [BN, QN]

        assert m_modes == [32, 4]
        assert n_modes == [32, 4]

        # Verify constructor gives same mode shapes
        ctor_shapes = self._get_constructor_mode_shapes("zz", (128, 128), (0, 1), (32, 32))
        assert ctor_shapes == [32, 4, 32, 4]

    def test_zn_shapes_from_divide(self):
        """ZN has same shapes as ZZ, just different strides."""
        ctor_shapes = self._get_constructor_mode_shapes("zn", (128, 128), (0, 1), (32, 32))
        # Same shapes as ZZ
        assert ctor_shapes == [32, 4, 32, 4]

    def test_4d_shapes(self):
        """4D attention-like: (B=2, S=8, N=4, D=128), tiled on axes 1,3
        with block (4, 32)."""
        shape = (2, 8, 4, 128)
        axes = (1, 3)
        block = (4, 32)

        ctor_shapes = self._get_constructor_mode_shapes("zz", shape, axes, block)
        # axis 1: BM=4, QM=ceil(8/4)=2
        # axis 3: BN=32, QN=ceil(128/32)=4
        # dim_levels: dim0=[2], dim1=[4,2], dim2=[4], dim3=[32,4]
        # mode_shape = (2, 4, 2, 4, 32, 4)
        assert ctor_shapes == [2, 4, 2, 4, 32, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
