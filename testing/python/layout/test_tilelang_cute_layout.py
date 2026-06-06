"""Tests for CuteLayout (CuteLayoutNode) and Sunmmio named constructors."""

import random
import pytest

from tilelang import tvm as tvm

tir = tvm.tir
_ffi = tvm.ffi


# ---------------------------------------------------------------------------
# FFI helpers
# ---------------------------------------------------------------------------


def _get(name):
    return _ffi.get_global_func(name)


make_cute = _get("tl.make_cute_layout")
make_row_major = _get("tl.sunmmio.make_row_major")
make_zz = _get("tl.sunmmio.make_zz")
make_zn = _get("tl.sunmmio.make_zn")
make_zzz = _get("tl.sunmmio.make_zzz")
make_nzz = _get("tl.sunmmio.make_nzz")
get_mode_shape = _get("tl.CuteLayout_mode_shape")
get_mode_stride = _get("tl.CuteLayout_mode_stride")
get_dim_levels = _get("tl.CuteLayout_dim_levels")
get_logical_shape = _get("tl.CuteLayout_logical_shape")
get_covered_shape = _get("tl.CuteLayout_covered_shape")
get_storage_size = _get("tl.CuteLayout_storage_size")
same_layout = _get("tl.CuteLayout_same_layout")
is_same_layout = _get("tl.IsSameLayout")
derive_layout_like = _get("tl.DeriveLayoutLike")
is_layout_match = _get("tl.IsLayoutMatch")

ANA = tvm.arith.Analyzer()


def _imm(v):
    return tir.IntImm("int32", v)


def _imms(*vals):
    return [_imm(v) for v in vals]


def _eval(layout, *indices):
    """Evaluate forward_index at concrete indices, return int."""
    idx = layout.map_forward_index(_imms(*indices))
    return int(ANA.simplify(idx[0]))


# ---------------------------------------------------------------------------
# 1. CuteLayout raw constructor
# ---------------------------------------------------------------------------


class TestCuteLayoutConstruction:
    def test_row_major_2d(self):
        layout = make_cute(_imms(128, 128), _imms(128, 128), _imms(128, 1), [1, 1])
        assert _eval(layout, 0, 0) == 0
        assert _eval(layout, 0, 1) == 1
        assert _eval(layout, 1, 0) == 128
        assert _eval(layout, 3, 5) == 3 * 128 + 5

    def test_blocked_2d(self):
        # ZZ-like: mode_shape=(BM=32, QM=4, BN=32, QN=4), strides=(32,4096,1,1024)
        layout = make_cute(
            _imms(128, 128),
            _imms(32, 4, 32, 4),
            _imms(32, 4096, 1, 1024),
            [2, 2],
        )
        # (33, 5): BM=33%32=1, QM=33//32=1, BN=5%32=5, QN=5//32=0
        # offset = 1*32 + 1*4096 + 5*1 + 0*1024 = 4133
        assert _eval(layout, 33, 5) == 4133
        assert _eval(layout, 0, 0) == 0

    def test_validation_dim_levels_mismatch(self):
        with pytest.raises(tvm.error.InternalError, match=r"sum\(dim_levels\) must equal mode_shape length"):
            make_cute(_imms(128, 128), _imms(128), _imms(128), [1, 1])

    def test_validation_sum_dim_levels(self):
        with pytest.raises(tvm.error.InternalError, match=r"sum\(dim_levels\) must equal mode_shape length"):
            make_cute(_imms(128, 128), _imms(128, 128), _imms(128, 1), [2, 1])


# ---------------------------------------------------------------------------
# 2. Structural accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_mode_shape(self):
        layout = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        ms = [int(x) for x in get_mode_shape(layout)]
        assert ms == [32, 4, 32, 4]

    def test_mode_stride(self):
        layout = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        ms = [int(x) for x in get_mode_stride(layout)]
        assert ms == [32, 4096, 1, 1024]

    def test_dim_levels(self):
        layout = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [2, 2]

    def test_logical_shape(self):
        layout = make_cute(_imms(128, 64), _imms(128, 64), _imms(64, 1), [1, 1])
        ls = [int(x) for x in get_logical_shape(layout)]
        assert ls == [128, 64]

    def test_covered_shape(self):
        layout = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        cs = [int(x) for x in get_covered_shape(layout)]
        assert cs == [128, 128]

    def test_storage_size(self):
        layout = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        ss = int(ANA.simplify(get_storage_size(layout)))
        assert ss == 128 * 128


# ---------------------------------------------------------------------------
# 3. SameLayout
# ---------------------------------------------------------------------------


class TestSameLayout:
    def test_identical(self):
        a = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        b = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        assert same_layout(a, b)

    def test_different_shape(self):
        a = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        b = make_cute(_imms(256, 128), _imms(32, 8, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        assert not same_layout(a, b)

    def test_different_stride(self):
        a = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(32, 4096, 1, 1024), [2, 2])
        b = make_cute(_imms(128, 128), _imms(32, 4, 32, 4), _imms(1, 4096, 32, 1024), [2, 2])
        assert not same_layout(a, b)


# ---------------------------------------------------------------------------
# 4. Sunmmio named constructors
# ---------------------------------------------------------------------------


class TestSunmmioConstructors:
    def test_row_major(self):
        layout = make_row_major(_imms(128, 64))
        assert _eval(layout, 0, 0) == 0
        assert _eval(layout, 0, 1) == 1
        assert _eval(layout, 1, 0) == 64
        assert _eval(layout, 2, 3) == 2 * 64 + 3

    def test_row_major_3d(self):
        layout = make_row_major(_imms(4, 128, 64))
        assert _eval(layout, 0, 0, 0) == 0
        assert _eval(layout, 1, 0, 0) == 128 * 64
        assert _eval(layout, 0, 1, 0) == 64
        assert _eval(layout, 0, 0, 1) == 1

    def test_zz_basic(self):
        layout = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [2, 2]
        ms = [int(x) for x in get_mode_shape(layout)]
        assert ms == [32, 4, 32, 4]

    def test_zn_basic(self):
        layout = make_zn(_imms(128, 128), [0, 1], _imms(32, 32))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [2, 2]
        # ZN: stride(BM) = 1, stride(BN) = BM = 32
        mstr = [int(x) for x in get_mode_stride(layout)]
        assert mstr[0] == 1  # stride_bm
        assert mstr[2] == 32  # stride_bn

    def test_zzz_basic(self):
        # cluster_shape=4 means 4 blocks per cluster; grid = ceildiv(8,4) = 2
        layout = make_zzz(_imms(128, 128), [0, 1], _imms(16, 16), _imms(4, 4))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [3, 3]

    def test_nzz_basic(self):
        layout = make_nzz(_imms(128, 128), [0, 1], _imms(16, 16), _imms(4, 4))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [3, 3]

    # --- Non-divisible (padded) shapes ---

    @pytest.mark.parametrize(
        "M,N,BM,BN",
        [
            (63, 127, 32, 32),
            (100, 200, 32, 32),
            (33, 65, 32, 32),
            (1, 1, 32, 32),
        ],
    )
    def test_zz_non_divisible(self, M, N, BM, BN):
        """ZZ with non-divisible shapes: ceildiv padding, correct offsets."""
        layout = make_zz(_imms(M, N), [0, 1], _imms(BM, BN))
        QM = (M + BM - 1) // BM
        QN = (N + BN - 1) // BN

        ms = [int(x) for x in get_mode_shape(layout)]
        assert ms == [BM, QM, BN, QN]

        cs = [int(x) for x in get_covered_shape(layout)]
        assert cs == [BM * QM, BN * QN]
        assert cs[0] >= M and cs[1] >= N

        ss = int(ANA.simplify(get_storage_size(layout)))
        assert ss == BM * QM * BN * QN

        # Spot-check: corners of the logical region
        assert _eval(layout, 0, 0) == 0
        if M > 1 and N > 1:
            off = _eval(layout, M - 1, N - 1)
            assert 0 <= off < ss

    @pytest.mark.parametrize(
        "M,N,BM,BN",
        [
            (63, 127, 32, 32),
            (100, 200, 32, 64),
        ],
    )
    def test_zn_non_divisible(self, M, N, BM, BN):
        """ZN with non-divisible shapes."""
        layout = make_zn(_imms(M, N), [0, 1], _imms(BM, BN))
        QM = (M + BM - 1) // BM
        QN = (N + BN - 1) // BN

        ms = [int(x) for x in get_mode_shape(layout)]
        assert ms == [BM, QM, BN, QN]

        # ZN inner-block strides: stride(BM)=1, stride(BN)=BM
        mst = [int(x) for x in get_mode_stride(layout)]
        assert mst[0] == 1  # stride_bm
        assert mst[2] == BM  # stride_bn

        assert _eval(layout, 0, 0) == 0
        if M > 1:
            assert _eval(layout, 1, 0) == 1  # column-major inner

    # --- Higher-rank with axis selection ---

    def test_zz_4d_attention(self):
        """ZZ on (seq_len, d_head) axes of a 4D attention buffer."""
        # (bsz=2, seq_len=128, nhead=4, d_head=64), ZZ on axes (1, 3)
        layout = make_zz(_imms(2, 128, 4, 64), [1, 3], _imms(32, 32))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [1, 2, 1, 2]  # row-major, blocked, row-major, blocked

        ms = [int(x) for x in get_mode_shape(layout)]
        assert ms == [2, 32, 4, 4, 32, 2]  # bsz, BM, QM, nhead, BN, QN

        # Inner block strides: ZZ means stride(BM)=BN=32, stride(BN)=1
        assert _eval(layout, 0, 0, 0, 0) == 0
        assert _eval(layout, 0, 0, 0, 1) == 1  # BN stride = 1
        assert _eval(layout, 0, 1, 0, 0) == 32  # BM stride = BN = 32

        # Non-selected axes are outermost row-major
        selected_size = 32 * 4 * 32 * 2  # BM*QM * BN*QN = 128*64 = 8192
        mst = [int(x) for x in get_mode_stride(layout)]
        assert mst[3] == selected_size  # nhead stride
        assert mst[0] == selected_size * 4  # bsz stride = nhead_stride * nhead

        # Last element
        ss = int(ANA.simplify(get_storage_size(layout)))
        assert _eval(layout, 1, 127, 3, 63) == ss - 1

    def test_zn_4d_attention(self):
        """ZN on (seq_len, d_head) axes of a 4D attention buffer."""
        layout = make_zn(_imms(2, 128, 4, 64), [1, 3], _imms(32, 32))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [1, 2, 1, 2]

        # ZN: inner-block is column-major
        assert _eval(layout, 0, 0, 0, 1) == 32  # BN stride = BM = 32
        assert _eval(layout, 0, 1, 0, 0) == 1  # BM stride = 1

    def test_zz_3d_non_divisible(self):
        """ZZ on axes (1,2) of a 3D buffer with non-divisible shape."""
        # (batch=8, M=100, N=200), ZZ on axes (1, 2), block=(32, 32)
        layout = make_zz(_imms(8, 100, 200), [1, 2], _imms(32, 32))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [1, 2, 2]

        ms = [int(x) for x in get_mode_shape(layout)]
        QM = (100 + 31) // 32  # 4
        QN = (200 + 31) // 32  # 7
        assert ms == [8, 32, QM, 32, QN]

        cs = [int(x) for x in get_covered_shape(layout)]
        assert cs == [8, 32 * QM, 32 * QN]
        assert cs[1] >= 100 and cs[2] >= 200

        # batch stride should be selected_size = 32*QM * 32*QN
        selected_size = 32 * QM * 32 * QN
        assert _eval(layout, 1, 0, 0) == selected_size
        assert _eval(layout, 0, 0, 0) == 0

    def test_zz_4d_non_divisible(self):
        """ZZ on non-adjacent axes with non-divisible shape."""
        # (2, 63, 4, 127), ZZ on axes (1, 3), block=(32, 32)
        layout = make_zz(_imms(2, 63, 4, 127), [1, 3], _imms(32, 32))
        dl = [int(x) for x in get_dim_levels(layout)]
        assert dl == [1, 2, 1, 2]

        QM = (63 + 31) // 32  # 2
        QN = (127 + 31) // 32  # 4
        ms = [int(x) for x in get_mode_shape(layout)]
        assert ms == [2, 32, QM, 4, 32, QN]

        cs = [int(x) for x in get_covered_shape(layout)]
        assert cs[1] >= 63 and cs[3] >= 127

        # All logical indices produce valid offsets
        ss = int(ANA.simplify(get_storage_size(layout)))
        assert _eval(layout, 0, 0, 0, 0) == 0
        off_last = _eval(layout, 1, 62, 3, 126)
        assert 0 <= off_last < ss


# ---------------------------------------------------------------------------
# 5. Equivalence: MakeZZ offset matches manual hdims/hstrides formula
# ---------------------------------------------------------------------------


class TestEquivalenceZZManual:
    @pytest.mark.parametrize(
        "M,N,BM,BN",
        [
            (128, 128, 32, 32),
            (64, 128, 16, 32),
            (256, 64, 32, 16),
        ],
    )
    def test_zz_matches_manual_computation(self, M, N, BM, BN):
        """Verify MakeZZ forward_index matches manual ZZ offset computation."""
        QN = N // BN

        zz = make_zz(_imms(M, N), [0, 1], _imms(BM, BN))

        # ZZ layout formula: decompose (i, j) into block and intra-block coords,
        # then compute offset = qm * (BM*BN*QN) + bm * BN + qn * (BM*BN) + bn
        # where qm = i // BM, bm = i % BM, qn = j // BN, bn = j % BN
        random.seed(42)
        for _ in range(200):
            i = random.randint(0, M - 1)
            j = random.randint(0, N - 1)
            off_zz = _eval(zz, i, j)

            qm, bm = divmod(i, BM)
            qn, bn = divmod(j, BN)
            expected = qm * (BM * BN * QN) + bm * BN + qn * (BM * BN) + bn
            assert off_zz == expected, f"Mismatch at ({i},{j}): zz={off_zz} expected={expected}"


# ---------------------------------------------------------------------------
# 6. IsSameLayout / IsLayoutMatch free functions
# ---------------------------------------------------------------------------


class TestLayoutRelations:
    def test_is_same_layout_true(self):
        a = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        b = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        assert is_same_layout(a, b)

    def test_is_same_layout_false(self):
        a = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        b = make_zn(_imms(128, 128), [0, 1], _imms(32, 32))
        assert not is_same_layout(a, b)

    def test_is_layout_match_same_shape(self):
        a = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        b = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        assert is_layout_match(a, b)

    def test_is_layout_match_different_shape(self):
        a = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        b = make_zz(_imms(256, 64), [0, 1], _imms(32, 32))
        assert is_layout_match(a, b)


# ---------------------------------------------------------------------------
# 7. DeriveLayoutLike
# ---------------------------------------------------------------------------


class TestDeriveLayoutLike:
    def test_derive_zz_to_larger(self):
        src = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        dst = derive_layout_like(src, _imms(256, 64), None)
        assert dst is not None
        # Should be ZZ with same block, different grid
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [2, 2]
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 32  # BM preserved
        assert ms[2] == 32  # BN preserved
        assert ms[1] == 8  # QM = 256/32
        assert ms[3] == 2  # QN = 64/32

    def test_derive_row_major(self):
        src = make_row_major(_imms(128, 64))
        dst = derive_layout_like(src, _imms(256, 32), None)
        assert dst is not None
        assert _eval(dst, 1, 0) == 32
        assert _eval(dst, 0, 1) == 1

    def test_derive_zn_to_different_shape(self):
        """DeriveLayoutLike preserves ZN stride pattern (col-major inner block)."""
        src = make_zn(_imms(128, 128), [0, 1], _imms(32, 32))
        dst = derive_layout_like(src, _imms(256, 64), None)
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [2, 2]
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 32  # BM preserved
        assert ms[2] == 32  # BN preserved
        assert ms[1] == 8  # QM = 256/32
        assert ms[3] == 2  # QN = 64/32
        # ZN inner-block: stride(BM)=1, stride(BN)=BM=32
        mst = [int(x) for x in get_mode_stride(dst)]
        assert mst[0] == 1  # stride_bm
        assert mst[2] == 32  # stride_bn

    def test_derive_zz_preserves_stride_pattern(self):
        """DeriveLayoutLike preserves ZZ stride pattern (row-major inner block)."""
        src = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        dst = derive_layout_like(src, _imms(256, 64), None)
        assert dst is not None
        # ZZ inner-block: stride(BN)=1, stride(BM)=BN=32
        mst = [int(x) for x in get_mode_stride(dst)]
        assert mst[2] == 1  # stride_bn
        assert mst[0] == 32  # stride_bm

    def test_derive_zzz_to_different_shape(self):
        """DeriveLayoutLike works for 3-level (ZZZ) layouts."""
        # cluster=4: [B=16, C=4, G=ceildiv(128,64)=2] per dim
        src = make_zzz(_imms(128, 128), [0, 1], _imms(16, 16), _imms(4, 4))
        dst = derive_layout_like(src, _imms(256, 64), None)
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [3, 3]
        ms = [int(x) for x in get_mode_shape(dst)]
        # dim0: B=16, C=4 (preserved), G=ceildiv(256, 16*4)=4
        assert ms[0] == 16  # BM preserved
        assert ms[1] == 4  # CM preserved (cluster)
        assert ms[2] == 4  # GM derived
        # dim1: B=16, C=4 (preserved), G=ceildiv(64, 16*4)=1
        assert ms[3] == 16  # BN preserved
        assert ms[4] == 4  # CN preserved (cluster)
        assert ms[5] == 1  # GN derived

    def test_derive_nzz_to_different_shape(self):
        """DeriveLayoutLike works for NZZ layouts."""
        src = make_nzz(_imms(128, 128), [0, 1], _imms(16, 16), _imms(4, 4))
        dst = derive_layout_like(src, _imms(64, 256), None)
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [3, 3]
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 16  # BM preserved
        assert ms[1] == 4  # CM preserved (cluster)
        assert ms[3] == 16  # BN preserved
        assert ms[4] == 4  # CN preserved (cluster)

    @pytest.mark.parametrize(
        "M,N,BM,BN",
        [
            (63, 127, 32, 32),
            (100, 200, 32, 64),
            (33, 65, 16, 16),
        ],
    )
    def test_derive_non_divisible(self, M, N, BM, BN):
        """DeriveLayoutLike with non-divisible destination shapes (ceildiv)."""
        src = make_zz(_imms(128, 128), [0, 1], _imms(BM, BN))
        dst = derive_layout_like(src, _imms(M, N), None)
        assert dst is not None
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == BM
        assert ms[2] == BN
        assert ms[1] == (M + BM - 1) // BM  # ceildiv
        assert ms[3] == (N + BN - 1) // BN

    def test_derive_3d_zz(self):
        """DeriveLayoutLike for a 3D buffer with ZZ on axes (1,2)."""
        src = make_zz(_imms(8, 128, 64), [1, 2], _imms(32, 32))
        dst = derive_layout_like(src, _imms(4, 256, 32), None)
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [1, 2, 2]
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 4  # batch dim
        assert ms[1] == 32  # BM preserved
        assert ms[2] == 8  # QM = 256/32
        assert ms[3] == 32  # BN preserved
        assert ms[4] == 1  # QN = 32/32

    def test_derive_4d_zz(self):
        """DeriveLayoutLike for a 4D attention buffer with ZZ on axes (1,3)."""
        src = make_zz(_imms(2, 128, 4, 64), [1, 3], _imms(32, 32))
        # Explicit axis_map to place blocked dims on dst axes 1 and 3
        # (default would place them on the last 2 axes: 2 and 3).
        dst = derive_layout_like(src, _imms(4, 64, 8, 128), [_imm(1), _imm(3)])
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [1, 2, 1, 2]
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 4  # bsz
        assert ms[1] == 32  # BM preserved
        assert ms[2] == 2  # QM = 64/32
        assert ms[3] == 8  # nhead
        assert ms[4] == 32  # BN preserved
        assert ms[5] == 4  # QN = 128/32

    def test_derive_zz_matches_constructor(self):
        """Derived layout matches direct constructor for same shape."""
        src = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        derived = derive_layout_like(src, _imms(256, 64), None)
        direct = make_zz(_imms(256, 64), [0, 1], _imms(32, 32))
        assert is_same_layout(derived, direct)

    def test_derive_zn_matches_constructor(self):
        """Derived ZN layout matches direct constructor for same shape."""
        src = make_zn(_imms(128, 128), [0, 1], _imms(32, 32))
        derived = derive_layout_like(src, _imms(256, 64), None)
        direct = make_zn(_imms(256, 64), [0, 1], _imms(32, 32))
        assert is_same_layout(derived, direct)

    def test_derive_zzz_matches_constructor(self):
        """Derived ZZZ layout matches direct constructor for same shape."""
        src = make_zzz(_imms(128, 128), [0, 1], _imms(16, 16), _imms(4, 4))
        derived = derive_layout_like(src, _imms(256, 64), None)
        direct = make_zzz(_imms(256, 64), [0, 1], _imms(16, 16), _imms(4, 4))
        assert is_same_layout(derived, direct)

    def test_derive_nzz_matches_constructor(self):
        """Derived NZZ layout matches direct constructor for same shape."""
        src = make_nzz(_imms(128, 128), [0, 1], _imms(16, 16), _imms(4, 4))
        derived = derive_layout_like(src, _imms(64, 256), None)
        direct = make_nzz(_imms(64, 256), [0, 1], _imms(16, 16), _imms(4, 4))
        assert is_same_layout(derived, direct)

    def test_derive_from_dynamic_2d_source(self):
        """Derive from a 2D source whose extents are dynamic (one symbolic
        stride)."""
        M = tir.Var("m", "int32")
        K = tir.Var("k", "int32")
        src = make_zz([M, K], [0, 1], _imms(32, 32))
        derived = derive_layout_like(src, _imms(256, 64), None)
        assert derived is not None
        direct = make_zz(_imms(256, 64), [0, 1], _imms(32, 32))
        assert is_same_layout(derived, direct)

    def test_derive_from_dynamic_higher_rank_source(self):
        """RecoverPhysicalOrder (Option A) handles several symbolic strides:
        derive from a 4D source whose two tiled extents are dynamic.  The
        non-tiled batch sits physically outside the tiled counts, so the
        source carries multiple symbolic strides."""
        S = tir.Var("s", "int32")
        D = tir.Var("d", "int32")
        src = make_zz([_imm(2), S, _imm(4), D], [1, 3], _imms(32, 32))
        derived = derive_layout_like(src, _imms(2, 128, 4, 64), [_imm(1), _imm(3)])
        assert derived is not None
        direct = make_zz(_imms(2, 128, 4, 64), [1, 3], _imms(32, 32))
        assert is_same_layout(derived, direct)

    def test_is_layout_match_dynamic_higher_rank(self):
        """IsLayoutMatch on a dynamic higher-rank layout: derivation +
        symbolic SameLayout over several symbolic strides (Option A)."""
        S = tir.Var("s", "int32")
        D = tir.Var("d", "int32")
        dyn = make_zz([_imm(2), S, _imm(4), D], [1, 3], _imms(32, 32))
        assert is_layout_match(dyn, dyn)


class TestDeriveLayoutLikeRankChanging:
    """Tests for DeriveLayoutLike with rank-mismatched src and dst."""

    def test_3d_zz_reduce_to_2d(self):
        """Reduce: 3D ZZ → 2D. Default axis_map places ZZ on last 2 axes."""
        src = make_zz(_imms(8, 128, 64), [1, 2], _imms(32, 32))
        dst = derive_layout_like(src, _imms(128, 64), None)
        assert dst is not None
        direct = make_zz(_imms(128, 64), [0, 1], _imms(32, 32))
        assert is_same_layout(dst, direct)

    def test_3d_zn_reduce_to_2d(self):
        """Reduce: 3D ZN → 2D. Col-major inner block preserved."""
        src = make_zn(_imms(4, 64, 128), [1, 2], _imms(32, 32))
        dst = derive_layout_like(src, _imms(64, 128), None)
        assert dst is not None
        direct = make_zn(_imms(64, 128), [0, 1], _imms(32, 32))
        assert is_same_layout(dst, direct)

    def test_3d_row_major_reduce_to_2d(self):
        """Reduce: 3D row-major → 2D. 0 blocked dims → all row-major."""
        src = make_row_major(_imms(8, 128, 64))
        dst = derive_layout_like(src, _imms(128, 64), None)
        assert dst is not None
        direct = make_row_major(_imms(128, 64))
        assert is_same_layout(dst, direct)

    def test_2d_zz_broadcast_to_3d(self):
        """Broadcast: 2D ZZ → 3D. Default places ZZ on last 2 axes."""
        src = make_zz(_imms(128, 64), [0, 1], _imms(32, 32))
        dst = derive_layout_like(src, _imms(8, 128, 64), None)
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [1, 2, 2]
        # Dim 0 is row-major, dims 1-2 preserve ZZ block structure
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 8  # batch, single-level
        assert ms[1] == 32  # BM preserved
        assert ms[3] == 32  # BN preserved

    def test_4d_reduce_to_3d(self):
        """Reduce: 4D ZZ(axes=[2,3]) → 3D. ZZ lands on last 2 of dst."""
        src = make_zz(_imms(2, 4, 128, 64), [2, 3], _imms(32, 32))
        dst = derive_layout_like(src, _imms(4, 128, 64), None)
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [1, 2, 2]

    def test_explicit_axis_map(self):
        """Explicit axis_map places ZZ on custom dst axes."""
        src = make_zz(_imms(128, 64), [0, 1], _imms(32, 32))
        dst = derive_layout_like(src, _imms(128, 8, 64), [_imm(0), _imm(2)])
        assert dst is not None
        dl = [int(x) for x in get_dim_levels(dst)]
        assert dl == [2, 1, 2]  # ZZ on dims 0,2; row-major dim 1
        ms = [int(x) for x in get_mode_shape(dst)]
        assert ms[0] == 32  # BM on dst dim 0
        assert ms[2] == 8  # row-major dst dim 1
        assert ms[3] == 32  # BN on dst dim 2

    def test_col_major_same_rank(self):
        """Column-major src → same-rank dst preserves col-major strides."""
        # Column-major 2D: stride pattern is (1, M) not (N, 1)
        src = make_cute(
            _imms(128, 64),  # logical_shape
            _imms(128, 64),  # mode_shape (single-level each)
            _imms(1, 128),  # mode_stride: dim 0 innermost
            [_imm(1), _imm(1)],  # dim_levels
        )
        dst = derive_layout_like(src, _imms(256, 32), None)
        assert dst is not None
        ms = [int(x) for x in get_mode_stride(dst)]
        # Column-major: dim 0 stride=1, dim 1 stride=256
        assert ms[0] == 1
        assert ms[1] == 256

    def test_col_major_3d_reduce_to_2d(self):
        """Column-major 3D → 2D preserves col-major stride ordering."""
        # 3D column-major: strides (1, 8, 8*128) — dim 0 innermost
        src = make_cute(
            _imms(8, 128, 64),
            _imms(8, 128, 64),
            _imms(1, 8, 1024),
            [_imm(1), _imm(1), _imm(1)],
        )
        dst = derive_layout_like(src, _imms(128, 64), None)
        assert dst is not None
        ms = [int(x) for x in get_mode_stride(dst)]
        # Should preserve col-major: dim 0 stride=1, dim 1 stride=128
        assert ms[0] == 1
        assert ms[1] == 128

    def test_col_major_2d_broadcast_to_3d(self):
        """Column-major 2D → 3D. Mapped dims preserve ordering, excess is outermost."""
        src = make_cute(
            _imms(128, 64),
            _imms(128, 64),
            _imms(1, 128),
            [_imm(1), _imm(1)],
        )
        dst = derive_layout_like(src, _imms(8, 128, 64), None)
        assert dst is not None
        ms = [int(x) for x in get_mode_stride(dst)]
        # Mapped single-level dims go to last 2 axes → dst dims 1,2
        # Col-major: dim 1 stride=1, dim 2 stride=128
        # Excess dim 0 is outermost row-major
        assert ms[1] == 1  # dim 1 innermost (from src dim 0)
        assert ms[2] == 128  # dim 2 (from src dim 1)
        assert ms[0] == 128 * 64  # dim 0 outermost


# ---------------------------------------------------------------------------
# 8. Inverse / Reshape throw
# ---------------------------------------------------------------------------


class TestUnsupported:
    def test_inverse_throws(self):
        layout = make_row_major(_imms(128, 64))
        with pytest.raises(Exception, match="Inverse"):
            layout.inverse()

    def test_reshape_throws(self):
        """Reshape is not directly exposed via Python, skip this test."""
        pass


# ---------------------------------------------------------------------------
# 9. Diverse hierarchical layout configurations (ported from old tests)
# ---------------------------------------------------------------------------


def _hdims_to_cute_params(hdims, hstrides, groups):
    """Convert old hdims/hstrides/groups (outermost-first) to CuteLayout params (innermost-first).

    Returns (logical_shape, mode_shape, mode_stride, dim_levels) as plain Python lists.
    """
    import math

    mode_shape = []
    mode_stride = []
    dim_levels = []
    logical_shape = []
    for start, end in groups:
        group_dims = list(hdims[start:end])
        group_strides = list(hstrides[start:end])
        group_dims.reverse()  # outermost-first → innermost-first
        group_strides.reverse()
        mode_shape.extend(group_dims)
        mode_stride.extend(group_strides)
        dim_levels.append(end - start)
        logical_shape.append(math.prod(hdims[start:end]))
    return logical_shape, mode_shape, mode_stride, dim_levels


def _reference_offset(logical_indices, hdims, hstrides, groups):
    """Compute expected physical offset via mixed-radix decomposition + dot product."""
    h_indices = []
    for dim, idx in enumerate(logical_indices):
        start, end = groups[dim]
        factors = hdims[start:end]
        indices = []
        rem = idx
        for f in reversed(factors):
            indices.append(rem % f)
            rem //= f
        h_indices.extend(reversed(indices))
    return sum(h * s for h, s in zip(h_indices, hstrides))


class TestDiverseLayouts:
    """Diverse CuteLayout mixed-radix configurations."""

    @pytest.mark.parametrize(
        "hdims, hstrides, groups, test_indices",
        [
            # 4-level layout: 2 groups of 2
            (
                [8, 128, 8, 128],
                [1024, 1, 8192, 128],
                [(0, 2), (2, 4)],
                [[1, 1], [129, 257], [0, 0], [7, 127]],
            ),
            # 4-level layout: 3 groups (1, 2, 1)
            (
                [4, 8, 16, 32],
                [1, 4, 32, 512],
                [(0, 1), (1, 3), (3, 4)],
                [[3, 20, 10], [0, 0, 0], [3, 127, 31]],
            ),
            # Single group of 3
            (
                [2, 4, 6],
                [24, 6, 1],
                [(0, 3)],
                [[29], [0], [47]],
            ),
            # 2x2 blocked (the ZZ docstring example)
            (
                [2, 2, 2, 2],
                [8, 2, 4, 1],
                [(0, 2), (2, 4)],
                [[2, 1], [0, 0], [3, 3], [0, 2], [3, 1]],
            ),
            # Degenerate dimensions (size-1 modes)
            (
                [4, 1, 1, 5],
                [5, 20, 20, 1],
                [(0, 1), (1, 3), (3, 4)],
                [[3, 0, 2], [0, 0, 0], [3, 0, 4]],
            ),
            # With a zero-stride mode
            (
                [2, 2],
                [1, 0],
                [(0, 2)],
                [[3], [0], [1], [2]],
            ),
        ],
    )
    def test_cute_layout_matches_reference(self, hdims, hstrides, groups, test_indices):
        logical_shape, mode_shape, mode_stride, dim_levels = _hdims_to_cute_params(hdims, hstrides, groups)
        layout = make_cute(
            _imms(*logical_shape),
            _imms(*mode_shape),
            _imms(*mode_stride),
            dim_levels,
        )

        for indices in test_indices:
            actual = _eval(layout, *indices)
            expected = _reference_offset(indices, hdims, hstrides, groups)
            assert actual == expected, (
                f"hdims={hdims}, hstrides={hstrides}, groups={groups}, indices={indices}: got {actual}, expected {expected}"
            )

    @pytest.mark.parametrize(
        "hdims, hstrides, groups",
        [
            ([8, 128, 8, 128], [1024, 1, 8192, 128], [(0, 2), (2, 4)]),
            ([4, 8, 16, 32], [1, 4, 32, 512], [(0, 1), (1, 3), (3, 4)]),
            ([2, 4, 6], [24, 6, 1], [(0, 3)]),
            ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)]),
            ([4, 1, 1, 5], [5, 20, 20, 1], [(0, 1), (1, 3), (3, 4)]),
        ],
    )
    def test_random_indices_match_reference(self, hdims, hstrides, groups):
        """Randomly sample 200 indices and verify offset matches reference."""
        logical_shape, mode_shape, mode_stride, dim_levels = _hdims_to_cute_params(hdims, hstrides, groups)
        layout = make_cute(
            _imms(*logical_shape),
            _imms(*mode_shape),
            _imms(*mode_stride),
            dim_levels,
        )

        random.seed(42)
        for _ in range(200):
            indices = [random.randint(0, s - 1) for s in logical_shape]
            actual = _eval(layout, *indices)
            expected = _reference_offset(indices, hdims, hstrides, groups)
            assert actual == expected, f"hdims={hdims}, indices={indices}: got {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# 10. ComputeContiguousTileSteps
# ---------------------------------------------------------------------------

compute_contiguous_steps = _get("tl.ComputeContiguousTileSteps")


def _steps_to_list(result):
    """Convert FFI result [[dim, extent], ...] to list of (dim, extent) tuples."""
    return [(int(pair[0]), int(pair[1])) for pair in result]


class TestComputeContiguousTileSteps:
    def test_row_major_2d(self):
        """Row-major(128, 256): contiguous W then H — entire buffer."""
        layout = make_row_major(_imms(128, 256))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        assert steps == [(1, 256), (0, 128)]

    def test_row_major_3d(self):
        """Row-major(4, 8, 16): contiguous from innermost outward."""
        layout = make_row_major(_imms(4, 8, 16))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        assert steps == [(2, 16), (1, 8), (0, 4)]

    def test_zz_32x32(self):
        """ZZ(128, 128) with block (32, 32): fully contiguous (ZZ preserves all)."""
        layout = make_zz(_imms(128, 128), [0, 1], _imms(32, 32))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # ZZ is fully contiguous: block inner (W:32, H:32), then outer (W:4, H:4)
        assert steps == [(1, 32), (0, 32), (1, 4), (0, 4)]

    def test_zz_32x64(self):
        """ZZ(128, 256) with block (32, 64): fully contiguous."""
        layout = make_zz(_imms(128, 256), [0, 1], _imms(32, 64))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # Block: W:64, H:32; Outer: W:4, H:4
        assert steps == [(1, 64), (0, 32), (1, 4), (0, 4)]

    def test_zn_32x64(self):
        """ZN(128, 256) with block (32, 64): fully contiguous, col-major inner."""
        layout = make_zn(_imms(128, 256), [0, 1], _imms(32, 64))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # ZN: col-major inner block (H:32 first, then W coalesces across boundary).
        # Dim 0 modes: (32, stride=1), (4, stride=8192)
        # Dim 1 modes: (64, stride=32), (4, stride=2048) — coalesces to (256, stride=32)
        # Walk: (0, 32), (1, 256), (0, 4)
        assert steps == [(0, 32), (1, 256), (0, 4)]

    def test_zzz_block_and_cluster(self):
        """ZZZ with block (32, 64) and cluster (2, 4): fully contiguous (all row-major)."""
        layout = make_zzz(_imms(256, 1024), [0, 1], _imms(32, 64), _imms(2, 4))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # ZZZ: all 3 levels row-major — fully contiguous.
        # Block: W:64, H:32; Cluster: W:4, H:2; Grid: W:4, H:4
        assert steps == [(1, 64), (0, 32), (1, 4), (0, 2), (1, 4), (0, 4)]

    def test_nzz_block_and_cluster(self):
        """NZZ with block (32, 64) and cluster (2, 4): col-major at group level."""
        layout = make_nzz(_imms(256, 1024), [0, 1], _imms(32, 64), _imms(2, 4))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # NZZ = {false, false, true}: block=row, tile=row, group=col
        # Col-major group level causes dim-0 modes CM(2) and GM(4) to coalesce.
        # Block: W:64, H:32; Tile: W:4; then coalesced H:8; then Grid W:4
        assert steps == [(1, 64), (0, 32), (1, 4), (0, 8), (1, 4)]

    def test_dynamic_outer_mode(self):
        """Dynamic logical shape: stride walk stops at symbolic mode boundary."""
        M = tir.Var("M", "int32")
        # Construct a ZZ-like layout with dynamic outer mode directly.
        # ZZ(M, 128) with block (32, 32):
        #   dim 0: [BM=32, QM=ceildiv(M,32)] strides [32, 4096]
        #   dim 1: [BN=32, QN=4] strides [1, 1024]
        cdiv = tir.ceildiv(M, _imm(32))
        layout = make_cute(
            [M, _imm(128)],
            [_imm(32), cdiv, _imm(32), _imm(4)],
            _imms(32, 4096, 1, 1024),
            [2, 2],
        )
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # Block modes (static): W:32 stride=1, H:32 stride=32
        # Dim 1 outer QN=4 stride=1024 is static, included.
        # Dim 0 outer QM=ceildiv(M,32) is symbolic, stops there.
        assert steps == [(1, 32), (0, 32), (1, 4)]

    def test_total_contiguous_elements(self):
        """ZZ is fully contiguous: total == buffer size."""
        for block_shape, buf_shape in [
            ((32, 32), (256, 256)),
            ((32, 64), (128, 256)),
            ((16, 128), (64, 128)),
        ]:
            layout = make_zz(_imms(*buf_shape), [0, 1], _imms(*block_shape))
            steps = _steps_to_list(compute_contiguous_steps(layout))
            total = 1
            for _, extent in steps:
                total *= extent
            expected_total = buf_shape[0] * buf_shape[1]
            assert total == expected_total, f"block_shape={block_shape}: total={total}, expected={expected_total}"

    def test_zz_3d_batch(self):
        """ZZ on last 2 dims of a 3D buffer: all modes fully contiguous."""
        layout = make_zz(_imms(4, 128, 128), [1, 2], _imms(32, 32))
        steps = _steps_to_list(compute_contiguous_steps(layout))
        # Block: (2, 32), (1, 32); Outer: (2, 4), (1, 4); Batch: (0, 4)
        assert steps == [(2, 32), (1, 32), (2, 4), (1, 4), (0, 4)]

    def test_block_boundary_is_first_step(self):
        """First two steps in ZZ always match the block shape."""
        for bh, bw in [(32, 32), (32, 64), (16, 128)]:
            layout = make_zz(_imms(256, 256), [0, 1], _imms(bh, bw))
            steps = _steps_to_list(compute_contiguous_steps(layout))
            # First step: width dim with block width
            assert steps[0] == (1, bw)
            # Second step: height dim with block height
            assert steps[1] == (0, bh)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
