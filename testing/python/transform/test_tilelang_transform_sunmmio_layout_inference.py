"""Tests for the standalone Sunmmio Layout Inference pass.

Verifies that SunmmioLayoutInference correctly assigns CuteLayout
to SRAM and DRAM buffers via priority-based inference (kStrict > kCommon > kFree).
"""

import tilelang
import pytest
import tvm_ffi
from tilelang import tvm as tvm
from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T
import tilelang.env as env
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout, make_zn_layout

tilelang.env.disable_cache()


# ---------------------------------------------------------------------------
# Test harness: the canonical layout-inference pipeline + a layout classifier.
# Every test runs `run_sunmmio_layout_inference` (so the pass order never
# drifts) and asserts with `assert_layout` (kind + block shape).
# ---------------------------------------------------------------------------

_dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
_mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")
_mode_stride = tvm_ffi.get_global_func("tl.CuteLayout_mode_stride")

# Populated by LayoutVisual; run_sunmmio_layout_inference returns a fresh copy.
collected_result = {}


@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):
    def visit_block_(self, op: tir.Block) -> None:
        anns = op.annotations
        # ApplyToIR annotates every block with the final maps; capture the SRAM
        # layout_map and the DRAM global_layout_map, merged by buffer name.
        if "layout_map" not in anns and "global_layout_map" not in anns:
            return
        collected_result.clear()
        for map_key in ("layout_map", "global_layout_map"):
            if map_key in anns:
                for key, layout in anns[map_key].items():
                    collected_result[key.name] = layout


def LayoutVisual():
    """Pass that records the final layout maps into ``collected_result``."""

    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)


def run_sunmmio_layout_inference(mod, target):
    """Run the canonical Sunmmio layout-inference pipeline and return the
    inferred per-buffer layouts as ``{buffer_name: Layout}``.

    Negative tests that expect a failure inside ``SunmmioLayoutInference`` should
    wrap this call in ``pytest.raises`` (they never reach the capture step).
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


def _maybe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x  # symbolic stride/shape; leave as-is


def _per_dim_modes(layout):
    """Split the flat (mode_shape, mode_stride) arrays into per-dim modes,
    innermost-mode-first (index 0 is innermost), matching the C++ convention."""
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


def layout_kind(layout):
    """Classify a CuteLayout as RowMajor, ZZ (any ZZ-like: ZZ/ZZZ/NZZ), or ZN.
    Mirrors C++ ``sunmmio::IsZZLike``: among the last two blocked dims, ZZ-like
    means the higher-indexed dim's innermost stride is smaller."""
    dims = _per_dim_modes(layout)
    blocked = [d for d, m in enumerate(dims) if len(m["shape"]) > 1]
    if len(blocked) < 2:
        return "RowMajor"
    ax0, ax1 = blocked[-2], blocked[-1]
    s0 = dims[ax0]["stride"][0]
    s1 = dims[ax1]["stride"][0]
    if isinstance(s0, int) and isinstance(s1, int):
        return "ZZ" if s1 < s0 else "ZN"
    return "ZZ" if s1 == 1 else "ZN"  # fallback: inner stride 1 is ZZ-like


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


# ---------------------------------------------------------------------------
# Test: GEMM layout values — ZZ for A/C, ZZ/ZN for B based on transB
# ---------------------------------------------------------------------------

M, N, K = 128, 128, 64
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
DTYPE = T.float16
ACCUM_DTYPE = T.float32


def gemm_kernel_notransB():
    @T.prim_func
    def main(
        A: T.Tensor((M, K), DTYPE),
        B: T.Tensor((K, N), DTYPE),
        C: T.Tensor((M, N), ACCUM_DTYPE),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
            C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), ACCUM_DTYPE)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=False)

            T.copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

    return tvm.IRModule({"main": main})


def gemm_kernel_transB():
    @T.prim_func
    def main(
        A: T.Tensor((M, K), DTYPE),
        B: T.Tensor((N, K), DTYPE),  # transposed shape
        C: T.Tensor((M, N), ACCUM_DTYPE),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), DTYPE)  # transposed tile
            C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), ACCUM_DTYPE)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=True)

            T.copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

    return tvm.IRModule({"main": main})


def test_gemm_layout_inference():
    """Verify Gemm assigns ZZ to A(ASRAM), ZN to B(WSRAM, transB=False), ZZ to C(RSRAM)."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(gemm_kernel_notransB(), target)

    # DRAM params are captured too (row-major by default).
    for name in ("A", "B", "C"):
        assert name in layouts, f"{name} (DRAM) not in layouts, got {sorted(layouts)}"

    # A (ASRAM): ZZ, B (WSRAM): ZN (transB=False, TMM.MN), C (RSRAM): ZZ — fp16 block (32,32)
    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))
    assert_layout(layouts, "C_shared", "ZZ", block=(32, 32))


def test_gemm_transB_layout_inference():
    """Verify Gemm with transB=True assigns ZZ (not ZN) to B(WSRAM)."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(gemm_kernel_transB(), target)

    # B (WSRAM): ZZ for transB=True (TMM.MT mode)
    assert_layout(layouts, "B_shared", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Copy-only kernel — default ZZ for RSRAM
# ---------------------------------------------------------------------------


def copy_only_kernel():
    M, N = 128, 128
    block_M, block_N = 64, 64
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def test_copy_only_layout_inference():
    """Copy-only kernel: RSRAM gets layout from Phase 3 scope-dependent defaults.

    Copy is the most flexible op — it bridges any layout mismatch via
    dma_layout_transform and does not propagate layouts.  In a copy-only
    kernel (no Gemm/Comm/Reduce to seed SRAM layouts), RSRAM receives the
    scope-dependent default: ZZ with (32, 32) block shape.
    """
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(copy_only_kernel(), target)

    # RSRAM buffer A_shared gets the Phase 3 default: ZZ(32, 32).
    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Every SRAM buffer gets a layout (completeness)
# ---------------------------------------------------------------------------


def test_all_sram_buffers_get_layouts():
    """Verify every allocated SRAM buffer ends up with a layout."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(gemm_kernel_notransB(), target)

    # A_shared (ASRAM), B_shared (WSRAM), C_shared (RSRAM) — one per SRAM scope.
    for name in ("A_shared", "B_shared", "C_shared"):
        assert name in layouts, f"{name} missing a layout, got {sorted(layouts)}"


# ---------------------------------------------------------------------------
# Test: Non-Sunmmio target raises error
# ---------------------------------------------------------------------------


def test_pass_requires_sunmmio_target():
    """SunmmioLayoutInference should error on non-Sunmmio targets."""
    target = determine_target("cuda", return_object=True)
    if target is None:
        pytest.skip("CUDA target not available")

    with tvm.target.Target(target):
        mod = gemm_kernel_notransB()
        mod = tvm.tir.transform.BindTarget(target)(mod)

        # Should raise an error because target is not Sunmmio
        with pytest.raises(tvm.error.InternalError):
            mod = tl.transform.SunmmioLayoutInference()(mod)


# ---------------------------------------------------------------------------
# Test: DRAM buffers get global_layout_map entries (row-major by default)
# ---------------------------------------------------------------------------


def test_dram_default_row_major():
    """Verify DRAM buffers without tensor_meta get row-major layout."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(copy_only_kernel(), target)

    # DRAM buffers without tensor_meta default to row-major.
    assert_layout(layouts, "A", "RowMajor")
    assert_layout(layouts, "B", "RowMajor")


# ---------------------------------------------------------------------------
# Test: Reduce along blockwise axis — dst should be row-major
# ---------------------------------------------------------------------------


def reduce_blockwise_kernel():
    """Gemm produces ZZ on C_shared (RSRAM), then reduce along axis 1 (blockwise)."""
    M, N, K = 128, 128, 64
    block_M, block_N, block_K = 64, 64, 32
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_reduce = T.alloc_shared((block_M,), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=False)

            T.reduce_sum(C_shared, C_reduce, dim=1)
            T.copy(C_reduce, C[bx * block_M])

    return tvm.IRModule({"main": main})


def test_reduce_blockwise_axis_gives_row_major():
    """Reduce along blockwise axis (dim=1 of 2D ZZ) → dst should be row-major."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(reduce_blockwise_kernel(), target)

    # C_shared (RSRAM, 2D) has ZZ from Gemm; C_reduce (1D) becomes row-major
    # because the reduced axis was a blocked dim.
    assert_layout(layouts, "C_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "C_reduce", "RowMajor")


# ---------------------------------------------------------------------------
# Test: Reduce along outer axis — dst should preserve ZZ
# ---------------------------------------------------------------------------


def reduce_outer_kernel():
    """3D buffer annotated with ZZ on last two dims, reduce along axis 0 (outer)."""
    block_M, block_N = 64, 64
    batch = 32
    dtype = T.float16

    # Create ZZ layout for the 3D buffer (ZZ on inner two dims)
    zz_3d = make_zz_layout((batch, block_M, block_N), axes=[1, 2], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((batch, block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((batch, block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: zz_3d})
            T.copy(A[0, 0, 0], A_shared)
            T.reduce_sum(A_shared, B_shared, dim=0)
            T.copy(B_shared, B[0, 0])

    return tvm.IRModule({"main": main})


def test_reduce_outer_axis_preserves_zz():
    """Reduce along outer axis (dim=0 of 3D ZZ) → dst should preserve ZZ on inner dims."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(reduce_outer_kernel(), target)

    # A_shared (3D) keeps its annotated ZZ on inner dims; reducing the outer
    # axis preserves ZZ on the 2D B_shared.
    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "B_shared", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: 3D ZZ, reduce along blocked axis → row-major
# ---------------------------------------------------------------------------


def reduce_3d_zz_blocked_axis_kernel():
    """3D ZZ on dims [1,2], reduce along dim 2 (a blocked axis)."""
    batch = 8
    block_M, block_N = 64, 64
    dtype = T.float16

    zz_3d = make_zz_layout((batch, block_M, block_N), axes=[1, 2], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((batch, block_M, block_N), dtype),
        B: T.Tensor((batch, block_M), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((batch, block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((batch, block_M), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: zz_3d})
            T.copy(A[0, 0, 0], A_shared)
            T.reduce_sum(A_shared, B_shared, dim=2)
            T.copy(B_shared, B[0, 0])

    return tvm.IRModule({"main": main})


def test_reduce_3d_zz_blocked_axis_gives_row_major():
    """3D ZZ, reduce along blocked axis (dim=2) → dst should be row-major."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(reduce_3d_zz_blocked_axis_kernel(), target)

    # Reducing a blocked axis destroys the block structure → row-major.
    assert_layout(layouts, "B_shared", "RowMajor")


# ---------------------------------------------------------------------------
# Test: 3D ZN, reduce along outer axis → preserves 2D ZN
# ---------------------------------------------------------------------------


def reduce_3d_zn_outer_axis_kernel():
    """3D ZN on dims [1,2], reduce along dim 0 (outer axis)."""
    batch = 8
    block_M, block_N = 64, 64
    dtype = T.float16

    zn_3d = make_zn_layout((batch, block_M, block_N), axes=[1, 2], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((batch, block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((batch, block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: zn_3d})
            T.copy(A[0, 0, 0], A_shared)
            T.reduce_sum(A_shared, B_shared, dim=0)
            T.copy(B_shared, B[0, 0])

    return tvm.IRModule({"main": main})


def test_reduce_3d_zn_outer_axis_preserves_zn():
    """3D ZN, reduce along outer axis (dim=0) → dst should preserve 2D ZN."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(reduce_3d_zn_outer_axis_kernel(), target)

    # Reducing the outer axis of a 3D ZN preserves ZN on the 2D result.
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: 3D row-major, reduce along outer axis → 2D row-major
# ---------------------------------------------------------------------------


def reduce_3d_row_major_outer_axis_kernel():
    """3D row-major, reduce along dim 0 (outer axis)."""
    batch = 8
    block_M, block_N = 64, 64
    dtype = T.float16

    from tilelang.layout.sunmmio_layouts import make_row_major

    rm_3d = make_row_major([batch, block_M, block_N])

    @T.prim_func
    def main(
        A: T.Tensor((batch, block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((batch, block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: rm_3d})
            T.copy(A[0, 0, 0], A_shared)
            T.reduce_sum(A_shared, B_shared, dim=0)
            T.copy(B_shared, B[0, 0])

    return tvm.IRModule({"main": main})


def test_reduce_3d_row_major_outer_axis_preserves_row_major():
    """3D row-major, reduce along outer axis (dim=0) → dst should be 2D row-major."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(reduce_3d_row_major_outer_axis_kernel(), target)

    # Row-major in, row-major out (outer-axis reduce).
    assert_layout(layouts, "B_shared", "RowMajor")


# ---------------------------------------------------------------------------
# Test: 4D ZZ, reduce along outer axis → 3D with ZZ on last 2 dims
# ---------------------------------------------------------------------------


def reduce_4d_zz_outer_axis_kernel():
    """4D ZZ on dims [2,3], reduce along dim 0 (outer axis)."""
    d0, d1 = 2, 4
    block_M, block_N = 64, 64
    dtype = T.float16

    zz_4d = make_zz_layout((d0, d1, block_M, block_N), axes=[2, 3], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((d0, d1, block_M, block_N), dtype),
        B: T.Tensor((d1, block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((d0, d1, block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((d1, block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: zz_4d})
            T.copy(A[0, 0, 0, 0], A_shared)
            T.reduce_sum(A_shared, B_shared, dim=0)
            T.copy(B_shared, B[0, 0, 0])

    return tvm.IRModule({"main": main})


def test_reduce_4d_zz_outer_axis_preserves_zz():
    """4D ZZ, reduce along outer axis (dim=0) → 3D dst with ZZ on last 2 dims."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(reduce_4d_zz_outer_axis_kernel(), target)

    # 4D ZZ, reduce outer axis → 3D with ZZ preserved on the inner two dims.
    assert_layout(layouts, "B_shared", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Broadcast propagates ZZ layout from src to dst
# ---------------------------------------------------------------------------


def broadcast_zz_kernel():
    """Gemm produces ZZ on C_shared (RSRAM), broadcast to B_shared (RSRAM)."""
    M, N, K = 128, 128, 64
    block_M, block_N, block_K = 64, 64, 32
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_bcast = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=False)

            T.comm.broadcast(C_shared, C_bcast, (1, 2), direction="all")
            T.copy(C_bcast, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def test_broadcast_propagates_zz():
    """Broadcast: src has ZZ from Gemm → dst should also get ZZ."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(broadcast_zz_kernel(), target)

    # C_shared has ZZ from Gemm; broadcast propagates ZZ to C_bcast.
    assert_layout(layouts, "C_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "C_bcast", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Broadcast propagates annotated ZN layout
# ---------------------------------------------------------------------------


def broadcast_zn_kernel():
    """Annotate src with ZN, broadcast to dst — dst should get ZN."""
    block_M, block_N = 64, 64
    dtype = T.float16

    zn_layout = make_zn_layout((block_M, block_N), axes=[0, 1], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: zn_layout})
            T.copy(A[0, 0], A_shared)
            T.comm.broadcast(A_shared, B_shared, (1, 2), direction="all")
            T.copy(B_shared, B[0, 0])

    return tvm.IRModule({"main": main})


def test_broadcast_propagates_zn():
    """Broadcast: src annotated ZN → dst should also get ZN (not ZZ)."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(broadcast_zn_kernel(), target)

    # Annotated ZN src → dst also gets ZN.
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Put propagates ZZ layout between src and dst
# ---------------------------------------------------------------------------


def put_zz_kernel():
    """Gemm produces ZZ on C_shared, put to C_put — C_put should get ZZ."""
    M, N, K = 128, 128, 64
    block_M, block_N, block_K = 64, 64, 32
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_put = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=False)

            T.comm.put(C_shared, C_put, (1, 2), (2, 3))
            T.copy(C_put, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def test_put_propagates_zz():
    """Put: src has ZZ from Gemm → dst should get ZZ."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(put_zz_kernel(), target)

    # put propagates the Gemm ZZ from C_shared to C_put.
    assert_layout(layouts, "C_put", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Allgather propagates ZZ layout (2D send → 3D recv)
# ---------------------------------------------------------------------------


def allgather_zz_kernel():
    """Gemm produces ZZ on C_shared, allgather to C_gather (3D)."""
    M, N, K = 128, 128, 64
    block_M, block_N, block_K = 64, 64, 32
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_gather = T.alloc_shared((16, block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=False)

            T.comm.all_gather(C_shared, C_gather, direction="all")

    return tvm.IRModule({"main": main})


def test_allgather_propagates_zz():
    """Allgather: 2D ZZ send → 3D recv. ZZ should land on last 2 dims of recv."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(allgather_zz_kernel(), target)

    # 2D ZZ send → 3D recv: ZZ lands on the last two dims.
    assert_layout(layouts, "C_gather", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: Allgather propagates row-major layout
# ---------------------------------------------------------------------------


def allgather_row_major_kernel():
    """Row-major src, allgather to 3D recv — recv should be row-major."""
    block_M, block_N = 64, 64
    dtype = T.float16

    from tilelang.layout.sunmmio_layouts import make_row_major

    rm_layout = make_row_major([block_M, block_N])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            A_gather = T.alloc_shared((16, block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: rm_layout})
            T.copy(A[0, 0], A_shared)
            T.comm.all_gather(A_shared, A_gather, direction="all")

    return tvm.IRModule({"main": main})


def test_allgather_propagates_row_major():
    """Allgather: 2D row-major send → 3D recv should be row-major."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(allgather_row_major_kernel(), target)

    # Row-major src → 3D recv stays row-major.
    assert_layout(layouts, "A_gather", "RowMajor")


# ---------------------------------------------------------------------------
# Test: Immutable conflict — T.annotate_layout vs Gemm hardware requirement
# ---------------------------------------------------------------------------


def gemm_annotate_conflict_asram_kernel():
    """Annotate ASRAM buffer with ZN, but Gemm.A requires ZZ → conflict."""
    block_M, block_K = 64, 32
    block_N = 64
    dtype = T.float16
    accum_dtype = T.float32

    # ZN layout — incompatible with Gemm.A's ZZ requirement
    wrong_layout = make_zn_layout((block_M, block_K), axes=[0, 1], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((block_K, block_N), dtype),
        C: T.Tensor((block_M, block_N), accum_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.annotate_layout({A_shared: wrong_layout})
            T.clear(C_shared)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, transpose_B=False)
            T.copy(C_shared, C[0, 0])

    return tvm.IRModule({"main": main})


def test_immutable_conflict_annotate_vs_gemm():
    """T.annotate_layout(ZN) on ASRAM buffer conflicts with Gemm's ZZ → hard error."""
    target = determine_target("Sunmmio", return_object=True)
    mod = gemm_annotate_conflict_asram_kernel()
    with pytest.raises(Exception, match="immutable layout"):
        run_sunmmio_layout_inference(mod, target)


def gemm_annotate_conflict_wsram_kernel():
    """Annotate WSRAM buffer with ZZ, but Gemm.B (transB=False) requires ZN → conflict."""
    block_M, block_K = 64, 32
    block_N = 64
    dtype = T.float16
    accum_dtype = T.float32

    # ZZ layout — incompatible with Gemm.B's ZN requirement when transB=False
    wrong_layout = make_zz_layout((block_K, block_N), axes=[0, 1], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((block_K, block_N), dtype),
        C: T.Tensor((block_M, block_N), accum_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.annotate_layout({B_shared: wrong_layout})
            T.clear(C_shared)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, transpose_B=False)
            T.copy(C_shared, C[0, 0])

    return tvm.IRModule({"main": main})


def test_immutable_conflict_wsram_annotate_vs_gemm():
    """T.annotate_layout(ZZ) on WSRAM buffer conflicts with Gemm.B(transB=False)'s ZN → hard error."""
    target = determine_target("Sunmmio", return_object=True)
    mod = gemm_annotate_conflict_wsram_kernel()
    with pytest.raises(Exception, match="immutable layout"):
        run_sunmmio_layout_inference(mod, target)


def gemm_annotate_conflict_rsram_kernel():
    """Annotate RSRAM buffer with ZN, but Gemm.C requires ZZ → conflict."""
    block_M, block_N = 64, 64
    block_K = 32
    dtype = T.float16
    accum_dtype = T.float32

    # ZN layout — incompatible with Gemm.C's ZZ requirement
    wrong_layout = make_zn_layout((block_M, block_N), axes=[0, 1], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((block_K, block_N), dtype),
        C: T.Tensor((block_M, block_N), accum_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.annotate_layout({C_shared: wrong_layout})
            T.clear(C_shared)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, transpose_B=False)
            T.copy(C_shared, C[0, 0])

    return tvm.IRModule({"main": main})


def test_immutable_conflict_rsram_annotate_vs_gemm():
    """T.annotate_layout(ZN) on RSRAM buffer conflicts with Gemm.C's ZZ → hard error."""
    target = determine_target("Sunmmio", return_object=True)
    mod = gemm_annotate_conflict_rsram_kernel()
    with pytest.raises(Exception, match="immutable layout"):
        run_sunmmio_layout_inference(mod, target)


# ---------------------------------------------------------------------------
# Test: DRAM ZN layout → Copy to ASRAM should fail IsZZLike validation
# ---------------------------------------------------------------------------


def dram_zn_to_asram_kernel():
    """DRAM buffer with ZN layout (via MeshTensor) copied to ASRAM → IsZZLike check should fail."""
    from tilelang.language.mesh_tensor import MeshShardingPolicy, MeshReplicationType

    M, K, N = 64, 32, 64
    block_M, block_K, block_N = 64, 32, 64
    dtype = "float16"
    accum_dtype = "float32"

    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)

    # ZN layout on DRAM — incompatible with ASRAM (Gemm.A requires ZZ-like)
    zn_dram = make_zn_layout((M, K), axes=[0, 1], block_shape=[32, 32])

    A_tensor = T.MeshTensor((M, K), policy, device_mesh, dtype=dtype, layout=zn_dram)
    B_tensor = T.MeshTensor((K, N), policy, device_mesh, dtype=dtype)
    C_tensor = T.MeshTensor((M, N), policy, device_mesh, dtype=accum_dtype)

    @T.prim_func
    def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, transpose_B=False)
            T.copy(C_shared, C[0, 0])

    return tvm.IRModule({"main": kernel})


def test_dram_zn_to_asram_succeeds_via_staged_rsram():
    """DRAM with ZN layout → Copy to ASRAM succeeds via LegalizeSunmmioDataPath staging through RSRAM.

    LegalizeSunmmioDataPath rewrites global→asram into global→rsram + rsram→asram,
    so layout inference assigns a ZZ layout to the ASRAM buffer regardless of the
    DRAM layout.
    """
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(dram_zn_to_asram_kernel(), target)

    # LegalizeSunmmioDataPath stages global→asram through RSRAM, so the ASRAM
    # buffer still gets an inferred layout despite the DRAM ZN source.
    assert "A_shared" in layouts, f"A_shared should have an inferred layout. Got: {sorted(layouts)}"


# ---------------------------------------------------------------------------
# Test: DRAM ZN layout → Copy to WSRAM should also fail IsZZLike validation
# ---------------------------------------------------------------------------


def dram_zn_to_wsram_kernel():
    """DRAM buffer with ZN layout (via MeshTensor) copied to WSRAM → IsZZLike check should fail."""
    from tilelang.language.mesh_tensor import MeshShardingPolicy, MeshReplicationType

    M, K, N = 64, 32, 64
    block_M, block_K, block_N = 64, 32, 64
    dtype = "float16"
    accum_dtype = "float32"

    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)

    # ZN layout on DRAM B — incompatible with WSRAM DMA (requires ZZ-like)
    zn_dram = make_zn_layout((K, N), axes=[0, 1], block_shape=[32, 32])

    A_tensor = T.MeshTensor((M, K), policy, device_mesh, dtype=dtype)
    B_tensor = T.MeshTensor((K, N), policy, device_mesh, dtype=dtype, layout=zn_dram)
    C_tensor = T.MeshTensor((M, N), policy, device_mesh, dtype=accum_dtype)

    @T.prim_func
    def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, transpose_B=False)
            T.copy(C_shared, C[0, 0])

    return tvm.IRModule({"main": kernel})


def test_dram_zn_to_wsram_fails():
    """DRAM with ZN layout → Copy to WSRAM should fail (IsZZLike rejects ZN)."""
    target = determine_target("Sunmmio", return_object=True)
    mod = dram_zn_to_wsram_kernel()
    with pytest.raises(Exception, match="ZZ-like DRAM layout"):
        run_sunmmio_layout_inference(mod, target)


# ---------------------------------------------------------------------------
# Test: Correct T.annotate_layout matching Gemm's requirement passes silently
# ---------------------------------------------------------------------------


def gemm_correct_annotate_kernel():
    """Annotate all three Gemm buffers with the correct hardware-mandated layouts.

    A (ASRAM): ZZ(32,32) for fp16
    B (WSRAM): ZN(32,32) for transB=False
    C (RSRAM): ZZ(32,32) for fp32
    """
    block_M, block_K = 64, 32
    block_N = 64
    dtype = T.float16
    accum_dtype = T.float32

    correct_a = make_zz_layout((block_M, block_K), axes=[0, 1], block_shape=[32, 32])
    correct_b = make_zn_layout((block_K, block_N), axes=[0, 1], block_shape=[32, 32])
    correct_c = make_zz_layout((block_M, block_N), axes=[0, 1], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((block_K, block_N), dtype),
        C: T.Tensor((block_M, block_N), accum_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: correct_a,
                    B_shared: correct_b,
                    C_shared: correct_c,
                }
            )
            T.clear(C_shared)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, transpose_B=False)
            T.copy(C_shared, C[0, 0])

    return tvm.IRModule({"main": main})


def test_correct_annotate_accepted():
    """T.annotate_layout matching Gemm's requirement should pass without error."""
    target = determine_target("Sunmmio", return_object=True)
    # Should not raise — correct annotations match Gemm's hardware requirement.
    layouts = run_sunmmio_layout_inference(gemm_correct_annotate_kernel(), target)

    # The annotated layouts are preserved.
    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))
    assert_layout(layouts, "C_shared", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Note: Same-level conflict (two ops proposing different layouts at the same
# InferLevel for the same buffer) is guarded by LOG(FATAL) in TryAssign, but
# cannot be triggered by end-to-end tests with current Sunmmio ops — only
# Gemm proposes at kStrict, and no op proposes at kCommon. The guard exists
# for future ops that may propose conflicting layouts.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Copy→Reduce→Copy — Reduce derives dst layout from src's kFree default
# ---------------------------------------------------------------------------


def copy_reduce_copy_kernel():
    """Copy→Reduce→Copy: the motivating case for phase reordering.

    With defaults assigned before BFS (Phase 3), A_shared gets kFree
    ZZ(32,32).  In Phase 5 BFS, Reduce sees A_shared has a layout and
    derives B_shared's layout.  Since dim=1 is blocked in ZZ(32,32),
    B_shared should get row-major (not a generic ZZ default).
    """
    M, N = 128, 128
    block_M, block_N = 64, 64
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M,), dtype, scope="shared.rsram")

            T.copy(A[bx * block_M, 0], A_shared)
            T.reduce_sum(A_shared, B_shared, dim=1)
            T.copy(B_shared, B[bx * block_M])

    return tvm.IRModule({"main": main})


def test_copy_reduce_copy_layout_derivation():
    """Copy→Reduce→Copy: Reduce derives B_shared from A_shared's kFree default.

    This test verifies the phase reordering fix.  Before the fix, defaults
    were assigned after BFS, so Reduce never saw A_shared's layout and
    both buffers got generic ZZ defaults.  Now:
      Phase 3: A_shared ← ZZ(32,32), B_shared ← row-major (1D default)
      Phase 5 BFS: Reduce derives B_shared from A_shared at kCommon —
                   dim=1 is blocked → row-major (overrides kFree)
    """
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(copy_reduce_copy_kernel(), target)

    # A_shared: 2D RSRAM ZZ(32,32) from the Phase 3 default.
    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))
    # B_shared: Reduce(dim=1) over a blocked ZZ dim → row-major, derived at
    # kCommon (Phase 5 BFS) overriding the kFree default.
    assert_layout(layouts, "B_shared", "RowMajor")


# ---------------------------------------------------------------------------
# Test: Broadcast with incompatible immutable annotations
# ---------------------------------------------------------------------------


def broadcast_mismatched_layouts_kernel():
    """Annotate src with ZZ and dst with ZN — both immutable, broadcast derives conflict."""
    block_M, block_N = 64, 64
    dtype = T.float16

    zz_layout = make_zz_layout((block_M, block_N), axes=[0, 1], block_shape=[32, 32])
    zn_layout = make_zn_layout((block_M, block_N), axes=[0, 1], block_shape=[32, 32])

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.annotate_layout({A_shared: zz_layout, B_shared: zn_layout})
            T.copy(A[0, 0], A_shared)
            T.comm.broadcast(A_shared, B_shared, (1, 2), direction="all")
            T.copy(B_shared, B[0, 0])

    return tvm.IRModule({"main": main})


def test_broadcast_mismatched_immutable_preserves_annotations():
    """Broadcast: both sides annotated with incompatible immutable layouts.

    When both src and dst have immutable annotations, the BFS proposals
    (derived from the other side) are silently rejected by TryAssign at
    kCommon level.  Both buffers keep their original annotations.  The
    incompatibility is a user error that downstream passes will catch.
    """
    target = determine_target("Sunmmio", return_object=True)
    # Both immutable annotations are preserved (no error at layout inference);
    # the BFS proposals derived from the other side are silently rejected.
    layouts = run_sunmmio_layout_inference(broadcast_mismatched_layouts_kernel(), target)

    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: GEMM operand layouts across gemm_v1 / gemm_v2 and block-size variants
# (formerly test_tilelang_transform_sunmmio_gemm_layout.py)
# ---------------------------------------------------------------------------


def matmul(M, N, K, block_M, block_N, block_K, version, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                if version == 1:
                    T.gemm_v1(A_shared, B_shared, C_shared)
                elif version == 2:
                    T.gemm_v2(A_shared, B_shared, C_shared)
                else:
                    raise ValueError(f"unsupported gemm version: {version}")

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


GEMM_TEST_CASES = [
    # (M, N, K, block_M, block_N, block_K, version)
    # gemm v1
    (128, 128, 128, 32, 32, 32, 1),
    (128, 128, 128, 64, 64, 64, 1),
    (128, 128, 128, 64, 32, 64, 1),
    (128, 128, 128, 32, 64, 64, 1),
    (128, 128, 128, 64, 64, 32, 1),
    (128, 128, 128, 64, 32, 32, 1),
    (128, 128, 128, 32, 64, 32, 1),
    (128, 128, 128, 32, 32, 64, 1),
    # gemm v2
    (128, 128, 128, 32, 32, 32, 2),
    (128, 128, 128, 64, 64, 64, 2),
    (128, 128, 128, 64, 32, 64, 2),
    (128, 128, 128, 32, 64, 64, 2),
    (128, 128, 128, 64, 64, 32, 2),
    (128, 128, 128, 64, 32, 32, 2),
    (128, 128, 128, 32, 64, 32, 2),
    (128, 128, 128, 32, 32, 64, 2),
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K, version",
    GEMM_TEST_CASES,
)
def test_tilelang_gemm_sunmmio_layout(M, N, K, block_M, block_N, block_K, version):
    # Enable v2
    env.TILELANG_USE_GEMM_V1 = 0
    assert not env.use_gemm_v1()
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = matmul(M, N, K, block_M, block_N, block_K, version)
    layouts = run_sunmmio_layout_inference(mod, target)

    # A (ASRAM): ZZ, B (WSRAM): ZN (transB=False), C (RSRAM): ZZ — block (32,32) for fp16
    assert_layout(layouts, "A_shared", "ZZ", block=(32, 32))
    assert_layout(layouts, "B_shared", "ZN", block=(32, 32))
    assert_layout(layouts, "C_shared", "ZZ", block=(32, 32))


def matmul_persistent(M, N, K, block_M, block_N, block_K, num_stages, dtype=T.bfloat16, accum_dtype=T.float32):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # Each core iterates its own sharded tile grid with plain nested
            # loops. The MeshTensor sharding already distributes A/B/C across
            # the core mesh, so no persistent core-distribution loop is used.
            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=num_stages):
                        # Stage each A/B tile into a shared buffer, then
                        # all-gather it across the core row / column.
                        # all_gather needs a Buffer source (not an indexed
                        # element) and an explicit concat axis.
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.comm.all_gather(A_shared, A_shared_dist, direction="horizontal", axis=-1)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.comm.all_gather(B_shared, B_shared_dist, direction="vertical", axis=0)
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)

                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return tvm.IRModule({"main": main})


def test_tilelang_sunmmio_persistent_gemm():
    target = determine_target("Sunmmio", return_object=True)
    block_M = block_N = block_K = 256
    with tvm.target.Target(target):
        mod = matmul_persistent(1024, 1024, 1024, block_M, block_N, block_K, num_stages=2)
    layouts = run_sunmmio_layout_inference(mod, target)

    # gemm(A_shared_dist, B_shared_dist, C_shared) operand layouts:
    assert_layout(layouts, "A_shared_dist", "ZZ", block=(32, 32))  # ASRAM, left operand
    assert_layout(layouts, "B_shared_dist", "ZN", block=(32, 32))  # WSRAM, right operand (transB=False)
    assert_layout(layouts, "C_shared", "ZZ", block=(32, 32))  # RSRAM, accumulator

    # The all_gather staging src into the ZN WSRAM operand stays ZZ — a ZN
    # WSRAM dst legally accepts a ZZ src, so comm must not force it to ZN.
    assert_layout(layouts, "B_shared", "ZZ", block=(32, 32))


# ---------------------------------------------------------------------------
# Test: RSRAM rank-<2 default, and the illegal rank-<2 ASRAM/WSRAM guard added
# alongside the alignment-padded row-major (MakeAlignedRowMajor) work.
# ---------------------------------------------------------------------------


def _rank1_copy_kernel(scope):
    """A rank-1 SRAM buffer touched only by copies, so it falls to the kFree
    scope default (no op-derived layout)."""

    @T.prim_func
    def main(A: T.Tensor((40,), "float16"), B: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            X = T.alloc_shared((40,), "float16", scope=scope)
            T.copy(A, X)
            T.copy(X, B)

    return tvm.IRModule({"main": main})


def test_rsram_rank1_copy_buffer_is_row_major():
    """A rank-1 RSRAM buffer touched only by copies is layout-matched to the
    unpadded DRAM by propagation, overriding the aligned kFree default -- correct,
    since a pure DMA passthrough is never tile-accessed."""
    target = determine_target("Sunmmio", return_object=True)
    layouts = run_sunmmio_layout_inference(_rank1_copy_kernel("shared.rsram"), target)
    assert_layout(layouts, "X", "RowMajor")


def test_rank1_asram_buffer_is_rejected():
    """ASRAM/WSRAM only accept rank-2 tensors; a rank-1 one must fail loudly."""
    target = determine_target("Sunmmio", return_object=True)
    with pytest.raises(Exception, match="rank-2"):
        run_sunmmio_layout_inference(_rank1_copy_kernel("shared.asram"), target)
