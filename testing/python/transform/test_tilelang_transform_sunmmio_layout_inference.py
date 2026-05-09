"""Tests for the standalone Sunmmio Layout Inference pass.

Verifies that SunmmioLayoutInference correctly assigns CuteLayout
to SRAM and DRAM buffers via priority-based inference (kStrict > kCommon > kFree).
"""

import tilelang
import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T
from tilelang.layout import make_zz_layout, make_zn_layout
from tilelang.layout.cute_layout import is_same_layout
from tvm.tir import Block
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()


def extract_layout_maps(func):
    """Extract layout_map and global_layout_map from block annotations."""
    layout_map = None
    global_layout_map = None

    def visit(node):
        nonlocal layout_map, global_layout_map
        if isinstance(node, Block):
            if "layout_map" in node.annotations:
                layout_map = node.annotations["layout_map"]
            if "global_layout_map" in node.annotations:
                global_layout_map = node.annotations["global_layout_map"]

    post_order_visit(func.body, visit)
    return layout_map, global_layout_map


def apply_passes_up_to_layout_inference(mod, target):
    """Apply passes up to and including SunmmioLayoutInference."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl.transform.AddWrapperForSingleBufStore()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LegalizeSunmmioDataPath()(mod)
    mod = tl.transform.LayoutReducer()(mod)
    mod = tl.transform.SunmmioLayoutInference()(mod)
    return mod


def get_buf_by_scope(layout_map, scope):
    """Return the first (buf, layout) pair whose buffer has the given scope."""
    for buf in layout_map:
        if buf.scope() == scope:
            return buf, layout_map[buf]
    return None, None


def get_buf_by_name(layout_map, name):
    """Return the first (buf, layout) pair whose buffer has the given name."""
    for buf in layout_map:
        if buf.name == name:
            return buf, layout_map[buf]
    return None, None


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
    with tvm.target.Target(target):
        mod = gemm_kernel_notransB()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, global_layout_map = extract_layout_maps(func)

        assert layout_map is not None, "layout_map annotation not found"
        assert global_layout_map is not None, "global_layout_map annotation not found"

        # Check SRAM buffer names present
        sram_buf_names = {buf.name for buf in layout_map}
        assert "A_shared" in sram_buf_names, f"A_shared not in layout_map, got {sram_buf_names}"
        assert "B_shared" in sram_buf_names, f"B_shared not in layout_map, got {sram_buf_names}"
        assert "C_shared" in sram_buf_names, f"C_shared not in layout_map, got {sram_buf_names}"

        # Check DRAM buffer names present
        dram_buf_names = {buf.name for buf in global_layout_map}
        assert "A" in dram_buf_names, f"A not in global_layout_map, got {dram_buf_names}"
        assert "B" in dram_buf_names, f"B not in global_layout_map, got {dram_buf_names}"
        assert "C" in dram_buf_names, f"C not in global_layout_map, got {dram_buf_names}"

        # A (ASRAM): ZZ with dtype-dependent block shape (32,32) for fp16
        buf_a, layout_a = get_buf_by_name(layout_map, "A_shared")
        expected_a = make_zz_layout(buf_a.shape, block_shape=[32, 32])
        assert is_same_layout(layout_a, expected_a), f"A_shared layout mismatch: got {layout_a}, expected ZZ(32,32)"

        # B (WSRAM): ZN for transB=False (TMM.MN mode)
        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        expected_b = make_zn_layout(buf_b.shape, axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_b), f"B_shared layout mismatch: got {layout_b}, expected ZN(32,32)"

        # C (RSRAM): ZZ with fixed (32,32) block shape
        buf_c, layout_c = get_buf_by_name(layout_map, "C_shared")
        expected_c = make_zz_layout(buf_c.shape, block_shape=[32, 32])
        assert is_same_layout(layout_c, expected_c), f"C_shared layout mismatch: got {layout_c}, expected ZZ(32,32)"


def test_gemm_transB_layout_inference():
    """Verify Gemm with transB=True assigns ZZ (not ZN) to B(WSRAM)."""
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = gemm_kernel_transB()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)

        assert layout_map is not None

        # B (WSRAM): ZZ for transB=True (TMM.MT mode)
        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        expected_b = make_zz_layout(buf_b.shape, block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_b), f"B_shared layout mismatch with transB=True: got {layout_b}, expected ZZ(32,32)"


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
    with tvm.target.Target(target):
        mod = copy_only_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, global_layout_map = extract_layout_maps(func)

        assert layout_map is not None, "layout_map annotation not found"
        assert global_layout_map is not None, "global_layout_map annotation not found"

        # RSRAM buffer should have a layout (Phase 3 default ZZ(32,32))
        buf, layout = get_buf_by_scope(layout_map, "shared.rsram")
        assert buf is not None, "No RSRAM buffer found in layout_map"
        assert layout is not None, "RSRAM buffer has no layout"
        # Phase 3 default for RSRAM: ZZ with (32, 32) block shape
        from tilelang.layout.sunmmio_layouts import make_zz_layout

        expected_zz = make_zz_layout(list(buf.shape), [0, 1], [32, 32])
        assert is_same_layout(layout, expected_zz), f"RSRAM layout should be ZZ(32,32) from Phase 3 defaults, got {layout}"


# ---------------------------------------------------------------------------
# Test: Every SRAM buffer gets a layout (completeness)
# ---------------------------------------------------------------------------


def test_all_sram_buffers_get_layouts():
    """Verify every allocated SRAM buffer ends up with a layout."""
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = gemm_kernel_notransB()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)

        assert layout_map is not None

        # Collect all SRAM buffer scopes
        sram_scopes = {"shared.asram", "shared.wsram", "shared.rsram"}
        found_scopes = set()
        for buf in layout_map:
            if buf.scope() in sram_scopes:
                found_scopes.add(buf.scope())
                assert layout_map[buf] is not None, f"Buffer {buf.name} (scope={buf.scope()}) has no layout"

        # All three SRAM types should be present in a GEMM kernel
        assert found_scopes == sram_scopes, f"Not all SRAM scopes covered: found {found_scopes}, expected {sram_scopes}"


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
    with tvm.target.Target(target):
        mod = copy_only_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        _, global_layout_map = extract_layout_maps(func)

        assert global_layout_map is not None

        # DRAM buffers should have entries
        dram_buf_names = {buf.name for buf in global_layout_map}
        assert "A" in dram_buf_names
        assert "B" in dram_buf_names


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
    with tvm.target.Target(target):
        mod = reduce_blockwise_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)

        assert layout_map is not None

        # C_shared (RSRAM, 2D) should have ZZ from Gemm
        buf_c, layout_c = get_buf_by_name(layout_map, "C_shared")
        assert buf_c is not None, "C_shared not in layout_map"
        expected_c = make_zz_layout(buf_c.shape, block_shape=[32, 32])
        assert is_same_layout(layout_c, expected_c), f"C_shared should be ZZ(32,32), got {layout_c}"

        # C_reduce (RSRAM, 1D) should be row-major (reduce along blockwise axis)
        buf_r, layout_r = get_buf_by_name(layout_map, "C_reduce")
        assert buf_r is not None, "C_reduce not in layout_map"
        from tilelang.layout.sunmmio_layouts import make_row_major

        expected_r = make_row_major(list(buf_r.shape))
        assert is_same_layout(layout_r, expected_r), f"C_reduce should be row-major after blockwise reduce, got {layout_r}"


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
    with tvm.target.Target(target):
        mod = reduce_outer_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)

        assert layout_map is not None

        # A_shared (RSRAM, 3D): annotated with ZZ on inner dims
        buf_a, layout_a = get_buf_by_name(layout_map, "A_shared")
        assert buf_a is not None, "A_shared not in layout_map"
        expected_a = make_zz_layout(buf_a.shape, axes=[1, 2], block_shape=[32, 32])
        assert is_same_layout(layout_a, expected_a), f"A_shared should be ZZ on inner dims (annotated), got {layout_a}"

        # B_shared (RSRAM, 2D): reduce along outer axis preserves ZZ
        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        expected_b = make_zz_layout(buf_b.shape, block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_b), f"B_shared should be ZZ(32,32) after outer-axis reduce, got {layout_b}"


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
    with tvm.target.Target(target):
        mod = reduce_3d_zz_blocked_axis_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        from tilelang.layout.sunmmio_layouts import make_row_major

        expected_b = make_row_major(list(buf_b.shape))
        assert is_same_layout(layout_b, expected_b), f"B_shared should be row-major after reducing blocked axis, got {layout_b}"


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
    with tvm.target.Target(target):
        mod = reduce_3d_zn_outer_axis_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        expected_b = make_zn_layout(list(buf_b.shape), axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_b), f"B_shared should be ZN(32,32) after outer-axis reduce, got {layout_b}"


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
    with tvm.target.Target(target):
        mod = reduce_3d_row_major_outer_axis_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        from tilelang.layout.sunmmio_layouts import make_row_major

        expected_b = make_row_major(list(buf_b.shape))
        assert is_same_layout(layout_b, expected_b), f"B_shared should be row-major after outer-axis reduce, got {layout_b}"


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
    with tvm.target.Target(target):
        mod = reduce_4d_zz_outer_axis_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        expected_b = make_zz_layout(list(buf_b.shape), axes=[1, 2], block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_b), f"B_shared should be 3D ZZ(32,32) on dims [1,2], got {layout_b}"


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
    with tvm.target.Target(target):
        mod = broadcast_zz_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        # C_shared: ZZ from Gemm
        buf_c, layout_c = get_buf_by_name(layout_map, "C_shared")
        assert buf_c is not None
        expected_zz = make_zz_layout(buf_c.shape, block_shape=[32, 32])
        assert is_same_layout(layout_c, expected_zz), f"C_shared should be ZZ(32,32), got {layout_c}"

        # C_bcast: should be ZZ propagated from C_shared via broadcast
        buf_b, layout_b = get_buf_by_name(layout_map, "C_bcast")
        assert buf_b is not None, "C_bcast not in layout_map"
        assert is_same_layout(layout_b, expected_zz), f"C_bcast should be ZZ(32,32) propagated from broadcast src, got {layout_b}"


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
    with tvm.target.Target(target):
        mod = broadcast_zn_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        expected_zn = make_zn_layout(list(buf_b.shape), axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_zn), f"B_shared should be ZN(32,32) from broadcast, got {layout_b}"


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
    with tvm.target.Target(target):
        mod = put_zz_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_p, layout_p = get_buf_by_name(layout_map, "C_put")
        assert buf_p is not None, "C_put not in layout_map"
        expected_zz = make_zz_layout(buf_p.shape, block_shape=[32, 32])
        assert is_same_layout(layout_p, expected_zz), f"C_put should be ZZ(32,32) from put, got {layout_p}"


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
    with tvm.target.Target(target):
        mod = allgather_zz_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_g, layout_g = get_buf_by_name(layout_map, "C_gather")
        assert buf_g is not None, "C_gather not in layout_map"
        expected_zz_3d = make_zz_layout(list(buf_g.shape), axes=[1, 2], block_shape=[32, 32])
        assert is_same_layout(layout_g, expected_zz_3d), f"C_gather should be 3D ZZ(32,32) on dims [1,2], got {layout_g}"


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
    with tvm.target.Target(target):
        mod = allgather_row_major_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        buf_g, layout_g = get_buf_by_name(layout_map, "A_gather")
        assert buf_g is not None, "A_gather not in layout_map"
        from tilelang.layout.sunmmio_layouts import make_row_major

        expected_rm_3d = make_row_major(list(buf_g.shape))
        assert is_same_layout(layout_g, expected_rm_3d), f"A_gather should be 3D row-major, got {layout_g}"


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
    with tvm.target.Target(target):
        mod = gemm_annotate_conflict_asram_kernel()
        with pytest.raises(Exception, match="immutable layout"):
            apply_passes_up_to_layout_inference(mod, target)


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
    with tvm.target.Target(target):
        mod = gemm_annotate_conflict_wsram_kernel()
        with pytest.raises(Exception, match="immutable layout"):
            apply_passes_up_to_layout_inference(mod, target)


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
    with tvm.target.Target(target):
        mod = gemm_annotate_conflict_rsram_kernel()
        with pytest.raises(Exception, match="immutable layout"):
            apply_passes_up_to_layout_inference(mod, target)


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
    with tvm.target.Target(target):
        mod = dram_zn_to_asram_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        layout_map = {}

        def visit(node):
            if isinstance(node, tvm.tir.Block) and "layout_map" in node.annotations:
                lm = node.annotations["layout_map"]
                for var in lm:
                    layout_map[var.name] = lm[var]

        post_order_visit(mod["main"].body, visit)

        assert "A_shared" in layout_map, f"A_shared should have an inferred layout. Got keys: {list(layout_map.keys())}"


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
    with tvm.target.Target(target):
        mod = dram_zn_to_wsram_kernel()
        with pytest.raises(Exception, match="ZZ-like DRAM layout"):
            apply_passes_up_to_layout_inference(mod, target)


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
    with tvm.target.Target(target):
        mod = gemm_correct_annotate_kernel()
        # Should not raise — correct annotations match Gemm's hardware requirement
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        # Verify the annotated layouts are preserved
        buf_a, layout_a = get_buf_by_name(layout_map, "A_shared")
        assert buf_a is not None, "A_shared not in layout_map"
        expected_a = make_zz_layout(list(buf_a.shape), axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_a, expected_a), f"A_shared should keep annotated ZZ, got {layout_a}"

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        expected_b = make_zn_layout(list(buf_b.shape), axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_b), f"B_shared should keep annotated ZN, got {layout_b}"

        buf_c, layout_c = get_buf_by_name(layout_map, "C_shared")
        assert buf_c is not None, "C_shared not in layout_map"
        expected_c = make_zz_layout(list(buf_c.shape), axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_c, expected_c), f"C_shared should keep annotated ZZ, got {layout_c}"


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
    with tvm.target.Target(target):
        mod = copy_reduce_copy_kernel()
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        # A_shared: 2D RSRAM, should have ZZ(32,32) from Phase 3 default
        buf_a, layout_a = get_buf_by_name(layout_map, "A_shared")
        assert buf_a is not None, "A_shared not in layout_map"
        expected_a = make_zz_layout(buf_a.shape, block_shape=[32, 32])
        assert is_same_layout(layout_a, expected_a), f"A_shared should be ZZ(32,32), got {layout_a}"

        # B_shared: 1D RSRAM, Reduce(dim=1) from ZZ(32,32) → blocked dim → row-major
        # This is derived by Reduce at kCommon (Phase 5 BFS), overriding the kFree default
        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None, "B_shared not in layout_map"
        from tilelang.layout.sunmmio_layouts import make_row_major

        expected_b = make_row_major(list(buf_b.shape))
        assert is_same_layout(layout_b, expected_b), f"B_shared should be row-major (Reduce derived from ZZ blocked dim), got {layout_b}"


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
    with tvm.target.Target(target):
        mod = broadcast_mismatched_layouts_kernel()
        # Both immutable annotations are preserved (no error at layout inference)
        mod = apply_passes_up_to_layout_inference(mod, target)

        func = list(mod.functions.values())[0]
        layout_map, _ = extract_layout_maps(func)
        assert layout_map is not None

        # Both buffers keep their original annotations
        buf_a, layout_a = get_buf_by_name(layout_map, "A_shared")
        assert buf_a is not None
        expected_zz = make_zz_layout(buf_a.shape, axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_a, expected_zz), f"A_shared should keep ZZ annotation, got {layout_a}"

        buf_b, layout_b = get_buf_by_name(layout_map, "B_shared")
        assert buf_b is not None
        expected_zn = make_zn_layout(buf_b.shape, axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(layout_b, expected_zn), f"B_shared should keep ZN annotation, got {layout_b}"
