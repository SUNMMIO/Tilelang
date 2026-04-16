"""Tests for automatic TileView inference in LowerTilesLoop.

Verifies that when no T.annotate_tileview() is used, LowerTilesLoop
derives feasible TileView candidates from each buffer access and picks a
common execution plan with the expected tile_size / execution_domain_axes.
"""

import pytest
import tilelang
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import make_blockwise_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir
from tvm import IRModule


def apply_sunmmio_passes(mod, target):
    """Apply the full SUNMMIO pass pipeline used for DMA copy lowering."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tl.transform.LowerTilesLoop()(mod)
    return mod


# ---------------------------------------------------------
# Helper: collect tile loop annotations from IR
# ---------------------------------------------------------
def collect_tile_annotations(func):
    """Return the scope-root plan annotations and execution-axis count."""
    scope_root = None
    execution_axes = []

    def visit(stmt):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann is None:
                return
            nonlocal scope_root
            if "tile.domain" in ann:
                scope_root = {
                    "tile.domain": ann["tile.domain"],
                    "tile.tile_size": ann["tile.tile_size"],
                    "tile.execution_domain_axes": ann["tile.execution_domain_axes"],
                }
            if "tile.execution_axis" in ann:
                execution_axes.append(int(ann["tile.execution_axis"]))

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    execution_axes.sort()
    return scope_root, execution_axes


def _to_int_list(arr):
    """Convert an Array<PrimExpr> annotation to a Python list of ints."""
    return [int(x) for x in arr]


def assert_scope_plan(mod, expected_tile_size, expected_execution_domain_axes):
    scope_root, execution_axes = collect_tile_annotations(mod["main"])

    assert scope_root is not None
    assert execution_axes == list(range(len(expected_tile_size)))
    assert _to_int_list(scope_root["tile.tile_size"]) == list(expected_tile_size)
    assert _to_int_list(scope_root["tile.execution_domain_axes"]) == list(expected_execution_domain_axes)


# ---------------------------------------------------------
# Test 1: 2D T.Tiles without annotate_tileview
# ---------------------------------------------------------
def test_infer_tileview_2d_no_annotation():
    """Row-major 2D pointwise access should choose the largest full-row tile."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            # No annotate_tileview — should be auto-inferred
            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[2, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_2d_with_layout_annotation():
    """Blockwise 2D pointwise access should choose the densest h x 32 tile."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            # No annotate_tileview — should be auto-inferred
            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[8, 32], expected_execution_domain_axes=[0, 1])


# ---------------------------------------------------------
# Test 2: 1D T.Tiles without annotate_tileview
# ---------------------------------------------------------
def test_infer_tileview_1d_no_annotation():
    """1D row-major fp32 buffers should fill the 4096-bit register."""
    N = 1024

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((N,), "float32")
            B_shared = T.alloc_shared((N,), "float32")

            T.copy(A[0:N], A_shared)

            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[i] * A_shared[i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[128], expected_execution_domain_axes=[0])


def test_infer_rank1_tileview_from_2d_buffer_access():
    """Layout inference may collapse a trailing-dim access to a blockwise rank-1 tile."""
    M, N = 64, 256

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((N,), "float32")

            T.copy(A[0:M, 0:N], A_shared)

            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[0, i] * A_shared[0, i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32], expected_execution_domain_axes=[0])


def test_infer_rank1_tileview_from_2d_buffer_access_with_outer_loop_var():
    """An outer serial loop var stays outside the 1D tile domain after layout inference."""
    M, N = 32, 256

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M, N), "float32")

            T.copy(A[0:M, 0:N], A_shared)

            for row in T.serial(M):
                for j in T.Tiles([N], parallel=True):
                    B_shared[row, j] = A_shared[row, j] * A_shared[row, j]

            T.copy(B_shared, B[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32], expected_execution_domain_axes=[0])


# ---------------------------------------------------------
# Test 3: Mixed-rank (1D + 2D) in same T.Tiles
# ---------------------------------------------------------
def test_infer_tileview_mixed_rank():
    """Layout inference should let mixed-rank accesses use the densest common blockwise plan."""
    M, N = 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M,), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M,), "float32")
            C_shared = T.alloc_shared((M, N), "float32")

            T.copy(A[0:M, 0:N], A_shared)

            # B_shared is 1D but used inside a 2D tile loop
            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] + B_shared[i]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[4, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_swapped_domain_binding():
    """Execution axes should be inferred from access bindings, not loop order."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for j, i in T.Tiles([N, M], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[8, 32], expected_execution_domain_axes=[1, 0])


# ---------------------------------------------------------
# Test 4: Manual annotation overrides inference
# ---------------------------------------------------------
def test_manual_annotation_overrides_inference():
    """When T.annotate_tileview is provided, it overrides inference.

    Without annotation, blockwise inference would produce tile_size=(8, 32).
    With annotation specifying a smaller but legal blockwise tile, we should
    preserve that override.
    """
    from tilelang.tileview import make_tileview

    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (4, 32), (-2, -1)),
                    C_shared: make_tileview(C_shared, (4, 32), (-2, -1)),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * A_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[4, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_3d_swapped_domain_binding():
    """Trailing 2D execution axes should be inferred independently of loop order."""
    K, M, N = 4, 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((K, M, N), "float16")
            B_shared = T.alloc_shared((K, M, N), "float16")
            C_shared = T.alloc_shared((K, M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared[0, :, :])
            T.copy(B[0:M, 0:N], B_shared[0, :, :])

            for k, j, i in T.Tiles([K, N, M], parallel=True):
                C_shared[k, i, j] = A_shared[k, i, j] * B_shared[k, i, j]

            T.copy(C_shared[0, :, :], C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[2, 128], expected_execution_domain_axes=[2, 1])


def test_infer_tileview_1d_fp16_fills_register():
    """1D row-major fp16 buffers should use 256 contiguous elements."""
    N = 1024

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float16"),
        B: T.Tensor((N,), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((N,), "float16")
            B_shared = T.alloc_shared((N,), "float16")

            T.copy(A[0:N], A_shared)

            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[i] + A_shared[i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[256], expected_execution_domain_axes=[0])


def test_infer_tileview_2d_rowmajor_fp32():
    """Row-major fp32 buffers should use full-row tiles limited by 128 elements."""
    M, N = 64, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M, N), "float32")
            C_shared = T.alloc_shared((M, N), "float32")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[1, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_2d_blockwise_fp32():
    """Blockwise fp32 buffers should use the densest legal h x 32 tile."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M, N), "float32")
            C_shared = T.alloc_shared((M, N), "float32")

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[4, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_blockwise_small_height():
    """Blockwise candidates should clamp height by the explicit tile domain."""
    domain_M, buffer_M, N = 4, 32, 128

    @T.prim_func
    def main(
        A: T.Tensor((buffer_M, N), "float16"),
        B: T.Tensor((buffer_M, N), "float16"),
        C: T.Tensor((domain_M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((buffer_M, N), "float16")
            B_shared = T.alloc_shared((buffer_M, N), "float16")
            C_shared = T.alloc_shared((buffer_M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )

            T.copy(A[0:buffer_M, 0:N], A_shared)
            T.copy(B[0:buffer_M, 0:N], B_shared)

            for i, j in T.Tiles([domain_M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared[0:domain_M, 0:N], C[0:domain_M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[4, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_rowmajor_wide_row_uses_single_row_tile():
    """Wide row-major buffers should fall back to a single-row contiguous tile."""
    M, N = 64, 512

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] + B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[1, 256], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_rowmajor_region_height_offset():
    """Aligned row-major height offsets should preserve the best full-row tile."""
    src_M, dst_M, N = 64, 32, 128

    @T.prim_func
    def main(
        A: T.Tensor((src_M, N), "float16"),
        C: T.Tensor((dst_M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, N), "float16")
            C_shared = T.alloc_shared((dst_M, N), "float16")

            T.copy(A[0:src_M, 0:N], A_shared)

            for i, j in T.Tiles([dst_M, N], parallel=True):
                C_shared[i, j] = A_shared[i + 16, j] + A_shared[i + 16, j]

            T.copy(C_shared, C[0:dst_M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[2, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_rowmajor_region_width_offset():
    """Wide row-major regions should use the best single-row contiguous tile."""
    src_M, src_N = 32, 512
    dst_M, dst_N = 32, 256

    @T.prim_func
    def main(
        A: T.Tensor((src_M, src_N), "float16"),
        C: T.Tensor((dst_M, dst_N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, src_N), "float16")
            C_shared = T.alloc_shared((dst_M, dst_N), "float16")

            T.copy(A[0:src_M, 0:src_N], A_shared)

            for i, j in T.Tiles([dst_M, dst_N], parallel=True):
                C_shared[i, j] = A_shared[i, j + 256] + A_shared[i, j + 256]

            T.copy(C_shared, C[0:dst_M, 0:dst_N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[1, 256], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_blockwise_region_height_and_width_offset():
    """Aligned blockwise regions should keep the densest h x 32 tile."""
    src_M, src_N = 64, 64
    dst_M, dst_N = 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((src_M, src_N), "float16"),
        C: T.Tensor((dst_M, dst_N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, src_N), "float16")
            C_shared = T.alloc_shared((dst_M, dst_N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )

            T.copy(A[0:src_M, 0:src_N], A_shared)

            for i, j in T.Tiles([dst_M, dst_N], parallel=True):
                C_shared[i, j] = A_shared[i + 16, j + 32] + A_shared[i + 16, j + 32]

            T.copy(C_shared, C[0:dst_M, 0:dst_N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[8, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_blockwise_region_misaligned_width_offset_rejected():
    """Misaligned blockwise width offsets should reject tileview planning."""
    src_M, src_N = 64, 64
    dst_M, dst_N = 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((src_M, src_N), "float16"),
        C: T.Tensor((dst_M, dst_N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, src_N), "float16")
            C_shared = T.alloc_shared((dst_M, dst_N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )

            T.copy(A[0:src_M, 0:src_N], A_shared)

            for i, j in T.Tiles([dst_M, dst_N], parallel=True):
                C_shared[i, j] = A_shared[i, j + 16] + A_shared[i, j + 16]

            T.copy(C_shared, C[0:dst_M, 0:dst_N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target), pytest.raises(Exception, match="Cannot infer any feasible TileView candidate"):
        apply_sunmmio_passes(mod, target)
