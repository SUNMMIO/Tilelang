"""Tests for automatic TileView inference in LegalizeTilesLoop.

Verifies that when no T.annotate_tileview() is used, LegalizeTilesLoop
auto-infers TileView from buffer shape and layout_map, and that the
inferred tile_size / dim_map values are correct.
"""

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
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    return mod


# ---------------------------------------------------------
# Helper: collect tile loop annotations from IR
# ---------------------------------------------------------
def collect_tile_annotations(func):
    """Walk IR and collect tile loop annotation values.

    Returns (loops, exec_count) where:
      loops: list of dicts, one per tile-level For loop (outermost first),
             each containing the annotation values.
      exec_count: number of loops marked tile.execution.
    """
    tile_execution_count = 0
    loops = []

    def visit(stmt):
        nonlocal tile_execution_count
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann is not None and "tile.loop_stage" in ann:
                info = {}
                for k in [
                    "tile.tile_size",
                    "tile.dim_map",
                    "tile.buffer_new_shape",
                    "tile.loop_parallel",
                    "tile.loop_stage",
                    "tile.tiled_buffer",
                ]:
                    info[k] = ann.get(k, None)
                info["tile.execution"] = "tile.execution" in ann
                if info["tile.execution"]:
                    tile_execution_count += 1
                loops.append(info)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return loops, tile_execution_count


def _to_int_list(arr):
    """Convert an Array<PrimExpr> annotation to a Python list of ints."""
    return [int(x) for x in arr]


# ---------------------------------------------------------
# Test 1: 2D T.Tiles without annotate_tileview
# ---------------------------------------------------------
def test_infer_tileview_2d_no_annotation():
    """2D buffers without layout -> inferred TileView (1, 32)."""
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
            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    # Omit layout inference to test rowmajor case
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)

    loops, exec_count = collect_tile_annotations(mod["main"])

    # Should have 2 tile loops (i, j)
    assert len(loops) == 4, f"Expected 4 tile loops, got {len(loops)}"
    assert exec_count == 2, f"Expected 2 tile.execution, got {exec_count}"

    # Verify inferred tile_size = (1, 32) — no layout, so row-major
    assert _to_int_list(loops[-1]["tile.tile_size"]) == [1, 32]
    # dim_map should be (-2, -1)
    assert _to_int_list(loops[-1]["tile.dim_map"]) == [-2, -1]
    # tiled shape: [num_tiles..., tile_shape...] = [256/1, 128/32, 1, 32]
    assert _to_int_list(loops[-1]["tile.buffer_new_shape"]) == [256, 4, 1, 32]


def test_infer_tileview_2d_with_layout_annotation():
    """2D buffers without layout -> inferred TileView (1, 32)."""
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
            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    loops, exec_count = collect_tile_annotations(mod["main"])

    # Should have 2 tile loops (i, j)
    assert len(loops) == 4, f"Expected 4 tile loops, got {len(loops)}"
    assert exec_count == 2, f"Expected 2 tile.execution, got {exec_count}"

    # Verify inferred tile_size = (32, 32) — blockwise layout
    assert _to_int_list(loops[-1]["tile.tile_size"]) == [32, 32]
    # dim_map should be (-2, -1)
    assert _to_int_list(loops[-1]["tile.dim_map"]) == [-2, -1]
    # tiled shape: [num_tiles..., tile_shape...] = [256/32, 128/32, 32, 32]
    assert _to_int_list(loops[-1]["tile.buffer_new_shape"]) == [8, 4, 32, 32]


# ---------------------------------------------------------
# Test 2: 1D T.Tiles without annotate_tileview
# ---------------------------------------------------------
def test_infer_tileview_1d_no_annotation():
    """1D buffers -> inferred TileView (32,)."""
    N = 256

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((N,), "float32")
            B_shared = T.alloc_shared((N,), "float32")

            T.copy(A[0:N], A_shared)

            for i in T.Tiles(A_shared, parallel=True):
                B_shared[i] = A_shared[i] * A_shared[i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    loops, exec_count = collect_tile_annotations(mod["main"])

    assert len(loops) == 2, f"Expected 2 tile loop, got {len(loops)}"
    assert exec_count == 1, f"Expected 1 tile.execution, got {exec_count}"

    # Verify inferred tile_size = (32,)
    assert _to_int_list(loops[-1]["tile.tile_size"]) == [32]
    # dim_map = (-1,)
    assert _to_int_list(loops[-1]["tile.dim_map"]) == [-1]
    # tiled shape: [num_tiles..., tile_shape...] = [256/32, 32]
    assert _to_int_list(loops[-1]["tile.buffer_new_shape"]) == [8, 32]


# ---------------------------------------------------------
# Test 3: Mixed-rank (1D + 2D) in same T.Tiles
# ---------------------------------------------------------
def test_infer_tileview_mixed_rank():
    """1D and 2D buffers coexisting — primary is 2D, 1D gets compatible tile."""
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
            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = A_shared[i, j] + B_shared[i]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    loops, exec_count = collect_tile_annotations(mod["main"])

    # Primary buffer is 2D -> 2 tile loops, both execution
    assert len(loops) == 4, f"Expected 4 tile loops, got {len(loops)}"
    assert exec_count == 2, f"Expected 2 tile.execution, got {exec_count}"

    # Loop structure follows primary (2D) buffer: tile_size = (32, 32)
    assert _to_int_list(loops[-1]["tile.tile_size"]) == [32, 32]
    assert _to_int_list(loops[-1]["tile.dim_map"]) == [-2, -1]
    # tiled shape: [num_tiles..., tile_shape...] = [128/32, 64/32, 32, 32]
    assert _to_int_list(loops[-1]["tile.buffer_new_shape"]) == [4, 2, 32, 32]


# ---------------------------------------------------------
# Test 4: Manual annotation overrides inference
# ---------------------------------------------------------
def test_manual_annotation_overrides_inference():
    """When T.annotate_tileview is provided, it overrides inference.

    Without annotation, inference would produce tile_size=(1, 32).
    With annotation specifying (32, 32), we should see that instead.
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

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (32, 32), (-2, -1)),
                    C_shared: make_tileview(C_shared, (32, 32), (-2, -1)),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)

            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = A_shared[i, j] * A_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    loops, exec_count = collect_tile_annotations(mod["main"])

    assert len(loops) == 4, f"Expected 4 tile loops, got {len(loops)}"
    assert exec_count == 2, f"Expected 2 tile.execution, got {exec_count}"

    # Must be (32, 32) from manual annotation, NOT (1, 32) from inference
    assert _to_int_list(loops[-1]["tile.tile_size"]) == [32, 32]
    assert _to_int_list(loops[-1]["tile.dim_map"]) == [-2, -1]
    # tiled shape: [num_tiles..., tile_shape...] = [256/32, 128/32, 32, 32]
    assert _to_int_list(loops[-1]["tile.buffer_new_shape"]) == [8, 4, 32, 32]
