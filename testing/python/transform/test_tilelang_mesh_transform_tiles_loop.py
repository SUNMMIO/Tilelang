import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.tileview import make_tileview
from tvm import tir
from tvm import IRModule

# =========================================================
# Helpers: build kernels
# =========================================================


def dot_mul_tiled_parallel_2d(
    M,
    N,
    block_M,
    block_N,
    tile_size,
    index_map,
    dtype="float16",
    accum_dtype="float16",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, index_map),
                    B_shared: make_tileview(B_shared, tile_size, index_map),
                    C_shared: make_tileview(C_shared, tile_size, index_map),
                }
            )

            T.clear(C_shared)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def dot_mul_tiled_parallel_3d(
    Batch,
    M,
    N,
    block_B,
    block_M,
    block_N,
    tile_size,
    index_map,
    dtype="float16",
    accum_dtype="float16",
):
    @T.prim_func
    def main(
        A: T.Tensor((Batch, M, N), dtype),
        B: T.Tensor((Batch, M, N), dtype),
        C: T.Tensor((Batch, M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            T.ceildiv(Batch, block_B),
            threads=128,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            C_shared = T.alloc_fragment((block_B, block_M, block_N), accum_dtype)

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, index_map),
                    B_shared: make_tileview(B_shared, tile_size, index_map),
                    C_shared: make_tileview(C_shared, tile_size, index_map),
                }
            )

            T.clear(C_shared)
            T.copy(
                A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                A_shared,
            )
            T.copy(
                B[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                B_shared,
            )

            for b, i, j in T.Tiles(A_shared, parallel=True):
                C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

            T.copy(
                C_shared,
                C[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
            )

    return main


# =========================================================
# Core test: TilesLoop
# =========================================================


@pytest.mark.parametrize(
    "prim_func_builder",
    [
        # 2D
        lambda: dot_mul_tiled_parallel_2d(
            M=512,
            N=1024,
            block_M=256,
            block_N=128,
            tile_size=(32, 32),
            index_map=(-2, -1),
        ),
        # 3D
        lambda: dot_mul_tiled_parallel_3d(
            Batch=64,
            M=512,
            N=1024,
            block_B=16,
            block_M=256,
            block_N=128,
            tile_size=(32, 32),
            index_map=(-2, -1),
        ),
    ],
)
def test_tiles_loop_insert_and_index_rewrite(prim_func_builder):
    """
    TilesLoop pass contract test.

    Verifies:
    1) tile.execution loops still represent tile counts
    2) serial(tile_size[0]) and vectorized(tile_size[1]) loops
       are inserted inside tile.execution loop subtrees
    3) index expressions are rewritten as:
         i * tile_size[0] + k
         j * tile_size[1] + l
    """

    tile_size = (32, 32)

    mod = IRModule.from_expr(prim_func_builder().with_attr("global_symbol", "main"))

    # Required pipeline
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)

    main_func = mod["main"]

    # -----------------------------------------------------
    # 1. Collect tile.execution loops
    # -----------------------------------------------------
    tile_exec_loops = []

    def collect_tile_exec(stmt, tile_exec_loops=tile_exec_loops):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann and ann.get("tile.execution", 0) == 1:
                tile_exec_loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_tile_exec)

    # Only i / j loops should be execution loops
    assert len(tile_exec_loops) == 2

    # -----------------------------------------------------
    # 2. Search each tile.execution subtree for k / l loops
    # -----------------------------------------------------
    for exec_loop in tile_exec_loops:
        found_serial = []
        found_vectorized = []

        def visit_subtree(
            stmt,
            found_serial=found_serial,
            found_vectorized=found_vectorized,
        ):
            if isinstance(stmt, tir.For):
                if stmt.kind == tir.ForKind.SERIAL and isinstance(stmt.extent, tir.IntImm) and stmt.extent.value == tile_size[0]:
                    found_serial.append(stmt)

                if stmt.kind == tir.ForKind.VECTORIZED and isinstance(stmt.extent, tir.IntImm) and stmt.extent.value == tile_size[1]:
                    found_vectorized.append(stmt)

        tvm.tir.stmt_functor.post_order_visit(exec_loop.body, visit_subtree)

        assert found_serial, "Expected serial(tile_size[0]) loop inside tile.execution subtree"
        assert found_vectorized, "Expected vectorized(tile_size[1]) loop inside tile.execution subtree"

    # -----------------------------------------------------
    # 3. Pattern check: index rewrite
    # -----------------------------------------------------
    index_exprs = []

    def collect_indices(stmt, index_exprs=index_exprs):
        if isinstance(stmt, tir.BufferStore):
            index_exprs.extend(stmt.indices)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_indices)

    def contains_mul(expr, factor):
        s = str(expr)
        return f"* {factor}" in s or f"*{factor}" in s

    assert any(contains_mul(e, tile_size[0]) for e in index_exprs), "Expected i * tile_size[0] in rewritten indices"

    assert any(contains_mul(e, tile_size[1]) for e in index_exprs), "Expected j * tile_size[1] in rewritten indices"


# =========================================================
# 1D test
# =========================================================


def dot_mul_tiled_parallel_1d(M, block_M, tile_size, index_map, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M,), dtype),
        B: T.Tensor((M,), dtype),
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M,), dtype)
            B_shared = T.alloc_shared((block_M,), dtype)
            C_shared = T.alloc_fragment((block_M,), dtype)

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, index_map),
                    B_shared: make_tileview(B_shared, tile_size, index_map),
                    C_shared: make_tileview(C_shared, tile_size, index_map),
                }
            )

            T.clear(C_shared)
            T.copy(A[bx * block_M], A_shared)
            T.copy(B[bx * block_M], B_shared)

            for i in T.Tiles(A_shared, parallel=True):
                C_shared[i] = A_shared[i] * B_shared[i]

            T.copy(C_shared, C[bx * block_M])

    return main


def test_tiles_loop_1d():
    """
    TilesLoop pass contract test for 1D tile_size.

    Verifies:
    1) A single tile.execution loop is present
    2) A vectorized(tile_size) loop is inserted inside it
    3) Index is rewritten as: i * tile_size + ki
    """
    tile_size = (32,)

    mod = IRModule.from_expr(
        dot_mul_tiled_parallel_1d(
            M=1024,
            block_M=256,
            tile_size=tile_size,
            index_map=(-1,),
        ).with_attr("global_symbol", "main")
    )

    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)

    main_func = mod["main"]

    # 1. Collect tile.execution loops
    tile_exec_loops = []

    def collect_tile_exec(stmt, tile_exec_loops=tile_exec_loops):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann and ann.get("tile.execution", 0) == 1:
                tile_exec_loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_tile_exec)

    assert len(tile_exec_loops) == 1, f"Expected 1 tile.execution loop, got {len(tile_exec_loops)}"

    # 2. Vectorized(tile_size) loop is inside the execution loop
    exec_loop = tile_exec_loops[0]
    found_vectorized = []

    def visit_subtree(stmt, found_vectorized=found_vectorized):
        if (
            isinstance(stmt, tir.For)
            and stmt.kind == tir.ForKind.VECTORIZED
            and isinstance(stmt.extent, tir.IntImm)
            and stmt.extent.value == tile_size[0]
        ):
            found_vectorized.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(exec_loop.body, visit_subtree)
    assert found_vectorized, "Expected vectorized(tile_size) loop inside tile.execution subtree"

    # 3. Index rewrite: i * tile_size + ki
    index_exprs = []

    def collect_indices(stmt, index_exprs=index_exprs):
        if isinstance(stmt, tir.BufferStore):
            index_exprs.extend(stmt.indices)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_indices)

    def contains_mul(expr, factor):
        s = str(expr)
        return f"* {factor}" in s or f"*{factor}" in s

    assert any(contains_mul(e, tile_size[0]) for e in index_exprs), "Expected i * tile_size in rewritten indices"


# =========================================================
# Annotation contract tests: tile.scope_entry, tile.interior
# =========================================================


def _collect_annotations(func):
    """Collect annotation info from all For loops in the function."""
    scope_entry_loops = []
    interior_loops = []

    def visitor(stmt):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann and ann.get("tile.scope_entry", 0) == 1:
                scope_entry_loops.append(stmt)
            if ann and ann.get("tile.interior", 0) == 1:
                interior_loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return scope_entry_loops, interior_loops


def test_annotations_2d():
    """
    For 2D tile_size=(32, 32):
    - Exactly 1 tile.scope_entry loop (the outermost tile.execution)
    - Exactly 2 tile.interior loops (ki axis=0, kj axis=1)
    - kj is vectorized, ki is serial
    """
    mod = IRModule.from_expr(
        dot_mul_tiled_parallel_2d(
            M=512,
            N=1024,
            block_M=256,
            block_N=128,
            tile_size=(32, 32),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    main_func = mod["main"]

    scope_entries, interiors = _collect_annotations(main_func)

    # 1. Exactly one scope entry
    assert len(scope_entries) == 1, f"Expected 1 tile.scope_entry, got {len(scope_entries)}"

    # 2. Exactly two interior loops
    assert len(interiors) == 2, f"Expected 2 tile.interior loops, got {len(interiors)}"

    # 3. Check axes and loop kinds
    axes = {int(loop.annotations["tile.interior_axis"]): loop for loop in interiors}
    assert 0 in axes, "Missing tile.interior_axis=0"
    assert 1 in axes, "Missing tile.interior_axis=1"
    assert axes[0].kind == tir.ForKind.SERIAL, "axis=0 should be serial"
    assert axes[1].kind == tir.ForKind.VECTORIZED, "axis=1 should be vectorized"

    # 4. Interior loops carry tiled_buffer annotation
    for loop in interiors:
        assert "tile.tiled_buffer" in loop.annotations, "tile.interior loop missing tile.tiled_buffer"


def test_annotations_1d():
    """
    For 1D tile_size=(32,):
    - Exactly 1 tile.scope_entry loop
    - Exactly 1 tile.interior loop (ki axis=0, vectorized)
    """
    mod = IRModule.from_expr(
        dot_mul_tiled_parallel_1d(
            M=1024,
            block_M=256,
            tile_size=(32,),
            index_map=(-1,),
        ).with_attr("global_symbol", "main")
    )
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    main_func = mod["main"]

    scope_entries, interiors = _collect_annotations(main_func)

    # 1. Exactly one scope entry
    assert len(scope_entries) == 1, f"Expected 1 tile.scope_entry, got {len(scope_entries)}"

    # 2. Exactly one interior loop
    assert len(interiors) == 1, f"Expected 1 tile.interior loop, got {len(interiors)}"

    # 3. Check axis=0, vectorized
    loop = interiors[0]
    assert int(loop.annotations["tile.interior_axis"]) == 0
    assert loop.kind == tir.ForKind.VECTORIZED

    # 4. Interior loop carries tiled_buffer annotation
    assert "tile.tiled_buffer" in loop.annotations


def test_annotations_3d():
    """
    For 3D buffer with tile_size=(32, 32), index_map=(-2, -1):
    - The batch dimension is NOT tile.execution, so only 1 scope_entry
    - Still 2 interior loops (same as 2D tile)
    """
    mod = IRModule.from_expr(
        dot_mul_tiled_parallel_3d(
            Batch=64,
            M=512,
            N=1024,
            block_B=16,
            block_M=256,
            block_N=128,
            tile_size=(32, 32),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    main_func = mod["main"]

    scope_entries, interiors = _collect_annotations(main_func)

    assert len(scope_entries) == 1, f"Expected 1 tile.scope_entry, got {len(scope_entries)}"
    assert len(interiors) == 2, f"Expected 2 tile.interior loops, got {len(interiors)}"


# =========================================================
# Nested T.Tiles rejection test
# =========================================================


def nested_tiles_different_buffers(M, N, block_M, block_N, tile_size, index_map, dtype="float16"):
    """Kernel with nested T.Tiles from different buffers — must be rejected."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_fragment((block_M, block_N), dtype)

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, index_map),
                    B_shared: make_tileview(B_shared, tile_size, index_map),
                    C_shared: make_tileview(C_shared, tile_size, index_map),
                }
            )

            T.clear(C_shared)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            # Nested T.Tiles from different buffers — should be rejected
            for i, j in T.Tiles(A_shared, parallel=True):
                for k, l in T.Tiles(B_shared, parallel=True):
                    C_shared[i, j] = A_shared[i, j] * B_shared[k, l]

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def test_nested_tiles_different_buffers_rejected():
    """LegalizeTilesLoop must reject nested T.Tiles from different buffers."""
    mod = IRModule.from_expr(
        nested_tiles_different_buffers(
            M=512,
            N=1024,
            block_M=256,
            block_N=128,
            tile_size=(32, 32),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )

    with pytest.raises(Exception, match="Nested T.Tiles from different buffers"):
        tl.transform.LegalizeTilesLoop()(mod)
