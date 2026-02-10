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

            T.annotate_tileview({
                A_shared: make_tileview(A_shared, tile_size, index_map),
                B_shared: make_tileview(B_shared, tile_size, index_map),
                C_shared: make_tileview(C_shared, tile_size, index_map),
            })

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

            T.annotate_tileview({
                A_shared: make_tileview(A_shared, tile_size, index_map),
                B_shared: make_tileview(B_shared, tile_size, index_map),
                C_shared: make_tileview(C_shared, tile_size, index_map),
            })

            T.clear(C_shared)
            T.copy(
                A[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M,
                  bx * block_N:(bx + 1) * block_N],
                A_shared,
            )
            T.copy(
                B[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M,
                  bx * block_N:(bx + 1) * block_N],
                B_shared,
            )

            for b, i, j in T.Tiles(A_shared, parallel=True):
                C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

            T.copy(
                C_shared,
                C[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M,
                  bx * block_N:(bx + 1) * block_N],
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
                if (stmt.kind == tir.ForKind.SERIAL and isinstance(stmt.extent, tir.IntImm) and
                        stmt.extent.value == tile_size[0]):
                    found_serial.append(stmt)

                if (stmt.kind == tir.ForKind.VECTORIZED and isinstance(stmt.extent, tir.IntImm) and
                        stmt.extent.value == tile_size[1]):
                    found_vectorized.append(stmt)

        tvm.tir.stmt_functor.post_order_visit(exec_loop.body, visit_subtree)

        assert found_serial, ("Expected serial(tile_size[0]) loop inside tile.execution subtree")
        assert found_vectorized, (
            "Expected vectorized(tile_size[1]) loop inside tile.execution subtree")

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

    assert any(contains_mul(e, tile_size[0])
               for e in index_exprs), "Expected i * tile_size[0] in rewritten indices"

    assert any(contains_mul(e, tile_size[1])
               for e in index_exprs), "Expected j * tile_size[1] in rewritten indices"
