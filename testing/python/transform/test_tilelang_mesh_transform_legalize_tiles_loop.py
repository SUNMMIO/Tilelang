from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm import tir
from tvm import IRModule
from tilelang.tileview import make_tileview


# ---------------------------------------------------------
# Helper: 2D tiled parallel kernel
# ---------------------------------------------------------
def dot_mul_tiled_parallel_2d(
    M,
    N,
    block_M,
    block_N,
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

            # Attach tileview metadata to buffers
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (32, 32), (-2, -1)),
                    B_shared: make_tileview(B_shared, (32, 32), (-2, -1)),
                    C_shared: make_tileview(C_shared, (32, 32), (-2, -1)),
                }
            )

            T.clear(C_shared)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            # Tile loop (target of LegalizeTilesLoop)
            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------
# Helper: 3D tiled parallel kernel
# ---------------------------------------------------------
def dot_mul_tiled_parallel_3d(
    B,
    M,
    N,
    block_B,
    block_M,
    block_N,
    dtype="float16",
    accum_dtype="float16",
):
    @T.prim_func
    def main(
        A: T.Tensor((B, M, N), dtype),
        B_: T.Tensor((B, M, N), dtype),
        C: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            T.ceildiv(B, block_B),
            threads=128,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            C_shared = T.alloc_fragment((block_B, block_M, block_N), accum_dtype)

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (32, 32), (-2, -1)),
                    B_shared: make_tileview(B_shared, (32, 32), (-2, -1)),
                    C_shared: make_tileview(C_shared, (32, 32), (-2, -1)),
                }
            )

            T.clear(C_shared)

            T.copy(
                A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                A_shared,
            )
            T.copy(
                B_[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                B_shared,
            )

            for b, i, j in T.Tiles(A_shared, parallel=True):
                C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

            T.copy(
                C_shared,
                C[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
            )

    return main


# ---------------------------------------------------------
# Core test: LegalizeTilesLoop (2D + 3D)
# ---------------------------------------------------------
def test_legalize_tiles_loop_attach_tileview_metadata():
    """
    Test that LegalizeTilesLoop correctly:
    1) reads tileview metadata from buffers
    2) attaches tile annotations to tile loops
    3) sets tile.execution only on inner tile loops
    """

    test_cases = [
        # (prim_func, expected_tile_execution_count)
        (
            dot_mul_tiled_parallel_2d(
                M=512,
                N=1024,
                block_M=256,
                block_N=128,
            ),
            2,  # (i, j)
        ),
        (
            dot_mul_tiled_parallel_3d(
                B=64,
                M=512,
                N=1024,
                block_B=16,
                block_M=256,
                block_N=128,
            ),
            2,  # (b, i, j) -> execution only on i, j
        ),
    ]

    for prim_func, expected_exec_count in test_cases:
        mod = IRModule.from_expr(prim_func.with_attr("global_symbol", "main"))

        # Apply the pass under test
        mod = tl.transform.LegalizeTilesLoop()(mod)

        main_func = mod["main"]

        # -------------------------------------------------
        # Collect loop annotations
        # -------------------------------------------------
        tile_execution_count = 0
        found_tile_metadata = {
            "tile.tile_size": False,
            "tile.dim_map": False,
            "tile.buffer_new_shape": False,
            "tile.loop_parallel": False,  # Was tile_level_loop
            "tile.loop_stage": False,  # New
            "tile.tiled_buffer": False,  # New
        }

        def visit(stmt, found_tile_metadata=found_tile_metadata):
            nonlocal tile_execution_count
            if isinstance(stmt, tir.For):
                ann = stmt.annotations
                if ann is not None:
                    # Check for keys
                    for k in found_tile_metadata:
                        if k in ann:
                            found_tile_metadata[k] = True

                    # Check tile.execution
                    if "tile.execution" in ann:
                        tile_execution_count += 1

        tvm.tir.stmt_functor.post_order_visit(main_func.body, visit)

        # -------------------------------------------------
        # Assertions
        # -------------------------------------------------
        for k, v in found_tile_metadata.items():
            assert v, f"Expected annotation '{k}' not found in tile loops"

        assert tile_execution_count == expected_exec_count, (
            f"Expected {expected_exec_count} tile.execution annotations, got {tile_execution_count}"
        )


if __name__ == "__main__":
    # tilelang.testing.main()
    test_legalize_tiles_loop_attach_tileview_metadata()
