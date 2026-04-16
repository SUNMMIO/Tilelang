import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.tileview import make_tileview
from tilelang.layout import make_blockwise_zz_layout
from tvm import tir
from tvm import IRModule


def apply_tiles_lowering(mod):
    return tl.transform.LowerTilesLoop()(mod)


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

            for i, j in T.Tiles([block_M, block_N], parallel=True):
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

            for b, i, j in T.Tiles([block_B, block_M, block_N], parallel=True):
                C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

            T.copy(
                C_shared,
                C[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
            )

    return main


# =========================================================
# Core test: LowerTilesLoop
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
            tile_size=(2, 128),
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
            tile_size=(2, 128),
            index_map=(-2, -1),
        ),
    ],
)
def test_tiles_loop_insert_and_index_rewrite(prim_func_builder):
    """
    LowerTilesLoop pass contract test.

    Verifies:
    1) execution loops are explicitly marked with tile.execution_axis
    2) serial(tile_size[0]) and vectorized(tile_size[1]) loops
       are inserted inside tile.execution loop subtrees
    3) index expressions are rewritten as:
         i * tile_size[0] + k
         j * tile_size[1] + l
    """

    tile_size = (2, 128)

    mod = IRModule.from_expr(prim_func_builder().with_attr("global_symbol", "main"))

    mod = apply_tiles_lowering(mod)

    main_func = mod["main"]

    # -----------------------------------------------------
    # 1. Collect execution loops
    # -----------------------------------------------------
    tile_exec_loops = []

    def collect_tile_exec(stmt, tile_exec_loops=tile_exec_loops):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann and "tile.execution_axis" in ann:
                tile_exec_loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_tile_exec)

    # Only i / j loops should be execution loops
    assert len(tile_exec_loops) == 2
    assert {int(loop.annotations["tile.execution_axis"]) for loop in tile_exec_loops} == {0, 1}

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

            for i in T.Tiles([block_M], parallel=True):
                C_shared[i] = A_shared[i] * B_shared[i]

            T.copy(C_shared, C[bx * block_M])

    return main


def copy_region_tiled_parallel_2d(tile_size=(8, 32), index_map=(-2, -1), dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), dtype),
        B: T.Tensor((32, 32), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 64), dtype)
            B_shared = T.alloc_shared((32, 32), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                }
            )

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, index_map),
                    B_shared: make_tileview(B_shared, tile_size, index_map),
                }
            )

            T.copy(A[0:64, 0:64], A_shared)

            for i, j in T.Tiles([32, 32], parallel=True):
                B_shared[i, j] = A_shared[i, j + 32]

            T.copy(B_shared, B[0:32, 0:32])

    return main


def dot_mul_tiled_parallel_2d_swapped_domain(M=256, N=128, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M, N), dtype)
            C_shared = T.alloc_shared((M, N), dtype)

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for j, i in T.Tiles([N, M], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    return main


def copy_region_tiled_parallel_2d_misaligned(tile_size=(8, 32), index_map=(-2, -1), dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), dtype),
        B: T.Tensor((32, 32), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 64), dtype)
            B_shared = T.alloc_shared((32, 32), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                }
            )

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, index_map),
                    B_shared: make_tileview(B_shared, tile_size, index_map),
                }
            )

            T.copy(A[0:64, 0:64], A_shared)

            for i, j in T.Tiles([32, 32], parallel=True):
                B_shared[i, j] = A_shared[i, j + 16]

            T.copy(B_shared, B[0:32, 0:32])

    return main


def conflicting_binding_tiled_parallel_2d(dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), dtype),
        B: T.Tensor((32, 32), dtype),
        C: T.Tensor((32, 32), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((32, 32), dtype)
            B_shared = T.alloc_shared((32, 32), dtype)
            C_shared = T.alloc_shared((32, 32), dtype)

            T.copy(A[0:32, 0:32], A_shared)
            T.copy(B[0:32, 0:32], B_shared)

            for j, i in T.Tiles([32, 32], parallel=True):
                C_shared[i, j] = A_shared[i, j] + B_shared[j, i]

            T.copy(C_shared, C[0:32, 0:32])

    return main


def nested_tiles_different_buffers(M, N, block_M, block_N, tile_size, index_map, dtype="float16"):
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

            for i, j in T.Tiles([block_M, block_N], parallel=True):
                for k, l in T.Tiles([block_M, block_N], parallel=True):
                    C_shared[i, j] = A_shared[i, j] * B_shared[k, l]

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def test_tiles_loop_1d():
    """
    LowerTilesLoop pass contract test for 1D tile_size.

    Verifies:
    1) A single tile.execution_axis loop is present
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

    mod = apply_tiles_lowering(mod)

    main_func = mod["main"]

    # 1. Collect execution loops
    tile_exec_loops = []

    def collect_tile_exec(stmt, tile_exec_loops=tile_exec_loops):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann and "tile.execution_axis" in ann:
                tile_exec_loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(main_func.body, collect_tile_exec)

    assert len(tile_exec_loops) == 1, f"Expected 1 tile execution loop, got {len(tile_exec_loops)}"
    assert int(tile_exec_loops[0].annotations["tile.execution_axis"]) == 0

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


def test_tiles_loop_region_offset_rewrite():
    """Explicit tile domains should allow offset regions without a primary buffer."""
    mod = IRModule.from_expr(copy_region_tiled_parallel_2d().with_attr("global_symbol", "main"))

    mod = apply_tiles_lowering(mod)

    index_exprs = []

    def collect_indices(stmt, index_exprs=index_exprs):
        if isinstance(stmt, tir.BufferStore):
            index_exprs.extend(stmt.indices)
        if isinstance(stmt, tir.BufferLoad):
            index_exprs.extend(stmt.indices)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, collect_indices)

    assert any(("+ 1" in str(expr) or "+1" in str(expr)) and ("* 32" in str(expr) or "*32" in str(expr)) for expr in index_exprs), (
        "Expected aligned region offset to be normalized into an explicit tile offset"
    )
    assert any("* 32" in str(expr) or "*32" in str(expr) for expr in index_exprs), "Expected tile-size rewrite in region indices"


def test_tiles_loop_swapped_domain_binding_rewrite():
    mod = IRModule.from_expr(dot_mul_tiled_parallel_2d_swapped_domain().with_attr("global_symbol", "main"))

    mod = apply_tiles_lowering(mod)

    c_store_indices = []

    def collect_c_indices(stmt, c_store_indices=c_store_indices):
        if isinstance(stmt, tir.BufferStore) and stmt.buffer.name == "C_shared":
            c_store_indices.append(stmt.indices)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, collect_c_indices)

    assert c_store_indices, "Expected a rewritten store to C_shared"
    assert any(
        ("i * 2" in str(indices[0]) or "i*2" in str(indices[0])) and ("j * 128" in str(indices[1]) or "j*128" in str(indices[1]))
        for indices in c_store_indices
    ), "Expected the rewritten indices to follow the inferred i/j axis binding rather than lexical loop order"


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


def _collect_root_and_exec_loops(func):
    scope_roots = []
    exec_loops = []
    legacy_attrs = []

    def visitor(stmt):
        if not isinstance(stmt, tir.For):
            return
        ann = stmt.annotations
        if ann and "tile.domain" in ann:
            scope_roots.append(stmt)
        if ann and "tile.execution_axis" in ann:
            exec_loops.append(stmt)
        if ann:
            for legacy in ["tile.loop_stage", "tile.execution", "tile.dim_map", "tile.loop_parallel"]:
                if legacy in ann:
                    legacy_attrs.append((legacy, stmt))

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return scope_roots, exec_loops, legacy_attrs


@pytest.mark.parametrize(
    "prim_func_builder, expected_domain_axes",
    [
        (
            lambda: dot_mul_tiled_parallel_2d(
                M=512,
                N=1024,
                block_M=256,
                block_N=128,
                tile_size=(2, 128),
                index_map=(-2, -1),
            ),
            [0, 1],
        ),
        (
            lambda: dot_mul_tiled_parallel_3d(
                Batch=64,
                M=512,
                N=1024,
                block_B=16,
                block_M=256,
                block_N=128,
                tile_size=(2, 128),
                index_map=(-2, -1),
            ),
            [1, 2],
        ),
    ],
)
def test_final_tile_attrs_are_explicit(prim_func_builder, expected_domain_axes):
    mod = apply_tiles_lowering(IRModule.from_expr(prim_func_builder().with_attr("global_symbol", "main")))

    scope_roots, exec_loops, legacy_attrs = _collect_root_and_exec_loops(mod["main"])

    assert len(scope_roots) == 1, f"Expected 1 tile.domain root, got {len(scope_roots)}"
    root = scope_roots[0]
    root_ann = root.annotations
    assert [int(x) for x in root_ann["tile.tile_size"]] == [2, 128]
    assert [int(x) for x in root_ann["tile.execution_domain_axes"]] == expected_domain_axes
    assert {int(loop.annotations["tile.execution_axis"]) for loop in exec_loops} == {0, 1}
    assert not legacy_attrs, f"Expected no legacy tile attrs after LowerTilesLoop, found {legacy_attrs}"


def test_annotations_2d():
    """
    For 2D tile_size=(2, 128):
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
            tile_size=(2, 128),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )
    mod = apply_tiles_lowering(mod)
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
    mod = apply_tiles_lowering(mod)
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


def test_annotations_3d():
    """
    For 3D buffer with tile_size=(2, 128), index_map=(-2, -1):
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
            tile_size=(2, 128),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )
    mod = apply_tiles_lowering(mod)
    main_func = mod["main"]

    scope_entries, interiors = _collect_annotations(main_func)

    assert len(scope_entries) == 1, f"Expected 1 tile.scope_entry, got {len(scope_entries)}"
    assert len(interiors) == 2, f"Expected 2 tile.interior loops, got {len(interiors)}"


def test_lower_tiles_loop_region_offset_rejects_misaligned_offset():
    mod = IRModule.from_expr(copy_region_tiled_parallel_2d_misaligned().with_attr("global_symbol", "main"))

    with pytest.raises(Exception, match="not divisible by tile size"):
        apply_tiles_lowering(mod)


def test_lower_tiles_loop_conflicting_binding_rejected():
    mod = IRModule.from_expr(conflicting_binding_tiled_parallel_2d().with_attr("global_symbol", "main"))

    with pytest.raises(Exception, match="do not share a common axis binding and tile shape"):
        apply_tiles_lowering(mod)


def test_lower_tiles_loop_nested_scopes_rejected():
    mod = IRModule.from_expr(
        nested_tiles_different_buffers(
            M=512,
            N=1024,
            block_M=256,
            block_N=128,
            tile_size=(2, 128),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )

    with pytest.raises(Exception, match="Nested T.Tiles scopes are not supported"):
        apply_tiles_lowering(mod)
