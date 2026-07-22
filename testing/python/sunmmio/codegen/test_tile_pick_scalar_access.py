import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_aligned_row_major, make_row_major, make_zz_layout
from tvm import tir

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")


def _buffer_load_count(func, buffer_name):
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, tir.BufferLoad) and node.buffer.name == buffer_name:
            count += 1

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return count


def _buffer_load_count_in_stores(func, buffer_name):
    count = 0

    def visit_expr(node):
        nonlocal count
        if isinstance(node, tir.BufferLoad) and node.buffer.name == buffer_name:
            count += 1

    def visit_stmt(node):
        if isinstance(node, tir.BufferStore):
            tvm.tir.stmt_functor.post_order_visit(node.value, visit_expr)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit_stmt)
    return count


def _assert_out_shared_store_dtype(func, dtype):
    expected = str(dtype)
    checked = False

    def visit(node):
        nonlocal checked
        if isinstance(node, tir.BufferStore) and node.buffer.name == "out_shared":
            checked = True
            assert str(node.buffer.dtype) == expected
            assert str(node.value.dtype) == expected
            if isinstance(node.value, tir.Add):
                assert str(node.value.a.dtype) == expected
                assert str(node.value.b.dtype) == expected

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert checked, func.script()


def _rsram_scalar_access_layout(shape, dtype, layout_kind, align_bytes=64):
    if layout_kind == "row_major":
        return make_row_major(shape)
    if layout_kind == "aligned_row_major":
        return make_aligned_row_major(shape, dtype, align_bytes=align_bytes)
    if layout_kind == "zz":
        return make_zz_layout(shape, [0, 1], (32, 32))
    raise ValueError(f"unknown layout_kind: {layout_kind}")


def _partitioned_tile_view_lines(src):
    return [line for line in src.splitlines() if "suvm.get_partitioned_tile_view" in line]


def _assert_has_rank2_partitioned_view(src):
    view_lines = _partitioned_tile_view_lines(src)
    assert view_lines, src
    for line in view_lines:
        if "!suvm.tile_view<" not in line:
            continue
        view_shape = line.split("!suvm.tile_view<", 1)[1].split(">", 1)[0]
        if "x" in view_shape:
            return
    raise AssertionError(src)


@target("Sunmmio")
def pick_1d_side_data_kernel(
    n=64,
    out_n=32,
    dtype=T.int32,
):
    shard_policy = T.MeshShardingPolicy()
    values_shape = (n,)
    out_shape = (out_n,)
    values_layout = make_aligned_row_major(values_shape, dtype, align_bytes=1024)
    out_layout = make_aligned_row_major(out_shape, dtype, align_bytes=1024)

    @T.prim_func
    def main(
        values: T.MeshTensor(values_shape, shard_policy, dtype, layout=values_layout),  # type: ignore
        out: T.MeshTensor(out_shape, shard_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel() as cid:
            local_n = values.local_shape[0]
            values_shared = T.alloc_shared((n,), dtype, scope="shared.rsram")
            out_shared = T.alloc_shared((out_n,), dtype)
            values_shared_layout = make_aligned_row_major((n,), dtype, align_bytes=1024)
            out_shared_layout = make_aligned_row_major((out_n,), dtype, align_bytes=1024)
            T.annotate_layout({values_shared: values_shared_layout, out_shared: out_shared_layout})

            T.copy(values[0:local_n], values_shared[0:local_n])

            idx = (cid + 7) % local_n
            picked = T.alloc_var(dtype)

            picked = values_shared[idx]

            for i in T.Tiles([out_n], parallel=True):
                out_shared[i] = picked + T.Cast(dtype, i)

            T.copy(out_shared, out[0:out_n])

    return main


@target("Sunmmio")
def pick_2d_side_data_kernel(
    rows=17,
    cols=19,
    out_n=32,
    dtype=T.int32,
):
    shard_policy = T.MeshShardingPolicy()
    table_shape = (rows, cols)
    out_shape = (out_n,)
    table_layout = make_aligned_row_major(table_shape, dtype, align_bytes=1024)
    out_layout = make_aligned_row_major(out_shape, dtype, align_bytes=1024)

    @T.prim_func
    def main(
        table: T.MeshTensor(table_shape, shard_policy, dtype, layout=table_layout),  # type: ignore
        out: T.MeshTensor(out_shape, shard_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            local_rows, local_cols = table.local_shape
            table_row_shared = T.alloc_shared((cols,), dtype, scope="shared.rsram")
            out_shared = T.alloc_shared((out_n,), dtype)
            row_layout = make_aligned_row_major((cols,), dtype, align_bytes=1024)
            out_shared_layout = make_aligned_row_major((out_n,), dtype, align_bytes=1024)
            T.annotate_layout({table_row_shared: row_layout, out_shared: out_shared_layout})
            picked = T.alloc_var(dtype)

            for row in T.serial(local_rows):
                T.copy(table[row, 0:local_cols], table_row_shared[0:local_cols])
                for col in T.serial(local_cols):
                    picked = table_row_shared[col]
                    for i in T.Tiles([out_n], parallel=True):
                        out_shared[i] = picked + T.Cast(dtype, row + col + i)

            T.copy(out_shared, out[0:out_n])

    return main


@target("Sunmmio")
def pick_3d_side_data_kernel(
    heads=2,
    q_blocks=500,
    k_blocks=500,
    out_tiles=4,
    out_n=32,
    dtype=T.int32,
):
    shard_policy = T.MeshShardingPolicy()
    mask_shape = (heads, q_blocks, k_blocks)
    out_shape = (out_tiles, out_n)
    mask_layout = make_aligned_row_major(mask_shape, dtype, align_bytes=1024)
    out_layout = make_aligned_row_major(out_shape, dtype, align_bytes=1024)

    @T.prim_func
    def main(
        block_sparse_mask: T.MeshTensor(mask_shape, shard_policy, dtype, layout=mask_layout),  # type: ignore
        out: T.MeshTensor(out_shape, shard_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            local_heads, local_q_blocks, local_k_blocks = block_sparse_mask.local_shape
            mask_row_shared = T.alloc_shared((k_blocks,), dtype, scope="shared.rsram")
            out_shared = T.alloc_shared((out_n,), dtype)
            mask_row_layout = make_aligned_row_major((k_blocks,), dtype, align_bytes=1024)
            out_shared_layout = make_aligned_row_major((out_n,), dtype, align_bytes=1024)
            T.annotate_layout({mask_row_shared: mask_row_layout, out_shared: out_shared_layout})
            mask_value = T.alloc_var(dtype, init=0)

            for t in T.serial(out_tiles):
                mask_value = T.Cast(dtype, 0)
                for h in T.serial(local_heads):
                    for q in T.serial(local_q_blocks):
                        T.copy(block_sparse_mask[h, q, :], mask_row_shared)
                        for k in T.serial(local_k_blocks):
                            mask_value += mask_row_shared[k]

                for i in T.Tiles([out_n], parallel=True):
                    out_shared[i] = mask_value + T.Cast(dtype, i)

                T.copy(out_shared, out[t, 0:out_n])

    return main


@target("Sunmmio")
def pick_1d_rsram_side_data_kernel(
    n=256,
    dtype=T.int32,
):
    @T.prim_func
    def main():
        with T.Kernel() as cid:
            side_data = T.alloc_shared((n,), dtype, scope="shared.rsram")
            first_tile_value = T.alloc_var(dtype)
            second_tile_value = T.alloc_var(dtype)

            for i in T.Tiles([n], parallel=True):
                side_data[i] = T.Cast(dtype, i)

            # Keep the access dynamic while targeting known 4096-bit register
            # tiles for int32: 0..127 and 128..255.
            first_tile_idx = cid % 128
            second_tile_idx = 128 + (cid % 128)

            first_tile_value = side_data[first_tile_idx]
            second_tile_value = side_data[second_tile_idx]

            T.evaluate(first_tile_value + second_tile_value)

    return main


@target("Sunmmio")
def pick_rank1_rsram_scalar_access_kernel(
    n=16,
    dtype=T.int32,
    layout_kind="row_major",
    align_bytes=64,
):
    side_shape = (n,)
    side_layout = _rsram_scalar_access_layout(side_shape, dtype, layout_kind, align_bytes)

    @T.prim_func
    def main():
        with T.Kernel() as cid:
            side_data = T.alloc_shared(side_shape, dtype, scope="shared.rsram")
            T.annotate_layout({side_data: side_layout})

            idx = (cid * 5 + 3) % n
            picked = T.alloc_var(dtype)
            picked = side_data[idx]
            T.evaluate(picked)

    return main


@target("Sunmmio")
def pick_rank2_rsram_scalar_access_kernel(
    rows=4,
    cols=32,
    dtype=T.int32,
    layout_kind="row_major",
    align_bytes=64,
):
    table_shape = (rows, cols)
    table_layout = _rsram_scalar_access_layout(table_shape, dtype, layout_kind, align_bytes)

    @T.prim_func
    def main():
        with T.Kernel() as cid:
            table = T.alloc_shared(table_shape, dtype, scope="shared.rsram")
            T.annotate_layout({table: table_layout})

            row = (cid + 1) % rows
            col = (cid * 17 + 3) % cols
            picked = T.alloc_var(dtype)
            picked = table[row, col]
            T.evaluate(picked)

    return main


@target("Sunmmio")
def pick_rank3_rsram_scalar_access_kernel(
    planes=2,
    rows=3,
    cols=500,
    dtype=T.int32,
    layout_kind="aligned_row_major",
    align_bytes=64,
):
    cube_shape = (planes, rows, cols)
    cube_layout = _rsram_scalar_access_layout(cube_shape, dtype, layout_kind, align_bytes)

    @T.prim_func
    def main():
        with T.Kernel() as cid:
            cube = T.alloc_shared(cube_shape, dtype, scope="shared.rsram")
            T.annotate_layout({cube: cube_layout})

            plane = cid % planes
            row = (cid + 1) % rows
            col = (cid * 17 + 3) % cols
            picked = T.alloc_var(dtype)
            picked = cube[plane, row, col]
            T.evaluate(picked)

    return main


@target("Sunmmio")
def set_rank1_rsram_scalar_access_kernel(
    n=16,
    dtype=T.int32,
    layout_kind="row_major",
    align_bytes=64,
    predicated=False,
):
    side_shape = (n,)
    side_layout = _rsram_scalar_access_layout(side_shape, dtype, layout_kind, align_bytes)

    @T.prim_func
    def main():
        with T.Kernel() as cid:
            side_data = T.alloc_shared(side_shape, dtype, scope="shared.rsram")
            T.annotate_layout({side_data: side_layout})

            idx = (cid * 7 + 1) % n
            value = T.Cast(dtype, cid)
            if predicated:
                side_data.vstore([idx], value, predicate=cid % 2 == 0)
            else:
                side_data[idx] = value
            T.evaluate(side_data[idx])

    return main


@target("Sunmmio")
def set_rank2_rsram_scalar_access_kernel(
    rows=4,
    cols=32,
    dtype=T.int32,
    layout_kind="row_major",
    align_bytes=64,
):
    table_shape = (rows, cols)
    table_layout = _rsram_scalar_access_layout(table_shape, dtype, layout_kind, align_bytes)

    @T.prim_func
    def main():
        with T.Kernel() as cid:
            table = T.alloc_shared(table_shape, dtype, scope="shared.rsram")
            T.annotate_layout({table: table_layout})

            row = (cid + 2) % rows
            col = (cid * 11 + 5) % cols
            table[row, col] = T.Cast(dtype, cid)
            T.evaluate(table[row, col])

    return main


@pytest.mark.parametrize(
    "factory,buffer_name,expected_loads",
    [
        (pick_1d_side_data_kernel, "values", 0),
        (pick_2d_side_data_kernel, "table", 0),
        (pick_3d_side_data_kernel, "block_sparse_mask", 0),
    ],
)
def test_pick_scalar_access_frontend_tir_stages_dram_before_scalar_bufferload(factory, buffer_name, expected_loads):
    kernel = factory()
    assert _buffer_load_count_in_stores(kernel, buffer_name) == expected_loads, kernel.script()


@pytest.mark.parametrize(
    "dtype",
    [
        T.int32,
        T.float32,
        T.bfloat16,
    ],
)
def test_pick_3d_out_shared_store_dtype_matches_scalar_expr(dtype):
    kernel = pick_3d_side_data_kernel(heads=1, q_blocks=2, k_blocks=3, out_tiles=1, dtype=dtype)
    _assert_out_shared_store_dtype(kernel, dtype)


def test_pick_1d_rsram_scalar_access_codegen_emits_tile_pick(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        pick_1d_rsram_side_data_kernel(),
        tmp_path,
        mlir_filename="pick_1d_rsram_side_data_suvm.mlir",
        expected_tokens=(
            "suvm.tile.load",
            "suvm.tile.pick",
        ),
    )


@pytest.mark.parametrize(
    "n,expected_tile_view",
    [
        (16, "!suvm.tile_view<16xi32>"),
        (32, "!suvm.tile_view<32xi32>"),
    ],
)
def test_pick_rank1_plain_row_major_rsram_codegen_emits_tile_pick(n, expected_tile_view, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        pick_rank1_rsram_scalar_access_kernel(n=n, layout_kind="row_major"),
        tmp_path,
        mlir_filename=f"pick_rank1_row_major_n{n}_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.pick", expected_tile_view),
        opt_args=("--verify-each",),
    )
    assert "fake_partitioned_tile_view" not in src
    assert "fake_missing_memtensor" not in src


@pytest.mark.parametrize("n", [1, 8, 40])
def test_pick_rank1_plain_row_major_rsram_rejects_illegal_transfer_sizes(n, tmp_path):
    with pytest.raises(Exception, match="cannot infer a legal tile.pick view"):
        validate_sunmmio_codegen_with_npuir_opt(
            pick_rank1_rsram_scalar_access_kernel(n=n, layout_kind="row_major"),
            tmp_path,
            mlir_filename=f"pick_rank1_row_major_n{n}_rejected_suvm.mlir",
            opt_args=("--verify-each",),
        )


def test_pick_rank1_aligned_row_major_rsram_uses_covered_extent(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        pick_rank1_rsram_scalar_access_kernel(n=40, layout_kind="aligned_row_major", align_bytes=64),
        tmp_path,
        mlir_filename="pick_rank1_aligned_row_major_covered_extent_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.pick", "!suvm.tile_view<48xi32>"),
        opt_args=("--verify-each",),
    )
    assert "#suvm.layout<(48), (1)>" in src


def test_pick_rank2_plain_row_major_rsram_codegen_emits_rank2_tile_pick(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        pick_rank2_rsram_scalar_access_kernel(rows=4, cols=32, layout_kind="row_major"),
        tmp_path,
        mlir_filename="pick_rank2_row_major_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.pick", "!suvm.tile_view<4x32xi32>"),
        opt_args=("--verify-each",),
    )
    _assert_has_rank2_partitioned_view(src)


def test_pick_rank2_aligned_row_major_rsram_uses_covered_width_for_500_cols(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        pick_rank2_rsram_scalar_access_kernel(rows=3, cols=500, layout_kind="aligned_row_major", align_bytes=64),
        tmp_path,
        mlir_filename="pick_rank2_aligned_row_major_500_cols_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.pick", "!suvm.tile_view<1x128xi32>"),
        opt_args=("--verify-each",),
    )
    assert "#suvm.layout<(3, 512), (512, 1)>" in src


def test_pick_rank3_aligned_row_major_rsram_projects_trailing_rank2_tile_view(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        pick_rank3_rsram_scalar_access_kernel(planes=2, rows=3, cols=500),
        tmp_path,
        mlir_filename="pick_rank3_aligned_row_major_projected_rank2_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.pick", "!suvm.tile_view<1x128xi32>"),
        opt_args=("--verify-each",),
    )
    assert "#suvm.layout<(2, 3, 512), (1536, 512, 1)>" in src
    assert "tiled_dims = [1, 2]" in src or "tiled_dims = array<i64: 1, 2>" in src
    _assert_has_rank2_partitioned_view(src)


@pytest.mark.parametrize(
    "dtype,expected_tile_view",
    [
        (T.int32, "!suvm.tile_view<4x32xi32>"),
        (T.bfloat16, "!suvm.tile_view<8x32xbf16>"),
    ],
)
def test_pick_rank2_zz_rsram_uses_inner_block_tile_view(dtype, expected_tile_view, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        pick_rank2_rsram_scalar_access_kernel(rows=64, cols=64, dtype=dtype, layout_kind="zz"),
        tmp_path,
        mlir_filename=f"pick_rank2_zz_{dtype}_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.pick", expected_tile_view),
        opt_args=("--verify-each",),
    )
    assert "!suvm.tile_view<64x64" not in src
    _assert_has_rank2_partitioned_view(src)


def test_set_rank1_plain_row_major_rsram_codegen_emits_tile_set(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        set_rank1_rsram_scalar_access_kernel(n=16, layout_kind="row_major"),
        tmp_path,
        mlir_filename="set_rank1_row_major_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.set", "suvm.tile.store", "!suvm.tile_view<16xi32>"),
        opt_args=("--verify-each",),
    )
    assert "suvm.tile.select" not in src


def test_set_rank2_plain_row_major_rsram_codegen_emits_rank2_tile_set(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        set_rank2_rsram_scalar_access_kernel(rows=4, cols=32, layout_kind="row_major"),
        tmp_path,
        mlir_filename="set_rank2_row_major_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.set", "suvm.tile.store", "!suvm.tile_view<4x32xi32>"),
        opt_args=("--verify-each",),
    )
    _assert_has_rank2_partitioned_view(src)


def test_predicated_set_rank1_rsram_scalar_access_uses_control_flow(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        set_rank1_rsram_scalar_access_kernel(n=16, layout_kind="row_major", predicated=True),
        tmp_path,
        mlir_filename="set_rank1_predicated_row_major_suvm.mlir",
        expected_tokens=("scf.if", "suvm.tile.set", "suvm.tile.store"),
        opt_args=("--verify-each",),
    )
    assert "suvm.tile.select" not in src


@pytest.mark.parametrize(
    "factory,mlir_filename",
    [
        (lambda: pick_1d_side_data_kernel(n=256), "pick_1d_side_data_suvm.mlir"),
        (lambda: pick_2d_side_data_kernel(rows=3, cols=256), "pick_2d_side_data_suvm.mlir"),
        (lambda: pick_3d_side_data_kernel(heads=2, q_blocks=5, k_blocks=256, out_tiles=2), "pick_3d_side_data_suvm.mlir"),
    ],
)
def test_pick_scalar_access_codegen_with_explicit_rsram_staging(factory, mlir_filename, tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        factory(),
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=("suvm.copy_async", "suvm.tile.pick"),
        opt_args=("--verify-each",),
    )


def test_pick_3d_predicated_1d_store_preserves_old_lanes(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        pick_3d_side_data_kernel(heads=2, q_blocks=5, k_blocks=256, out_tiles=2),
        tmp_path,
        mlir_filename="pick_3d_predicated_store_suvm.mlir",
        expected_tokens=("suvm.tile.load", "suvm.tile.select", "suvm.tile.store"),
        opt_args=("--verify-each",),
    )


if __name__ == "__main__":
    tilelang.testing.main()
