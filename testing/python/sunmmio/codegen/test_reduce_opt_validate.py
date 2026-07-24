import os
import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_row_major, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


REDUCE_IN_TILE_CASES = [
    ((32, 1024), 1, False),
    ((32, 1024), 0, False),
    ((1024, 32), 1, True),
    ((256, 128), 1, True),
    ((256, 128), 0, False),
    ((256, 128), 1, False),
    ((64, 256, 128), 0, True),
    ((32, 256, 128), 2, True),
    ((32, 256, 128), 1, False),
]

LOOSE_OPT_ARGS = ("--verify-each",)


def validate_sunmmio_codegen_loose(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=LOOSE_OPT_ARGS,
    )


def _dram_input_layout(shape):
    if len(shape) >= 2:
        return make_zz_layout(shape, [len(shape) - 2, len(shape) - 1], (32, 32))
    return make_row_major(shape)


def _dram_reduce_output_layout(shape):
    return make_row_major(shape)


def _valid_extent(tile_index, block, total):
    start = tile_index * block
    remaining = total - start
    return T.min(block, remaining)


@target("Sunmmio")
def reduce_kernel_builder(shape, reduce_axis, dtype="bfloat16", clear=True):
    shape = tuple(shape)
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]
    out_shape = tuple(out_shape)

    shard_policy = T.MeshShardingPolicy()
    input_layout = _dram_input_layout(shape)
    output_layout = _dram_reduce_output_layout(out_shape)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            if len(shape) == 3:
                for bb in T.serial(shape[0]):
                    T.copy(A[bb, :, :], A_shared[bb, :, :])
            else:
                T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis, clear=clear)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_tail_region_kernel(
    m=1000,
    n=2000,
    block_m=256,
    block_n=96,
    dtype="bfloat16",
):
    shard_policy = T.MeshShardingPolicy()
    input_shape = (block_m, block_n)
    output_shape = (block_m,)
    input_layout = make_zz_layout(input_shape, [0, 1], (32, 32))
    output_layout = make_row_major(output_shape)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(input_shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(output_shape, shard_policy, dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared((block_m,), dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.fill(Out_shared, -T.infinity(dtype))

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
                    T.reduce_max(
                        A_shared[
                            0 : _valid_extent(by, block_m, m),
                            0 : _valid_extent(bx, block_n, n),
                        ],
                        Out_shared[0 : _valid_extent(by, block_m, m)],
                        dim=1,
                        clear=False,
                    )

            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_tiled_test(
    b=64,
    m=512,
    n=1024,
    block_b=32,
    block_m=256,
    block_n=128,
    reduce_axis=1,
    dtype="bfloat16",
    clear=False,
):
    out_shape_full = (b, m) if reduce_axis == 2 else (b, n) if reduce_axis == 1 else (m, n)
    out_shape_block = (block_b, block_m) if reduce_axis == 2 else (block_b, block_n) if reduce_axis == 1 else (block_m, block_n)
    input_shape = (b, m, n)

    shard_policy = T.MeshShardingPolicy()
    input_layout = make_zz_layout(input_shape, [1, 2], (32, 32))
    output_layout = _dram_reduce_output_layout(out_shape_full)
    grid_b = T.ceildiv(b, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(input_shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(out_shape_full, shard_policy, dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape_block, dtype, scope="shared.rsram")

            if reduce_axis == 2:
                for bz in T.serial(grid_b):
                    for by in T.serial(grid_m):
                        if clear:
                            T.fill(Out_shared, 0)
                        else:
                            T.copy(
                                Out[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                ],
                                Out_shared,
                            )
                        for bx in T.serial(grid_n):
                            for bb in T.serial(block_b):
                                T.copy(
                                    A[
                                        bz * block_b + bb,
                                        by * block_m : (by + 1) * block_m,
                                        bx * block_n : (bx + 1) * block_n,
                                    ],
                                    A_shared[bb, :, :],
                                )
                            T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
                        T.copy(
                            Out_shared,
                            Out[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                            ],
                        )
            elif reduce_axis == 1:
                for bz in T.serial(grid_b):
                    for bx in T.serial(grid_n):
                        if clear:
                            T.fill(Out_shared, 0)
                        else:
                            T.copy(
                                Out[
                                    bz * block_b : (bz + 1) * block_b,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                Out_shared,
                            )
                        for by in T.serial(grid_m):
                            for bb in T.serial(block_b):
                                T.copy(
                                    A[
                                        bz * block_b + bb,
                                        by * block_m : (by + 1) * block_m,
                                        bx * block_n : (bx + 1) * block_n,
                                    ],
                                    A_shared[bb, :, :],
                                )
                            T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
                        T.copy(
                            Out_shared,
                            Out[
                                bz * block_b : (bz + 1) * block_b,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )
            else:
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        if clear:
                            T.fill(Out_shared, 0)
                        else:
                            T.copy(
                                Out[
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                Out_shared,
                            )
                        for bz in T.serial(grid_b):
                            for bb in T.serial(block_b):
                                T.copy(
                                    A[
                                        bz * block_b + bb,
                                        by * block_m : (by + 1) * block_m,
                                        bx * block_n : (bx + 1) * block_n,
                                    ],
                                    A_shared[bb, :, :],
                                )
                            T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
                        T.copy(
                            Out_shared,
                            Out[
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )

    return main


@pytest.mark.parametrize("shape,reduce_axis,clear", REDUCE_IN_TILE_CASES)
def test_reduce_generic_in_tile_codegen_generates_expected_ops(tmp_path, shape, reduce_axis, clear):
    shape_label = "x".join(str(dim) for dim in shape)
    expected_tokens = ("suvm.copy_async",)
    if reduce_axis != 0:
        expected_tokens = (*expected_tokens, "suvm.tile.reduce")
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder(shape, reduce_axis, clear=clear),
        tmp_path,
        mlir_filename=f"reduce_shape_{shape_label}_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=expected_tokens,
    )
    if reduce_axis == 0:
        assert_source_contains(src, ("suvm.tile.addf", "suvm.tile.store"))
    else:
        assert_source_contains(src, ("suvm.tile.reduce", "sum"))
    if shape == (32, 256, 128) and reduce_axis == 2:
        aligned_view_lines = [
            line
            for line in src.splitlines()
            if "suvm.get_partitioned_tile_view" in line
            and "!suvm.memtensor<32x256xbf16" in line
            and "tiled_dims = [1]" in line
            and "-> !suvm.tile_view<32xbf16>" in line
        ]
        assert aligned_view_lines
        assert any("indices = [%arg2," in line or "(%3, %arg2," in line for line in aligned_view_lines)


@pytest.mark.parametrize("reduce_axis,clear", [(1, False), (2, True)])
def test_reduce_tiled_in_tile_codegen_generates_expected_ops(tmp_path, reduce_axis, clear):
    src = validate_sunmmio_codegen_loose(
        reduce_tiled_test(reduce_axis=reduce_axis, clear=clear),
        tmp_path,
        mlir_filename=f"reduce_tiled_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.reduce"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "sum"))


def test_reduce_small_1d_result_uses_aligned_store_bridge(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder((32, 64, 256), 2, clear=True),
        tmp_path,
        mlir_filename="reduce_small_1d_result_aligned_store_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "suvm.tile.insert_slice", "suvm.tile.store"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "!suvm.tile<8x1xbf16>", "!suvm.tile<32x1xbf16>"))
    assert "fake_tile_insert_slice" not in src
    assert "suvm.tile.store" in src
    assert "fake_tile_store" not in src
    assert "!suvm.tile_view<32x1xbf16>" not in src
    assert "fake_partitioned_tile_view" not in src


def test_reduce_dim0_multisegment_store_uses_partition_coordinate(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder((4, 128), 0, dtype="float32", clear=True),
        tmp_path,
        mlir_filename="reduce_dim0_multisegment_store_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "suvm.tile.store"),
    )

    div_results = {
        match.group(1)
        for line in src.splitlines()
        if (match := re.match(r"\s*(%[\w.]+) = arith\.divsi\b", line))
    }
    destination_views = [
        line
        for line in src.splitlines()
        if "suvm.get_partitioned_tile_view" in line
        and "!suvm.memtensor<128xf32" in line
        and "-> !suvm.tile_view<32xf32>" in line
    ]
    assert len(destination_views) == 1, destination_views

    index_match = re.search(r"indices = \[(%[\w.]+)\]", destination_views[0])
    assert index_match, destination_views[0]
    assert index_match.group(1) in div_results, (
        "The reduction destination must use the tile partition coordinate "
        "(element offset / tile extent), not the raw element offset.\n"
        f"destination view: {destination_views[0]}"
    )


def test_reduce_tail_region_codegen_uses_identity_select(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_tail_region_kernel(),
        tmp_path,
        mlir_filename="reduce_tail_region_suvm.mlir",
        expected_tokens=(
            "suvm.tile.reduce  max",
            "suvm.tile.select",
            "0xFF80",
            "suvm.tile.insert_slice",
            "suvm.tile.store",
        ),
    )
    assert "fake_missing" not in src
    assert_source_contains(src, ("!suvm.tile<8x32xbf16>", "!suvm.tile<8x1xbf16>", "!suvm.tile<32xbf16>"))


if __name__ == "__main__":
    tilelang.testing.main()
