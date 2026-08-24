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
from tilelang.tileview import make_tileview


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
STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")


def validate_sunmmio_codegen_loose(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=LOOSE_OPT_ARGS,
    )


def validate_sunmmio_codegen_strict(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=STRICT_OPT_ARGS,
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
def reduce_kernel_builder(
    shape,
    reduce_axis,
    dtype="bfloat16",
    clear=True,
    tile_size=None,
    out_dtype=None,
):
    shape = tuple(shape)
    out_dtype = out_dtype or dtype
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]
    out_shape = tuple(out_shape)

    shard_policy = T.placement.replicated()
    input_layout = _dram_input_layout(shape)
    output_layout = _dram_reduce_output_layout(out_shape)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, out_dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, out_dtype, scope="shared.rsram")

            if tile_size is not None:
                T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, (-2, -1))})
            if len(shape) == 3:
                for bb in T.serial(shape[0]):
                    T.copy(A[bb, :, :], A_shared[bb, :, :])
            else:
                T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis, clear=clear)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_keepdim_kernel_builder(shape=(32, 128), reduce_axis=1, dtype="float32"):
    out_shape = list(shape)
    out_shape[reduce_axis] = 1
    out_shape = tuple(out_shape)
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)

    return main


@target("Sunmmio")
def reduce_dynamic_region_kernel_builder(max_k=128, dtype="float32"):
    shape = (32, max_k)
    out_shape = (32,)
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
        k: T.int32,
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared[:, 0:k], Out_shared, dim=1)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_manual_dst_tileview_kernel_builder(dst_tile_size):
    shape = (32, 128)
    out_shape = (32,)
    dtype = "float32"
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, (dst_tile_size,), (-1,))})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=1)
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
    shard_policy = T.placement.replicated()
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

    shard_policy = T.placement.replicated()
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
    kernel = (
        reduce_tiled_test(
            b=4,
            m=512,
            n=128,
            block_b=4,
            block_m=512,
            block_n=128,
            reduce_axis=reduce_axis,
            clear=clear,
        )
        if reduce_axis == 2
        else reduce_tiled_test(reduce_axis=reduce_axis, clear=clear)
    )
    validate = validate_sunmmio_codegen_strict if reduce_axis == 2 else validate_sunmmio_codegen_loose
    src = validate(
        kernel,
        tmp_path,
        mlir_filename=f"reduce_tiled_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.reduce"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "sum"))
    if reduce_axis == 2:
        assert_source_contains(
            src,
            (
                "!suvm.tile<32x32xbf16>",
                "suvm.tile.squeeze",
                "suvm.tile.unsqueeze",
                "!suvm.tile<1x32xbf16>",
            ),
        )


def test_reduce_layout_bounded_zz_block_lowers_to_llvm(tmp_path):
    src = validate_sunmmio_codegen_strict(
        reduce_kernel_builder((512, 128), 1, dtype="bfloat16", clear=True),
        tmp_path,
        mlir_filename="reduce_layout_bounded_zz_block_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "!suvm.tile<32x32xbf16>"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "!suvm.tile<32x32xbf16>"))


@pytest.mark.parametrize("reduce_axis", [0, 1])
def test_reduce_bf16_source_uses_fp32_semantic_accumulator(tmp_path, reduce_axis):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder(
            (32, 128),
            reduce_axis,
            dtype="bfloat16",
            out_dtype="float32",
        ),
        tmp_path,
        mlir_filename=f"reduce_bf16_to_fp32_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.tile.cast", "suvm.tile.reduce", "xf32>"),
    )
    cast = re.search(
        r"suvm\.tile\.cast .*!suvm\.tile<(\d+x\d+)xbf16> -> !suvm\.tile<\1xf32>",
        src,
    )
    assert cast, src
    assert re.search(
        rf"suvm\.tile\.reduce\s+sum, .*: !suvm\.tile<{cast.group(1)}xf32>",
        src,
    )


@pytest.mark.parametrize("reduce_axis", [0, 1])
def test_reduce_keepdim_preserves_unit_destination_axis(tmp_path, reduce_axis):
    src = validate_sunmmio_codegen_loose(
        reduce_keepdim_kernel_builder(reduce_axis=reduce_axis),
        tmp_path,
        mlir_filename=f"reduce_keepdim_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "xf32>"),
    )
    expected_out_shape = "1x128" if reduce_axis == 0 else "32x1"
    assert f"!suvm.memtensor<{expected_out_shape}xf32" in src
    expected_result_shape = r"1x\d+" if reduce_axis == 0 else r"\d+x1"
    assert re.search(
        rf"suvm\.tile\.reduce\s+sum, .*\{{axis = {reduce_axis} : i64\}}.*"
        rf"-> !suvm\.tile<{expected_result_shape}xf32>",
        src,
    )
    surviving_axis = 1 - reduce_axis
    assert re.search(
        rf"get_partitioned_tile_view .* tiled_dims = \[{surviving_axis}\].*"
        rf"!suvm\.memtensor<{expected_out_shape}xf32",
        src,
    )


def test_reduce_dynamic_k_is_preserved_in_raw_suvm(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_dynamic_region_kernel_builder(),
        tmp_path,
        mlir_filename="reduce_dynamic_k_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "scf.for"),
    )
    assert re.search(r"func\.func .*\bi32\b", src)
    assert re.search(r"scf\.for .*%arg\d+", src)
    assert_source_contains(src, ("suvm.tile.select", "suvm.tile.reduce  sum"))
    assert re.search(r"suvm\.tile\.reduce\s+sum, .*: !suvm\.tile<\d+x32xf32>", src)


def test_reduce_manual_destination_tileview_is_reflected_in_raw_suvm(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_manual_dst_tileview_kernel_builder(16),
        tmp_path,
        mlir_filename="reduce_manual_dst_tile_16_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "!suvm.tile<16x1xf32>"),
    )
    assert_source_contains(src, ("!suvm.tile<16x32xf32>", "!suvm.tile<16x1xf32>"))


def test_reduce_small_1d_result_uses_aligned_store_bridge(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder((32, 64, 256), 2, clear=True, tile_size=(8, 32)),
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

    definitions = {match.group(1): match.group(2) for line in src.splitlines() if (match := re.match(r"\s*(%[\w.]+)\s*=\s*(.+)", line))}
    source_views = [
        line
        for line in src.splitlines()
        if "suvm.get_partitioned_tile_view" in line
        and "!suvm.memtensor<4x128xf32" in line
        and "#suvm.memory_space<rsram>" in line
        and "-> !suvm.tile_view<4x32xf32>" in line
    ]
    destination_stores = [
        line for line in src.splitlines() if "suvm.tile.store" in line and "!suvm.tile<32xf32>, !suvm.tile_view<32xf32>" in line
    ]
    assert len(source_views) == 1, source_views
    assert len(destination_stores) == 1, destination_stores

    source_indices = re.search(r"indices = \[(%[\w.]+), (%[\w.]+)\]", source_views[0])
    assert source_indices, source_views[0]
    segment_index = source_indices.group(2)

    store_match = re.search(
        r"suvm\.tile\.store\s+%[\w.]+,\s+(%[\w.]+)\s+:",
        destination_stores[0],
    )
    assert store_match, destination_stores[0]
    destination_view = store_match.group(1)
    destination_view_definition = definitions.get(destination_view, "")
    assert "suvm.get_partitioned_tile_view" in destination_view_definition
    assert "!suvm.memtensor<128xf32" in destination_view_definition
    assert "-> !suvm.tile_view<32xf32>" in destination_view_definition

    index_match = re.search(r"indices = \[(%[\w.]+)\]", destination_view_definition)
    assert index_match, destination_view_definition
    destination_index = index_match.group(1)

    # Canonicalization may fold `(segment * 32) / 32` to `segment`.  Before
    # that fold, require the complete def-use chain instead of accepting an
    # unrelated `arith.divsi` result.
    if destination_index == segment_index:
        return

    div_match = re.fullmatch(
        r"arith\.divsi (%[\w.]+), (%[\w.]+) : index",
        definitions.get(destination_index, ""),
    )
    assert div_match, f"destination index is not defined by divsi: {destination_view_definition}"
    numerator, divisor = div_match.groups()
    assert definitions.get(divisor) == "arith.constant 32 : index"

    numerator_cast = re.fullmatch(
        r"arith\.index_cast (%[\w.]+) : i32 to index",
        definitions.get(numerator, ""),
    )
    assert numerator_cast, definitions.get(numerator)
    product = numerator_cast.group(1)

    multiply = re.fullmatch(
        r"arith\.muli (%[\w.]+), (%[\w.]+) : i32",
        definitions.get(product, ""),
    )
    assert multiply, definitions.get(product)
    lhs, rhs = multiply.groups()
    if definitions.get(lhs) == "arith.constant 32 : i32":
        factor, segment_i32 = lhs, rhs
    else:
        segment_i32, factor = lhs, rhs
    assert definitions.get(factor) == "arith.constant 32 : i32"
    assert definitions.get(segment_i32) == (f"arith.index_cast {segment_index} : index to i32"), (
        "The reduction destination must be indexed by "
        "(segment_index * 32) / 32.\n"
        f"source view: {source_views[0]}\n"
        f"destination view: {destination_view_definition}"
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
