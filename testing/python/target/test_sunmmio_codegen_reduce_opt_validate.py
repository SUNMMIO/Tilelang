import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver

from compile_pipeline import target
from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


REDUCE_IN_TILE_CASES = [
    ((256, 128), 1, True),
    ((256, 128), 1, False),
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


@target("Sunmmio")
def reduce_kernel_builder(shape, reduce_axis, dtype="float16", clear=True):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis, clear=clear)
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
    dtype="float16",
    clear=False,
):
    out_shape_full = (b, m) if reduce_axis == 2 else (b, n) if reduce_axis == 1 else (m, n)
    out_shape_block = (block_b, block_m) if reduce_axis == 2 else (block_b, block_n) if reduce_axis == 1 else (block_m, block_n)

    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(b, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(A: T.Tensor((b, m, n), dtype), Out: T.Tensor(out_shape_full, dtype)):
        with T.Kernel(ncores):
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
                            T.copy(
                                A[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                A_shared,
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
                            T.copy(
                                A[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                A_shared,
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
                            T.copy(
                                A[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                A_shared,
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
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder(shape, reduce_axis, clear=clear),
        tmp_path,
        mlir_filename=f"reduce_shape_{shape_label}_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.reduce"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "sum"))


@pytest.mark.parametrize("reduce_axis,clear", [(1, False), (2, True)])
def test_reduce_tiled_in_tile_codegen_generates_expected_ops(tmp_path, reduce_axis, clear):
    src = validate_sunmmio_codegen_loose(
        reduce_tiled_test(reduce_axis=reduce_axis, clear=clear),
        tmp_path,
        mlir_filename=f"reduce_tiled_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.reduce"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "sum"))


if __name__ == "__main__":
    tilelang.testing.main()
