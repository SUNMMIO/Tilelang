import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver

from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ["SUNMMIO_TEST_PRINT"] = "0"
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


def validate_sunmmio_codegen_loose(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=LOOSE_OPT_ARGS,
    )


def dot_mul_tiled_parallel_3d(
    batch=64,
    m=512,
    n=1024,
    block_b=2,
    block_m=256,
    block_n=128,
    dtype="bfloat16",
    accum_dtype="bfloat16",
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(batch, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, n), dtype),
        B: T.Tensor((batch, m, n), dtype),
        C: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), accum_dtype)

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        T.copy(
                            A[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                            A_shared,
                        )
                        T.copy(
                            B[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                            B_shared,
                        )

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                            B_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
                            C_shared[b, i, j] = T.exp(A_shared[b, i, j]) + T.exp(B_shared[b, i, j])

                        T.copy(
                            C_shared,
                            C[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )

    return main


def dot_mul_tiled_parallel_2d(
    m=512,
    n=1024,
    block_m=256,
    block_n=512,
    dtype="bfloat16",
    accum_dtype="bfloat16",
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.Tensor((m, n), dtype),
        B: T.Tensor((m, n), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_m, block_n), accum_dtype)

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
                    T.copy(
                        A[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                        A_shared,
                    )
                    T.copy(
                        B[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                        B_shared,
                    )

                    for i, j in T.Tiles(A_shared, parallel=True):
                        A_shared[i, j] = A_shared[i, j] * T.float32(2.0)
                        B_shared[i, j] = A_shared[i, j] * B_shared[i, j]
                        C_shared[i, j] = T.exp(A_shared[i, j]) + T.exp(B_shared[i, j])

                    T.copy(
                        C_shared,
                        C[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                    )

    return main


def tiles_broadcast(
    batch=64,
    m=512,
    n=1024,
    block_b=2,
    block_m=256,
    block_n=128,
    dtype="bfloat16",
    accum_dtype="bfloat16",
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(batch, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, n), dtype),
        B: T.Tensor((batch, m, n), dtype),
        C: T.Tensor((batch, m, n), dtype),
        D: T.Tensor((m,), dtype),
    ):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), accum_dtype)
            D_shared = T.alloc_shared((block_m,), dtype)

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        T.copy(
                            A[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                            A_shared,
                        )
                        T.copy(
                            B[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                            B_shared,
                        )
                        T.copy(D[by * block_m : (by + 1) * block_m], D_shared)

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            A_shared[b, i, j] = A_shared[b, i, j] + D_shared[i]
                            A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                            C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            A_shared[b, i, j] = A_shared[b, i, j] + D_shared[j]
                            A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                            C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

                        T.copy(
                            C_shared,
                            C[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )

    return main


def tiles_1d(m=512, block_m=256, dtype="bfloat16", accum_dtype="bfloat16"):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_m = T.ceildiv(m, block_m)

    @T.prim_func
    def main(
        A: T.Tensor((m,), dtype),
        B: T.Tensor((m,), dtype),
        C: T.Tensor((m,), dtype),
    ):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared((block_m,), dtype)
            B_shared = T.alloc_shared((block_m,), dtype)
            C_shared = T.alloc_shared((block_m,), accum_dtype)

            for by in T.serial(grid_m):
                T.clear(C_shared)
                T.copy(A[by * block_m : (by + 1) * block_m], A_shared)
                T.copy(B[by * block_m : (by + 1) * block_m], B_shared)
                for i in T.Tiles(A_shared, parallel=True):
                    C_shared[i] = A_shared[i] * B_shared[i]
                T.copy(C_shared, C[by * block_m : (by + 1) * block_m])

    return main


def test_dot_mul_tiled_parallel_2d_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        dot_mul_tiled_parallel_2d(),
        tmp_path,
        mlir_filename="dot_mul_tiled_parallel_2d_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.mulf", "suvm.tile.exp"),
    )
    assert_source_contains(src, ("suvm.tile.mulf", "suvm.tile.exp"))


@pytest.mark.parametrize(
    "case_name,kernel_factory,expected_tokens",
    [
        ("dot_mul_tiled_parallel_3d", dot_mul_tiled_parallel_3d, ("suvm.tile.mulf", "suvm.tile.exp")),
        ("tiles_broadcast", tiles_broadcast, ("suvm.tile.mulf",)),
        ("tiles_1d", tiles_1d, ("suvm.tile.mulf",)),
    ],
)
def test_tiles_codegen_validates_loose_with_npuir_opt(tmp_path, case_name, kernel_factory, expected_tokens):
    src = validate_sunmmio_codegen_loose(
        kernel_factory(),
        tmp_path,
        mlir_filename=f"{case_name}_suvm.mlir",
        expected_tokens=expected_tokens,
    )
    assert_source_contains(src, expected_tokens)


if __name__ == "__main__":
    tilelang.testing.main()
