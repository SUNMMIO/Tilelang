import os

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
def tile_elementwise_ops_test(
    batch=2,
    m=256,
    n=128,
    block_b=2,
    block_m=256,
    block_n=128,
    dtype="float16",
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
            C_shared = T.alloc_shared((block_b, block_m, block_n), dtype)

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
                            x = T.max(A_shared[b, i, j], B_shared[b, i, j])
                            y = T.min(x, T.exp(B_shared[b, i, j]))
                            z = T.log(T.abs(y) + T.float32(1.0))
                            C_shared[b, i, j] = T.if_then_else(
                                z > T.float32(0.5),
                                T.rsqrt(z + T.float32(1.0)),
                                T.ieee_frcp(z + T.float32(2.0)),
                            )

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            neg_a = T.float32(0.0) - T.Cast("float32", A_shared[b, i, j])
                            rem_b = T.fmod(T.Cast("float32", B_shared[b, i, j]), T.float32(3.0))
                            C_shared[b, i, j] = (
                                T.ceil(C_shared[b, i, j])
                                + T.floor(A_shared[b, i, j])
                                + T.round(B_shared[b, i, j])
                                + T.trunc(A_shared[b, i, j])
                                + neg_a
                                + rem_b
                            )

                        T.copy(
                            C_shared,
                            C[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )

    return main


@target("Sunmmio")
def tile_elementwise_ops_2d_test(
    m=256,
    n=1024,
    block_m=256,
    block_n=512,
    dtype="float16",
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
            C_shared = T.alloc_shared((block_m, block_n), dtype)

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
                        x = T.max(A_shared[i, j], B_shared[i, j])
                        y = T.min(x, T.exp(B_shared[i, j]))
                        z = T.log(T.abs(y) + T.float32(1.0))
                        C_shared[i, j] = T.if_then_else(
                            z > T.float32(0.5),
                            T.rsqrt(z + T.float32(1.0)),
                            T.ieee_frcp(z + T.float32(2.0)),
                        )

                    for i, j in T.Tiles(A_shared, parallel=True):
                        neg_a = T.float32(0.0) - T.Cast("float32", A_shared[i, j])
                        rem_b = T.fmod(T.Cast("float32", B_shared[i, j]), T.float32(3.0))
                        C_shared[i, j] = (
                            T.ceil(C_shared[i, j])
                            + T.floor(A_shared[i, j])
                            + T.round(B_shared[i, j])
                            + T.trunc(A_shared[i, j])
                            + neg_a
                            + rem_b
                        )

                    T.copy(
                        C_shared,
                        C[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                    )

    return main


def test_tile_elementwise_ops_2d_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        tile_elementwise_ops_2d_test(),
        tmp_path,
        mlir_filename="tile_elementwise_ops_2d_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.store"),
    )
    assert_source_contains(
        src,
        (
            "suvm.tile.abs",
            "suvm.tile.ceil",
            "suvm.tile.cmpf",
            "suvm.tile.floor",
            "suvm.tile.ln",
            "suvm.tile.maxf",
            "suvm.tile.minf",
            "suvm.tile.neg",
            "suvm.tile.recip",
            "suvm.tile.remf",
            "suvm.tile.round",
            "suvm.tile.rsqrt",
            "suvm.tile.select",
            "suvm.tile.trunc",
        ),
    )


def test_tile_elementwise_ops_codegen_validates_loose_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_loose(
        tile_elementwise_ops_test(),
        tmp_path,
        mlir_filename="tile_elementwise_ops_suvm.mlir",
        expected_tokens=("suvm.tile.store",),
    )
    assert_source_contains(
        src,
        (
            "suvm.tile.abs",
            "suvm.tile.ceil",
            "suvm.tile.cmpf",
            "suvm.tile.floor",
            "suvm.tile.ln",
            "suvm.tile.maxf",
            "suvm.tile.minf",
            "suvm.tile.neg",
            "suvm.tile.recip",
            "suvm.tile.remf",
            "suvm.tile.round",
            "suvm.tile.rsqrt",
            "suvm.tile.select",
            "suvm.tile.trunc",
        ),
    )


if __name__ == "__main__":
    tilelang.testing.main()
