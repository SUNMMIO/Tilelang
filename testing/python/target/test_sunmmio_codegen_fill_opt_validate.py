import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver

from compile_pipeline import target
from sunmmio_codegen_validation_utils import validate_sunmmio_codegen_with_npuir_opt


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
def fill_tiled_test(
    b=64,
    m=512,
    n=1024,
    block_b=16,
    block_m=256,
    block_n=128,
    dtype="float16",
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(b, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(A: T.Tensor((b, m, n), dtype)):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        T.fill(A_shared, T.float16(1.0))
                        T.fill(A_shared[0:block_b, 0 : block_m // 2, 0:block_n], T.float16(2.0))
                        T.fill(A_shared[0:block_b, block_m // 2 : block_m, 0:block_n], T.float16(3.0))
                        T.clear(A_shared)
                        T.copy(
                            A_shared,
                            A[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )

    return main


@target("Sunmmio")
def fill_tiled_2d_test(
    m=512,
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
    def main(A: T.Tensor((m, n), dtype)):
        with T.Kernel(ncores):
            A_shared = T.alloc_shared((block_m, block_n), dtype)

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
                    T.fill(A_shared, T.float16(1.0))
                    T.fill(A_shared[0 : block_m // 2, 0:block_n], T.float16(2.0))
                    T.fill(A_shared[block_m // 2 : block_m, 0:block_n], T.float16(3.0))
                    T.clear(A_shared)
                    T.copy(
                        A_shared,
                        A[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                    )

    return main


def test_fill_tiled_2d_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        fill_tiled_2d_test(),
        tmp_path,
        mlir_filename="fill_tiled_2d_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.fill"),
    )
    assert src.count("suvm.tile.fill") >= 4


def test_fill_tiled_codegen_validates_loose_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_loose(
        fill_tiled_test(),
        tmp_path,
        mlir_filename="fill_tiled_suvm.mlir",
        expected_tokens=("suvm.tile.fill",),
    )
    assert src.count("suvm.tile.fill") >= 4


if __name__ == "__main__":
    tilelang.testing.main()
