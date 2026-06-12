import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_row_major, make_zz_layout
from tilelang.tileview import make_tileview

from compile_pipeline import target
from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
# os.environ["SUNMMIO_TEST_PRINT"] = "0"
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


@target("Sunmmio")
def per_token_cast_to_fp8_kernel(
    M=1024,
    N=4096,
    block_M=32,
    group_size=128,
):
    dtype = T.float32
    fp8_min = -448.0
    fp8_max = 448.0
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, group_size)

    assert M % block_M == 0
    assert N % group_size == 0
    assert block_M == 32
    assert group_size == 128

    shard_policy = T.MeshShardingPolicy(x=1, y=0)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    X_shape = (M, N)
    X_amax_shape = (M, grid_n)
    X_layout = make_zz_layout(X_shape, [0, 1], (32, 32))
    X_fp8_layout = make_zz_layout(X_shape, [0, 1], (32, 32))
    X_amax_layout = make_zz_layout(X_amax_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        X: T.MeshTensor(X_shape, shard_policy, device_mesh_config, dtype, layout=X_layout),  # type: ignore
        X_fp8: T.MeshTensor(X_shape, shard_policy, device_mesh_config, T.float8_e4m3fn, layout=X_fp8_layout),  # type: ignore
        X_amax: T.MeshTensor(X_amax_shape, shard_policy, device_mesh_config, dtype, layout=X_amax_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            x_reduce_shared = T.alloc_shared((block_M, group_size), dtype)
            x_quant_shared = T.alloc_shared((block_M, group_size), dtype)
            x_fp8_shared = T.alloc_shared((block_M, group_size), T.float8_e4m3fn)
            amax_shared = T.alloc_shared((block_M,), dtype)
            scale_shared = T.alloc_shared((block_M,), dtype)

            for bx in T.serial(grid_m):
                for by in T.serial(grid_n):
                    T.copy(X[bx * block_M, by * group_size], x_reduce_shared)
                    T.copy(X[bx * block_M, by * group_size], x_quant_shared)
                    T.reduce_absmax(x_reduce_shared, amax_shared, dim=1)

                    for i in T.Tiles([block_M], parallel=True):
                        amax_shared[i] = T.max(amax_shared[i], T.float32(1e-4))
                        scale_shared[i] = amax_shared[i] / T.float32(fp8_max)

                    for i, j in T.Tiles([block_M, group_size], parallel=True):
                        q_scaled = x_quant_shared[i, j] / scale_shared[i]
                        q_clamped = T.clamp(q_scaled, T.float32(fp8_min), T.float32(fp8_max))
                        x_quant_shared[i, j] = q_clamped

                    for i, j in T.Tiles([block_M, group_size], parallel=True):
                        x_fp8_shared[i, j] = x_quant_shared[i, j]

                    T.copy(scale_shared, X_amax[bx * block_M : (bx + 1) * block_M, by : by + 1])
                    T.copy(x_fp8_shared, X_fp8[bx * block_M, by * group_size])

    return main


@target("Sunmmio")
def per_token_cast_to_fp8_kernel_rowmajor(
    M=1024,
    N=4096,
    block_M=32,
    group_size=128,
):
    dtype = T.float32
    fp8_min = -448.0
    fp8_max = 448.0
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, group_size)

    assert M % block_M == 0
    assert N % group_size == 0
    assert block_M == 32
    assert group_size == 128

    shard_policy = T.MeshShardingPolicy(x=1, y=0)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    X_shape = (M, N)
    X_amax_shape = (M, grid_n)
    X_layout = make_zz_layout(X_shape, [0, 1], (32, 32))
    X_fp8_layout = make_zz_layout(X_shape, [0, 1], (32, 32))
    #
    X_amax_layout = make_row_major(X_amax_shape)

    @T.prim_func
    def main(
        X: T.MeshTensor(X_shape, shard_policy, device_mesh_config, dtype, layout=X_layout),  # type: ignore
        X_fp8: T.MeshTensor(X_shape, shard_policy, device_mesh_config, T.float8_e4m3fn, layout=X_fp8_layout),  # type: ignore
        X_amax: T.MeshTensor(X_amax_shape, shard_policy, device_mesh_config, dtype, layout=X_amax_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            x_reduce_shared = T.alloc_shared((block_M, group_size), dtype)
            x_quant_shared = T.alloc_shared((block_M, group_size), dtype)
            x_fp8_shared = T.alloc_shared((block_M, group_size), T.float8_e4m3fn)
            amax_shared = T.alloc_shared((block_M,), dtype)
            scale_shared = T.alloc_shared((block_M,), dtype)

            for bx in T.serial(grid_m):
                for by in T.serial(grid_n):
                    T.copy(X[bx * block_M, by * group_size], x_reduce_shared)
                    T.copy(X[bx * block_M, by * group_size], x_quant_shared)
                    T.reduce_absmax(x_reduce_shared, amax_shared, dim=1)

                    for i in T.Tiles([block_M], parallel=True):
                        amax_shared[i] = T.max(amax_shared[i], T.float32(1e-4))
                        scale_shared[i] = amax_shared[i] / T.float32(fp8_max)

                    for i, j in T.Tiles([block_M, group_size], parallel=True):
                        q_scaled = x_quant_shared[i, j] / scale_shared[i]
                        q_clamped = T.clamp(q_scaled, T.float32(fp8_min), T.float32(fp8_max))
                        x_quant_shared[i, j] = q_clamped

                    for i, j in T.Tiles([block_M, group_size], parallel=True):
                        x_fp8_shared[i, j] = x_quant_shared[i, j]

                    T.copy(scale_shared, X_amax[bx * block_M : (bx + 1) * block_M, by : by + 1])
                    T.copy(x_fp8_shared, X_fp8[bx * block_M, by * group_size])

    return main


@target("Sunmmio")
def per_token_cast_to_fp8_transition_kernel(
    M=1024,
    N=4096,
    block_M=32,
    group_size=128,
):
    dtype = T.float32
    fp8_min = -448.0
    fp8_max = 448.0
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, group_size)

    assert M % block_M == 0
    assert N % group_size == 0
    assert block_M == 32
    assert group_size == 128

    shard_policy = T.MeshShardingPolicy(x=1, y=0)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    X_shape = (M, N)
    X_amax_shape = (M, grid_n)
    X_layout = make_zz_layout(X_shape, [0, 1], (32, 32))
    X_fp8_layout = make_zz_layout(X_shape, [0, 1], (32, 64))
    X_amax_layout = make_zz_layout(X_amax_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        X: T.MeshTensor(X_shape, shard_policy, device_mesh_config, dtype, layout=X_layout),  # type: ignore
        X_fp8: T.MeshTensor(X_shape, shard_policy, device_mesh_config, T.float8_e4m3fn, layout=X_fp8_layout),  # type: ignore
        X_amax: T.MeshTensor(X_amax_shape, shard_policy, device_mesh_config, dtype, layout=X_amax_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            x_reduce_shared = T.alloc_shared((block_M, group_size), dtype)
            x_quant_shared = T.alloc_shared((block_M, group_size), dtype)
            x_cast_input_shared = T.alloc_shared((block_M, group_size), dtype)
            x_cast_shared = T.alloc_shared((block_M, group_size), dtype)
            x_fp8_shared = T.alloc_shared((block_M, group_size), T.float8_e4m3fn)
            amax_shared = T.alloc_shared((block_M,), dtype)
            scale_shared = T.alloc_shared((block_M,), dtype)

            T.annotate_layout(
                {
                    x_reduce_shared: make_zz_layout(x_reduce_shared),
                    x_quant_shared: make_zz_layout(x_quant_shared),
                    x_cast_input_shared: make_zz_layout(x_cast_input_shared, block_shape=(32, 64)),
                    x_cast_shared: make_zz_layout(x_cast_shared, block_shape=(32, 64)),
                    x_fp8_shared: make_zz_layout(x_fp8_shared, block_shape=(32, 64)),
                }
            )
            T.annotate_tileview(
                {
                    x_reduce_shared: make_tileview(x_reduce_shared, (4, 32), (-2, -1)),
                    x_quant_shared: make_tileview(x_quant_shared, (4, 32), (-2, -1)),
                    x_cast_input_shared: make_tileview(x_cast_input_shared, (2, 64), (-2, -1)),
                    x_cast_shared: make_tileview(x_cast_shared, (2, 64), (-2, -1)),
                    x_fp8_shared: make_tileview(x_fp8_shared, (2, 64), (-2, -1)),
                }
            )

            for bx in T.serial(grid_m):
                for by in T.serial(grid_n):
                    T.copy(X[bx * block_M, by * group_size], x_reduce_shared)
                    T.copy(X[bx * block_M, by * group_size], x_quant_shared)
                    T.copy(X[bx * block_M, by * group_size], x_cast_input_shared)
                    T.reduce_absmax(x_reduce_shared, amax_shared, dim=1)

                    for i in T.Tiles([block_M], parallel=True):
                        amax_shared[i] = T.max(amax_shared[i], T.float32(1e-4))
                        scale_shared[i] = amax_shared[i] / T.float32(fp8_max)

                    for i, j in T.Tiles([block_M, group_size], parallel=True):
                        q_scaled = x_quant_shared[i, j] / scale_shared[i]
                        q_clamped = T.clamp(q_scaled, T.float32(fp8_min), T.float32(fp8_max))
                        x_quant_shared[i, j] = q_clamped

                    for i, j in T.Tiles([block_M, group_size], parallel=True):
                        q_scaled = x_cast_input_shared[i, j] / scale_shared[i]
                        q_clamped = T.clamp(q_scaled, T.float32(fp8_min), T.float32(fp8_max))
                        x_cast_shared[i, j] = q_clamped
                        x_fp8_shared[i, j] = x_cast_shared[i, j]

                    T.copy(scale_shared, X_amax[bx * block_M : (bx + 1) * block_M, by : by + 1])
                    T.copy(x_fp8_shared, X_fp8[bx * block_M, by * group_size])

    return main


def test_per_token_cast_to_fp8_transition_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        # per_token_cast_to_fp8_transition_kernel(),
        per_token_cast_to_fp8_kernel_rowmajor(),
        # per_token_cast_to_fp8_kernel(),
        tmp_path,
        mlir_filename="per_token_cast_to_fp8_transition_suvm.mlir",
        expected_tokens=(
            # "suvm.copy_async",
            # "suvm.tile.reduce",
            # "suvm.tile.maxf",
            # "suvm.tile.minf",
            # "suvm.tile.cast",
            # "f8E4M3FN",
        ),
        opt_args=("--verify-each",),
    )
    assert_source_contains(
        src,
        (
            # "suvm.copy_async",
            # "suvm.tile.reduce",
            # "suvm.tile.cast",
            # "f8E4M3FN",
        ),
    )


if __name__ == "__main__":
    tilelang.testing.main()
